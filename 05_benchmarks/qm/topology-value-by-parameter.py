"""Label topology values by force field parameter ID and compute MM vs QM differences.

This script processes computed topology values (e.g., bond lengths, angles, torsions)
for molecules optimized with both QM and MM methods. It assigns parameter IDs based
on SMIRKS patterns from a reference force field, then computes the difference between
MM-optimized and QM-reference values for each internal coordinate.

Workflow:
1. Load topology values from parquet dataset (output of compute-topology-values.py)
2. Load parameter labels (SMIRKS assignments) from reference force field
3. For each force field in the dataset:
   - Match each topology instance to its parameter ID via SMILES + atom indices
   - Retrieve corresponding QM reference value
   - Compute difference (MM - QM), handling angle periodicity
   - Write labeled results with differences to parquet files

Inputs:
- topology-values/: Parquet dataset with MM and QM topology values
- topology-labels/: Parquet dataset mapping (SMILES, atom_indices) → parameter_id

Outputs:
- Parquet files with columns: [original columns] + parameter_id, reference_value,
  difference, abs_difference

Note: Angles and torsions use periodic difference computation to handle wraparound
(e.g., 359° and 1° are 2° apart, not 358°).
"""
from collections import defaultdict
import typing
import pathlib

import click
import tqdm

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq

def compute_angle_difference(mm_value, qm_value):
    """Compute periodic angle difference in degrees.
    
    Handles wraparound at 360°/0° boundary. For example, 359° and 1°
    have a difference of 2°, not 358°.
    
    Parameters
    ----------
    mm_value : float
        MM-computed angle in degrees.
    qm_value : float
        QM reference angle in degrees.
    
    Returns
    -------
    float
        Signed difference in degrees, constrained to [-180, 180].
    """
    difference = mm_value - qm_value
    if difference > 180:
        difference -= 360
    if difference < -180:
        difference += 360
    return difference


@click.command()
@click.option(
    "--input-directory",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="topology-values",
    help="Directory with computed topology values (parquet dataset from compute-topology-values.py).",
)
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="topology-labeled-by-parameter",
    help="Directory to write labeled parquet files with MM-QM differences.",
)
@click.option(
    "--topology-group",
    "-t",
    type=click.Choice(["Bonds", "Angles", "ProperTorsions", "ImproperTorsions"], case_sensitive=True),
    default="Bonds",
    help="Type of internal coordinate to process.",
)
@click.option(
    "--input-labels-directory",
    "-l",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="topology-labels/fb-fit-v3-single-mean-k100_unconstrained/",
    help="Directory with parameter labels mapping (SMILES, atom indices) to parameter IDs.",
)
@click.option(
    "--forcefield",
    "-ff",
    "forcefields",
    multiple=True,
    type=str,
    help="Specific force field(s) to process. If omitted, processes all non-QM methods.",
)
def main(
    input_directory: str = "topology-values",
    output_directory: str = "topology-labeled-by-parameter",
    topology_group: typing.Literal["Bonds", "Angles", "ProperTorsions", "ImproperTorsions"] = "Bonds",
    input_labels_directory: str = "topology-labels/fb-fit-v3-single-mean-k100_unconstrained/",
    forcefields: list | None = None,
):
    """Label topology values by parameter ID and compute MM-QM differences.
    
    Assigns parameter IDs to each topology instance based on SMILES and atom
    indices, then computes differences between MM and QM values.
    """
    # Load topology values filtered by type (Bonds, Angles, etc.).
    dataset = ds.dataset(input_directory).filter(
        pc.field("topology") == topology_group
    )
    
    # Load parameter labels: (mapped_smiles, atom_indices) → parameter_id.
    labels = ds.dataset(input_labels_directory).filter(
        pc.field("topology") == topology_group
    )
    
    # Build lookup: smiles → {atom_indices: parameter_id}.
    ALL_LABELS = defaultdict(dict)
    for row in labels.to_table().to_pylist():
        ALL_LABELS[row["mapped_smiles"]][tuple(row["atom_indices"])] = row["parameter_id"]

    # Determine which force fields to process.
    if not forcefields:
        forcefields = dataset.to_table(
            columns=["method"]
        ).to_pydict()["method"]
        forcefields = sorted(set(forcefields) - {"qm"})

    # Build QM reference lookup: qcarchive_id → {atom_indices: value}.
    qm_pylist = dataset.filter(
        pc.field("method") == "qm"
    ).to_table().to_pylist()
    qm_pylist_by_smiles = defaultdict(dict)
    for row in qm_pylist:
        qm_pylist_by_smiles[row["qcarchive_id"]][tuple(row["atom_indices"])] = row["value"]

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Process each force field.
    for ff in forcefields:
        ffdir = output_directory / ff
        ffdir.mkdir(parents=True, exist_ok=True)

        subset = dataset.filter(
            pc.field("method") == ff
        )
        entries = []
        for row in tqdm.tqdm(subset.to_table().to_pylist(), desc=f"Labeling {topology_group} for {ff}"):
            parameter_labels_for_molecule = ALL_LABELS[row["mapped_smiles"]]
            qm_values_for_molecule = qm_pylist_by_smiles[row["qcarchive_id"]]
            value = row["value"]
            
            # Improper torsions: reorder atom indices with central atom second.
            if topology_group == "ImproperTorsions":
                central = row["atom_indices"][1]
                others = sorted([idx for idx in row["atom_indices"] if idx != central])
                atom_indices = tuple([others[0], central, others[1], others[2]])
            else:
                atom_indices = tuple(row["atom_indices"])

            parameter_id = parameter_labels_for_molecule[atom_indices]
            qm_value = qm_values_for_molecule[atom_indices]
            
            # Compute difference (handle periodicity for angles/torsions).
            entry = dict(row)
            if topology_group == "Bonds":
                difference = value - qm_value
            else:
                difference = compute_angle_difference(value, qm_value)
            entry.update(
                {
                    "parameter_id": parameter_id,
                    "reference_value": qm_value,
                    "difference": difference,
                    "abs_difference": abs(difference),
                }
            )
            entries.append(entry)
        
        # Write labeled results to parquet.
        table = pa.Table.from_pylist(entries)
        filename = ffdir / f"{topology_group}.parquet"
        pq.write_table(table, filename)
        print(f"Wrote {len(entries)} entries to {filename}")

if __name__ == "__main__":
    main()
