

"""Compute double-delta energies (ddE) from all-to-all RMSD-matched conformers.

This script analyzes conformational energy landscapes by comparing MM and QM
energy differences relative to their respective minimum energy conformers.
For each molecule and force field, it identifies which conformers have low
RMSD matches between MM and QM geometries, then computes ddE values.

ddE (double-delta energy):
    ddE = (E_MM - E_MM_min) - (E_QM - E_QM_min)

Where:
- E_MM: MM energy for a specific conformer
- E_MM_min: MM energy of the MM-optimized lowest-energy conformer
- E_QM: QM energy for the RMSD-matched QM reference conformer
- E_QM_min: QM energy of the QM lowest-energy conformer

Workflow:
1. Load all-to-all RMSD data (MM conformer matched to QM conformer)
2. Filter pairs by RMSD threshold (default 0.3 Å)
3. For each molecule:
   - Find MM minimum energy conformer
   - Find QM minimum energy conformer
   - Compute ddE for all other conformers relative to these minima
4. Write ddE results to parquet files (one per force field)

Inputs:
- all-to-all-rmsd/: Parquet dataset with MM-QM conformer pairs and RMSDs
  Required columns: ff_qcarchive_id, qm_qcarchive_id, ff_energy, qm_energy,
  rmsd, method, mapped_smiles, inchi

Outputs:
- ddEs/{force-field-name}.parquet: ddE values for each force field
  Columns: inchi, ff_qcarchive_id, qm_qcarchive_id, ff_min_qcarchive_id,
  qm_min_qcarchive_id, ddE, ff_de, qm_de, method, Force field, n_conformers
"""
import pathlib
import sys
import click

from loguru import logger

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def get_ddEs(
    df: pd.DataFrame,
) -> list[dict]:
    """Calculate ddE values for all conformers of a single force field.
    
    For each molecule, identifies the MM and QM minimum energy conformers,
    then computes double-delta energies for all other conformers:
        ddE = (E_MM - E_MM_min) - (E_QM - E_QM_min)
    
    A positive ddE indicates the force field overestimates the relative energy
    of that conformer compared to QM. A negative ddE indicates underestimation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame for a single force field containing:
        - Force field: Force field name
        - method: Force field stem name
        - inchi: Molecule InChI identifier
        - ff_qcarchive_id: QCArchive ID of MM-optimized conformer
        - qm_qcarchive_id: QCArchive ID of matched QM conformer
        - ff_energy: MM energy (kcal/mol)
        - qm_energy: QM energy (kcal/mol)

    Returns
    -------
    list[dict]
        ddE entries, each containing:
        - inchi: Molecule identifier
        - ff_qcarchive_id: MM conformer ID
        - qm_qcarchive_id: Matched QM conformer ID
        - ff_min_qcarchive_id: MM minimum energy conformer ID
        - qm_min_qcarchive_id: QM minimum energy conformer ID
        - ddE: Double-delta energy (kcal/mol)
        - ff_de: MM energy difference from minimum (kcal/mol)
        - qm_de: QM energy difference from minimum (kcal/mol)
        - method: Force field stem name
        - Force field: Force field display name
        - n_conformers: Number of conformers for this molecule
    """
    assert len(df["Force field"].unique()) == 1, "DataFrame should only contain one force field"
    ff = df["Force field"].unique()[0]
    method = df["method"].unique()[0]

    entries: list[dict] = []

    # Process each molecule (grouped by InChI).
    for inchi, subdf in df.groupby("inchi"):
        # Find QM minimum energy conformer.
        lowest_qm_energy_idx: int = subdf["qm_energy"].idxmin()
        lowest_qm_energy: float = subdf.loc[lowest_qm_energy_idx, "qm_energy"]
        lowest_energy_qm_id: int = subdf.loc[lowest_qm_energy_idx, "qm_qcarchive_id"]
        
        # Find MM minimum energy conformer.
        # We need the MM energy at the QM minimum, then find the actual MM minimum.
        lowest_ff_energy: float = subdf.loc[
            subdf.qm_qcarchive_id == lowest_energy_qm_id,
            "ff_energy"
        ].values.min()
        assert len(subdf[subdf.ff_energy == lowest_ff_energy]) == 1, \
            "There should be only one conformer with the lowest energy"
        lowest_energy_ff_id: int = subdf[subdf.ff_energy == lowest_ff_energy]["ff_qcarchive_id"].values[0]
        
        logger.info(
            f"Lowest energy conformer ID for {inchi}: {lowest_energy_ff_id} ({ff}) -> "
            f"{lowest_energy_qm_id} (QM) "
            f"with QM energy {lowest_qm_energy:.3f} kcal/mol"
        )

        # Compute ddE for all conformers relative to the minima.
        for _, row in subdf.iterrows():
            # Skip self-to-self comparisons (minimum energy conformer).
            if row["ff_qcarchive_id"] == lowest_energy_ff_id:
                continue
            mm_de = row["ff_energy"] - lowest_ff_energy
            qm_de = row["qm_energy"] - lowest_qm_energy
            entry = {
                "inchi": inchi,
                "ff_qcarchive_id": row["ff_qcarchive_id"],
                "qm_qcarchive_id": row["qm_qcarchive_id"],
                "ff_min_qcarchive_id": lowest_energy_ff_id,
                "qm_min_qcarchive_id": lowest_energy_qm_id,
                "ddE": mm_de - qm_de,
                "ff_de": mm_de,
                "qm_de": qm_de,
                "method": method,
                "Force field": ff,
                "n_conformers": subdf.shape[0]
            }
            entries.append(entry)
    return entries


@click.command()
@click.option(
    "--ff-name-and-stem",
    "-ff",
    "ff_name_and_stems",
    type=(str, str),
    multiple=True,
    default=None,
    required=False,
    help=(
        "Force field name and stem pairs for custom labeling. "
        "First arg: display name, second arg: stem (e.g., 'openff_unconstrained-2.2.1'). "
        "Can be specified multiple times. If omitted, all force fields in input "
        "directory are included with stems as labels."
    )
)
@click.option(
    "--input-directory",
    "-i",
    "input_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="all-to-all-rmsd",
    help="Directory with all-to-all RMSD parquet data (from get-all-to-all-rmsds-and-tfds.py).",
)
@click.option(
    "--rmsd-threshold",
    "-t",
    "rmsd_threshold",
    type=float,
    default=0.3,
    help="Heavy atom RMSD threshold (Angstrom) for filtering MM-QM conformer pairs.",
)
@click.option(
    "--output-directory",
    "-o",
    "output_directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="ddEs",
    help="Directory to write ddE parquet files (one per force field).",
)
def main(
    ff_name_and_stems: list[tuple[str, str]],
    input_directory: str = "all-to-all-rmsd",
    rmsd_threshold: float = 0.3,
    output_directory: str = "ddEs",
):
    """Compute double-delta energies (ddE) from RMSD-matched conformer pairs.
    
    Loads all-to-all RMSD data, filters by RMSD threshold, identifies minimum
    energy conformers for MM and QM, then computes ddE for all conformers.
    """
    # Load all-to-all RMSD data.
    dataset = ds.dataset(input_directory)
    logger.info(f"Loaded {dataset.count_rows()} rows from {input_directory}")

    # Set up force field name mapping (stem -> display name).
    FF_STEM_TO_NAME = {}
    if ff_name_and_stems:
        FF_STEM_TO_NAME = {
            stem: name for name, stem in ff_name_and_stems
        }
        stem_to_name_str = ", ".join([
            f"{stem} -> {name}"
            for stem, name in FF_STEM_TO_NAME.items()
        ])
        logger.info(f"Mapping {stem_to_name_str}")
    else:
        # Use stem names as display names if no custom mapping provided.
        FF_STEM_TO_NAME = {
            stem: stem
            for stem in set(
                dataset.to_table(columns=["method"]).to_pydict()["method"]
            )
        }

    # Filter dataset by RMSD threshold.
    filtered_dataset = dataset.filter(
        pc.field("rmsd") < rmsd_threshold
    )
    logger.info(f"Filtered dataset to {filtered_dataset.count_rows()} rows with RMSD < {rmsd_threshold} A")

    # Convert to pandas and map force field names.
    df = filtered_dataset.to_table(
        columns=[
            "mapped_smiles",
            "ff_qcarchive_id", "ff_energy",
            "qm_qcarchive_id", "qm_energy",
            "inchi", "method"
        ]
    ).to_pandas()
    logger.info(f"Found unique methods: {', '.join(df.method.unique())}")
    df["Force field"] = df["method"].map(FF_STEM_TO_NAME)
    unique_ffs = df["Force field"].unique()
    logger.info(f"Found unique FFs: {', '.join(map(str, unique_ffs))}")

    # Log entry counts per force field.
    counts = df.groupby("Force field").size().reset_index(name='count')
    count_str = ", ".join(
        f"{row['Force field']}: {row['count']}" for _, row in counts.iterrows()
    )
    logger.info(f"Counts of entries per force field -- {count_str}")

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Compute and write ddE values for each force field.
    for ff_name, subdf in df.groupby("Force field"):
        entries = get_ddEs(subdf)
        table = pa.Table.from_pylist(entries)
        table_file = output_directory / f"{ff_name}.parquet"
        pq.write_table(table, table_file)
        logger.info(f"Wrote {len(entries)} ddE entries for {ff_name} to {table_file}")


if __name__ == "__main__":
    main()
