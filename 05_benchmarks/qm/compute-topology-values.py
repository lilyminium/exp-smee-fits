"""Compute internal coordinate values from molecular conformations.

This script extracts bond lengths, angles, proper torsions, and improper
torsions from molecular geometries and saves them to parquet files for
analysis. Uses MDAnalysis for efficient coordinate calculations and supports
distributed processing with Dask.

Workflow:
1. Load molecular conformations from parquet dataset (QM or MM-optimized)
2. For each molecule:
   - Create MDAnalysis universe from coordinates
   - Compute bond lengths (Angstrom)
   - Compute angles (degrees)
   - Compute proper torsions (degrees)
   - Compute improper torsions (degrees, with special ordering)
3. Write results to parquet files for downstream analysis

Inputs:
- Parquet dataset with columns: qcarchive_id, mapped_smiles, coordinates, method
- coordinates: flattened array in Angstrom

Outputs:
- Parquet files with topology values:
  - qcarchive_id: Molecule identifier
  - mapped_smiles: Mapped SMILES string
  - method: QM or force field name
  - topology: Type (Bonds, Angles, ProperTorsions, ImproperTorsions)
  - atom_indices: Atom indices defining the internal coordinate
  - value: Computed value (Angstrom for bonds, degrees for angles/torsions)

Note: Improper torsions use OpenFF ordering convention (central atom second),
whereas MDAnalysis may use different ordering.
"""
import pathlib
import typing
import click
import tqdm
import time
import sys
from loguru import logger

from click_option_group import optgroup

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import MDAnalysis as mda
from openff.toolkit import Molecule

logger.remove()
logger.add(sys.stdout)

# Map MDAnalysis topology group names to OpenFF handler names.
MDA_TO_OPENFF = {
    "bonds": "Bonds",
    "angles": "Angles",
    "dihedrals": "ProperTorsions",
    "impropers": "ImproperTorsions",
}

def compute_single(
    row,
):
    """Compute all internal coordinate values for a single molecule.
    
    Uses MDAnalysis to compute bond lengths, angles, and torsions from
    the molecular coordinates. Improper torsions are handled specially
    to match OpenFF ordering conventions.
    
    Parameters
    ----------
    row : dict
        Molecule data containing:
        - qcarchive_id: Molecule identifier
        - mapped_smiles: Mapped SMILES string
        - coordinates: Flattened array of coordinates (Angstrom)
        - method: QM or force field name
    
    Returns
    -------
    list[dict]
        List of topology value entries, each containing:
        - qcarchive_id, mapped_smiles, method (copied from input)
        - topology: Handler type (Bonds/Angles/ProperTorsions/ImproperTorsions)
        - atom_indices: Atom indices defining the coordinate
        - value: Computed value (Angstrom or degrees)
    """
    """Compute all internal coordinate values for a single molecule.
    
    Uses MDAnalysis to compute bond lengths, angles, and torsions from
    the molecular coordinates. Improper torsions are handled specially
    to match OpenFF ordering conventions.
    
    Parameters
    ----------
    row : dict
        Molecule data containing:
        - qcarchive_id: Molecule identifier
        - mapped_smiles: Mapped SMILES string
        - coordinates: Flattened array of coordinates (Angstrom)
        - method: QM or force field name
    
    Returns
    -------
    list[dict]
        List of topology value entries, each containing:
        - qcarchive_id, mapped_smiles, method (copied from input)
        - topology: Handler type (Bonds/Angles/ProperTorsions/ImproperTorsions)
        - atom_indices: Atom indices defining the coordinate
        - value: Computed value (Angstrom or degrees)
    """
    from openff.units import unit
    
    # Create OpenFF molecule and set coordinates.
    mol = Molecule.from_mapped_smiles(
        row["mapped_smiles"],
        allow_undefined_stereo=True
    )
    positions = np.array(row["coordinates"]).reshape((-1, 3))
    mol._conformers = [positions * unit.angstrom]

    # Base entry with metadata copied to all topology values.
    base_entry = {
        "qcarchive_id": row["qcarchive_id"],
        "mapped_smiles": row["mapped_smiles"],
        "method": row["method"]
    }

    # Create MDAnalysis universe for coordinate calculations.
    u = mda.Universe(mol.to_rdkit(), to_guess=["angles", "dihedrals"])
    topology_groups = ["bonds", "angles", "dihedrals"]

    # Compute standard topology values (bonds, angles, proper torsions).
    entries = []
    for topology_group in topology_groups:
        group = getattr(u, topology_group)
        if not len(group):
            continue
        atom_indices = group.indices
        values = group.values()
        # Convert angles and torsions from radians to degrees.
        if topology_group != "bonds":
            values = np.rad2deg(values)
        for ix, val in zip(atom_indices, values):
            entry = dict(base_entry)
            entry.update(
                {
                    "topology": MDA_TO_OPENFF[topology_group],
                    "atom_indices": list(ix),
                    "value": val,
                }
            )
            entries.append(entry)

    # Handle improper torsions with OpenFF ordering (central atom second).
    # MDAnalysis may use different ordering, so we reorder to match OpenFF.
    seen_impropers = set()
    for atoms in mol.smirnoff_impropers:
        central = atoms[1].molecule_atom_index
        others = tuple(
            sorted([
                atoms[0].molecule_atom_index,
                atoms[2].molecule_atom_index,
                atoms[3].molecule_atom_index
            ])
        )
        openff_key = (others[0], central, others[1], others[2])
        mda_key = [central] + list(others)
        if openff_key in seen_impropers:
            continue
        seen_impropers.add(openff_key)
        value = np.rad2deg(u.atoms[mda_key].improper.value())
        entry = dict(base_entry)
        entry.update(
            {
                "topology": "ImproperTorsions",
                "atom_indices": list(openff_key),
                "value": value,
            }
        )
        entries.append(entry)

    return entries




def batch_optimize(
    qcarchive_ids: list[str],
    data_directory: str,
):
    """Compute topology values for a batch of molecules.
    
    Filters dataset to specified QCArchive IDs and computes all internal
    coordinate values for distributed processing.
    
    Parameters
    ----------
    qcarchive_ids : list[str]
        QCArchive IDs to process in this batch.
    data_directory : str
        Path to parquet dataset with molecular coordinates.
    
    Returns
    -------
    list[dict]
        Flattened list of all topology value entries from all molecules.
    """
    dataset = ds.dataset(data_directory)
    subset = dataset.filter(
        pc.field("qcarchive_id").isin(qcarchive_ids)
    )
    rows = subset.to_table(
        columns=["qcarchive_id", "mapped_smiles", "coordinates", "method"]
    ).to_pylist()

    entries = []
    for row in tqdm.tqdm(rows, desc=f"Computing {len(rows)} molecules"):
        entries.extend(compute_single(row))
    return entries



@click.command()
@click.option(
    "--input-directory",
    "-i",
    "input_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="data",
    help="Directory with parquet dataset of molecular coordinates.",
)
@click.option(
    "--output-directory",
    "-o",
    "output_directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="topology-values",
    help="Directory to write topology value parquet files.",
)
@optgroup.group("Parallelization configuration")
@optgroup.option(
    "--n-workers",
    help="Number of parallel workers. Use -1 for one worker per batch.",
    type=int,
    default=1,
    show_default=True,
)
@optgroup.option(
    "--worker-type",
    help="Execution backend: local, SLURM, or LSF cluster.",
    type=click.Choice(["lsf", "local", "slurm"]),
    default="local",
    show_default=True,
)
@optgroup.option(
    "--batch-size",
    help="Number of molecules per batch (distributed to each worker).",
    type=int,
    default=500,
    show_default=True,
)
@optgroup.group("Cluster configuration", help="Options for cluster workers.")
@optgroup.option(
    "--memory",
    help="Memory per worker in GB.",
    type=int,
    default=3,
    show_default=True,
)
@optgroup.option(
    "--walltime",
    help="Maximum wall-clock time per worker in hours.",
    type=int,
    default=2,
    show_default=True,
)
@optgroup.option(
    "--queue",
    help="SLURM/LSF queue/partition name.",
    type=str,
    default="cpuqueue",
    show_default=True,
)
@optgroup.option(
    "--conda-environment",
    help="Conda environment name for cluster workers.",
    type=str,
)
def main(
    method_name: str = None,
    input_directory: str = "data",
    output_directory: str = "topology-values",
    worker_type: typing.Literal["slurm", "local"] = "local",
    queue: str = "free",
    conda_environment: str = "ib-dev",
    memory: int = 4,  # GB
    walltime: int = 32,  # hours
    batch_size: int = 300,
    n_workers: int = -1,
):
    """Compute internal coordinate values from molecular conformations.
    
    Loads molecular geometries, computes bond/angle/torsion values using
    MDAnalysis, and writes results to parquet files.
    """
    from openff.nagl.utils._parallelization import batch_distributed
    from dask import distributed

    logger.info(f"{time.ctime()} - Starting topology value computation")
    start_time = time.time()

    # Load input dataset and optionally filter by method.
    input_directory = pathlib.Path(input_directory)
    input_dataset = ds.dataset(input_directory)
    if method_name:
        input_dataset = input_dataset.filter(
            pc.field("method") == method_name
        )
    logger.info(f"Loaded {input_dataset.count_rows()} rows from {input_directory}")
    
    # Extract QCArchive IDs to process.
    input_qcarchive_ids = input_dataset.to_table(
        columns=["qcarchive_id"]
    ).to_pydict()["qcarchive_id"]
    input_qcarchive_ids = set(input_qcarchive_ids)

    # Set up output directory and skip already-processed molecules.
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_dataset = ds.dataset(output_directory)
    n_files = 0
    if method_name and output_dataset.count_rows():
        output_dataset = output_dataset.filter(
            pc.field("method") == method_name
        )
    if output_dataset.count_rows():
        existing_qcarchive_ids = output_dataset.to_table(
            columns=["qcarchive_id"]
        ).to_pydict()["qcarchive_id"]
        logger.info(f"Found {len(existing_qcarchive_ids)} existing results in {output_directory}")
        input_qcarchive_ids -= set(existing_qcarchive_ids)
        logger.info(f"Remaining to process: {len(input_qcarchive_ids)} molecules")
        n_files = len(output_dataset.files)

    input_qcarchive_ids = sorted(input_qcarchive_ids)

    # Distribute computation across workers.
    with batch_distributed(
        input_qcarchive_ids,
        batch_size=batch_size,
        worker_type=worker_type,
        queue=queue,
        conda_environment=conda_environment,
        memory=memory,
        walltime=walltime,
        n_workers=n_workers,
    ) as batcher:
        futures = list(batcher(
            batch_optimize,
            data_directory=str(input_directory.resolve()),
        ))
        for future in tqdm.tqdm(
            distributed.as_completed(futures, raise_errors=False),
            total=len(futures),
            desc="Calculating topology batches",
        ):
            entries = future.result()
            if len(entries):
                table = pa.Table.from_pylist(entries)
                table_file = output_directory / f"batch-{n_files:04d}.parquet"
                pq.write_table(table, table_file)
                logger.info(f"Wrote {len(entries)} entries to {table_file}")
                n_files += 1

    logger.info(f"{time.ctime()} - Finished topology value computation")
    elapsed_time = time.time() - start_time
    logger.info(f"Total elapsed time: {elapsed_time / 60:.2f} minutes")
    logger.info(f"Wrote {n_files} output files to {output_directory}")


if __name__ == "__main__":
    main()
