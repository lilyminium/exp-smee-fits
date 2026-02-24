"""Benchmark MM optimization of molecules using OpenMM and a specified force field.

This script reads QM-optimized molecular conformations from a dataset, re-optimizes
them using a specified MM force field, and saves the resulting coordinates and
energies for comparison. Results are batch-processed with Dask for distributed
computing on SLURM clusters.

Directory structure:
- data/qm/: Input QM reference data (parquet files)
- data/{forcefield-name}/: Output MM-optimized data (parquet files)

Input data requirements:
- qcarchive_id: Unique identifier for each molecule
- cmiles: Canonical SMILES string
- mapped_smiles: Mapped SMILES with atom indices
- coordinates: Initial QM coordinates (Angstrom)
- dataset: Source dataset name

Output format (parquet files):
- qcarchive_id (int): QCArchive ID
- cmiles (str): Canonical SMILES
- mapped_smiles (str): Mapped SMILES
- coordinates (list[float]): MM-optimized coordinates (Angstrom)
- energy (float): MM-optimized energy (kcal/mol)
- method (str): Force field name
- dataset (str): Source dataset

The script automatically skips already-processed molecules by comparing QCArchive
IDs in input vs output directories.
"""

import pathlib
import typing
import click
import tqdm
import time

from click_option_group import optgroup

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import openmm
import openmm.app
import openmm.unit
from openff.toolkit import Molecule, ForceField

from loguru import logger


def optimize_single(
    row,
    forcefield: ForceField,
    ff_name: str,
) -> typing.Optional[dict[str, typing.Any]]:
    """Optimize a single molecule using OpenMM local energy minimizer.

    Converts the molecule to an OpenMM system, performs energy minimization
    using the Reference platform, and returns the optimized geometry and energy.

    Parameters
    ----------
    row : dict[str, typing.Any]
        Molecule data containing:
        - qcarchive_id: QCArchive ID
        - cmiles: Canonical SMILES
        - mapped_smiles: Mapped SMILES (for topology creation)
        - coordinates: Initial coordinates (flat list in Angstrom)
        - dataset: Source dataset name
    forcefield : ForceField
        OpenFF force field for parameterization.
    ff_name : str
        Force field name for output labeling.

    Returns
    -------
    dict[str, typing.Any] or None
        Optimized result with coordinates (Angstrom) and energy (kcal/mol),
        or None if optimization fails.
    """
    # Create molecule from mapped SMILES and reshape coordinates.
    mol = Molecule.from_mapped_smiles(
        row["mapped_smiles"],
        allow_undefined_stereo=True
    )
    positions = np.array(row["coordinates"]).reshape((-1, 3))
    
    # Parameterize molecule with force field.
    try:
        system = forcefield.create_interchange(mol.to_topology()).to_openmm(
            combine_nonbonded_forces=False,
        )
    except Exception as e:
        logger.warning(f"Skipping record {row['qcarchive_id']}: {e}")
        return None
    
    # Set up OpenMM context for energy minimization.
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.1 * openmm.unit.femtoseconds),
        openmm.Platform.getPlatformByName("Reference"),
    )

    # Set initial positions and minimize.
    context.setPositions(
        (positions * openmm.unit.angstrom).in_units_of(openmm.unit.nanometer),
    )
    openmm.LocalEnergyMinimizer.minimize(
        context=context,
        tolerance=10,
        maxIterations=0,
    )
    
    # Extract optimized coordinates and energy.
    state = context.getState(getPositions=True, getEnergy=True)
    coordinates = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
    energy = state.getPotentialEnergy().value_in_unit(openmm.unit.kilocalorie_per_mole)

    return {
        "qcarchive_id": row["qcarchive_id"],
        "cmiles": row["cmiles"],
        "mapped_smiles": row["mapped_smiles"],
        "coordinates": coordinates.flatten().tolist(),
        "energy": energy,
        "method": ff_name,
        "dataset": row["dataset"],
    }


def batch_optimize(
    qcarchive_ids: list[str],
    qm_directory: str,
    forcefield_path: str,
) -> list[dict[str, typing.Any]]:
    """Optimize a batch of molecules for distributed processing.

    Loads QM data for specified molecules, removes H-X constraints from the
    force field (if present), and optimizes each molecule sequentially.
    Designed to be called in parallel across multiple workers.

    Parameters
    ----------
    qcarchive_ids : list[str]
        QCArchive IDs to process in this batch.
    qm_directory : str
        Path to directory with input QM parquet dataset.
    forcefield_path : str
        Path to OpenFF force field .offxml file.

    Returns
    -------
    list[dict[str, typing.Any]]
        List of optimization results, each containing:
        - qcarchive_id: Molecule identifier
        - cmiles: Canonical SMILES
        - mapped_smiles: Mapped SMILES
        - coordinates: Optimized coordinates (Angstrom, flattened)
        - energy: Optimized energy (kcal/mol)
        - method: Force field name
        - dataset: Source dataset
    """
    # Load QM data for this batch.
    dataset = ds.dataset(qm_directory)
    subset = dataset.filter(
        pc.field("qcarchive_id").isin(qcarchive_ids)
    )
    
    # Load force field and remove H-X constraints if present.
    forcefield = ForceField(forcefield_path)
    constraints = forcefield.get_parameter_handler("Constraints")
    if constraints.parameters and constraints.parameters[0].smirks == "[#1:1]-[*:2]":
        del constraints._parameters[0]

    ff_name = pathlib.Path(forcefield_path).stem
    rows = subset.to_table().to_pylist()

    # Optimize each molecule in the batch.
    entries = []
    for row in tqdm.tqdm(rows, desc=f"Optimizing {len(rows)} molecules"):
        entry = optimize_single(row, forcefield, ff_name)
        if entry is not None:
            entries.append(entry)
    return entries



@click.command()
@click.option(
    "--forcefield",
    "-ff",
    "forcefield",
    default="openff_unconstrained-2.2.1.offxml",
    help="Path to OpenFF force field .offxml file for MM optimization.",
)
@click.option(
    "--data",
    "-d",
    "data_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="data",
    help="Root data directory (must contain 'qm/' subdirectory with QM parquet files).",
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
    forcefield: str = "openff_unconstrained-2.2.1.offxml",
    data_directory: str = "data",
    worker_type: typing.Literal["slurm", "local"] = "local",
    queue: str = "free",
    conda_environment: str = "ib-dev",
    memory: int = 4,  # GB
    walltime: int = 32,  # hours
    batch_size: int = 300,
    n_workers: int = -1,
):
    """Optimize molecules with MM force field using distributed computing.
    
    Loads QM reference data, distributes optimization across workers,
    and saves results to parquet files.
    """
    from openff.nagl.utils._parallelization import batch_distributed
    from dask import distributed

    logger.info(f"{time.ctime()} - Starting batch optimization")
    start_time = time.time()

    # Load input QM data.
    data_directory = pathlib.Path(data_directory)
    input_directory = data_directory / "qm"
    input_dataset = ds.dataset(input_directory)
    logger.info(f"Loaded {input_dataset.count_rows()} rows from {input_directory}")
    
    # Extract QCArchive IDs to process.
    input_qcarchive_ids = input_dataset.to_table(
        columns=["qcarchive_id"]
    ).to_pydict()["qcarchive_id"]
    input_qcarchive_ids = set(input_qcarchive_ids)

    # Set up output directory and skip already-processed molecules.
    ff_name = pathlib.Path(forcefield).stem
    output_directory = data_directory / ff_name
    output_directory.mkdir(parents=True, exist_ok=True)
    output_dataset = ds.dataset(output_directory)
    n_files = 0
    
    if output_dataset.count_rows():
        existing_qcarchive_ids = output_dataset.to_table(
            columns=["qcarchive_id"]
        ).to_pydict()["qcarchive_id"]
        logger.info(f"Found {len(existing_qcarchive_ids)} existing results in {output_directory}")
        input_qcarchive_ids -= set(existing_qcarchive_ids)
        logger.info(f"Remaining to process: {len(input_qcarchive_ids)} molecules")
        n_files = len(output_dataset.files)

    input_qcarchive_ids = sorted(input_qcarchive_ids)

    # Distribute optimization across workers.
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
            forcefield_path=forcefield,
            qm_directory=str(input_directory.resolve()),
        ))
        for future in tqdm.tqdm(
            distributed.as_completed(futures, raise_errors=False),
            total=len(futures),
            desc="Optimizing batches",
        ):
            entries = future.result()
            table = pa.Table.from_pylist(entries)
            table_file = output_directory / f"batch-{n_files:04d}.parquet"
            pq.write_table(table, table_file)
            logger.info(f"Wrote {len(entries)} entries to {table_file}")
            n_files += 1

    logger.info(f"{time.ctime()} - Finished batch optimization")
    elapsed_time = time.time() - start_time
    logger.info(f"Total elapsed time: {elapsed_time / 60:.2f} minutes")
    logger.info(f"Wrote {n_files} output files to {output_directory}")


if __name__ == "__main__":
    main()
