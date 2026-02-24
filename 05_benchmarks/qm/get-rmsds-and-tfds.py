"""Compute RMSD and TFD between MM-optimized and QM conformers.

This script compares MM-optimized molecular conformations against their
QM reference conformers by computing RMSD (root-mean-square deviation of
heavy atom positions) and TFD (torsion fingerprint deviation) for conformers
with matching QCArchive IDs.

Workflow:
1. Load QM reference conformers and MM-optimized conformers
2. Match conformers by QCArchive ID (1:1 matching)
3. For each matched pair:
   - Compute RMSD between QM and MM coordinates
   - Compute TFD between QM and MM torsion angles
4. Write results to parquet files for analysis

Metrics:
- RMSD: Heavy-atom coordinate deviation in Angstroms
- TFD: Torsion fingerprint deviation (dimensionless, 0-1 scale)

Inputs:
- data/qm/: QM reference conformers (parquet)
- data/{forcefield}/: MM-optimized conformers (parquet)

Outputs:
- {rmsd_directory}/{forcefield}/: Parquet files with columns:
  - qcarchive_id: QCArchive conformer ID
  - rmsd: Heavy-atom RMSD (Angstrom)
  - tfd: Torsion fingerprint deviation
  - method: Force field name

Note: Unlike get-all-to-all-rmsds-and-tfds.py, this script assumes
QCArchive IDs match exactly between QM and MM datasets (same starting
conformer, different optimization).
"""

import pathlib
import sys
import typing
import click
import tqdm
import time

from loguru import logger
from click_option_group import optgroup

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


from openff.toolkit import Molecule, ForceField

from yammbs.analysis import get_rmsd, get_tfd

logger.remove()
logger.add(sys.stdout)

def batch_get_rmsd(
    qcarchive_ids: list[int],
    qm_directory: str = None,
    ff_directory: str = None,
) -> list[dict]:
    """Compute RMSD and TFD for a batch of conformer pairs.
    
    Matches QM and MM conformers by QCArchive ID and computes structural
    similarity metrics.
    
    Parameters
    ----------
    qcarchive_ids : list[int]
        QCArchive IDs to process in this batch.
    qm_directory : str
        Path to QM reference conformer parquet dataset.
    ff_directory : str
        Path to MM-optimized conformer parquet dataset.
    
    Returns
    -------
    list[dict]
        List of results, each containing:
        - qcarchive_id (int): QCArchive conformer ID
        - rmsd (float): Heavy-atom RMSD in Angstroms
        - tfd (float): Torsion fingerprint deviation
        - method (str): Force field name
    """
    # Load QM reference conformers for this batch.
    qm_dataset = ds.dataset(qm_directory)
    qm_subset = qm_dataset.filter(
        pc.field("qcarchive_id").isin(qcarchive_ids)
    )
    qm_rows = qm_subset.to_table(
        columns=["qcarchive_id", "mapped_smiles", "coordinates"]
    ).to_pylist()
    qm_coordinates_by_qcarchive_id = {
        row["qcarchive_id"]: np.array(row["coordinates"]).reshape((-1, 3))
        for row in qm_rows
    }

    # Load MM-optimized conformers for this batch.
    ff_subset = ds.dataset(ff_directory).filter(
        pc.field("qcarchive_id").isin(qcarchive_ids)
    )
    ff_rows = ff_subset.to_table(
        columns=["qcarchive_id", "mapped_smiles", "coordinates", "method"]
    ).to_pylist()

    # Match QM and MM conformers by QCArchive ID and compute metrics.
    entries = []
    for row in tqdm.tqdm(ff_rows, desc="Computing RMSD/TFD"):
        try:
            qm_coordinates = qm_coordinates_by_qcarchive_id[row["qcarchive_id"]]
        except KeyError:
            logger.warning(f"Skipping {row['qcarchive_id']}: No matching QM conformer found")
            continue
        
        mol = Molecule.from_mapped_smiles(
            row["mapped_smiles"],
            allow_undefined_stereo=True
        )
        ff_coordinates = np.array(row["coordinates"]).reshape((-1, 3))
        
        # Compute RMSD between QM and MM conformers.
        rmsd = get_rmsd(mol, qm_coordinates, ff_coordinates)
        
        # Compute TFD (may fail for some molecules with unusual bonding).
        try:
            tfd = get_tfd(mol, qm_coordinates, ff_coordinates)
        except Exception as e:
            logger.debug(f"TFD computation failed for {row['qcarchive_id']}: {e}")
            tfd = np.nan
        
        entry = {
            "qcarchive_id": row["qcarchive_id"],
            "rmsd": rmsd,
            "tfd": tfd,
            "method": row["method"],
        }
        entries.append(entry)
    return entries



@click.command()
@click.option(
    "--forcefield",
    "-ff",
    "forcefield",
    default="openff_unconstrained-2.2.1.offxml",
    help="Force field name/stem (used to locate MM-optimized conformers directory).",
)
@click.option(
    "--data",
    "-d",
    "data_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="data",
    help="Root data directory containing 'qm/' and '{forcefield}/' subdirectories.",
)
@click.option(
    "--rmsd",
    "-r",
    "rmsd_directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="rmsd",
    help="Directory to write RMSD/TFD parquet files.",
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
    help="Number of conformers per batch (distributed to each worker).",
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
    rmsd_directory: str = "rmsd",
    worker_type: typing.Literal["slurm", "local"] = "local",
    queue: str = "free",
    conda_environment: str = "ib-dev",
    memory: int = 4,  # GB
    walltime: int = 32,  # hours
    batch_size: int = 300,
    n_workers: int = -1,
):
    """Compute RMSD and TFD for MM-QM conformer pairs.
    
    Matches conformers by QCArchive ID and computes structural similarity metrics.
    """
    from openff.nagl.utils._parallelization import batch_distributed
    from dask import distributed

    logger.info(f"{time.ctime()} - Starting RMSD/TFD computation")
    start_time = time.time()

    # Load QM and MM datasets.
    data_directory = pathlib.Path(data_directory)
    qm_directory = data_directory / "qm"
    qm_dataset = ds.dataset(qm_directory)
    logger.info(f"Loaded {qm_dataset.count_rows()} QM conformers from {qm_directory}")

    ff_name = pathlib.Path(forcefield).stem
    ff_directory = data_directory / ff_name
    ff_dataset = ds.dataset(ff_directory)
    logger.info(f"Loaded {ff_dataset.count_rows()} MM conformers from {ff_directory}")

    # Set up output directory.
    rmsd_directory = pathlib.Path(rmsd_directory) / ff_name
    rmsd_directory.mkdir(parents=True, exist_ok=True)
    
    # Get conformers to process and skip already-completed ones.
    input_qcarchive_ids = ff_dataset.to_table(
        columns=["qcarchive_id"]
    ).to_pydict()["qcarchive_id"]
    input_qcarchive_ids = set(input_qcarchive_ids)
    logger.info(f"Found {len(input_qcarchive_ids)} conformers to process")

    output_dataset = ds.dataset(rmsd_directory)
    n_files = 0
    if output_dataset.count_rows():
        existing_qcarchive_ids = output_dataset.to_table(
            columns=["qcarchive_id"]
        ).to_pydict()["qcarchive_id"]
        logger.info(f"Found {len(existing_qcarchive_ids)} existing results in {rmsd_directory}")
        input_qcarchive_ids -= set(existing_qcarchive_ids)
        logger.info(f"Remaining to process: {len(input_qcarchive_ids)} conformers")
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
            batch_get_rmsd,
            qm_directory=str(qm_directory.resolve()),
            ff_directory=str(ff_directory.resolve()),
        ))
        for future in tqdm.tqdm(
            distributed.as_completed(futures, raise_errors=False),
            total=len(futures),
            desc="Computing RMSD/TFD batches",
        ):
            entries = future.result()
            table = pa.Table.from_pylist(entries)
            table_file = rmsd_directory / f"batch-{n_files:04d}.parquet"
            pq.write_table(table, table_file)
            logger.info(f"Wrote {len(entries)} entries to {table_file}")
            n_files += 1

    logger.info(f"{time.ctime()} - Finished RMSD/TFD computation")
    elapsed_time = time.time() - start_time
    logger.info(f"Total elapsed time: {elapsed_time / 60:.2f} minutes")
    logger.info(f"Wrote {n_files} output files to {rmsd_directory}")


if __name__ == "__main__":
    main()
