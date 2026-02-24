"""Compute all-to-all RMSD and TFD between MM and QM conformers.

This script compares MM-optimized molecular conformations against their
QM reference conformers by computing pairwise RMSDs (root-mean-square
deviations of heavy atom positions) and TFDs (torsion fingerprint deviations).
For each MM conformer, it identifies the best-matching QM conformer based on
lowest RMSD.

Workflow:
1. Load QM reference conformers and MM-optimized conformers
2. Group by molecule (mapped SMILES)
3. For each MM conformer:
   - Compute RMSD to ALL QM conformers of the same molecule
   - Identify QM conformer with lowest RMSD as the match
   - Compute TFD for the best match
   - Record the match with energies, RMSD, and TFD
4. Write results to parquet files for analysis

Metrics:
- RMSD: Heavy-atom coordinate deviation in Angstroms
- TFD: Torsion fingerprint deviation (dimensionless, 0-1 scale)

Inputs:
- data/qm/: QM reference conformers (parquet)
- data/{forcefield}/: MM-optimized conformers (parquet)

Outputs:
- {rmsd_directory}/{forcefield}/: Parquet files with columns:
  - mapped_smiles, cmiles, inchi: Molecule identifiers
  - ff_qcarchive_id: MM conformer ID
  - qm_qcarchive_id: Best-matching QM conformer ID
  - ff_energy, qm_energy: Energies (kcal/mol)
  - rmsd: Heavy-atom RMSD (Angstrom)
  - tfd: Torsion fingerprint deviation
  - method: Force field name

Note: This is computationally expensive for molecules with many conformers
(scales as N_MM × N_QM per molecule).
"""
import collections
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

from openff.toolkit import Molecule

from yammbs.analysis import get_rmsd, get_tfd

logger.remove()
logger.add(sys.stdout)


def batch_get_rmsd(
    batch_mapped_smiles: list[str],
    qm_directory: str = None,
    ff_directory: str = None,
):
    """Compute all-to-all RMSD/TFD for a batch of molecules.
    
    For each MM conformer in the batch, finds the best-matching QM conformer
    by computing RMSD to all QM conformers of the same molecule.
    
    Parameters
    ----------
    batch_mapped_smiles : list[str]
        Mapped SMILES strings to process in this batch.
    qm_directory : str
        Path to QM reference conformer parquet dataset.
    ff_directory : str
        Path to MM-optimized conformer parquet dataset.
    
    Returns
    -------
    list[dict]
        List of matches, each containing:
        - mapped_smiles, cmiles, inchi: Molecule identifiers
        - ff_qcarchive_id: MM conformer ID
        - qm_qcarchive_id: Best-matching QM conformer ID
        - ff_energy, qm_energy: Energies (kcal/mol)
        - rmsd: Heavy-atom RMSD (Angstrom)
        - tfd: Torsion fingerprint deviation
        - method: Force field name
    """
    """Compute all-to-all RMSD/TFD for a batch of molecules.
    
    For each MM conformer in the batch, finds the best-matching QM conformer
    by computing RMSD to all QM conformers of the same molecule.
    
    Parameters
    ----------
    batch_mapped_smiles : list[str]
        Mapped SMILES strings to process in this batch.
    qm_directory : str
        Path to QM reference conformer parquet dataset.
    ff_directory : str
        Path to MM-optimized conformer parquet dataset.
    
    Returns
    -------
    list[dict]
        List of matches, each containing:
        - mapped_smiles, cmiles, inchi: Molecule identifiers
        - ff_qcarchive_id: MM conformer ID
        - qm_qcarchive_id: Best-matching QM conformer ID
        - ff_energy, qm_energy: Energies (kcal/mol)
        - rmsd: Heavy-atom RMSD (Angstrom)
        - tfd: Torsion fingerprint deviation
        - method: Force field name
    """
    # Load QM reference conformers for this batch.
    qm_dataset = ds.dataset(qm_directory)
    qm_subset = qm_dataset.filter(
        pc.field("mapped_smiles").isin(batch_mapped_smiles)
    )
    qm_rows = qm_subset.to_table(
        columns=["qcarchive_id", "mapped_smiles", "coordinates", "energy"]
    ).to_pylist()
    
    # Build lookups for QM data.
    qm_coordinates_by_qcarchive_id = {
        row["qcarchive_id"]: np.array(row["coordinates"]).reshape((-1, 3))
        for row in qm_rows
    }
    qcarchive_id_by_mapped_smiles = collections.defaultdict(list)
    for row in qm_rows:
        qcarchive_id_by_mapped_smiles[row["mapped_smiles"]].append(row["qcarchive_id"])

    qm_energy = {
        row["qcarchive_id"]: row["energy"]
        for row in qm_rows
    }

    # Load MM-optimized conformers for this batch.
    ff_subset = ds.dataset(ff_directory).filter(
        pc.field("mapped_smiles").isin(batch_mapped_smiles)
    )
    ff_df = ff_subset.to_table(
        columns=["qcarchive_id", "cmiles", "mapped_smiles", "coordinates", "method", "energy"]
    ).to_pandas()
    
    # Compute all-to-all RMSD and find best QM match for each MM conformer.
    rows = []
    for ff, ff_subdf in ff_df.groupby("method"):
        for mapped_smiles, smiles_df in tqdm.tqdm(ff_subdf.groupby("mapped_smiles"), desc=f"Processing {ff}"):
            qcarchive_ids = qcarchive_id_by_mapped_smiles[mapped_smiles]
            mol = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
            inchi = mol.to_inchi(fixed_hydrogens=True)

            # For each MM conformer, find the best-matching QM conformer.
            for _, row in smiles_df.iterrows():
                ff_coordinates = np.array(row["coordinates"]).reshape((-1, 3))
                assert row["qcarchive_id"] in qcarchive_ids
                
                # Compute RMSD to all QM conformers and find minimum.
                matched_qcarchive_id = -1
                current_rmsd = np.inf
                current_tfd = np.nan
                
                for qca_id in qcarchive_ids:
                    qm_coordinates = qm_coordinates_by_qcarchive_id[qca_id]
                    rmsd = get_rmsd(mol, qm_coordinates, ff_coordinates)
                    logger.debug(f"Comparing FF {row['qcarchive_id']} to QM {qca_id}: RMSD = {rmsd:.4f} Å")
                    if rmsd < current_rmsd:
                        matched_qcarchive_id = qca_id
                        current_rmsd = rmsd
                        try:
                            current_tfd = get_tfd(mol, qm_coordinates, ff_coordinates)
                        except Exception as e:
                            logger.warning(f"Could not compute TFD for {inchi}: {e}")
                            current_tfd = np.nan
            
                # Record the best match for this MM conformer.
                entry = {
                    "mapped_smiles": mapped_smiles,
                    "cmiles": row["cmiles"],
                    "inchi": inchi,
                    "ff_qcarchive_id": row["qcarchive_id"],
                    "ff_energy": row["energy"],
                    "rmsd": current_rmsd,
                    "tfd": current_tfd,
                    "qm_qcarchive_id": matched_qcarchive_id,
                    "qm_energy": qm_energy[matched_qcarchive_id],
                    "method": ff,
                }
                rows.append(entry)

    return rows

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
    rmsd_directory: str = "rmsd",
    worker_type: typing.Literal["slurm", "local"] = "local",
    queue: str = "free",
    conda_environment: str = "ib-dev",
    memory: int = 4,  # GB
    walltime: int = 32,  # hours
    batch_size: int = 300,
    n_workers: int = -1,
):
    """Compute all-to-all RMSD and TFD between MM and QM conformers.
    
    For each MM conformer, finds the best-matching QM conformer by RMSD
    and records the match with energy, RMSD, and TFD data.
    """
    from openff.nagl.utils._parallelization import batch_distributed
    from dask import distributed

    logger.info(f"{time.ctime()} - Starting all-to-all RMSD/TFD computation")
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

    rmsd_directory = pathlib.Path(rmsd_directory) / ff_name
    rmsd_directory.mkdir(parents=True, exist_ok=True)
    
    input_mapped_smiles = ff_dataset.to_table(
        columns=["mapped_smiles"]
    ).to_pydict()["mapped_smiles"]
    input_mapped_smiles = set(input_mapped_smiles)
    print(f"Loaded {len(input_mapped_smiles)} mapped smiles to process")

    output_dataset = ds.dataset(rmsd_directory)
    n_files = 0
    if output_dataset.count_rows():
        existing_mapped_smiles = output_dataset.to_table(
            columns=["mapped_smiles"]
        ).to_pydict()["mapped_smiles"]
        logger.info(f"Found {len(existing_mapped_smiles)} existing results in {rmsd_directory}")
        input_mapped_smiles -= set(existing_mapped_smiles)
        logger.info(f"Remaining to process: {len(input_mapped_smiles)} molecules")
        n_files = len(output_dataset.files)

    input_mapped_smiles = sorted(input_mapped_smiles)

    # Distribute computation across workers.
    with batch_distributed(
        input_mapped_smiles,
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

    logger.info(f"{time.ctime()} - Finished all-to-all RMSD/TFD computation")
    elapsed_time = time.time() - start_time
    logger.info(f"Total elapsed time: {elapsed_time / 60:.2f} minutes")
    logger.info(f"Wrote {n_files} output files to {rmsd_directory}")


if __name__ == "__main__":
    main()
