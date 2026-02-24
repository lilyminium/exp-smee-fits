"""
Download and process QCArchive torsiondrive data for smee force field fitting.

This script downloads quantum chemistry torsiondrive scan trajectories from QCArchive
and processes them into two HuggingFace datasets suitable for training neural
network force fields using the smee/descent framework. Torsiondrives provide
energy profiles along dihedral rotation coordinates, critical for accurate
torsional parameter fitting.

Workflow
--------
1. Load pre-downloaded torsiondrive metadata from Sage 2.3.0 fitting workflow
2. Download torsiondrive records from QCArchive in parallel batches
3. Extract optimization IDs for each torsiondrive (one per grid angle)
4. Download optimization records to access their trajectories
5. Fetch trajectory singlepoint IDs for each optimization (subsampled to max 3 points)
6. Process each unique mapped SMILES in parallel to extract:
   - Molecular geometries (coordinates) at each grid point
   - Quantum mechanical energies
   - Quantum mechanical forces (from gradients)
7. Create two datasets:
   - Minimum/equilibrium: Only final optimized geometries at each grid point
   - Trajectory: All geometries along optimization paths at each grid point
8. Save datasets and generate metadata/statistics

Data Hierarchy
--------------
TorsionDrive → Optimizations → Singlepoints
- Each torsiondrive has multiple optimizations (one per grid angle)
- Each optimization has a trajectory of singlepoint calculations
- Subsampling: Max 3 points per optimization trajectory (vs 10 for regular optimizations)

Input
-----
- PyArrow dataset directory containing torsiondrive records with columns:
  - cmiles: Canonical mapped SMILES strings
  - id: QCArchive torsiondrive record IDs
  - dataset_name: Source dataset names
  
Expected from: Sage 2.3.0 data pool downloaded via download_sage_pool.py

Output
------
1. HuggingFace Datasets (in output-data/):
   - minimum/: Equilibrium geometries at each grid point
     * Smaller dataset, provides torsion energy profiles
     * Useful for fitting torsional parameters
     * Contains: smiles, coords, energy, forces
   
   - trajectory/: Full optimization trajectories at each grid point
     * Larger dataset, includes non-equilibrium geometries
     * Better coverage of conformational space near torsion barriers
     * Useful for training on diverse geometries
     * Contains: smiles, coords, energy, forces

2. Metadata (in info-data/):
   - mapped_smiles.parquet: Per-molecule statistics
     * mapped_smiles, n_torsiondrives, n_optimizations
     * torsiondrive_ids, n_trajectory_points
     * minimum_singlepoint_ids, trajectory_singlepoint_ids
   
   - n_optimizations_histogram.png: Distribution visualization
     * Shows total grid points across all torsiondrives per molecule
   
   - n_torsiondrives_histogram.png: Distribution visualization
     * Shows number of torsiondrive scans per molecule

Unit Conversions
----------------
- Coordinates: Bohr → Angstrom
- Energies: Hartree → kcal/mol
- Forces: Hartree/Bohr → kcal/mol/Angstrom (with gradient sign flip)

Performance
-----------
Uses ThreadPoolExecutor for parallel downloads (default 30 workers):
- Torsiondrive record downloading in 1000-record chunks
- Optimization record downloading in 1000-record chunks
- Trajectory ID fetching per optimization
- Singlepoint record processing per mapped SMILES

Memory Management
-----------------
- Explicit garbage collection after processing each SMILES
- Important for torsiondrives due to large number of singlepoints
- Aggressive trajectory subsampling (3 points) to manage dataset size

Notes
-----
- Subsamples optimization trajectories to maximum 3 evenly-spaced points
- Always includes final (optimized) geometry at each grid angle
- Skips records without gradient information
- One molecule may have multiple torsiondrive scans (different dihedrals)
- Total grid points = n_torsiondrives × grid_points_per_scan
"""
# Standard library imports
import gc
import pathlib
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import click
import descent.targets.energy
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import qcportal
import tqdm
from loguru import logger
from openff.units import unit

QCFRACTAL_URL = "https://api.qcarchive.molssi.org:443/"

HARTREE_TO_KCAL: float = (1 * unit.hartree * unit.avogadro_constant).m_as(
    unit.kilocalories_per_mole
)
BOHR_TO_ANGSTROM: float = (1.0 * unit.bohr).m_as(unit.angstrom)


def convert_to_entry(
    client: qcportal.PortalClient,
    mapped_smiles: str,
    singlepoint_ids: list[int],
) -> dict[str, typing.Any] | None:
    """
    Process QCArchive singlepoint records into a smee-compatible dataset entry.

    Parameters
    ----------
    client : qcportal.PortalClient
        QCPortal client for fetching records from QCArchive.
    mapped_smiles : str
        Canonical isomeric explicit hydrogen mapped SMILES string.
    singlepoint_ids : list[int]
        List of singlepoint record IDs to process.

    Returns
    -------
    dict[str, typing.Any] | None
        Dictionary containing 'smiles', 'coords', 'energy', and 'forces' keys,
        or None if no valid records with gradients were found.

    Notes
    -----
    This function performs the following operations:
    - Fetches singlepoint records from QCArchive
    - Skips records without gradient information
    - Converts coordinates from Bohr to Angstrom
    - Converts energies from Hartree to kcal/mol
    - Converts gradients to forces (kcal/mol/Angstrom)
    - Creates descent-compatible dataset entries
    - Explicitly clears memory to handle large batches
    """

    # Fetch all singlepoint records with properties and molecule geometry
    # Unpack generator into list for processing
    records = [
        *client.get_records(
            record_ids=singlepoint_ids, include=["properties", "molecule"]
        )
    ]
    
    all_coords = []
    all_energy = []
    all_forces = []
    
    for record in records:
        # Skip records without gradient information (can't compute forces)
        if "scf_total_gradient" not in record.properties:
            continue
        
        # Extract gradient and reshape to (N_atoms, 3) array
        gradient = np.array(record.properties["scf_total_gradient"]).reshape((-1, 3))
        
        # Convert gradient to forces: F = -∇E, with unit conversion
        # (Hartree/Bohr → kcal/mol/Angstrom)
        forces = (-gradient) * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM
        
        # Convert coordinates from Bohr to Angstrom
        coords = record.molecule.geometry * BOHR_TO_ANGSTROM
        
        # Convert energy from Hartree to kcal/mol
        energy = record.properties["return_energy"] * HARTREE_TO_KCAL

        all_coords.append(coords)
        all_energy.append(energy)
        all_forces.append(forces)

    logger.info(
        f"Processed {len(all_energy)} / {len(records)} records for SMILES: {mapped_smiles}"
    )
    if not len(all_energy):
        return None
    
    # Explicitly clear memory - important for large torsiondrive datasets
    # Each SMILES can have many singlepoint records
    del records
    gc.collect()

    return {
        "smiles": mapped_smiles,
        "coords": all_coords,
        "energy": all_energy,
        "forces": all_forces,
    }


def process_mapped_smiles(
    args: tuple[str, list[list[int]]],
) -> dict[str, dict[str, typing.Any]] | None:
    """
    Process torsiondrive optimization trajectories for a single mapped SMILES.

    For torsiondrives, each molecule has multiple optimizations (one per grid point
    on the torsion scan), and each optimization has a trajectory of singlepoints.
    This function creates two datasets:
    1. Equilibrium dataset: Only the final (optimized) single-point geometries from each grid point
    2. Trajectory dataset: All sampled geometries along all optimization paths

    Parameters
    ----------
    args : tuple[str, list[list[int]]]
        Tuple containing (mapped_smiles, all_singlepoint_ids), where:
        - mapped_smiles: Canonical isomeric explicit hydrogen mapped SMILES
        - all_singlepoint_ids: List of singlepoint ID lists, one per optimization
          (multiple optimizations per torsiondrive, one for each grid angle)

    Returns
    -------
    dict[str, dict[str, typing.Any]] | None
        Dictionary with 'equilibrium' and 'trajectory' entries, or None if
        no valid equilibrium data was found.
    """
    mapped_smiles, all_singlepoint_ids = args
    
    client = qcportal.PortalClient(address=QCFRACTAL_URL, cache_dir=".")

    # Extract final (minimum energy) singlepoint ID from each optimization
    # For torsiondrives, these represent the optimized geometry at each grid angle
    minimum_singlepoint_ids = [
        singlepoint_ids[-1] for singlepoint_ids in all_singlepoint_ids
    ]
    
    # Flatten all singlepoint IDs across all optimizations for trajectory dataset
    trajectory_singlepoint_ids = [
        id_ for singlepoint_ids in all_singlepoint_ids for id_ in singlepoint_ids
    ]
    
    # Create equilibrium dataset entry (only optimized geometries at each grid point)
    equilibrium_entry = convert_to_entry(
        client=client,
        mapped_smiles=mapped_smiles,
        singlepoint_ids=minimum_singlepoint_ids,
    )
    if not equilibrium_entry:
        return None

    # Create trajectory dataset entry (all geometries along all optimization paths)
    trajectory_entry = convert_to_entry(
        client=client,
        mapped_smiles=mapped_smiles,
        singlepoint_ids=trajectory_singlepoint_ids,
    )
    
    return {
        "equilibrium": equilibrium_entry,
        "trajectory": trajectory_entry,
    }

    

@click.command()
@click.option(
    "--input-data",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    default="sage-data-pool/torsiondrive",
    help="Path to input dataset directory.",
)
@click.option(
    "--output-data",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
    default="output/torsiondrives/dataset",
    help="Path to output dataset directory.",
)
@click.option(
    "--info-data",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
    default="metadata/torsiondrives",
    help="Path to output metadata directory.",
)
@click.option(
    "--n-threads",
    type=int,
    default=30,
    help="Number of threads to use for downloading data.",
)
def main(
    input_data: pathlib.Path = "sage-data-pool/torsiondrive",
    output_data: pathlib.Path = "output/torsiondrives/dataset",
    info_data: pathlib.Path = "metadata/torsiondrives",
    n_threads: int = 30,
):
    # Load PyArrow dataset containing pre-downloaded torsiondrive records
    # from the Sage 2.3.0 fitting workflow
    torsiondrive_datasets = ds.dataset(input_data)

    # Initialize QCPortal client for fetching detailed record information
    client = qcportal.PortalClient(address=QCFRACTAL_URL, cache_dir=".")

    # Extract unique mapped SMILES to determine how many distinct molecules
    # we need to process (one molecule may have multiple torsiondrive records)
    unique_mapped_smiles: set[str] = set(
        torsiondrive_datasets.to_table(columns=["cmiles"]).to_pydict()["cmiles"]
    )

    logger.info(f"Found {len(unique_mapped_smiles)} unique mapped SMILES entries.")

    # Extract unique QCArchive torsiondrive record IDs for batch downloading
    unique_qcarchive_ids: set[str] = set(
        torsiondrive_datasets.to_table(columns=["id"]).to_pydict()["id"]
    )
    logger.info(f"Downloading {len(unique_qcarchive_ids)} unique QCArchive record IDs.")

    # Download torsiondrive records in parallel batches for efficiency
    # Include minimum_optimizations to get optimization record IDs
    chunk_size = 1_000
    unique_qcarchive_ids = list(unique_qcarchive_ids)
    chunks = [
        unique_qcarchive_ids[i : i + chunk_size]
        for i in range(0, len(unique_qcarchive_ids), chunk_size)
    ]

    def fetch_torsiondrive_chunk(chunk_ids):
        """Fetch a chunk of torsiondrive records from QCArchive."""
        return client.get_records(
            record_ids=chunk_ids, include=["minimum_optimizations"]
        )

    torsiondrives_iterable = []
    logger.info(
        f"Using {n_threads} parallel workers for downloading "
        f"{len(chunks)} torsiondrive record chunks."
    )
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(fetch_torsiondrive_chunk, chunk) for chunk in chunks]
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading TorsionDrive records",
        ):
            records_chunk = future.result()
            torsiondrives_iterable.extend(records_chunk)

    logger.info(f"Downloaded {len(torsiondrives_iterable)} unique TorsionDrive record IDs.")

    def fetch_minimum_optimization_ids(record):
        """
        Get optimization IDs for a given torsiondrive record.

        Each torsiondrive contains multiple optimization records, one for each
        grid point on the torsion scan. This function extracts those optimization
        IDs so we can later fetch their trajectories.

        Note: Uses internal API as there's no public convenience method yet.
        """
        # Returns dict mapping grid angles to optimization IDs
        trajectory_ids = record._client.make_request(
            "get",
            f"api/v1/records/torsiondrive/{record.id}/minimum_optimizations",
            typing.Dict[str, int],
        )
        return list(trajectory_ids.values())

    # Create mapping from torsiondrive IDs to their constituent optimization IDs
    # Each torsiondrive has multiple optimizations (one per grid angle)
    torsiondrive_to_optimization_ids: dict[int, list[int]] = {}
    logger.info(
        f"Using {n_threads} parallel workers for fetching minimum optimization IDs."
    )
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(fetch_minimum_optimization_ids, record): record
            for record in torsiondrives_iterable
        }
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fetching minimum optimization IDs",
        ):
            record = futures[future]
            optimization_ids = future.result()
            torsiondrive_to_optimization_ids[record.id] = optimization_ids

    # Download optimization records to access their trajectory singlepoint IDs
    # Hierarchy: TorsionDrive → Optimizations → Singlepoints
    # We need optimization records to get the full trajectory of each grid point
    logger.info("Downloading associated optimization records...")
    
    # Collect all unique optimization IDs across all torsiondrives
    optimization_ids: list[int] = sorted(
        set(
            [
                opt_id
                for opt_ids in torsiondrive_to_optimization_ids.values()
                for opt_id in opt_ids
            ]
        )
    )

    def fetch_optimization_chunk(chunk_ids):
        """Fetch a chunk of optimization records from QCArchive."""
        return client.get_records(record_ids=chunk_ids)

    # Download optimization records in parallel chunks
    optimization_records_iterable = []
    chunks = [
        optimization_ids[i : i + chunk_size]
        for i in range(0, len(optimization_ids), chunk_size)
    ]
    logger.info(
        f"Using {n_threads} parallel workers for downloading "
        f"{len(chunks)} optimization record chunks."
    )
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(fetch_optimization_chunk, chunk) for chunk in chunks]
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading Optimization records",
        ):
            records_chunk = future.result()
            optimization_records_iterable.extend(records_chunk)
    
    # Create lookup dictionary for fast access to optimization records by ID
    optimization_records_by_id = {
        record.id: record for record in optimization_records_iterable
    }
    logger.info(
        f"Downloaded {len(optimization_records_by_id)} unique optimization record IDs."
    )

    # Each optimization record contains a trajectory of singlepoint calculations
    # For torsiondrives, we have many optimizations (one per grid angle), so we
    # subsample more aggressively than regular optimizations to keep dataset size
    # manageable while still capturing trajectory information
    
    # Maximum number of trajectory points to keep per optimization
    # Using 3 for torsiondrives (vs 10 for regular optimizations) due to higher
    # total number of optimizations per molecule
    N_MAX_POINTS = 3

    def fetch_trajectory_ids(record):
        """
        Get trajectory singlepoint IDs for a given optimization record.

        For torsiondrives, each grid point has an optimization trajectory.
        We subsample to N_MAX_POINTS evenly-spaced points, always including
        the final (optimized) point.

        Note: Uses internal API as there's no public convenience method yet.
        """
        # Fetch all singlepoint IDs along the optimization trajectory
        trajectory_ids = record._client.make_request(
            "get",
            f"api/v1/records/optimization/{record.id}/trajectory",
            typing.List[int],
        )
        
        # Subsample long trajectories to N_MAX_POINTS evenly-spaced points
        # Always keep the last point (optimized geometry at this grid angle)
        if len(trajectory_ids) > N_MAX_POINTS:
            indices = np.linspace(0, len(trajectory_ids) - 1, N_MAX_POINTS, dtype=int)
            assert indices[-1] == len(trajectory_ids) - 1, "Must include final point"
            trajectory_ids = [trajectory_ids[i] for i in indices]
        
        return record.id, trajectory_ids

    # Fetch trajectory IDs in parallel for all optimization records
    logger.info(f"Using {n_threads} parallel workers for trajectory ID fetching.")
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(fetch_trajectory_ids, record): record
            for record in optimization_records_by_id.values()
        }
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Collecting singlepoint IDs",
        ):
            record_id, trajectory_ids = future.result()
            # Attach trajectory IDs to the record for later use
            optimization_records_by_id[record_id].trajectory_ids_ = trajectory_ids

    # Prepare data structures for tracking processed results
    mapped_smiles_data = []  # Metadata about each SMILES
    equilibrium_data = []    # Final optimized geometries only (all grid points)
    trajectory_data = []     # Sampled optimization trajectories (all grid points)

    # Create mapping from CMILES (mapped SMILES) to torsiondrive record IDs
    # For torsiondrives, one molecule may have multiple scans (different dihedrals)
    df = torsiondrive_datasets.to_table(columns=["id", "cmiles"]).to_pandas()
    smiles_to_td_ids: dict[str, set[int]] = {}
    for smiles, subdf in df.groupby("cmiles"):
        smiles_to_td_ids[smiles] = set(subdf["id"].tolist())

    # Prepare arguments for parallel processing of each unique mapped SMILES
    logger.info(
        f"Using {n_threads} parallel workers for processing "
        f"{len(smiles_to_td_ids)} mapped SMILES."
    )
    args_list = []
    for mapped_smiles in unique_mapped_smiles:
        # Get all torsiondrive record IDs for this molecule
        torsiondrive_ids = smiles_to_td_ids[mapped_smiles]
        
        # Collect all optimization IDs across all torsiondrives for this molecule
        # Multiple torsiondrives × multiple grid points per torsiondrive
        optimization_ids = [
            opt_id
            for td_id in torsiondrive_ids
            for opt_id in torsiondrive_to_optimization_ids[td_id]
        ]
        
        # Collect trajectory IDs for all optimizations
        # This is a list of lists: one list per optimization
        singlepoint_ids = [
            optimization_records_by_id[opt_id].trajectory_ids_
            for opt_id in optimization_ids
        ]
        
        # Flatten all trajectory IDs for metadata tracking
        flat_trajectory_ids = [
            id_
            for opt_id in optimization_ids
            for id_ in optimization_records_by_id[opt_id].trajectory_ids_
        ]
        
        # Prepare arguments for process_mapped_smiles function
        args_list.append((mapped_smiles, singlepoint_ids))
        
        # Store comprehensive metadata about this molecule's torsiondrives
        mapped_smiles_data.append(
            {
                "mapped_smiles": mapped_smiles,
                "n_torsiondrives": len(torsiondrive_ids),  # Number of torsiondrive scans
                "n_optimizations": len(optimization_ids),  # Total grid points across all scans
                "torsiondrive_ids": list(torsiondrive_ids),
                # Last point in each optimization trajectory = optimized geometry at that grid angle
                "minimum_singlepoint_ids": [
                    optimization_records_by_id[opt_id].trajectory_ids_[-1]
                    for opt_id in optimization_ids
                ],
                "n_singlepoint_ids": len(singlepoint_ids),
                # All singlepoint IDs across all optimization trajectories
                "trajectory_singlepoint_ids": flat_trajectory_ids,
                "n_trajectory_points": len(flat_trajectory_ids),
            }
        )

    # Process all mapped SMILES in parallel, fetching singlepoint data
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(process_mapped_smiles, args) for args in args_list]
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing mapped SMILES",
            position=0,
            leave=True,
        ):
            result = future.result()
            if result is not None:
                equilibrium_data.append(result["equilibrium"])
                trajectory_data.append(result["trajectory"])

    # Prepare output directory for datasets
    output_data = pathlib.Path(output_data)
    output_data.mkdir(parents=True, exist_ok=True)

    assert len(equilibrium_data) == len(trajectory_data)
    logger.info(f"Processed {len(equilibrium_data)} unique mapped SMILES entries.")

    # Save two datasets for different use cases:
    # 1. Minimum (equilibrium) dataset: Only optimized geometries at each grid point
    #    - For torsiondrives, includes all grid points but only final optimized geometry
    #    - Smaller dataset, provides torsion energy profiles
    minimum_data_path = output_data / "minimum"
    minimum_dataset = descent.targets.energy.create_dataset(equilibrium_data)
    logger.info(f"Saving HuggingFace dataset to: {minimum_data_path.resolve()}")
    minimum_dataset.save_to_disk(minimum_data_path)

    # 2. Trajectory dataset: Optimization paths at each grid point
    #    - Includes non-equilibrium geometries from optimization trajectories
    #    - Much larger dataset, better coverage of conformational space
    #    - Useful for training on diverse geometries around torsion barriers
    trajectory_data_path = output_data / "trajectory"
    trajectory_dataset = descent.targets.energy.create_dataset(trajectory_data)
    logger.info(f"Saving HuggingFace dataset to: {trajectory_data_path.resolve()}")
    trajectory_dataset.save_to_disk(trajectory_data_path)

    # Save metadata about the processed molecules
    info_data = pathlib.Path(info_data)
    info_data.mkdir(parents=True, exist_ok=True)
    
    # Sort by number of torsiondrives and optimizations to identify most scanned molecules
    mapped_smiles_data = sorted(
        mapped_smiles_data,
        key=lambda x: (x["n_torsiondrives"], x["n_optimizations"]),
    )
    
    # Report the molecule with the most torsiondrive data
    last = mapped_smiles_data[-1]
    logger.info(
        f"Most common SMILES: {last['mapped_smiles']} "
        f"with {last['n_optimizations']} optimizations "
        f"across {last['n_torsiondrives']} torsiondrives"
    )
    
    # Save metadata as Parquet file for later analysis
    table = pa.Table.from_pylist(mapped_smiles_data)
    pq.write_table(table, info_data / "mapped_smiles.parquet")

    # Generate histogram showing distribution of optimization counts per molecule
    # For torsiondrives, this reflects total grid points across all scans
    n_optimizations = [entry["n_optimizations"] for entry in mapped_smiles_data]
    plt.hist(n_optimizations, bins=50, log=True)
    plt.xlabel("Number of Optimizations per Mapped SMILES")
    plt.ylabel("Frequency (log scale)")
    plt.title("Distribution of Optimizations per Mapped SMILES")
    plt.savefig(info_data / "n_optimizations_histogram.png")
    plt.close()

    # histplot of number of torsiondrives per mapped_smiles
    n_torsiondrives = [entry["n_torsiondrives"] for entry in mapped_smiles_data]
    plt.hist(n_torsiondrives, bins=50, log=True)
    plt.xlabel("Number of Torsiondrives per Mapped SMILES")
    plt.ylabel("Frequency (log scale)")
    plt.title("Distribution of Torsiondrives per Mapped SMILES")
    plt.savefig(info_data / "n_torsiondrives_histogram.png")
    plt.close()


if __name__ == "__main__":
    main()
