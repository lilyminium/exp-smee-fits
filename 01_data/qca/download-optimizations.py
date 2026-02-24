"""
Download and process QCArchive optimization data for smee force field fitting.

This script downloads quantum chemistry optimization trajectories from QCArchive
and processes them into two HuggingFace datasets suitable for training neural
network force fields using the smee/descent framework.

Workflow
--------
1. Load pre-downloaded optimization metadata from Sage 2.3.0 fitting workflow
2. Filter out Industry Benchmark dataset (reserved for validation)
3. Download optimization records from QCArchive in parallel batches
4. Fetch trajectory singlepoint IDs for each optimization (subsampled to max 10 points)
5. Process each unique mapped SMILES in parallel to extract:
   - Molecular geometries (coordinates)
   - Quantum mechanical energies
   - Quantum mechanical forces (from gradients)
6. Create two datasets:
   - Minimum/equilibrium: Only final optimized geometries
   - Trajectory: All geometries along optimization paths
7. Save datasets and generate metadata/statistics

Input
-----
- PyArrow dataset directory containing optimization records with columns:
  - cmiles: Canonical mapped SMILES strings
  - id: QCArchive optimization record IDs
  - dataset_name: Source dataset names
  
Expected from: Sage 2.3.0 data pool downloaded via download_sage_pool.py

Output
------
1. HuggingFace Datasets (in output-data/):
   - minimum/: Equilibrium geometries only
     * Smaller, faster for equilibrium property fitting
     * Contains: smiles, coords, energy, forces
   
   - trajectory/: Full optimization trajectories  
     * Larger, includes non-equilibrium geometries
     * Better coverage of conformational space
     * Contains: smiles, coords, energy, forces

2. Metadata (in info-data/):
   - mapped_smiles.parquet: Per-molecule statistics
     * mapped_smiles, n_optimizations, n_trajectory_points
     * minimum_singlepoint_ids, trajectory_singlepoint_ids
   
   - n_optimizations_histogram.png: Distribution visualization
     * Shows how many optimizations exist per molecule

Unit Conversions
----------------
- Coordinates: Bohr → Angstrom
- Energies: Hartree → kcal/mol
- Forces: Hartree/Bohr → kcal/mol/Angstrom (with gradient sign flip)

Performance
-----------
Uses ThreadPoolExecutor for parallel downloads (default 30 workers):
- Optimization record downloading in 1000-record chunks
- Trajectory ID fetching per optimization
- Singlepoint record processing per mapped SMILES

Notes
-----
- Subsamples long trajectories to maximum 10 evenly-spaced points
- Always includes final (optimized) geometry in trajectory
- Skips records without gradient information
- Only processes records with complete status
"""
# Standard library imports
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
    """

    all_coords = []
    all_energy = []
    all_forces = []
    
    # Fetch all singlepoint records with properties and molecule geometry
    singlepoint_records = client.get_records(
        record_ids=singlepoint_ids, include=["properties", "molecule"]
    )
    
    for record in singlepoint_records:
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
        f"Processed {len(all_energy)} / {len(singlepoint_records)} records for SMILES: {mapped_smiles}"
    )
    if not len(all_energy):
        return None
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
    Process optimization trajectories for a single mapped SMILES string.

    This function creates two datasets:
    1. Equilibrium dataset: Only the final (optimized) geometries
    2. Trajectory dataset: All geometries along the optimization path

    Parameters
    ----------
    args : tuple[str, list[list[int]]]
        Tuple containing (mapped_smiles, all_trajectory_ids), where:
        - mapped_smiles: Canonical isomeric explicit hydrogen mapped SMILES
        - all_trajectory_ids: List of singlepoint ID lists,
          one per optimization trajectory

    Returns
    -------
    dict[str, dict[str, typing.Any]] | None
        Dictionary with 'equilibrium' and 'trajectory' entries, or None if
        no valid equilibrium data was found.
    """
    mapped_smiles, all_trajectory_ids = args
    
    client = qcportal.PortalClient(address=QCFRACTAL_URL, cache_dir=".")

    # Extract final (minimum energy) singlepoint ID from each optimization trajectory
    minimum_singlepoint_ids: list[int] = [
        trajectory_ids[-1] for trajectory_ids in all_trajectory_ids
    ]
    
    # Flatten all trajectory IDs into a single list for the full trajectory dataset
    trajectory_singlepoint_ids: list[int] = [
        id_ for trajectory_ids in all_trajectory_ids for id_ in trajectory_ids
    ]
    
    # Create equilibrium dataset entry (only optimized geometries)
    equilibrium_entry = convert_to_entry(
        client=client,
        mapped_smiles=mapped_smiles,
        singlepoint_ids=minimum_singlepoint_ids,
    )
    if not equilibrium_entry:
        return None
    
    # Create trajectory dataset entry (all geometries along optimization paths)
    trajectory_entry = convert_to_entry(
        client=client,
        mapped_smiles=mapped_smiles,
        singlepoint_ids=trajectory_singlepoint_ids,
    )
    
    return {
        "equilibrium": equilibrium_entry,
        "trajectory": trajectory_entry,
    }


@click.command(help=__doc__.split("\n\n")[0])
@click.option(
    "--input-data",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    default="sage-data-pool/optimization",
    help="Path to input dataset directory.",
)
@click.option(
    "--output-data",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
    default="output/optimizations/dataset",
    help="Path to output dataset directory.",
)
@click.option(
    "--info-data",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
    default="metadata/optimizations",
    help="Path to output metadata directory.",
)
@click.option(
    "--n-threads",
    type=int,
    default=30,
    help="Number of threads to use for downloading data.",
)
def main(
    input_data: pathlib.Path = "sage-data-pool/optimization",
    output_data: pathlib.Path = "output/optimizations/dataset",
    info_data: pathlib.Path = "metadata/optimizations",
    n_threads: int = 30,
):
    # Load PyArrow dataset containing pre-downloaded optimization records
    # from the Sage 2.3.0 fitting workflow
    optimization_datasets = ds.dataset(input_data)

    # Initialize QCPortal client for fetching detailed record information
    client = qcportal.PortalClient(address=QCFRACTAL_URL, cache_dir=".")

    # Filter out Industry Benchmark dataset - reserved for benchmarking/validation
    # rather than training
    optimization_datasets = optimization_datasets.filter(
        ~(pc.field("dataset_name") == "OpenFF Industry Benchmark Season 1 v1.2")
    )
    
    # Extract unique mapped SMILES to determine how many distinct molecules
    # we need to process (one molecule may have multiple optimization records)
    unique_mapped_smiles: set[str] = set(
        optimization_datasets.to_table(columns=["cmiles"]).to_pydict()["cmiles"]
    )

    logger.info(f"Found {len(unique_mapped_smiles)} unique mapped SMILES entries.")

    # Extract unique QCArchive optimization record IDs for batch downloading
    unique_qcarchive_ids: set[str] = set(
        optimization_datasets.to_table(columns=["id"]).to_pydict()["id"]
    )
    logger.info(f"Downloading {len(unique_qcarchive_ids)} unique QCArchive record IDs.")
    
    # Download optimization records in parallel batches for efficiency
    def fetch_optimization_chunk(chunk_ids):
        """Fetch a chunk of optimization records from QCArchive."""
        return client.get_records(record_ids=chunk_ids)
    
    # Split record IDs into manageable chunks of 1000
    chunk_size = 1_000
    unique_qcarchive_ids = list(unique_qcarchive_ids)
    chunks = [
        unique_qcarchive_ids[i : i + chunk_size]
        for i in range(0, len(unique_qcarchive_ids), chunk_size)
    ]
    
    records_iterable = []
    logger.info(
        f"Using {n_threads} parallel workers for downloading "
        f"{len(chunks)} optimization record chunks."
    )
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(fetch_optimization_chunk, chunk) for chunk in chunks]
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading QCArchive records",
        ):
            records_chunk = future.result()
            records_iterable.extend(records_chunk)
    
    # Create lookup dictionary for fast access to records by ID
    records_by_id = {record.id: record for record in records_iterable}
    logger.info(f"Downloaded {len(records_by_id)} unique QCArchive record IDs.")

    # Each optimization record contains a trajectory of singlepoint calculations
    # (geometry optimization steps). We need to fetch these singlepoint IDs to
    # get the full optimization path, not just the final optimized structure.
    
    # Maximum number of trajectory points to keep per optimization
    # Subsampling reduces dataset size while maintaining trajectory shape
    N_MAX = 10
    
    def fetch_trajectory_ids(record):
        """
        Get trajectory singlepoint IDs for a given optimization record.
        
        An optimization trajectory contains all singlepoint calculations from
        the optimization path. This function fetches those IDs and subsamples
        if there are too many points, ensuring the final (optimized) point
        is always included.
        
        Note: Uses internal API as there's no public convenience method yet.
        """
        # Fetch all singlepoint IDs along the optimization trajectory
        trajectory_ids = record._client.make_request(
            "get",
            f"api/v1/records/optimization/{record.id}/trajectory",
            typing.List[int],
        )
        
        # Subsample long trajectories to N_MAX evenly-spaced points
        # Always keep the last point (optimized geometry)
        if len(trajectory_ids) > N_MAX:
            indices = np.linspace(0, len(trajectory_ids) - 1, N_MAX, dtype=int)
            assert indices[-1] == len(trajectory_ids) - 1, "Must include final point"
            trajectory_ids = [trajectory_ids[i] for i in indices]
        
        return record.id, trajectory_ids

    # Fetch trajectory IDs in parallel for all optimization records
    logger.info(f"Using {n_threads} parallel workers for trajectory ID fetching.")
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(fetch_trajectory_ids, record): record
            for record in records_by_id.values()
        }
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Collecting singlepoint IDs",
        ):
            record_id, trajectory_ids = future.result()
            # Attach trajectory IDs to the record for later use
            records_by_id[record_id].trajectory_ids_ = trajectory_ids

    # Create mapping from CMILES (mapped SMILES) to optimization record IDs
    # Multiple optimizations may exist for the same molecule (e.g., different
    # conformers or from different datasets)
    optimization_cmiles_to_ids = {}
    df = optimization_datasets.to_table(columns=["cmiles", "id"]).to_pandas()
    for smiles, subdf in df.groupby("cmiles"):
        optimization_cmiles_to_ids[smiles] = set(subdf["id"].values.tolist())

    # Prepare data structures for tracking processed results
    mapped_smiles_data = []  # Metadata about each SMILES
    equilibrium_data = []    # Final optimized geometries only
    trajectory_data = []     # Sampled optimization trajectories
    
    # Prepare arguments for parallel processing of each unique mapped SMILES
    logger.info(
        f"Using {n_threads} parallel workers for processing "
        f"{len(unique_mapped_smiles)} mapped SMILES."
    )
    args_list = []
    for mapped_smiles in unique_mapped_smiles:
        # Get all optimization record IDs for this molecule
        optimization_ids = list(optimization_cmiles_to_ids[mapped_smiles])
        
        # Collect trajectory IDs for all optimizations of this molecule
        all_trajectory_ids = [
            records_by_id[opt_id].trajectory_ids_ for opt_id in optimization_ids
        ]
        
        # Prepare arguments for process_mapped_smiles function
        args_list.append((mapped_smiles, all_trajectory_ids))

        # Flatten all trajectory IDs for metadata tracking
        flat_trajectory_ids = [
            id_
            for opt_id in optimization_ids
            for id_ in records_by_id[opt_id].trajectory_ids_
        ]

        # Store metadata about this molecule's optimizations
        mapped_smiles_data.append(
            {
                "mapped_smiles": mapped_smiles,
                "n_optimizations": len(optimization_ids),
                # Last point in each trajectory = optimized geometry
                "minimum_singlepoint_ids": [
                    records_by_id[opt_id].trajectory_ids_[-1]
                    for opt_id in optimization_ids
                ],
                # All points along all optimization paths
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
    # 1. Minimum (equilibrium) dataset: Only optimized geometries
    #    - Used for fitting to equilibrium structures
    #    - Smaller dataset, faster training
    minimum_data_path = output_data / "minimum"
    minimum_dataset = descent.targets.energy.create_dataset(equilibrium_data)
    logger.info(f"Saving HuggingFace dataset to: {minimum_data_path.resolve()}")
    minimum_dataset.save_to_disk(minimum_data_path)

    # 2. Trajectory dataset: Full optimization paths
    #    - Used for fitting to both equilibrium and non-equilibrium structures
    #    - Provides more diverse training data
    #    - Helps improve force field accuracy across conformational space
    trajectory_data_path = output_data / "trajectory"
    trajectory_dataset = descent.targets.energy.create_dataset(trajectory_data)
    logger.info(f"Saving HuggingFace dataset to: {trajectory_data_path.resolve()}")
    trajectory_dataset.save_to_disk(trajectory_data_path)

    # Save metadata about the processed molecules
    info_data = pathlib.Path(info_data)
    info_data.mkdir(parents=True, exist_ok=True)
    
    # Sort by number of optimizations to identify most frequently optimized molecules
    mapped_smiles_data = sorted(
        mapped_smiles_data,
        key=lambda x: x["n_optimizations"],
    )
    
    # Report the molecule with the most optimization records
    last = mapped_smiles_data[-1]
    logger.info(
        f"Most common SMILES: {last['mapped_smiles']} "
        f"with {last['n_optimizations']} optimizations"
    )
    
    # Save metadata as Parquet file for later analysis
    table = pa.Table.from_pylist(mapped_smiles_data)
    pq.write_table(table, info_data / "mapped_smiles.parquet")

    # Generate histogram showing distribution of optimization counts per molecule
    # Useful for understanding dataset composition and identifying over-represented molecules
    n_optimizations = [entry["n_optimizations"] for entry in mapped_smiles_data]
    plt.hist(n_optimizations, bins=50, log=True)
    plt.xlabel("Number of Optimizations per Mapped SMILES")
    plt.ylabel("Frequency (log scale)")
    plt.title("Distribution of Optimizations per Mapped SMILES")
    plt.savefig(info_data / "n_optimizations_histogram.png")
    plt.close()


if __name__ == "__main__":
    main()
