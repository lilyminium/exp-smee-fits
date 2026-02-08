"""
Download and process QCArchive optimization data for smee fitting workflows.

This takes advantage of existing processed datasets
downloaded for the Sage 2.3.0 fitting workflow.
"""
import typing
import pathlib
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import descent.targets.energy
import numpy as np
from loguru import logger
from openff.units import unit
import tqdm

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.dataset as ds

import matplotlib.pyplot as plt

import click
import qcportal

QCFRACTAL_URL = "https://api.qcarchive.molssi.org:443/"

HARTREE_TO_KCAL: float = (1 * unit.hartree * unit.avogadro_constant).m_as(
    unit.kilocalories_per_mole
)
BOHR_TO_ANGSTROM: float = (1.0 * unit.bohr).m_as(unit.angstrom)


def convert_to_entry(
    client,
    mapped_smiles: str,
    singlepoint_ids: list[int],
) -> dict[str, typing.Any] | None:
    """
    Process QCArchive records into a smee entry

    Parameters
    ----------
    mapped_smiles : str
        Canonical isomeric explicit hydrogen mapped SMILES string.
    singlepoint_ids : list[int]
        List of singlepoint record IDs to process.

    Notes
    -----
    This function performs the following operations:
    - Converts coordinates from Bohr to Angstrom
    - Converts energies from Hartree to kcal/mol
    - Converts gradients to forces (kcal/mol/Angstrom)
    - Creates descent-compatible dataset entries
    """

    records = [*client.get_records(record_ids=singlepoint_ids, include=["properties", "molecule"])]
    all_coords = []
    all_energy = []
    all_forces = []
    for record in records:
        # skip records with no gradients
        if "scf_total_gradient" not in record.properties:
            continue
        gradient = np.array(record.properties["scf_total_gradient"]).reshape((-1, 3))
        forces = (-gradient) * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM
        coords = record.molecule.geometry * BOHR_TO_ANGSTROM
        energy = record.properties["return_energy"] * HARTREE_TO_KCAL

        all_coords.append(coords)
        all_energy.append(energy)
        all_forces.append(forces)

    logger.info(
        f"Processed {len(all_energy)} / {len(records)} records for SMILES: {mapped_smiles}"
    )
    if not len(all_energy):
        return None
    
    # try to force clear memory
    del records
    gc.collect()

    return {
        "smiles": mapped_smiles,
        "coords": all_coords,
        "energy": all_energy,
        "forces": all_forces,
    }


def process_mapped_smiles(args):
    """Process a single mapped SMILES entry."""
    mapped_smiles, all_singlepoint_ids = args
    
    client = qcportal.PortalClient(address=QCFRACTAL_URL, cache_dir=".")

    # get singlepoint IDs from optimization records
    minimum_singlepoint_ids = [
        singlepoint_ids[-1] for singlepoint_ids in all_singlepoint_ids
    ]
    trajectory_singlepoint_ids = [
        id_ for singlepoint_ids in all_singlepoint_ids
        for id_ in singlepoint_ids
    ]
    # convert to entries
    equilibrium_entry = convert_to_entry(
        client=client,
        mapped_smiles=mapped_smiles,
        singlepoint_ids=minimum_singlepoint_ids,
    )
    if not equilibrium_entry:
        return None

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
    torsiondrive_datasets = ds.dataset(input_data)

    client = qcportal.PortalClient(address=QCFRACTAL_URL, cache_dir=".")

    # torsiondrives -- download both just the optimized output,
    # and the full trajectory
    unique_mapped_smiles: set[str] = set(
        torsiondrive_datasets.to_table(
            columns=["cmiles"]
        ).to_pydict()["cmiles"]
    )

    logger.info(f"Found {len(unique_mapped_smiles)} unique mapped SMILES entries.")

    unique_qcarchive_ids: set[str] = set(
        torsiondrive_datasets.to_table(
            columns=["id"]
        ).to_pydict()["id"]
    )
    logger.info(f"Downloading {len(unique_qcarchive_ids)} unique QCArchive record IDs.")
    

    # Parallelize torsiondrive record downloads
    chunk_size = 1_000
    unique_qcarchive_ids = list(unique_qcarchive_ids)
    chunks = [unique_qcarchive_ids[i:i + chunk_size] for i in range(0, len(unique_qcarchive_ids), chunk_size)]
    
    def fetch_torsiondrive_chunk(chunk_ids):
        return client.get_records(record_ids=chunk_ids, include=["minimum_optimizations"])
    
    torsiondrives_iterable = []
    logger.info(f"Using {n_threads} parallel workers for downloading {len(chunks)} optimization record chunks.")
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(fetch_torsiondrive_chunk, chunk) for chunk in chunks]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Downloading TorsionDrive records", ):
            records_chunk = future.result()
            torsiondrives_iterable.extend(records_chunk)
    
    logger.info(f"Downloaded {len(torsiondrives_iterable)} unique TorsionDrive record IDs.")

    def fetch_minimum_optimization_ids(record):
        """
        Get optimization IDs for a given torsion record.        
        There's no convenient API point for this at the moment
        """
        trajectory_ids = record._client.make_request(
            "get",
            f"api/v1/records/torsiondrive/{record.id}/minimum_optimizations",
            typing.Dict[str, int],
        )
        return list(trajectory_ids.values())

    # map torsion drive IDs to optimization IDs
    torsiondrive_to_optimization_ids: dict[int, list[int]] = {}
    logger.info(f"Using {n_threads} parallel workers for fetching minimum optimization IDs.")
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(fetch_minimum_optimization_ids, record): record for record in torsiondrives_iterable}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fetching minimum optimization IDs", ):
            record = futures[future]
            optimization_ids = future.result()
            torsiondrive_to_optimization_ids[record.id] = optimization_ids

    # now download optimization records
    logger.info("Downloading associated optimization records...")
    optimization_ids: list[int] = sorted(set(
        [
            opt_id
            for opt_ids in torsiondrive_to_optimization_ids.values()
            for opt_id in opt_ids
        ]
    ))
    
    def fetch_optimization_chunk(chunk_ids):
        return client.get_records(record_ids=chunk_ids)
    
    optimization_records_iterable = []
    chunks = [optimization_ids[i:i + chunk_size] for i in range(0, len(optimization_ids), chunk_size)]
    logger.info(f"Using {n_threads} parallel workers for downloading {len(chunks)} optimization record chunks.")
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(fetch_optimization_chunk, chunk) for chunk in chunks]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Downloading Optimization records", ):
            records_chunk = future.result()
            optimization_records_iterable.extend(records_chunk)
    
    optimization_records_by_id = {record.id: record for record in optimization_records_iterable}
    logger.info(f"Downloaded {len(optimization_records_by_id)} unique optimization record IDs.")

    # download all single points in batches as well
    # Fetch trajectory IDs in parallel for speed
    def fetch_trajectory_ids(record):
        trajectory_ids = record._client.make_request(
            "get",
            f"api/v1/records/optimization/{record.id}/trajectory",
            typing.List[int],
        )
        # keep a max of 3? to keep numbers reasonable
        # but make sure we have the last one...
        n_max_points = 3
        if len(trajectory_ids) > n_max_points:
            indices = np.linspace(0, len(trajectory_ids) - 1, n_max_points, dtype=int)
            assert indices[-1] == len(trajectory_ids) - 1
            trajectory_ids = [trajectory_ids[i] for i in indices]
        return record.id, trajectory_ids

    logger.info(f"Using {n_threads} parallel workers for trajectory ID fetching.")
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(fetch_trajectory_ids, record): record for record in optimization_records_by_id.values()}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Collecting singlepoint IDs", ):
            record_id, trajectory_ids = future.result()
            optimization_records_by_id[record_id].trajectory_ids_ = trajectory_ids

    # track mapped_smiles data
    mapped_smiles_data = []
    equilibrium_data = []
    trajectory_data = []

    # group everything by smiles
    df = torsiondrive_datasets.to_table(columns=["id", "cmiles"]).to_pandas()
    smiles_to_td_ids: dict[str, set[int]] = {}
    for smiles, subdf in df.groupby("cmiles"):
        smiles_to_td_ids[smiles] = set(subdf["id"].tolist())

    # Parallelize processing of mapped SMILES
    logger.info(f"Using {n_threads} parallel workers for processing {len(smiles_to_td_ids)} mapped SMILES.")
    args_list = []
    for mapped_smiles in unique_mapped_smiles:
        torsiondrive_ids = smiles_to_td_ids[mapped_smiles]
        optimization_ids = [
            opt_id
            for td_id in torsiondrive_ids
            for opt_id in torsiondrive_to_optimization_ids[td_id]
        ]
        singlepoint_ids = [
            optimization_records_by_id[opt_id].trajectory_ids_
            for opt_id in optimization_ids
        ]
        flat_trajectory_ids = [
            id_ for opt_id in optimization_ids
            for id_ in optimization_records_by_id[opt_id].trajectory_ids_
        ]
        args_list.append((mapped_smiles, singlepoint_ids))
        mapped_smiles_data.append({
            "mapped_smiles": mapped_smiles,
            "n_torsiondrives": len(torsiondrive_ids),
            "n_optimizations": len(optimization_ids),
            "torsiondrive_ids": list(torsiondrive_ids),
            "minimum_singlepoint_ids": [
                optimization_records_by_id[opt_id].trajectory_ids_[-1]
                for opt_id in optimization_ids
            ],
            "n_singlepoint_ids": len(singlepoint_ids),
            "trajectory_singlepoint_ids": flat_trajectory_ids,
            "n_trajectory_points": len(flat_trajectory_ids),
        })

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(process_mapped_smiles, args)
            for args in args_list
        ]
        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Processing mapped SMILES", position=0, leave=True):
            result = future.result()
            if result is not None:
                equilibrium_data.append(result["equilibrium"])
                trajectory_data.append(result["trajectory"])

    output_data = pathlib.Path(output_data)
    output_data.mkdir(parents=True, exist_ok=True)

    assert len(equilibrium_data) == len(trajectory_data)
    logger.info(f"Processed {len(equilibrium_data)} unique mapped SMILES entries.")

    # save datasets
    # minimum dataset
    minimum_data_path = output_data / "minimum"
    minimum_dataset = descent.targets.energy.create_dataset(equilibrium_data)
    logger.info(f"Saving HuggingFace dataset to: {minimum_data_path.resolve()}")
    minimum_dataset.save_to_disk(minimum_data_path)

    # trajectory dataset
    trajectory_data_path = output_data / "trajectory"
    trajectory_dataset = descent.targets.energy.create_dataset(trajectory_data)
    logger.info(f"Saving HuggingFace dataset to: {trajectory_data_path.resolve()}")
    trajectory_dataset.save_to_disk(trajectory_data_path)

    # save mapped_smiles info
    info_data = pathlib.Path(info_data)
    info_data.mkdir(parents=True, exist_ok=True)
    mapped_smiles_data = sorted(
        mapped_smiles_data,
        key=lambda x: (x["n_torsiondrives"], x["n_optimizations"]),
    )
    last = mapped_smiles_data[-1]
    logger.info(
        f"Most common SMILES: {last['mapped_smiles']} "
        f"with {last['n_optimizations']} optimizations "
        f"across {last['n_torsiondrives']} torsiondrives"
    )
    table = pa.Table.from_pylist(mapped_smiles_data)
    pq.write_table(table, info_data / "mapped_smiles.parquet")

    # histplot of number of optimizations per mapped_smiles
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
