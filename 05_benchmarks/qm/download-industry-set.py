"""
This script downloads the OpenFF Industry Benchmark dataset from QCArchive
and saves it in a specified output directory under the "qm" subdirectory as a parquet file.

It postprocesses the data into entries with the following format:
- `qcarchive_id` (int): The unique identifier for the record in QCArchive.
- `cmiles` (str): The canonical SMILES representation of the molecule.
- `mapped_smiles` (str): The mapped SMILES representation of the molecule.
- `coordinates` (list of float): The 3D coordinates of the molecule in Angstroms.
- `energy` (float): The energy of the molecule in kcal/mol.
- `method` (str): The method used for the calculation, set to "qm".
- `dataset` (str): The name of the dataset, set to "OpenFF Industry Benchmark Season 1 v1.2".
"""

import pathlib
import sys
import pickle
import click
import tqdm


import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from loguru import logger

import qcelemental
import qcportal as ptl
from openff.units import unit
from openff.qcsubmit.utils.utils import portal_client_manager
from openff.qcsubmit.results import OptimizationResultCollection

QCFRACTAL_URL = "https://api.qcarchive.molssi.org:443/"

hartree2kcalmol = qcelemental.constants.hartree2kcalmol

logger.remove()
logger.add(sys.stdout)


def get_client(url=QCFRACTAL_URL):
    return ptl.PortalClient(
        url,
        cache_dir="../../03_fit-valence/02_curate-data/"
    )


@click.command()
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    default="data/optimization",
    help="Path to the output directory.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force reprocessing of existing data.",
)
@click.option(
    "--dataset-name",
    "-d",
    type=str,
    default="OpenFF Industry Benchmark Season 1 v1.2",
    help="Name of the dataset to process.",
)
def main(
    output_directory: str,
    force: bool = False,
    dataset_name: str = "OpenFF Industry Benchmark Season 1 v1.2",
):
    # check for existing results
    output_directory = pathlib.Path(output_directory) / "qm"
    output_directory.mkdir(parents=True, exist_ok=True)

    existing_dataset = ds.dataset(output_directory)
    existing_qcarchive_ids = []
    n_files = 0
    if existing_dataset.count_rows():
        existing_qcarchive_ids = existing_dataset.to_table(
            columns=["qcarchive_id"]
        ).to_pydict()["qcarchive_id"]
        n_files = len(existing_dataset.files)
    
    # get dataset
    qc_client = get_client()

    collection = OptimizationResultCollection.from_server(
        client=qc_client,
        datasets=[dataset_name],
        spec_name="default",
    )
    qcarchive_id_to_cmiles = {
        record.record_id: record.cmiles
        for record in collection.entries[QCFRACTAL_URL]
    }

    # check for existing pickle file
    pickle_file = f"{dataset_name}.pkl"
    if pathlib.Path(pickle_file).exists() and not force:
        logger.info(f"Loading records and molecules from {pickle_file}")
        with open(pickle_file, "rb") as f:
            records_and_molecules = pickle.load(f)
    else:
        with portal_client_manager(get_client):
            records_and_molecules = collection.to_records()
        
        with open(pickle_file, "wb") as f:
            pickle.dump(records_and_molecules, f)
    
    entries = []
    for record, molecule in tqdm.tqdm(records_and_molecules):
        if record.id in existing_qcarchive_ids:
            continue
        assert len(molecule.conformers) == 1
        geometry = molecule.conformers[0].m_as(unit.angstrom)
        entry = {
            "qcarchive_id": record.id,
            "cmiles": qcarchive_id_to_cmiles[record.id],
            "mapped_smiles": molecule.to_smiles(mapped=True),
            "coordinates": geometry.flatten().tolist(),
            "energy": record.energies[-1] * hartree2kcalmol,
            "method": "qm",
            "dataset": dataset_name,
        }
        entries.append(entry)
    
    table = pa.Table.from_pylist(entries)
    table_file = output_directory / f"{dataset_name}.parquet"
    pq.write_table(table, table_file)
    logger.info(f"Wrote {len(entries)} entries to {table_file}")


if __name__ == "__main__":
    main()
