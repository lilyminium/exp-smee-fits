"""
This script downloads optimization and torsion drive datasets from QCArchive.
It saves the data as pyarrow tables in a specified output directory.
Each dataset is saved as a separate .parquet file.
The script uses multiprocessing to speed up the download process.
It also filters out iodine-containing molecules from certain datasets.

PyArrow datasets follow the following schema:
- `id` (int): Unique identifier for the record.
- `inchi_key` (str): InChI key of the molecule.
- `cmiles` (str): CMILES (mapped SMILES) of the molecule.
- `smiles` (str): Sanitized unmapped SMILES of the molecule.
- `dataset_name` (str): Name of the dataset.

For Optimization datasets, the following additional fields are included:
- `energy` (float): Final energy (a.u.) of the optimization record.

For TorsionDrive datasets, the following additional fields are included:
- `dihedral` (list of int): Atom indices defining the dihedral angles rotated around.
- `n_dihedrals` (int): Number of dihedrals rotated around in the TorsionDrive scan.
- `angles` (list of int): Grid angles (in degrees) sampled during the torsion scan.
- `energies` (list of float): Energies (kcal/mol) corresponding to each angle in the torsion scan.
- `coordinates` (list of float): Flattened 3D coordinates of all conformers in angstroms.

"""

import functools
import pathlib
import click
import logging
import tqdm
import multiprocessing

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd

from rdkit import Chem
import qcportal as ptl
from openff.toolkit import Molecule
from openff.units import unit

from openff.qcsubmit.results import (
    BasicResultCollection,
    OptimizationResultCollection,
    TorsionDriveResultCollection,
)
from openff.qcsubmit.utils.utils import portal_client_manager
from openff.qcsubmit.results.filters import (
    ConnectivityFilter,
    UnperceivableStereoFilter,
    HydrogenBondFilter,
)

import qcelemental

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


QCFRACTAL_URL = "https://api.qcarchive.molssi.org:443/"

IGNORE_IODINE = [
    "OpenFF Discrepancy Benchmark 1",
    "OpenFF Gen 2 Opt Set 2 Coverage",
    "OpenFF Gen 2 Opt Set 4 eMolecules Discrepancy",
    "SMIRNOFF Coverage Set 1",
    "OpenFF Ehrman Informative Optimization v0.2",
    "FDA optimization dataset 1",
    "Kinase Inhibitors: WBO Distributions",

    "OpenFF Gen 2 Torsion Set 2 Coverage 2",
    "OpenFF Gen 2 Torsion Set 4 eMolecules Discrepancy 2",
]

OPTIMIZATION_WHITELISTS = [
    "OpenFF Optimization Set 1",
    "SMIRNOFF Coverage Set 1",
    "OpenFF VEHICLe Set 1",
    "OpenFF Discrepancy Benchmark 1",
    "OpenFF Ehrman Informative Optimization v0.2",
    "Pfizer discrepancy optimization dataset 1",
    "FDA optimization dataset 1",
    "Kinase Inhibitors: WBO Distributions",
    "OpenFF Gen 2 Opt Set 1 Roche",
    "OpenFF Gen 2 Opt Set 2 Coverage",
    "OpenFF Gen 2 Opt Set 3 Pfizer Discrepancy",
    "OpenFF Gen 2 Opt Set 4 eMolecules Discrepancy",
    "OpenFF Gen 2 Opt Set 5 Bayer",
    "OpenFF Sandbox CHO PhAlkEthOH v1.0",
    "OpenFF Aniline Para Opt v1.0",
    "OpenFF Industry Benchmark Season 1 v1.2",
    "OpenFF Gen2 Optimization Dataset Protomers v1.0",
    "OpenFF Protein Capped 1-mers 3-mers Optimization Dataset v1.0",
    "OpenFF Iodine Chemistry Optimization Dataset v1.0",
    "XtalPi Shared Fragments OptimizationDataset v1.0",
    "XtalPi 20-percent Fragments OptimizationDataset v1.0",
    "OpenFF Torsion Benchmark Supplement v1.0",
    "OpenFF Torsion Multiplicity Optimization Training Coverage Supplement v1.0",
    "OpenFF Torsion Multiplicity Optimization Benchmarking Coverage Supplement v1.0",
    "OpenFF Iodine Fragment Opt v1.0",
    "OpenFF Sulfur Optimization Training Coverage Supplement v1.0",
    "OpenFF Sulfur Optimization Benchmarking Coverage Supplement v1.0",
    "OpenFF Lipid Optimization Training Supplement v1.0",
    "OpenFF Lipid Optimization Benchmark Supplement v1.0",
    "OpenFF Cresset Additional Coverage Optimizations v4.0",
    "OpenFF Protein PDB 4-mers v4.0",
    "OpenFF Additional Generated ChEMBL Optimizations v4.0"
]

TORSIONDRIVE_WHITELISTS = [
    "OpenFF Group1 Torsions",
    "OpenFF Group1 Torsions 2",
    "OpenFF Group1 Torsions 3",
    "SMIRNOFF Coverage Torsion Set 1",
    "OpenFF Substituted Phenyl Set 1",
    "Pfizer discrepancy torsion dataset 1",
    "OpenFF Primary Benchmark 1 Torsion Set",
    "OpenFF Gen 2 Torsion Set 1 Roche 2",
    "OpenFF Gen 2 Torsion Set 2 Coverage 2",
    "OpenFF Gen 2 Torsion Set 3 Pfizer Discrepancy 2",
    "OpenFF Gen 2 Torsion Set 4 eMolecules Discrepancy 2",
    "OpenFF Gen 2 Torsion Set 5 Bayer 2",
    "OpenFF Gen 2 Torsion Set 6 supplemental 2",
    "OpenFF Fragmenter Validation 1.0",
    "OpenFF DANCE 1 eMolecules t142 v1.0",
    "OpenFF Rowley Biaryl v1.0",
    "OpenFF-benchmark-ligand-fragments-v1.0",
    "OpenFF Protein Fragments TorsionDrives v1.0",
    "OpenFF WBO Conjugated Series v1.0",
    "OpenFF Amide Torsion Set v1.0",
    "OpenFF-benchmark-ligand-fragments-v2.0",
    "OpenFF multiplicity correction torsion drive data v1.1",
    "OpenFF Protein Capped 3-mer Omega v1.0",
    "XtalPi Shared Fragments TorsiondriveDataset v1.0",
    "OpenFF Torsion Coverage Supplement v1.0",
    "OpenFF RNA Dinucleoside Monophosphate TorsionDrives v1.0",
    "XtalPi 20-percent Fragments TorsiondriveDataset v1.0",
    "OpenFF Torsion Drive Supplement v1.0",
    "OpenFF Torsion Multiplicity Torsion Drive Coverage Supplement v1.0",
    "OpenFF Phosphate Torsion Drives v1.0",
    "OpenFF Alkane Torsion Drives v1.0",
    "OpenFF Cresset Additional Coverage TorsionDrives v4.0",
    "OpenFF Additional Generated ChEMBL TorsionDrives 4.0",
    "OpenFF Additional Generated Guanidine Amidine Derivative and O-Linker TorsionDrives 4.0",
    "OpenFF Gen3 Torsion Set v1.0",
    "OpenFF Lipid Torsion Drives v4.0"
]

@functools.cache
def sanitize_smiles(smiles: str) -> str:
    """Sanitize a SMILES string by removing atom map numbers and Hs."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def get_canonical_smiles(entry) -> str:
    """
    Get the CMILES from the entry.
    This checks a couple different places.
    """
    KEY = "canonical_isomeric_explicit_hydrogen_mapped_smiles"
    try:
        return entry.attributes[KEY]
    except KeyError:
        return getattr(entry.initial_molecule.identifiers, KEY)

@functools.cache
def cmiles_to_inchi(cmiles: str) -> str:
    """Convert a CMILES string to an InChI key."""
    return Molecule.from_mapped_smiles(
        cmiles,
        allow_undefined_stereo=True
    ).to_inchikey(fixed_hydrogens=True)


def single_process_optimization(item):
    """
    This is a helper function to process a single record and molecule into an entry.
    This record is filtered to ensure it is complete, that connectivity has not changed,
    and that stereochemistry is perceivable.

    Parameters
    ----------
    item : tuple[OptimizationRecord, Molecule, str, str]
        A tuple containing the OptimizationRecord, Molecule, cmiles, and dataset name.
    
    Returns
    -------
    dict or None
        A dictionary containing the processed data entry, or None if the entry is invalid.
    """
    record, molecule, cmiles, dataset_name = item
    try:
        if not record.status.upper() == "COMPLETE":
            return
        if not ConnectivityFilter()._filter_function(None, None, molecule):
            return
        if not UnperceivableStereoFilter()._filter_function(None, None, molecule):
            return
    # can have some rdkit connection errors
    except:
        return

    try:
        smiles = sanitize_smiles(cmiles)
    except ValueError as e:
        logger.info(e)
        return []
    try:
        inchi_key = cmiles_to_inchi(cmiles)
    except:
        return []
    energy = record.energies[-1]
    coords = molecule.conformers[0].m_as(unit.angstrom)
    data_entry = {
        "id": record.id,
        "inchi_key": inchi_key,
        "cmiles": cmiles,
        "smiles": smiles,
        "dataset_name": dataset_name,
        "energy": energy,
        # "coordinates": coords.flatten().tolist()
    }
    return data_entry



def download_optimization(client, dataset_name: str, n_processes: int = 4) -> pa.Table:
    """
    Download an Optimization dataset from QCArchive and return a pyarrow table.

    This checks for two possible spec names: "default" and "spec_1",
    as some datasets were renamed to "spec_1" during the QCArchive migration.

    Entries are filtered to ensure they are complete, that connectivity has not changed,
    and that stereochemistry is perceivable.

    Parameters
    ----------
    client : ptl.PortalClient
        An instance of a QCFractal PortalClient.
    dataset_name : str
        The name of the dataset to download.
    n_processes : int, optional
        The number of processes to use for downloading, by default 4.

    Returns
    -------
    pa.Table
        A pyarrow Table containing the dataset.
        The table will have the following columns:
        - `id` (int): Unique identifier for the record.
        - `inchi_key` (str): InChI key of the molecule.
        - `cmiles` (str): CMILES (mapped SMILES) of the molecule.
        - `smiles` (str): Sanitized SMILES of the molecule.
        - `dataset_name` (str): Name of the dataset.
        - `energy` (float): Final energy (a.u.) of the optimization record.
    """
    data_entries = []
    try:
        # ideally would be default spec
        optimization_result_collection = OptimizationResultCollection.from_server(
            client=client,
            datasets=[dataset_name],
            spec_name="default"
        )
    except KeyError:
        # some datasets were renamed to spec_1 during migration
        optimization_result_collection = OptimizationResultCollection.from_server(
            client=client,
            datasets=[dataset_name],
            spec_name="spec_1"
        )

    ids_to_cmiles: dict[int, str] = {
        entry.record_id: entry.cmiles
        for entries in optimization_result_collection.entries.values()
        for entry in entries
    }
    with portal_client_manager(lambda x: ptl.PortalClient(x, cache_dir=".")):
        records_and_molecules = optimization_result_collection.to_records()

    items = [
        (record, molecule, ids_to_cmiles[record.id], dataset_name)
        for record, molecule in records_and_molecules
    ]
    with multiprocessing.Pool(n_processes) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(single_process_optimization, items),
                total=len(items),
                desc="Processing records and molecules"
            )
        )
    data_entries = [result for result in results if result is not None]

    table = pa.Table.from_pylist(data_entries)
    return table

def single_process_torsiondrive(item):
    """
    This is a helper function to process a single record and molecule into an entry.
    This record is filtered to ensure it is complete, that connectivity has not changed,
    that stereochemistry is perceivable, and that there are no intramolecular hydrogen bonds.

    Parameters
    ----------
    item : tuple[OptimizationRecord, Molecule, str, str]
        A tuple containing the OptimizationRecord, Molecule, cmiles, and dataset name.

    Returns
    -------
    dict or None
        A dictionary containing the processed data entry, or None if the entry is invalid.
    """
    record, molecule, cmiles, dataset_name = item
    try:
        if not record.status.upper() == "COMPLETE":
            return
        if not ConnectivityFilter()._filter_function(None, None, molecule):
            return
        if not UnperceivableStereoFilter()._filter_function(None, None, molecule):
            return
        if not HydrogenBondFilter()._filter_function(None, None, molecule):
            return
    except:
        return

    try:
        smiles = sanitize_smiles(cmiles)
    except ValueError as e:
        logger.info(e)
        return []
    try:
        inchi_key = cmiles_to_inchi(cmiles)
    except:
        return []
    dihedrals = record.specification.keywords.dihedrals
    dihedral = []
    for dih in dihedrals:
        dihedral.extend(dih)

    coordinates = np.concatenate(
        [conf.m_as(unit.angstrom).flatten() for conf in molecule.conformers],
    ).flatten().tolist()

    # grab energies...
    hartree2kcalmol = qcelemental.constants.hartree2kcalmol
    grid_ids = []
    energies = []
    for grid_id, opt in record.minimum_optimizations.items():
        grid_ids.append(grid_id[0])
        energies.append(opt.energies[-1] * hartree2kcalmol)

    indices = np.argsort(grid_ids)
    grid_ids = [grid_ids[i] for i in indices]
    molecule_grid_ids = [x[0] for x in molecule.properties["grid_ids"]]
    assert grid_ids == molecule_grid_ids, "Grid IDs do not match!"
    energies = [energies[i] for i in indices]

    data_entry = {
        "id": record.id,
        "inchi_key": inchi_key,
        "cmiles": cmiles,
        "smiles": smiles,
        "dataset_name": dataset_name,
        "dihedral": list(dihedral),
        "n_dihedrals": len(dihedrals),
        "angles": molecule_grid_ids,
        "energies": energies,
        "coordinates": coordinates
    }
    return data_entry


def download_torsiondrive(client, dataset_name: str, n_processes: int = 4) -> pa.Table:
    """
    Download a TorsionDrive dataset from QCArchive and return a pyarrow table.
    This checks for two possible spec names: "default" and "spec_1",
    as some datasets were renamed to "spec_1" during the QCArchive migration.

    Entries are filtered to ensure they are complete, that connectivity has not changed,
    that stereochemistry is perceivable, and that there are no intramolecular hydrogen bonds.

    Parameters
    ----------
    client : ptl.PortalClient
        An instance of a QCFractal PortalClient.
    dataset_name : str
        The name of the dataset to download.
    n_processes : int, optional
        The number of processes to use for downloading, by default 4.

    Returns
    -------
    pa.Table
        A pyarrow Table containing the dataset.
        The table will have the following columns:
        - `id` (int): Unique identifier for the record.
        - `inchi_key` (str): InChI key of the molecule.
        - `cmiles` (str): CMILES (mapped SMILES) of the molecule.
        - `smiles` (str): Sanitized unmapped SMILES of the molecule.
        - `dataset_name` (str): Name of the dataset.
        - `dihedral` (list of int): Dihedral angles rotated around
        - `n_dihedrals` (int): Number of dihedrals rotated around
    """
    data_entries = []

    try:
        # ideally would be default spec
        torsion_result_collection = TorsionDriveResultCollection.from_server(
            client=client,
            datasets=[dataset_name],
            spec_name="default"
        )
    except KeyError:
        # some datasets were renamed to spec_1 during migration
        torsion_result_collection = TorsionDriveResultCollection.from_server(
            client=client,
            datasets=[dataset_name],
            spec_name="spec_1"
        )

    ids_to_cmiles: dict[int, str] = {
        entry.record_id: entry.cmiles
        for entries in torsion_result_collection.entries.values()
        for entry in entries
    }
    with portal_client_manager(lambda x: ptl.PortalClient(x, cache_dir=".")):
        records_and_molecules = torsion_result_collection.to_records()

    items = [
        (record, molecule, ids_to_cmiles[record.id], dataset_name)
        for record, molecule in records_and_molecules
    ]
    with multiprocessing.Pool(n_processes) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(single_process_torsiondrive, items),
                total=len(items),
                desc="Processing records and molecules"
            )
        )
    data_entries = [result for result in results if result is not None]

    table = pa.Table.from_pylist(data_entries)
    return table


@click.command()
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="data/tables",
    help=(
        "Directory to save the downloaded tables. "
        "This will create subdirectories for optimization and torsiondrive data."
        "Each dataset will be saved as a .parquet file."
    )
)
def main(
    output_directory: str = "data"
):
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    client = ptl.PortalClient(address=QCFRACTAL_URL, cache_dir=".")
    
    # # download optimizations
    # opt_directory = output_directory / "optimization"
    # opt_directory.mkdir(parents=True, exist_ok=True)
    # for dsname in tqdm.tqdm(OPTIMIZATION_WHITELISTS, desc="Downloading Optimizations"):
    #     table = download_optimization(client, dsname)
    #     # filter out iodines
    #     if dsname in IGNORE_IODINE:
    #         df = table.to_pandas()
    #         mask = np.array(["I" in smi for smi in df.smiles.values])
    #         df = pd.DataFrame(df[~mask])
    #         table = pa.Table.from_pandas(df)

    #     table_file = opt_directory / f"{dsname}.parquet"
    #     pq.write_table(table, table_file)
    #     logger.info(f"Saved {table_file}")

    # download torsiondrives
    td_directory = output_directory / "torsiondrive"
    td_directory.mkdir(parents=True, exist_ok=True)
    for dsname in tqdm.tqdm(TORSIONDRIVE_WHITELISTS, desc="Downloading TorsionDrives"):
        table = download_torsiondrive(client, dsname)
        if dsname in IGNORE_IODINE:
            df = table.to_pandas()
            mask = np.array(["I" in smi for smi in df.smiles.values])
            df = pd.DataFrame(df[~mask])
            table = pa.Table.from_pandas(df)

        table_file = td_directory / f"{dsname}.parquet"
        pq.write_table(table, table_file)
        logger.info(f"Saved {table_file}")

    
        
if __name__ == "__main__":
    main()
