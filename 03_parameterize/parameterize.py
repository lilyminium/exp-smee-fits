"""
Generate Interchanges from SMILES and a force field and
pickle the resulting smee force field and Interchange objects.
\b
This saves it to a SINGLE file where the output object is
(smee_force_field, dict[smiles, Interchange]).
\b
Code borrowed and modified from Jennifer Clark:
https://github.com/openforcefield/back-to-school-jen/blob/main/5_setup_train_ff_topologies/setup_train_ff_topo.py
"""

import functools
import multiprocessing
import pathlib
import pickle

import click
import tqdm

import datasets
import smee
import smee.converters
from openff.toolkit import Molecule, ForceField
from openff.interchange import Interchange

from loguru import logger


def smiles_to_interchange(smiles: str, offxml: str) -> tuple[str, Interchange] | None:
    """Convert SMILES string to OpenFF Interchange object using force field.

    Creates an OpenFF Interchange object from a SMILES string by generating
    a molecule and applying the force field parameters.

    Parameters
    ----------
    smiles : str
        SMILES molecular representation string.
    offxml : str
        Path to OpenFF force field object for parameterization.

    Returns
    -------
    Interchange or None
        OpenFF Interchange object if successful, None if parameterization fails.

    Notes
    -----
    Failed molecules return None. Uses `allow_undefined_stereo=True`
    for molecule creation to handle molecules with undefined stereochemistry.

    Examples
    --------
    >>> interchange = smiles_to_interchange("CCO", "openff-2.2.1.offxml")
    >>> interchange is not None
    True

    >>> # Failed case returns None
    >>> interchange = smiles_to_interchange("invalid_smiles", "openff-2.2.1.offxml")
    >>> interchange is None
    True
    """
    try:
        forcefield = ForceField(offxml)
        mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        interchange = forcefield.create_interchange(mol.to_topology())
        return (smiles, interchange)
    except Exception as e:
        # Log the specific error for debugging
        if "stereochemistry" in str(e).lower():
            logger.debug(f"Stereochemistry error for '{smiles}': {str(e)}")
        else:
            logger.debug(f"Failed to create interchange for '{smiles}': {str(e)}")
        return None


@click.command(help=__doc__)
@click.option(
    "--input-force-field",
    "-ff",
    required=True,
    type=str,
    help="Path to input force field file",
)
@click.option(
    "--input-data-path",
    "-i",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    help="Path to input dataset file",
)
@click.option(
    "--output-path",
    "-o",
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
    ),
    help="Path to output file",
)
@click.option(
    "--n-processes",
    "-np",
    required=True,
    type=int,
    default=8,
    help="Number of processes to use for parallel processing",
)
def main(
    input_force_field: str,
    input_data_path: str,
    output_path: str,
    n_processes: int,
):
    input_data_path = pathlib.Path(input_data_path)
    if input_data_path.is_dir():
        dataset = datasets.load_from_disk(input_data_path)
        if isinstance(dataset, datasets.Dataset):
            smiles = dataset["smiles"]
        elif isinstance(dataset, datasets.DatasetDict):
            smiles = []
            for split in dataset:
                smiles.extend(dataset[split]["smiles"])
        else:
            raise ValueError("Unsupported dataset type")
    elif input_data_path.is_file():
        with open(input_data_path, "r") as f:
            smiles = [line.strip() for line in f.readlines()]
    else:
        raise ValueError("Input data path is neither a file nor a directory")

    logger.info(f"Loaded {len(smiles)} SMILES strings from dataset")
    logger.info(f"Using force field: {input_force_field}")

    interchanger = functools.partial(smiles_to_interchange, offxml=input_force_field)

    with multiprocessing.Pool(n_processes) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(interchanger, smiles),
                total=len(smiles),
                desc="Generating Interchanges",
            )
        )

    failed = [result for result in results if result is None]
    logger.info(f"Failed to create interchange for {len(failed)} molecules")

    successful = [result for result in results if result is not None]
    logger.info(f"Successfully created interchange for {len(successful)} molecules")

    all_smiles, all_interchanges = zip(*successful)

    logger.info("Converting successful interchanges to SMEE format...")
    smee_force_field, smee_topologies = smee.converters.convert_interchange(
        all_interchanges
    )
    logger.info("Conversion to SMEE format completed.")

    smiles_to_topologies = dict(zip(all_smiles, smee_topologies))

    with open(output_path, "wb") as f:
        pickle.dump((smee_force_field, smiles_to_topologies), f)
    logger.info(f"Successfully saved SMEE force field and topologies to {output_path}")


if __name__ == "__main__":
    main()
