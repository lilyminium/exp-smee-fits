"""
Select only the SMILES parameterisable by Sage 2.3.0
"""
import functools
import multiprocessing
import pathlib
import click

import tqdm
from loguru import logger
import datasets

from openff.toolkit import Molecule, ForceField

def is_parameterizable(smiles: str, ff: ForceField) -> str | None:
    """
    Check if a SMILES is parameterizable by the given force field

    Parameters
    ----------
    smiles : str
        The SMILES string to check
    ff : ForceField
        The OpenFF ForceField object

    Returns
    -------
    str | None
        The original SMILES if parameterizable, None otherwise
    """
    try:
        mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        ff.create_interchange(mol.to_topology())
        return smiles
    except Exception as e:
        logger.info(f"SMILES {smiles} is not parameterizable: {str(e)}")
        return None

@click.command()
@click.option(
    "--forcefield",
    "-ff",
    type=str,
    default="openff-2.3.0.offxml",
    help="Path to the force field file (e.g., openff-2.3.0.offxml).",
)
@click.option(
    "--input-path",
    "-i",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    required=True,
    help="Path to input dataset in HuggingFace datasets format, or SMILES.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(dir_okay=False, file_okay=True),
    required=True,
    help="Path to output file to write parameterizable SMILES.",
)
@click.option(
    "--n-processes",
    "-np",
    type=int,
    default=1,
    help="Number of processes to use for checking parameterizability.",
)
def main(
    forcefield: str,
    input_path: str,
    output_path: str,
    n_processes: int = 1,
):
    input_path = pathlib.Path(input_path)
    if input_path.is_dir():
        dataset = datasets.load_from_disk(input_path)
        all_smiles = list(dataset["smiles"])
    else:
        # assume it's a SMILES file
        with open(input_path, "r") as f:
            all_smiles = [line.strip() for line in f.readlines() if line.strip()]

    logger.info(f"Loaded {len(all_smiles)} molecules from {input_path}")

    ff = ForceField(forcefield)

    # multiprocess
    check_func = functools.partial(is_parameterizable, ff=ff)
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(
                    check_func,
                    all_smiles,
                ),
                total=len(all_smiles),
                desc="Checking parameterizability",
            )
        )

    parameterizable_smiles = [s for s in results if s is not None]
    logger.info(
        f"Found {len(parameterizable_smiles)} parameterizable "
        f"molecules out of {len(all_smiles)}"
    )

    output_path = pathlib.Path(output_path)
    with open(output_path, "w") as f:
        f.write("\n".join(parameterizable_smiles))

    logger.info(f"Wrote parameterizable SMILES to {output_path}")


if __name__ == "__main__":
    main()
