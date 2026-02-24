"""
This script compares molecules based on SMILES fingerprints,
and curates a broadly diverse set of molecules.
"""

import click
import pathlib
import datasets
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers

import tqdm

from loguru import logger

@click.command()
@click.option(
    "--n-molecules",
    "-n",
    type=int,
    required=True,
    help="Number of diverse molecules to select.",
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
    help="Path to output file to write selected SMILES.",
)
def main(
    n_molecules: int,
    input_path: str,
    output_path: str,
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

    rdmols = [Chem.MolFromSmiles(smiles) for smiles in all_smiles]
    
    # Generate Morgan fingerprints using the fingerprint generator
    fp_gen = rdFingerprintGenerator.GetMorganGenerator()
    fingerprints = [
        fp_gen.GetFingerprint(mol)
        for mol in tqdm.tqdm(rdmols, desc="Generating fingerprints")
    ]
    
    # Use RDKit's LazyPick to select diverse molecules
    mmp = rdSimDivPickers.MaxMinPicker()
    selected_indices = mmp.LazyBitVectorPick(
        fingerprints,
        len(fingerprints),
        n_molecules,
        seed=0,
    )
    
    # Filter dataset to selected molecules
    selected_smiles = [all_smiles[i] for i in selected_indices]
    with open(output_path, "w") as f:
        f.write("\n".join(selected_smiles))

    logger.info(f"Wrote {len(selected_smiles)} selected molecules to {output_path}")
        

if __name__ == "__main__":
    main()
