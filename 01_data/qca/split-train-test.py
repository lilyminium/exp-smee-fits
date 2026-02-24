"""
Split datasets into training and test.

We avoid deepchem because they only support up to Python 3.10 ...
"""

import click
import json

import datasets
import tqdm
from loguru import logger

import pathlib
import datasets
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers


@click.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    required=True,
    help="Path to input dataset in SMILES format.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(dir_okay=False, file_okay=True),
    required=True,
    help="Path to output JSON to write train/valid/test SMILES.",
)
@click.option(
    "--train-fraction",
    "-tf",
    type=float,
    default=0.8,
    help="Fraction of data to use for training.",
)
@click.option(
    "--valid-fraction",
    "-vf",
    type=float,
    default=0.1,
    help="Fraction of data to use for validation.",
)
def main(
    input_path: str,
    output_path: str,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
):
    input_path = pathlib.Path(input_path)
    if input_path.is_dir():
        dataset = datasets.load_from_disk(input_path)
        all_smiles = list(dataset["smiles"])
    else:
        # assume it's a SMILES file
        with open(input_path, "r") as f:
            all_smiles = [line.strip() for line in f.readlines() if line.strip()]

    n_molecules = len(all_smiles)
    logger.info(f"Loaded {n_molecules} molecules from {input_path}")

    rdmols = [Chem.MolFromSmiles(smiles) for smiles in all_smiles]
    # Generate Morgan fingerprints using the fingerprint generator
    fp_gen = rdFingerprintGenerator.GetMorganGenerator()
    fingerprints = [
        fp_gen.GetFingerprint(mol)
        for mol in tqdm.tqdm(rdmols, desc="Generating fingerprints")
    ]

    n_test = int(n_molecules * (1 - train_fraction - valid_fraction) + 0.1)
    logger.info(f"Selecting {n_test} test molecules")
    n_valid = int(n_molecules * valid_fraction)
    logger.info(f"Selecting {n_valid} validation molecules")
    
    # Use RDKit's LazyPick to select diverse molecules
    mmp = rdSimDivPickers.MaxMinPicker()
    logger.info("Selecting test molecules")
    test_indices = mmp.LazyBitVectorPick(
        fingerprints,
        n_molecules,
        n_test,
        seed=0,
    )
    logger.info("Selecting validation molecules")
    valid_and_test_indices = mmp.LazyBitVectorPick(
        fingerprints,
        n_molecules,
        n_valid + n_test,
        firstPicks=test_indices,
        seed=0,
    )
    all_indices = set(list(range(n_molecules)))
    test_indices_set = set(test_indices)
    valid_indices_set = set(valid_and_test_indices) - test_indices_set
    train_indices_set = all_indices - test_indices_set - valid_indices_set

    train_smiles = [all_smiles[i] for i in train_indices_set]
    valid_smiles = [all_smiles[i] for i in valid_indices_set]
    test_smiles = [all_smiles[i] for i in test_indices_set]

    data = {
        "train": train_smiles,
        "validation": valid_smiles,
        "test": test_smiles,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"Wrote train/valid/test split to {output_path}")
    



if __name__ == "__main__":
    main()
