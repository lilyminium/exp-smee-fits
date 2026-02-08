"""
Filter a dataset by a list of SMILES

This is a fairly straightforward function but perhaps should be done multiple times
"""

import json
import click
import pathlib

import datasets

from loguru import logger


@click.command()
@click.option(
    "--input-dataset",
    "-i",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=True,
    help="Path to input dataset in HuggingFace datasets format.",
)
@click.option(
    "--input-split",
    "-s",
    "smiles_split",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
    help="Path to input JSON file containing SMILES split.",
)
@click.option(
    "--output-dataset",
    "-o",
    type=click.Path(dir_okay=True, file_okay=False),
    required=True,
    help="Path to output dataset directory in HuggingFace datasets format.",
)
def main(
    input_dataset: str,
    smiles_split: str,
    output_dataset: str,
):
    input_dataset = datasets.load_from_disk(input_dataset)
    with open(smiles_split, "r") as f:
        smiles_data = json.load(f)

    train_smiles = set(smiles_data.get("train", []))
    valid_smiles = set(smiles_data.get("validation", []))
    test_smiles = set(smiles_data.get("test", []))

    logger.info(f"Splitting {len(train_smiles)} train SMILES")
    logger.info(f"Splitting {len(valid_smiles)} validation SMILES")
    logger.info(f"Splitting {len(test_smiles)} test SMILES")

    train_dataset = input_dataset.filter(
        lambda example: example["smiles"] in train_smiles
    )
    valid_dataset = input_dataset.filter(
        lambda example: example["smiles"] in valid_smiles
    )
    test_dataset = input_dataset.filter(
        lambda example: example["smiles"] in test_smiles
    )
    output_dataset = pathlib.Path(output_dataset)
    output_dataset.mkdir(parents=True, exist_ok=True)
    train_path = output_dataset / "train"
    valid_path = output_dataset/ "validation"
    test_path = output_dataset / "test"

    train_dataset.save_to_disk(train_path)
    valid_dataset.save_to_disk(valid_path)
    test_dataset.save_to_disk(test_path)

    logger.info(f"Saved filtered datasets to {output_dataset}")


if __name__ == "__main__":
    main()
