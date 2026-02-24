"""
Extract SMILES from OpenFF Industry Benchmark Set.

This script reads the OpenFF Industry Benchmark Season 1 v1.2 dataset (stored in Parquet
format) and extracts all canonical SMILES strings, saving them as a JSON list. The Industry
Benchmark Set contains molecules relevant to pharmaceutical and materials science applications,
selected by OpenFF's industry partners for force field validation.

Output is a simple JSON array of SMILES strings for downstream diversity analysis or filtering.
"""

import json

import click
import pyarrow.dataset as ds
from loguru import logger


@click.command(help=__doc__)
@click.option(
    "--input-path",
    "-i",
    type=str,
    default="../qca/sage-data-pool/optimization/OpenFF Industry Benchmark Season 1 v1.2.parquet",
    help="Path to the input Parquet dataset containing the Industry Benchmark Set",
)
@click.option(
    "--output-path",
    "-o",
    type=str,
    default="smiles.json",
    help="Path to save the output JSON file containing the list of SMILES",
)
def main(input_path: str, output_path: str):
    """Extract SMILES from Industry Benchmark dataset and save to JSON."""
    # Load the parquet dataset
    dataset = ds.dataset(input_path, format="parquet")
    
    # Extract only the 'cmiles' column (canonical SMILES) and convert to list
    smiles_list = dataset.to_table(columns=["cmiles"]).to_pydict()["cmiles"]
    
    logger.info(f"Extracted {len(smiles_list)} SMILES from {input_path}")
    
    # Save SMILES list as JSON array
    with open(output_path, "w") as f:
        json.dump(smiles_list, f, indent=2)


if __name__ == "__main__":
    main()