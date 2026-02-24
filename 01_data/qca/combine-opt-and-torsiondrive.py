"""
Combine optimizations and torsiondrives (or really any split dataset)

The expensive part is deduplication...
"""

from collections import defaultdict
import datasets
import pathlib

import click
from loguru import logger
import tqdm
import numpy as np
from openff.toolkit import Molecule
from descent.targets.energy import create_dataset


@click.command()
@click.option(
    "--input-dataset",
    "-i",
    "input_datasets",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    multiple=True,
    required=True,
    help="Paths to input datasets in HuggingFace datasets format.",
)
@click.option(
    "--output-dataset",
    "-o",
    type=click.Path(dir_okay=True, file_okay=False),
    required=True,
    help="Path to output dataset in HuggingFace datasets format.",
)
def main(
    input_datasets: list[str],
    output_dataset: str,
):
    
    all_entries = defaultdict(lambda: defaultdict(list))
    for input_dataset in input_datasets:
        dataset = datasets.load_from_disk(input_dataset)
        logger.info(f"Loaded dataset from {input_dataset} with {len(dataset)} records")
        for row in tqdm.tqdm(dataset, desc="Processing records"):
            smiles = row["smiles"]
            mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
            n_atoms = len(mol.atoms)
            coords = np.array(row["coords"]).reshape((-1, n_atoms, 3))
            forces = np.array(row["forces"]).reshape((-1, n_atoms, 3))
            energies = np.array(row["energy"]).reshape((-1,))

            for i, new_coord in enumerate(coords):
                for j, old_coord in enumerate(all_entries[smiles]["coords"]):
                    if np.allclose(new_coord, old_coord, atol=1e-5):
                        logger.info(f"Skipping -- conf {i} similar to existing conf {j} for smiles {smiles}")
                        break
                else:
                    all_entries[smiles]["coords"].append(new_coord)
                    all_entries[smiles]["forces"].append(forces[i])
                    all_entries[smiles]["energies"].append(energies[i])

    # Now create a new dataset
    all_final_entries = []
    for smiles, data in all_entries.items():
        n_atoms = len(Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True).atoms)
        coords = np.array(data["coords"]).reshape((-1, n_atoms * 3))
        forces = np.array(data["forces"]).reshape((-1, n_atoms * 3))
        energies = np.array(data["energies"])

        all_final_entries.append({
            "smiles": smiles,
            "coords": coords,
            "forces": forces,
            "energy": energies,
        })

    final_dataset = create_dataset(all_final_entries)
    logger.info(f"Final combined dataset has {len(final_dataset)} records")
    output_dataset = pathlib.Path(output_dataset)
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.save_to_disk(output_dataset)


if __name__ == "__main__":
    main()
