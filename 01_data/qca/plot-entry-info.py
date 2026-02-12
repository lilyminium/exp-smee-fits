"""
Analyze and visualize QCArchive dataset statistics.

This script processes a HuggingFace dataset containing molecular conformers with
QM energies and forces, generating comprehensive statistics and visualizations.

Outputs:
    - metadata.csv: Per-conformer statistics (SMILES, force RMS, energy)
    - smiles_to_inchi.json: Mapping of SMILES strings to InChI identifiers
    - n_conformers_hist.png: Distribution of conformers per molecule
    - energy_boxplot.png: Distribution of QM energies (kcal/mol)
    - force_rms_boxplot.png: Distribution of force RMS values (kcal/mol/Å)

The force RMS (root mean square) is calculated as the L2 norm of all atomic forces
in a conformer, providing a measure of force magnitude.
"""
import pathlib
import json

import click
import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from loguru import logger
from matplotlib import pyplot as plt
from openff.toolkit import Molecule


@click.command(help=__doc__.split("\n\n")[0])
@click.option(
    "--input-path",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to input dataset in HuggingFace datasets format.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    required=True,
    help="Path to output directory to write plots and metadata.",
)
def main(
    input_path: str,
    output_path: str,
):
    """Generate dataset statistics and visualizations."""
    dataset = datasets.Dataset.load_from_disk(input_path)
    logger.info(f"Loaded dataset with {len(dataset)} entries from {input_path}")
    
    metadata = []
    smiles_to_inchi = {}
    
    for entry in tqdm.tqdm(dataset, desc="Processing conformers"):
        mol = Molecule.from_mapped_smiles(entry["smiles"], allow_undefined_stereo=True)
        smiles_to_inchi[entry["smiles"]] = mol.to_inchi()
        
        n_atoms = mol.n_atoms
        forces = np.array(entry["forces"]).reshape((-1, n_atoms, 3))
        energies = np.array(entry["energy"])

        for conf_idx, (force, energy) in enumerate(zip(forces, energies)):
            force_rms = np.sqrt(np.mean(force**2))
            metadata.append({
                "smiles": entry["smiles"],
                "force_rms": force_rms,
                "energy": energy,
                "conf_index": conf_idx,
            })
    
    df = pd.DataFrame(metadata)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metadata CSV
    output_file = output_path / "metadata.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved metadata to {output_file}")

    # Save SMILES to InChI mapping
    inchi_mapping_file = output_path / "smiles_to_inchi.json"
    with open(inchi_mapping_file, "w") as f:
        json.dump(smiles_to_inchi, f, indent=2)
    logger.info(f"Saved SMILES to InChI mapping to {inchi_mapping_file}")

    # Log dataset statistics
    unique_inchis = set(smiles_to_inchi.values())
    logger.info(f"Dataset contains {len(unique_inchis)} unique molecules (by InChI)")
    logger.info(f"Dataset contains {len(smiles_to_inchi)} unique SMILES")
    logger.info(f"Total conformers: {len(df)}")

    # Plot histogram of conformers per molecule
    plt.figure(figsize=(8, 6))
    n_confs = df.groupby("smiles")["conf_index"].max() + 1
    ax = sns.histplot(n_confs, bins=50)
    ax.set_xlabel("Number of Conformers per Molecule")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    n_conf_file = output_path / "n_conformers_hist.png"
    plt.savefig(n_conf_file, dpi=300)
    plt.close()
    logger.info(f"Saved histogram of conformers to {n_conf_file}")

    # Plot energy distribution
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x=df["energy"])
    ax.set_xlabel("Energy (kcal/mol)")
    ax.set_xscale("log")
    plt.tight_layout()
    energy_file = output_path / "energy_boxplot.png"
    plt.savefig(energy_file, dpi=300)
    plt.close()
    logger.info(f"Saved energy boxplot to {energy_file}")

    # Plot force RMS distribution
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x=df["force_rms"])
    ax.set_xlabel("Force RMS (kcal/mol/Å)")
    ax.set_xscale("log")
    plt.tight_layout()
    force_file = output_path / "force_rms_boxplot.png"
    plt.savefig(force_file, dpi=300)
    plt.close()
    logger.info(f"Saved force RMS boxplot to {force_file}")


if __name__ == "__main__":
    main()
