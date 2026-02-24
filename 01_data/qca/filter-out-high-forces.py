"""
Filter out conformers with high RMS forces from QCArchive datasets.

This script processes HuggingFace datasets containing quantum chemistry data
(coordinates, energies, forces) and removes conformers where the RMS atomic force
exceeds a specified threshold. High RMS forces typically indicate:
- Geometries far from equilibrium
- Poor optimization convergence
- Numerical instabilities in QM calculations

Filtering these conformers improves dataset quality and prevents training
instabilities in neural network force fields.

Purpose
-------
High-force conformers can:
- Dominate loss functions during training
- Lead to unstable gradient updates
- Represent unphysical or poorly converged structures
- Reduce model generalization

This script removes such outliers while preserving the dataset structure.

Input
-----
HuggingFace Dataset containing:
- smiles: Mapped SMILES strings
- coords: Flattened coordinates (N_conformers × N_atoms × 3) in Angstrom
- energy: Energies for each conformer (N_conformers) in kcal/mol
- forces: Atomic forces for each conformer (N_conformers × N_atoms × 3) in kcal/mol/Å

Output
------
Filtered HuggingFace Dataset with same structure:
- Conformers with RMS force > threshold removed
- Molecules with no remaining valid conformers removed entirely
- Statistics logged showing number and percentage of conformers filtered

Filtering Criterion
-------------------
Uses RMS (root-mean-square) force per conformer:
  RMS_force = sqrt(mean(force_x^2 + force_y^2 + force_z^2))

Default Threshold: 5.0 kcal/mol/Angstrom
- Typical equilibrium forces: < 1 kcal/mol/Å
- Well-converged optimizations: < 2-3 kcal/mol/Å
- Threshold captures problematic conformers while preserving most data
"""

import pathlib

# Third-party imports
import click
import datasets
import descent
import descent.targets.energy
import numpy as np
import tqdm
from loguru import logger
from openff.toolkit import Molecule


@click.command(help=__doc__)
@click.option(
    "--input-path",
    "-i",
    type=str,
    default="qca/output/combined/trajectory-diverse",
    help="Path to the input dataset (a directory containing a HuggingFace dataset)"
)
@click.option(
    "--output-path",
    "-o",
    type=str,
    default="qca/output/combined/trajectory-diverse-filtered",
    help="Path to save the output filtered dataset"
)
@click.option(
    "--force-threshold",
    "-t",
    type=float,
    default=5.0,
    help="Force threshold (in kcal/mol/Angstrom) above which conformers will be filtered out"
)
def main(
    input_path: str,
    output_path: str,
    force_threshold: float = 5.0
):
    # Load the HuggingFace dataset from disk
    dataset = datasets.Dataset.load_from_disk(input_path)
    
    # Track statistics for filtering summary
    n_original = 0  # Total conformers before filtering
    n_removed = 0   # Conformers filtered out
    
    # Store entries that pass the force threshold filter
    edited_entries = []
    
    # Process each molecule entry in the dataset
    for entry in tqdm.tqdm(dataset, desc="Processing"):
        # Create OpenFF Molecule to get atom count
        # (needed to reshape flattened coordinate/force arrays)
        mol = Molecule.from_mapped_smiles(entry["smiles"], allow_undefined_stereo=True)
        n_atoms = mol.n_atoms
        
        # Reshape flattened arrays back to (N_conformers, N_atoms, 3) format
        # Input format: [x1, y1, z1, x2, y2, z2, ...] for all conformers
        # Output format: [[atom1_xyz, atom2_xyz, ...], [conf2...], ...]
        coords = np.array(entry["coords"]).reshape((-1, n_atoms, 3))
        forces = np.array(entry["forces"]).reshape((-1, n_atoms, 3))
        energies = np.array(entry["energy"])
        
        n_original += len(energies)

        # Store conformers that pass the force threshold filter
        new_coords = []
        new_forces = []
        new_energies = []
        
        # Check each conformer's forces against threshold
        for i, fc in enumerate(forces):
            # Calculate RMS force across all atoms in this conformer
            # RMS = sqrt(mean(force_x^2 + force_y^2 + force_z^2))
            force_rms = np.sqrt(np.mean(fc**2))
            
            # Keep conformer if RMS force is below threshold
            if force_rms <= force_threshold:
                new_coords.append(coords[i])
                new_forces.append(forces[i])
                new_energies.append(energies[i])
            else:
                # Filter out high-force conformer and log it
                n_removed += 1
                logger.info(
                    f"Filtering conformer for {entry['smiles']} "
                    f"with RMS force {force_rms:.2f} > {force_threshold}"
                )
        
        # Only keep this molecule if at least one conformer passed the filter
        if len(new_energies):
            # Create new entry with filtered conformers
            edited_entry = descent.targets.energy.Entry({"smiles": entry["smiles"]})
            
            # Flatten arrays back to 1D format for storage
            # (N_conformers, N_atoms, 3) → [x1, y1, z1, x2, y2, z2, ...]
            edited_entry["coords"] = np.array(new_coords).flatten().tolist()
            edited_entry["forces"] = np.array(new_forces).flatten().tolist()
            edited_entry["energy"] = np.array(new_energies).tolist()
            
            edited_entries.append(edited_entry)
    
    # Create new HuggingFace dataset from filtered entries
    new_dataset = descent.targets.energy.create_dataset(edited_entries)

    # Save filtered dataset to disk
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_dataset.save_to_disk(output_path)
    
    # Log summary statistics
    logger.info(f"Saved filtered dataset with {len(new_dataset)} entries to {output_path}")
    logger.info(
        f"Filtering summary: {n_removed} conformers removed out of {n_original} total "
        f"({100*n_removed/n_original:.2f}% removed)"
    )


if __name__ == "__main__":
    main()
