"""Visualize MM-QM differences in internal coordinates by force field parameter.

This script generates box plots comparing MM and QM internal coordinates (bonds,
angles, proper torsions, improper torsions) for each force field parameter. The
plots show the distribution of (MM - QM) differences for all instances where
each parameter is applied, enabling identification of systematic parameter biases.

Workflow:
1. Load force field to map parameter IDs to SMIRKS patterns
2. Load topology-labeled dataset with MM and QM internal coordinates
3. Filter to conformers present in all MM methods (common set)
4. Group by topology type (Bonds, Angles, ProperTorsions, ImproperTorsions)
5. For each parameter ID:
   - Create box plot of (MM - QM) differences across all instances
   - Show distributions for each MM method (force field)
   - Annotate with SMIRKS pattern
6. Save individual plots organized by topology type

Inputs:
- {input_force_field}: OpenFF force field XML file
- {input_dataset}/: Parquet dataset with topology values labeled by parameter
  Required columns: qcarchive_id, parameter_id, topology, difference, method

Outputs (saved to {output_directory}/):
- bonds/{parameter_id}.png: Box plots for each bond parameter
- angles/{parameter_id}.png: Box plots for each angle parameter
- propers/{parameter_id}.png: Box plots for each proper torsion parameter
- impropers/{parameter_id}.png: Box plots for each improper torsion parameter

Box plot interpretation:
- Box shows interquartile range (25th-75th percentile)
- Whiskers show 5th-95th percentile range
- Outliers shown as individual points
- Zero line indicates perfect MM-QM agreement
- Systematic shifts indicate parameter bias
"""
import pathlib
import tqdm
import click

import pyarrow.dataset as ds
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

from openff.toolkit import Molecule, ForceField

from loguru import logger

sns.set_context("paper")
sns.set_style("ticks")
mpl.rcParams['font.sans-serif'] = ["muli"]


def plot_with_handler(
    df,
    handler,
    unit: str = "$\AA$",
    output_directory: str = "images/mm-vs-qm/bonds"
):
    """Generate box plots of MM-QM differences for each parameter.
    
    Creates one plot per parameter ID showing the distribution of
    (MM - QM) differences across all conformers where the parameter
    is applied.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: parameter_id, difference, method
    handler : ParameterHandler
        Force field parameter handler (e.g., Bonds, Angles)
    unit : str, optional
        Unit string for x-axis label (LaTeX formatted), by default "$\AA$"
    output_directory : str, optional
        Directory to save plots, by default "images/mm-vs-qm/bonds"
    """
    bond_unique = set(df.parameter_id.unique())
    for pid in tqdm.tqdm(bond_unique, desc="Plotting parameters"):
        parameter = handler.get_parameter({"id": pid})[0]
        
        # Create box plot with outliers at 5th-95th percentile whiskers.
        ax = sns.boxplot(
            data=df[df.parameter_id == pid],
            y="parameter_id",
            x="difference",
            hue="method",
            fliersize=2,
            whis=(5, 95),  # Whiskers at 5th and 95th percentiles
        )
        
        # Add reference line at zero (perfect agreement).
        ax.axvline(0, ls="--", lw=1, color="gray")
        ax.grid(axis="x", color="gray", alpha=0.3, ls="--", lw=1)
        ax.grid(axis="y", color="gray", alpha=0.3, ls="--", lw=1)
        ax.set_ylabel("Parameter ID")
        ax.set_xlabel(f"MM - QM [{unit}]")
        plt.title(parameter.smirks)
        plt.tight_layout()
        
        fnm = f"{output_directory}/{pid}.png"
        plt.savefig(fnm, dpi=300)
        logger.debug(f"Saved plot to {fnm}")
        plt.close()
    
    logger.info(f"Saved {len(bond_unique)} parameter plots to {output_directory}")
    
    
@click.command()
@click.option(
    "--input-force-field",
    "-ff",
    type=str,
    default="openff-2.3.0.offxml",
    help="OpenFF force field XML file (for parameter ID to SMIRKS mapping).",
)
@click.option(
    "--input-dataset",
    "-i",
    default="topology-labeled-by-parameter",
    help="Parquet dataset directory with topology values labeled by parameter ID.",
)
@click.option(
    "--output-directory",
    "-o",
    default="images/mm-vs-qm",
    help="Directory to write output plots (organized by topology type).",
)
def main(
    input_force_field: str = "openff-2.3.0.offxml",
    input_dataset: str = "topology-labeled-by-parameter",
    output_directory: str = "images/mm-vs-qm",
):
    """Generate MM-QM parameter comparison plots.
    
    Creates box plots showing (MM - QM) differences for each
    force field parameter across all topology types.
    """
    # Load force field to map parameter IDs to SMIRKS patterns.
    ff = ForceField(input_force_field)
    logger.info(f"Loaded force field: {input_force_field}")
    
    # Load dataset with topology values labeled by parameter ID.
    dataset = ds.dataset(input_dataset)
    df = dataset.to_table().to_pandas()
    logger.info(f"Loaded {len(df)} topology value records from {input_dataset}")

    # Filter to conformers present in all MM methods (common set).
    qcarchive_ids = set(df.qcarchive_id.values)
    logger.info(f"Starting with {len(qcarchive_ids)} unique conformers")
    for ff_name, subdf in df.groupby("method"):
        qcarchive_ids &= set(subdf.qcarchive_id.values)
        logger.info(f"Filtered to {len(qcarchive_ids)} conformers present in {ff_name}")

    df = df[df.qcarchive_id.isin(qcarchive_ids)]
    logger.info(f"Using {len(df)} records from {len(qcarchive_ids)} common conformers")
    
    # Separate data by topology type.
    angles = df[df.topology == "Angles"]
    bonds = df[df.topology == "Bonds"]
    torsions = df[df.topology == "ProperTorsions"]
    impropers = df[df.topology == "ImproperTorsions"]
    logger.info(f"Topology counts: {len(bonds)} bonds, {len(angles)} angles, "
                f"{len(torsions)} proper torsions, {len(impropers)} improper torsions")

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Generate plots for bonds.
    logger.info("Plotting bond parameters...")
    bond_directory = output_directory / "bonds"
    bond_directory.mkdir(parents=True, exist_ok=True)
    handler = ff.get_parameter_handler("Bonds")
    plot_with_handler(
        bonds,
        handler,
        unit="$\AA$",
        output_directory=str(bond_directory)
    )
    
    # Generate plots for angles.
    logger.info("Plotting angle parameters...")
    angle_directory = output_directory / "angles"
    angle_directory.mkdir(parents=True, exist_ok=True)
    handler = ff.get_parameter_handler("Angles")
    plot_with_handler(
        angles,
        handler,
        unit="°",
        output_directory=str(angle_directory)
    )

    # Generate plots for proper torsions.
    logger.info("Plotting proper torsion parameters...")
    torsion_directory = output_directory / "propers"
    torsion_directory.mkdir(parents=True, exist_ok=True)
    handler = ff.get_parameter_handler("ProperTorsions")
    plot_with_handler(
        torsions,
        handler,
        unit="°",
        output_directory=str(torsion_directory)
    )
    
    # Generate plots for improper torsions.
    logger.info("Plotting improper torsion parameters...")
    improper_directory = output_directory / "impropers"
    improper_directory.mkdir(parents=True, exist_ok=True)
    handler = ff.get_parameter_handler("ImproperTorsions")
    plot_with_handler(
        impropers,
        handler,
        unit="°",
        output_directory=str(improper_directory)
    )
    
    logger.info(f"All plots saved to {output_directory}")

if __name__ == "__main__":
    main()
