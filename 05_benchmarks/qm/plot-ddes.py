"""Visualize double-delta energy (ddE) distributions from QM benchmarks.

This script generates publication-quality plots comparing force field accuracy
using ddE data computed from RMSD-matched QM and MM conformers. The ddE metric
represents the relative energy error between MM and QM conformational preferences:
    ddE = (E_MM - E_MM_min) - (E_QM - E_QM_min)

Workflow:
1. Load ddE data from parquet files
2. Optionally filter to specific force fields
3. Generate comparison plots:
   - ECDF plots showing cumulative |ddE| distributions
   - Histogram step plots showing ddE distributions around zero
4. Save plots to output directory

Inputs:
- {input_directory}/: Parquet dataset with ddE data (from get-all-to-all-dde.py)
  Required columns: Force field, ddE, n_conformers

Outputs (saved to {output_directory}/):
- dde.png: ECDF plot of |ddE| up to 12 kcal/mol (full range)
- dde-close.png: ECDF plot of |ddE| up to 5 kcal/mol (close-up)
- dde-step.png: Histogram step plot of ddE without legend
- dde-step-legend.png: Histogram step plot of ddE with legend

The ECDF plots show what fraction of conformers have errors below a given
threshold, making it easy to compare force field accuracy. The step plots
show the distribution symmetry and identify systematic biases.
"""

import functools
import pathlib
import click

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from openff.toolkit import Molecule
import seaborn as sns
from matplotlib import pyplot as plt
from loguru import logger


@functools.cache
def smiles_to_inchi(smiles: str) -> str:
    """Convert SMILES string to InChI identifier.
    
    Uses caching to avoid repeated conversions of the same molecule.
    
    Parameters
    ----------
    smiles : str
        SMILES string to convert.
    
    Returns
    -------
    str
        InChI identifier with fixed hydrogens.
    """
    mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    return mol.to_inchi(fixed_hydrogens=True)


def plot_ecdf(
    data: pd.DataFrame,
    palette: dict,
    xmax=1,
    xlabel: str = r"|ddE| ($\mathrm{kcal \, mol^{-1}}$)",
    imgfile: pathlib.Path = None,
):
    """Generate ECDF plot of absolute ddE values.
    
    Creates an empirical cumulative distribution function plot showing
    the count of conformers below each |ddE| threshold.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns: |ddE|, Force field
    palette : dict
        Color mapping for force fields.
    xmax : float, optional
        Maximum x-axis value (kcal/mol), by default 1
    xlabel : str, optional
        X-axis label with LaTeX formatting
    imgfile : pathlib.Path, optional
        Output path for saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    rmsd_ax = sns.ecdfplot(
        ax=ax,
        data=data,
        x="|ddE|",
        hue="Force field",
        # palette=palette,  # Uncomment to use custom colors
        stat="count",
    )
    rmsd_ax.set_xlim(0, xmax)
    rmsd_ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(imgfile, dpi=300)
    logger.info(f"Saved ddE ECDF plot to {imgfile}")
    plt.close()


@click.command()
@click.option(
    "--ff-name-and-stem",
    "-ff",
    "ff_name_and_stems",
    type=(str, str),
    multiple=True,
    default=None,
    required=False,
    help=(
        "Force field display name and dataset stem name (e.g., 'Sage 2.2.1' 'openff_unconstrained-2.2.1'). "
        "Can be specified multiple times for multiple force fields. "
        "If omitted, all force fields in the input directory will be included with their dataset names as labels."
    ),
)
@click.option(
    "--n-last",
    "-n",
    "n_last",
    type=int,
    default=12,
    help="If specified, only include the last N force fields sorted.",
)
@click.option(
    "--must-include",
    "-m",
    "must_include",
    multiple=True,
    default=["openff-2.3.0"],
    help="Force fields that must always be included, regardless of sorting or filtering.",
)
@click.option(
    "--input-directory",
    "-i",
    "input_directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="ddEs",
    help="Directory containing ddE parquet dataset.",
)
@click.option(
    "--output-directory",
    "-o",
    "output_directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="images",
    help="Directory to write output plot files.",
)
def main(
    ff_name_and_stems: list[tuple[str, str]],
    n_last: int = None,
    must_include: list[str] = ["openff-2.3.0"],
    input_directory: str = "ddEs",
    output_directory: str = "images",
):
    """Generate ddE distribution plots for force field comparison.
    
    Creates ECDF and histogram plots comparing force field accuracy
    based on double-delta energies.
    """
    # Load ddE data from parquet dataset.
    dataset = ds.dataset(input_directory)
    
    # Determine which force fields to include.
    if not ff_name_and_stems:
        if n_last > 0:
            # Get all unique force fields sorted alphabetically.
            all_ffs = sorted(
                dataset.to_table(columns=["Force field"]).to_pydict()["Force field"]
            )
            # Select the last N force fields.
            selected_ffs = all_ffs[-n_last:]
            # Ensure must-include force fields are included.
            for ff in must_include:
                if ff not in selected_ffs:
                    selected_ffs.append(ff)
            ff_name_and_stems = [
                (ff_name, ff_name) for ff_name in selected_ffs
            ]
        else:
            # If not specified, include all force fields in the dataset.
            ff_name_and_stems = [
                (ff_name, ff_name) for ff_name in dataset.to_table(columns=["Force field"]).to_pydict()["Force field"]
            ]
    unique_ffs = [
        ff_name_and_stem[1] for ff_name_and_stem in ff_name_and_stems
    ]

    stem_to_name = {
        ff_name_and_stem[1]: ff_name_and_stem[0]
        for ff_name_and_stem in ff_name_and_stems
    }
    
    # Filter dataset to requested force fields.
    dataset = dataset.filter(
        pc.field("Force field").isin(unique_ffs)
    )
    df = dataset.to_table().to_pandas()
    df["FF"] = df["Force field"].map(stem_to_name)
    unique_ffs = df["FF"].unique()
    df["|ddE|"] = df["ddE"].abs()
    logger.info(f"Loaded {len(df)} ddE records from {input_directory}")

    # Log summary statistics about loaded data.
    n_ddes = df.groupby("Force field").size().to_dict()
    count_str = ", ".join(
        [f"{ff}: {count}" for ff, count in n_ddes.items()]
    )
    logger.info(f"ddE entries per force field: {count_str}")

    n_conformers = df.groupby("Force field")["n_conformers"].sum().to_dict()
    count_str = ", ".join(
        [f"{ff}: {count}" for ff, count in n_conformers.items()]
    )
    logger.info(f"Total conformers per force field: {count_str}")

    n_unique_ffs = len(unique_ffs)
    logger.info(f"Found {n_unique_ffs} unique force fields in {input_directory}")

    # Generate color palette for force fields.
    palette = sns.color_palette(n_colors=len(unique_ffs))
    PALETTE = {
        unique_ffs[i]: palette[i] for i in range(len(unique_ffs))
    }

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Generate ECDF plots at different scales.
    for xmax, filename in [(12, "dde.png"), (5, "dde-close.png")]:
        plot_ecdf(
            data=df,
            palette=PALETTE,
            xmax=xmax,
            xlabel=r"ddE ($\mathrm{kcal \, mol^{-1}}$)",
            imgfile=output_directory / filename,
        )

    # Generate histogram step plots (with and without legend).
    for legend in [True, False]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            ax=ax,
            data=df,
            x="ddE",
            hue="Force field",
            palette=PALETTE,
            binrange=(-10.25, 10.25),
            binwidth=0.5,
            element="step",  # Line-based histogram
            fill=False,
            legend=legend,
        )
        ax.set_xlabel("ddE (kcal/mol)")
        plt.tight_layout()
        imgfile = output_directory / f"dde-step{'-legend' if legend else ''}.png"
        plt.savefig(imgfile, dpi=300)
        logger.info(f"Saved ddE step plot to {imgfile}")
        plt.close()
    
    logger.info(f"All plots saved to {output_directory}")


if __name__ == "__main__":
    main()
