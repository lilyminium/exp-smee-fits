"""Visualize RMSD and TFD distributions from QM benchmarks.

This script generates publication-quality plots comparing force field accuracy
using RMSD (root-mean-square deviation of heavy atoms) and TFD (torsion fingerprint
deviation) metrics computed from MM-optimized and QM reference conformers.

Workflow:
1. Load RMSD/TFD data from parquet files
2. Optionally filter to specific force fields
3. Filter to conformers present in all force fields (common set)
4. Generate comparison plots:
   - ECDF plots showing cumulative distributions
   - Box plots showing distribution summaries
5. Save plots to output directory

Metrics:
- RMSD: Heavy-atom coordinate deviation in Angstroms (lower is better)
  - Good: <0.5 Å, Excellent: <0.2 Å
- TFD: Torsion angle deviation, dimensionless 0-1 scale (lower is better)
  - Good: <0.2, Excellent: <0.1

Inputs:
- {input_directory}/: Parquet dataset with RMSD/TFD data
  Required columns: qcarchive_id (or other ID), rmsd, tfd, method

Outputs (saved to {output_directory}/):
- rmsd{suffix}.png: ECDF plot of RMSD up to 1.0 Å
- rmsd-close{suffix}.png: ECDF plot of RMSD up to 0.4 Å (close-up)
- rmsd-box{suffix}.png: Box plot of RMSD distributions
- tfd{suffix}.png: ECDF plot of TFD up to 0.4
- tfd-close{suffix}.png: ECDF plot of TFD up to 0.1 (close-up)
- tfd-box{suffix}.png: Box plot of TFD distributions

The ECDF plots show what fraction of conformers have errors below a given
threshold. Box plots show median, quartiles, and outliers for each force field.
"""

import functools
import pathlib
import click
import time


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
    x: str,
    palette: dict,
    xmax=1,
    xlabel: str = r"Heavy atom RMSD ($\AA$)",
    imgfile: pathlib.Path = None,
):
    """Generate ECDF plot of RMSD or TFD values.
    
    Creates an empirical cumulative distribution function plot showing
    the count of conformers below each metric threshold.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns: {x}, Force field
    x : str
        Column name to plot (e.g., 'rmsd' or 'tfd')
    palette : dict
        Color mapping for force fields.
    xmax : float, optional
        Maximum x-axis value, by default 1
    xlabel : str, optional
        X-axis label with LaTeX formatting
    imgfile : pathlib.Path, optional
        Output path for saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    rmsd_ax = sns.ecdfplot(
        ax=ax,
        data=data,
        x=x,
        hue="Force field",
        # palette=palette,  # Uncomment to use custom colors
        stat="count",
    )
    rmsd_ax.set_xlim(0, xmax)
    rmsd_ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(imgfile, dpi=300)
    logger.info(f"Saved {x.upper()} ECDF plot to {imgfile}")
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
    default="rmsd",
    help="Directory containing RMSD/TFD parquet dataset.",
)
@click.option(
    "--output-directory",
    "-o",
    "output_directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default="images",
    help="Directory to write output plot files.",
)
@click.option(
    "--suffix",
    "-s",
    "suffix",
    type=str,
    default="",
    help="Suffix to append to output file names (e.g., '-v2' for rmsd-v2.png).",
)
@click.option(
    "--qca-id-col",
    "qcarchive_id_col",
    type=str,
    default="qcarchive_id",
    help="Column name to use as unique conformer identifier.",
)
def main(
    ff_name_and_stems: list[tuple[str, str]],
    n_last: int = None,
    must_include: list[str] = ["openff-2.3.0"],
    input_directory: str = "rmsd",
    output_directory: str = "images",
    suffix: str = "",
    qcarchive_id_col: str = "qcarchive_id",
):
    """Generate RMSD and TFD distribution plots for force field comparison.
    
    Creates ECDF and box plots comparing force field accuracy based on
    structural similarity metrics (RMSD and TFD).
    """
    logger.info(f"{time.ctime()} - Starting RMSD/TFD visualization")
    start_time = time.time()

    # Load RMSD/TFD data from parquet dataset.
    rmsd_dataset = ds.dataset(input_directory)
    unique_methods = sorted(
        set(
            rmsd_dataset.to_table(columns=["method"]).to_pydict()["method"]
        )
    )
    logger.info(f"Available force fields in dataset: {unique_methods}")
    
    # Build mapping from force field stem names to display names.
    FF_STEM_TO_NAME = {}
    if ff_name_and_stems:
        FF_STEM_TO_NAME = {
            stem: name for name, stem in ff_name_and_stems
        }
        # Filter to requested force fields.
        rmsd_dataset = rmsd_dataset.filter(
            pc.field("method").isin(FF_STEM_TO_NAME.keys())
        )
    elif n_last > 0:
        # Select last N force fields sorted alphabetically, plus any must-include.
        selected_stems = sorted(unique_methods)[-n_last:]
        for stem in must_include:
            if stem not in selected_stems and stem in unique_methods:
                selected_stems.append(stem)
        FF_STEM_TO_NAME = {
            stem: stem for stem in selected_stems
        }
        rmsd_dataset = rmsd_dataset.filter(
            pc.field("method").isin(FF_STEM_TO_NAME.keys())
        )
    else:
        # If not specified, include all force fields in the dataset.
        unique_stems = rmsd_dataset.to_table(
            columns=["method"]
        ).to_pydict()["method"]
        FF_STEM_TO_NAME = {
            stem: stem for stem in unique_stems
        }
    logger.info(f"Force fields and display labels: {FF_STEM_TO_NAME}")

    rmsd_df = rmsd_dataset.to_table(
        columns=[qcarchive_id_col, "rmsd", "tfd", "method"]
    ).to_pandas()
    logger.info(f"Loaded {len(rmsd_df)} RMSD/TFD records from {input_directory}")

    # Count unique force fields.
    unique_ffs = rmsd_df["method"].unique()
    n_unique_ffs = len(unique_ffs)
    logger.info(f"Found {n_unique_ffs} unique force fields")

    # Filter to conformers present in all force fields (common set).
    counts = rmsd_df.groupby(qcarchive_id_col)["method"].nunique()
    qcarchive_ids = counts[counts == n_unique_ffs].index
    logger.info(f"Found {len(qcarchive_ids)} conformers present in all force fields")
    rmsd_df = rmsd_df[rmsd_df[qcarchive_id_col].isin(qcarchive_ids)]
    rmsd_df["Force field"] = rmsd_df["method"].map(FF_STEM_TO_NAME)
    logger.info(f"Using {len(rmsd_df)} total records from common conformer set")

    # Generate color palette for force fields.
    palette = sns.color_palette(n_colors=len(FF_STEM_TO_NAME))
    ff_names = list(FF_STEM_TO_NAME.values())
    PALETTE = {
        ff_names[i]: palette[i] for i in range(len(FF_STEM_TO_NAME))
    }

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Generate RMSD plots (ECDF at different scales and box plot).
    logger.info("Generating RMSD plots...")
    for xmax, filename in [(1, f"rmsd{suffix}.png"), (0.4, f"rmsd-close{suffix}.png")]:
        plot_ecdf(
            data=rmsd_df,
            x="rmsd",
            palette=PALETTE,
            xmax=xmax,
            xlabel=r"Heavy atom RMSD ($\AA$)",
            imgfile=output_directory / filename,
        )
    
    # Generate RMSD box plot.
    ax = sns.boxplot(data=rmsd_df, y="method", hue="Force field", palette=PALETTE, x="rmsd")
    ax.set_ylim(0, 10)
    imgfile = output_directory / f"rmsd-box{suffix}.png"
    plt.savefig(imgfile, dpi=300)
    logger.info(f"Saved RMSD box plot to {imgfile}")
    plt.close()

    # Generate TFD plots (ECDF at different scales and box plot).
    logger.info("Generating TFD plots...")
    for xmax, filename in [(0.4, f"tfd{suffix}.png"), (0.1, f"tfd-close{suffix}.png")]:
        plot_ecdf(
            data=rmsd_df,
            x="tfd",
            palette=PALETTE,
            xmax=xmax,
            xlabel="TFD",
            imgfile=output_directory / filename,
        )
    
    # Generate TFD box plot.
    ax = sns.boxplot(data=rmsd_df, y="method", hue="Force field", palette=PALETTE, x="tfd")
    ax.set_ylim(0, 1)
    imgfile = output_directory / f"tfd-box{suffix}.png"
    plt.savefig(imgfile, dpi=300)
    logger.info(f"Saved TFD box plot to {imgfile}")
    plt.close()
    
    elapsed_time = time.time() - start_time
    logger.info(f"{time.ctime()} - Finished in {elapsed_time:.2f} seconds")
    logger.info(f"All plots saved to {output_directory}")



if __name__ == "__main__":
    main()
