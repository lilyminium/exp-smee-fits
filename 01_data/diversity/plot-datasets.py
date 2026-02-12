"""
Compare two lists of SMILES in chemical space using RDKit.

We use Morgan fingerprint to characterize each molecule,
then visualize the diversity using t-SNE.
"""
import json
import pathlib
import pickle
import tqdm

import click
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from loguru import logger
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import datasets


@click.command()
@click.option(
    "--input-file",
    "-i",
    type=str,
    default="output.pkl",
    help="Path to save the output pickle file containing fingerprints and t-SNE results"
)
@click.option(
    "--output-image",
    "-o",
    type=str,
    default="output.png",
    help="Path to save  the t-SNE plot image"
)
def main(
    input_file: str = "output.pkl",
    output_image: str = "output.png",
):

    # obj_to_pickle = {
    #     "spice_smiles": spice_data,
    #     "other_smiles": other_data,
    #     "pca": pca,
    #     "pca_results": features_pca,
    #     "tsne": tsne,
    #     "tsne_results": tsne_results,
    # }

    with open(input_file, "rb") as f:
        obj = pickle.load(f)
        
    # quickly construct df and plot
    df = pd.DataFrame(obj["tsne_results"], columns=["tsne1", "tsne2"])
    df["dataset"] = ["SPICE"] * len(obj["spice_smiles"]) + ["QCArchive"] * len(obj["other_smiles"])
    ax = sns.scatterplot(data=df, x="tsne1", y="tsne2", hue="dataset", alpha=0.5, s=3)
    ax.set_title("t-SNE of SPICE vs QCArchive SMILES")
    plt.legend()
    plt.savefig(output_image, dpi=300)
    logger.info(f"Saved t-SNE plot to {output_image}")


if __name__ == "__main__":
    main()
