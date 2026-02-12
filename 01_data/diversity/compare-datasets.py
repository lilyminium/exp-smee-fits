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
    "--other-dataset",
    "-od",
    type=str,
    default="qca/output/combined/minimum-all",
    help="Path to the other dataset (either a text file or a HuggingFace dataset)"
)
@click.option(
    "--spice-dataset",
    "-sd",
    type=str,
    default="spice/smiles.json",
    help="Path to the SPICE dataset (a JSON file containing a list of SMILES)"
)
@click.option(
    "--output-file",
    "-of",
    type=str,
    default="output.pkl",
    help="Path to save the output pickle file containing fingerprints and t-SNE results"
)
@click.option(
    "--output-image",
    "-oi",
    type=str,
    default="output.png",
    help="Path to save  the t-SNE plot image"
)
def main(
    other_dataset: str = "qca/output/combined/minimum-all",
    spice_dataset: str = "spice/smiles.json",
    output_file: str = "output.pkl",
    output_image: str = "output.png",
):
    with open(spice_dataset, "r") as f:
        spice_data = json.load(f)

    logger.info(f"Loaded {len(spice_data)} entries from SPICE dataset")

    if pathlib.Path(other_dataset).is_file():
        with open(other_dataset, "r") as f:
            other_data = [x.strip() for x in f.readlines()]
    else:
        other_data = datasets.Dataset.load_from_disk(other_dataset)["smiles"]

    logger.info(f"Loaded {len(other_data)} entries from other dataset")

    spice_fingerprints = []
    failed = []
    mfpgen = rdFingerprintGenerator.GetMorganGenerator()
    for smiles in tqdm.tqdm(spice_data, desc="Computing SPICE fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            failed.append(smiles)
            continue
        fp = mfpgen.GetFingerprintAsNumPy(mol)
        spice_fingerprints.append(fp)

    other_fingerprints = []
    failed_other = []
    for smiles in tqdm.tqdm(other_data, desc="Computing other dataset fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            failed_other.append(smiles)
            continue
        fp = mfpgen.GetFingerprintAsNumPy(mol)
        other_fingerprints.append(fp)

    # PCA preprocessing before t-SNE
    features = np.array(spice_fingerprints + other_fingerprints)
    logger.info("Starting PCA...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    logger.info("PCA completed.")
    logger.info("Starting t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
    tsne_results = tsne.fit_transform(features_pca)
    logger.info("t-SNE completed.")

    spice_data = [x for x in spice_data if x not in failed]
    other_data = [x for x in other_data if x not in failed_other]

    obj_to_pickle = {
        "spice_smiles": spice_data,
        "other_smiles": other_data,
        "pca": pca,
        "pca_results": features_pca,
        "tsne": tsne,
        "tsne_results": tsne_results,
    }

    with open(output_file, "wb") as f:
        pickle.dump(obj_to_pickle, f)
        
    # quickly construct df and plot
    df = pd.DataFrame(tsne_results, columns=["tsne1", "tsne2"])
    df["dataset"] = ["SPICE"] * len(spice_fingerprints) + ["QCArchive"] * len(other_fingerprints)
    df["smiles"] = list(spice_data) + list(other_data)
    csv_file = str(output_file).rsplit(".", maxsplit=1)[0] + ".csv"
    df.to_csv(csv_file)
    logger.info(f"Saved dataframe to {csv_file}")
    ax = sns.scatterplot(data=df, x="tsne1", y="tsne2", hue="dataset", alpha=0.5, s=3)
    ax.set_title("t-SNE of SPICE vs QCArchive SMILES")
    plt.legend()
    plt.savefig(output_image, dpi=300)
    logger.info(f"Saved t-SNE plot to {output_image}")


if __name__ == "__main__":
    main()
