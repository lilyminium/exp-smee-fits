"""
Compare multiple SMILES datasets in chemical space using RDKit fingerprints.

Workflow:
1) Load each dataset from a JSON list, a plain-text file of SMILES (one per
    line), or a HuggingFace dataset on disk with a `smiles` column.
2) Compute Morgan fingerprints with RDKit.
3) Reduce dimensionality with PCA and then t-SNE.
4) Save a pickle of intermediate artifacts, a CSV with t-SNE coordinates and
    labels, and a set of PNG plots. A seed can be provided for reproducibility.

The script also tags molecules by element presence, charge, and a small set
of functional-group SMARTS patterns (defined in `PATTERNS`) to support
additional faceted plots.

Outputs:
- output.pkl: SMILES, dataset labels, PCA/t-SNE models, coordinates, and
  label columns used for plotting.
- tsne_results.csv: t-SNE coordinates with labels.
- PNG images: overall scatter and faceted views by dataset/label.
"""
import collections
import json
import pathlib
import pickle

import click
import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from loguru import logger
from rdkit import Chem
from openff.toolkit import Molecule
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PATTERNS = {
    "Nitro": "[#7X3:1](~[#8X1:2])~[#8X1:3]",
    "Alcohol": "[#6X4:1]-[#8X2H1:2]",
    "Amide": "[#6X3:1](=[#8X1:2])-[#7X3:3]",
    "Ester": "[#6X3:1](=[#8X1:2])-[#8X2:3]-[#6,#1]",
    "Ether": "[#6X4:1]-[#8X2:2]-[#6,#1]",
    "Ketone": "[#6X3:1](=[#8X1:2])-[#6,#1]",
    "Aldehyde": "[#6X3:1](=[#8X1:2])-[#1]",
    "Nitrile": "[#6X2:1]#[#7X1:2]",
    "Sulfonamide": "[#6]-[#16X4:1](~[#8X1:2])(~[#8X1:2])-[#7X3:3](-[#6,#1])-[#6,#1]",
    "Sulfone": "[#6]-[#16X4:1](~[#8X1:2])(~[#8X1:2])-[#6,#1]",
    "Phosphate": "[#6]-[#15X4:1](~[#8X1:2])(~[#8X1:2])(~[#8X1:2])-[#6,#1]",
}


@click.command()
@click.option(
    "--input-dataset",
    "-i",
    "input_datasets",
    type=(str, str),
    multiple=True,
    default=None,
    help=(
        "Path to the other dataset (either a JSON file or a HuggingFace dataset), "
        "and name to label it in the plot. Can be specified multiple times for multiple datasets."
    )
)
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="output",
    help="Path to save the output pickle files, csv, and images"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for t-SNE (and numpy) for reproducible layouts",
)
def main(
    input_datasets: list[tuple[str, str]] = None,
    output_directory: str = "output",
    seed: int | None = None,
):
    """Load datasets, compute embeddings, and write plots/artifacts."""
    datasets_by_label: dict[str, list[str]] = {}
    for input_dataset, label in input_datasets:
        input_dataset = pathlib.Path(input_dataset)
        if input_dataset.is_file():
            if input_dataset.suffix.lower() == ".json":
                logger.info(f"Loading {label} dataset from JSON file: {input_dataset}")
                with open(input_dataset, "r") as f:
                    data = json.load(f)
                datasets_by_label[label] = list(set(data))
            else:
                with open(input_dataset, "r") as f:
                    data = [x.strip() for x in f.readlines()]
                logger.info(f"Loaded {len(data)} entries from {label} dataset")
                datasets_by_label[label] = list(set(data))
        else:
            logger.info(f"Loading {label} dataset from HuggingFace dataset: {input_dataset}")
            data = datasets.Dataset.load_from_disk(input_dataset)["smiles"]
            logger.info(f"Loaded {len(data)} entries from {label} dataset")
            datasets_by_label[label] = data

    fingerprints = []
    all_labels = []
    all_smiles = []
    label_flags = collections.defaultdict(list)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator()
    for label, smiles_list in datasets_by_label.items():
        for smiles in tqdm.tqdm(smiles_list, desc=f"Computing fingerprints for {label}"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                continue
            fp = mfpgen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
            all_labels.append(label)
            all_smiles.append(smiles)

            # Element presence flags for simple faceting.
            elements_in_mol = {atom.GetSymbol() for atom in mol.GetAtoms()}
            for element in ["N", "O", "S", "F", "Cl", "Br", "I", "P"]:
                label_flags[element].append(element in elements_in_mol)

            # Use OpenFF for charge and SMARTS matching on mapped SMILES.
            offmol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
            label_flags["Positive"].append(offmol.total_charge.m > 0.1)
            label_flags["Negative"].append(offmol.total_charge.m < -0.1)

            for pattern_name, pattern_smarts in PATTERNS.items():
                label_flags[pattern_name].append(
                    bool(offmol.chemical_environment_matches(pattern_smarts))
                )
            

    logger.info(
        f"Computed fingerprints for {len(fingerprints)} molecules across {len(datasets_by_label)} datasets",
    )

    # PCA preprocessing before t-SNE
    features = np.array(fingerprints)
    logger.info("Starting PCA...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    logger.info("PCA completed.")
    logger.info("Starting t-SNE...")
    if seed is not None:
        np.random.seed(seed)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=seed)
    tsne_results = tsne.fit_transform(features_pca)
    logger.info("t-SNE completed.")

    # Save artifacts for later inspection or reuse.
    obj_to_pickle = {
        "smiles": all_smiles,
        "labels": all_labels,
        "pca": pca,
        "pca_results": features_pca,
        "tsne": tsne,
        "tsne_results": tsne_results,
        "seed": seed,
    }
    obj_to_pickle.update(label_flags)

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    pickle_file = output_directory / "output.pkl"

    with open(pickle_file, "wb") as f:
        pickle.dump(obj_to_pickle, f)
    logger.info(f"Saved intermediate results to {pickle_file}")
        
    # Construct dataframe and save
    df = pd.DataFrame(tsne_results, columns=["tsne1", "tsne2"])
    df["Dataset"] = all_labels
    df["smiles"] = all_smiles
    for label in label_flags.keys():
        df[label] = label_flags[label]
    df["Charged"] = df["Positive"] | df["Negative"]
    df["Halogen"] = df[["F", "Cl", "Br", "I"]].any(axis=1)

    csvfile = output_directory / "tsne_results.csv"
    df.to_csv(csvfile, index=False)
    logger.info(f"Saved t-SNE results and labels to {csvfile}")

    # plot overall
    fig, ax = plt.subplots(figsize=(7, 7))
    ax = sns.scatterplot(ax=ax, data=df, x="tsne1", y="tsne2", hue="Dataset", alpha=0.5, s=3)
    xmin = round(min([df.tsne1.min(), df.tsne2.min()]) - 5, -1)
    xmax = round(max([df.tsne1.max(), df.tsne2.max()]) + 5, -1)
    plt.xticks(np.arange(xmin, xmax, step=10), minor=True)
    plt.yticks(np.arange(xmin, xmax, step=10), minor=True)
    ax.grid()
    ax.grid(ls="--", which="minor")

    ax.set_title("t-SNE of SPICE vs QCArchive SMILES")
    plt.legend()
    output_image = output_directory / "overall.png"
    plt.savefig(output_image, dpi=300)
    logger.info(f"Saved overall t-SNE plot to {output_image}")

    # plot by interesting labels
    cols_to_plot = ["Charged", "Halogen"] + list(PATTERNS.keys())
    # first, split on dataset
    for col in cols_to_plot:
        g = sns.FacetGrid(
            data=df,
            col="Dataset",
            hue=col,
            aspect=1,
            height=4,
        )
        g.map_dataframe(sns.scatterplot, x="tsne1", y="tsne2", s=3, alpha=0.5)
        g.add_legend()
        g.set_titles(col_template="{col_name}")
        for ax in g.axes.flatten():
            ax.set_xlabel("TSNE Component 1")
            ax.set_ylabel("TSNE Component 2")
            ax.grid(ls="--")
        
        pngfile = output_directory / f"all-{col.replace(' ', '-')}-by-dataset.png"
        plt.savefig(pngfile, dpi=300)
        logger.info(f"Saved t-SNE plot split by dataset and colored by {col} to {pngfile}")

    # next on label/element
    for col in cols_to_plot:
        g = sns.FacetGrid(
            data=df,
            col=col,
            hue="Dataset",
            aspect=1,
            height=4,
        )
        g.map_dataframe(sns.scatterplot, x="tsne1", y="tsne2", s=3, alpha=0.5)
        g.add_legend()
        g.figure.suptitle(col)
        g.set_titles(col_template="{col_name}")
        for ax in g.axes.flatten():
            ax.set_xlabel("TSNE Component 1")
            ax.set_ylabel("TSNE Component 2")
            ax.grid(ls="--")
        pngfile = output_directory / f"all-{col.replace(' ', '-')}-by-element.png"
        plt.savefig(pngfile, dpi=300)
        logger.info(f"Saved t-SNE plot split by {col} and colored by dataset to {pngfile}")



if __name__ == "__main__":
    main()
