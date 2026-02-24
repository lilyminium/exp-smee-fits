"""
Plot intermediate bond, angle, and torsion parameter values for each parameter type,
highlighting the evolution of parameters during training and focusing on the biggest
differences. Intermediate values are saved to CSV/JSON, and torsion profiles are
plotted to visualize energy as a function of dihedral angle.
"""

import json
from collections import defaultdict
import pathlib

import click
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from loguru import logger
from matplotlib import pyplot as plt


def retrieve_ff_values(
    optimized_ff_path: pathlib.Path | str,
) -> tuple[pd.DataFrame, dict[str, dict[int, float]]]:
    """
    Read optimized force field parameters from the checkpoint
    and convert to a DataFrame for plotting.
    """

    # Load the optimized force field that was saved during training
    optimized_ff_path = pathlib.Path(optimized_ff_path)
    logger.info(f"Loading optimized force field from: {optimized_ff_path}")

    if not optimized_ff_path.exists():
        raise FileNotFoundError(
            f"Optimized force field not found at {optimized_ff_path}. "
            "Ensure training has completed successfully."
        )
    
    if torch.cuda.is_available():
        smee_force_field = torch.load(optimized_ff_path)
    else:
        smee_force_field = torch.load(optimized_ff_path, map_location=torch.device('cpu'))

    bond_angle_rows = []
    torsion_data: dict[str, dict[int, float]] = defaultdict(dict)

    for potential in smee_force_field.potentials:
        handler_name = potential.parameter_keys[0].associated_handler
        if handler_name is None:
            logger.warning("Skipping potential with no associated handler")
            continue

        parameter_attrs = potential.parameter_cols
        parameter_units = potential.parameter_units

        if handler_name in ["Bonds", "Angles"]:
            # Update force constants and equilibrium values directly
            for param_key, opt_parameters in zip(
                potential.parameter_keys, potential.parameters
            ):
                smirks = param_key.id
                opt_parameters = opt_parameters.detach().cpu().numpy()

                for param_name, param_value, param_unit in zip(
                    parameter_attrs, opt_parameters, parameter_units
                ):
                    row = {
                        "parameter_type": handler_name,
                        "smirks": smirks,
                        "attribute": param_name,
                        "value": param_value,
                        "unit": str(param_unit),
                    }
                    bond_angle_rows.append(row)

        elif handler_name == "ProperTorsions":
            # Collect k values by periodicity for each SMIRKS pattern
            k_index = parameter_attrs.index("k")
            p_index = parameter_attrs.index("periodicity")

            for param_key, opt_parameters in zip(
                potential.parameter_keys, potential.parameters
            ):
                opt_parameters = opt_parameters.detach().cpu().numpy()
                k = opt_parameters[k_index]
                periodicity = int(opt_parameters[p_index])
                torsion_data[param_key.id][periodicity] = k

        elif handler_name == "ImproperTorsions":
            # Only fit v2 terms for improper torsions
            k_index = parameter_attrs.index("k")
            for param_key, opt_parameters in zip(
                potential.parameter_keys, potential.parameters
            ):
                smirks = param_key.id
                opt_parameters = opt_parameters.detach().cpu().numpy()
                row = {
                    "parameter_type": handler_name,
                    "smirks": smirks,
                    "attribute": "k",
                    "value": opt_parameters[k_index],
                    "unit": str(parameter_units[k_index]),
                }
                bond_angle_rows.append(row)
        else:
            continue

        logger.info(f"Updated {len(potential.parameters)} {handler_name} parameters")

    return pd.DataFrame(bond_angle_rows), torsion_data


def get_all_intermediate_values(
    directory
) -> tuple[pd.DataFrame, dict[str, dict[int | str, dict[int, float]]]]:
    directory = pathlib.Path(directory)
    intermediate_files = sorted(directory.glob("force-field-epoch-*.pt"))

    all_torsions = {}
    all_data = []

    for intermediate_file in tqdm.tqdm(
        intermediate_files,
        desc="Processing intermediate files"
    ):
        epoch = int(intermediate_file.stem.split("-")[-1])
        bai, propers = retrieve_ff_values(intermediate_file)
        bai["epoch"] = epoch
        all_data.append(bai)

        for smirks, values in propers.items():
            if smirks not in all_torsions:
                all_torsions[smirks] = {}
            all_torsions[smirks][epoch] = values

    # add final file
    final_ff = directory.parent / "final-force-field.pt"
    if final_ff.exists():
        epoch = "final"
        bai, propers = retrieve_ff_values(final_ff)
        bai["epoch"] = epoch
        all_data.append(bai)

        for smirks, values in propers.items():
            if smirks not in all_torsions:
                all_torsions[smirks] = {}
            all_torsions[smirks][epoch] = values
    else:
        logger.warning(f"Final force field not found at {final_ff}, skipping.")

    all_data_df = pd.concat(all_data, ignore_index=True)
    return all_data_df, all_torsions

def get_largest_difference_smirks(df: pd.DataFrame, n_top: int = 10) -> pd.DataFrame:
    final = df[df["epoch"] == "final"]
    initial = df[df["epoch"] == 0]

    merged = pd.merge(
        initial,
        final,
        on=["parameter_type", "smirks", "attribute"],
        suffixes=("_initial", "_final")
    )
    merged["difference"] = (merged["value_final"] - merged["value_initial"]).abs()
    sorted_by_diff = merged.sort_values("difference", ascending=False)
    return sorted_by_diff


def plot_all_highlight_highest(
    df: pd.DataFrame,
    n_top: int = 10,
    output_directory: str = "images"
):
    parameter_types = df.parameter_type.unique()
    if not len(parameter_types):
        raise ValueError("DataFrame should contain only one parameter type for plotting")
    parameter_type = parameter_types[0]

    attribute = df.attribute.unique()
    if len(attribute) != 1:
        raise ValueError("DataFrame should contain only one attribute for plotting")
    attribute = attribute[0]

    sorted_by_diff = get_largest_difference_smirks(df, n_top)

    integer_epochs = sorted(df[df["epoch"] != "final"]["epoch"].unique())
    epoch_step = integer_epochs[1] - integer_epochs[0]
    # replace final with integer
    df["epoch"] = df["epoch"].replace("final", integer_epochs[-1] + epoch_step)

    # now plot all values, highlighting the ones with the largest difference
    top_smirks = sorted_by_diff["smirks"].values[:n_top]
    PALETTE = {
        smirks: color
        for smirks, color in zip(
            top_smirks,
            sns.color_palette("husl", n_colors=n_top)
        )
    }
    for smirks in sorted_by_diff["smirks"].values[n_top:]:
        PALETTE[smirks] = "lightgray"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.lineplot(
        data=df,
        x="epoch",
        y="value",
        hue="smirks",
        palette=PALETTE,
    )
    # Only show legend for the top n_top SMIRKS with largest differences
    handles, labels = ax.get_legend_handles_labels()
    top_smirks_set = set(top_smirks)
    filtered = [(h, l) for h, l in zip(handles, labels) if l in top_smirks_set]
    if filtered:
        filtered_handles, filtered_labels = zip(*filtered)
        ax.legend(filtered_handles, filtered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.get_legend().remove()
    ax.set_title(f"{parameter_type} {attribute} values over epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{attribute} ({df['unit'].iloc[0]})")
    plt.tight_layout()

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_path = output_directory / f"{parameter_type}_{attribute}_evolution.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved plot to {plot_path}")
    plt.close()


def get_torsion_profile(
    values: dict[int, float]
) -> tuple[np.ndarray, np.ndarray]:
    def torsion(x):
        energy = 0.0
        for periodicity, k in values.items():
            energy += k * (1 + np.cos(periodicity * np.radians(x)))
        return energy
    
    xs = np.linspace(-180, 180, 361)
    ys = np.array([torsion(x) for x in xs])
    return xs, ys


def plot_torsion_profiles(
    torsion_data: dict[str, dict[int | str, dict[int, float]]],
    output_directory: str = "images"
):
    """Plot torsion energy profiles for each SMIRKS pattern across epochs."""
    torsion_directory = pathlib.Path(output_directory) / "torsion_profiles"
    torsion_directory.mkdir(parents=True, exist_ok=True)

    for i, (smirks, epoch_values) in enumerate(torsion_data.items(), 1):
        # Convert 'final' epoch to an integer for plotting
        final = epoch_values.pop("final")
        all_epochs = sorted(epoch_values.keys())
        step = all_epochs[1] - all_epochs[0]
        epoch_values[all_epochs[-1] + step] = final

        df_rows = []
        for epoch, values in epoch_values.items():
            xs, ys = get_torsion_profile(values)
            for x, y in zip(xs, ys):
                df_rows.append({
                    "Epoch": epoch,
                    "Angle (°)": x,
                    "Energy (kcal/mol)": y
                })
        df = pd.DataFrame(df_rows)
        ax = sns.lineplot(
            data=df,
            x="Angle (°)",
            y="Energy (kcal/mol)",
            hue="Epoch",
            palette=sns.color_palette("viridis_r", n_colors=len(epoch_values)),
        )
        # ax.set_title(smirks)
        # get first and last epochs and just restrict legend to those
        epochs = sorted(epoch_values.keys())
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l in [str(epochs[0]), str(epochs[-1])]]
        if filtered:
            filtered_handles, filtered_labels = zip(*filtered)
            ax.legend(filtered_handles, filtered_labels, title="Epoch", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.get_legend().remove()
        ax.set_title(smirks.replace("$", r"\$"))
        plt.tight_layout()
        pngfile = torsion_directory / f"torsion-{i:03d}.png"
        plt.savefig(pngfile, dpi=300)
        logger.info(f"Saved torsion profile plot to {pngfile}")
        plt.close()

@click.command()
@click.option(
    "--intermediate-directory",
    "-i",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to the directory containing intermediate force field files."
)
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    help="Path to the output directory for images."
)
@click.option(
    "--n-top",
    "-n",
    type=int,
    default=10,
    help="Number of top parameters with the largest differences to highlight in the plots."
)
def main(
    intermediate_directory: str,
    output_directory: str,
    n_top: int = 10
):
    """
    Plot intermediate bond, angle, and torsion parameter values for each parameter type,
    highlighting the evolution of parameters during training and focusing on the biggest
    differences. Intermediate values are saved to CSV/JSON, and torsion profiles are plotted.
    """
    
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    intermediate_directory = pathlib.Path(intermediate_directory)

    final_ff = intermediate_directory.parent / "final-force-field.pt"
    assert final_ff.exists(), f"Final force field not found at {final_ff}. Ensure training has completed successfully."

    all_data_df, all_torsions = get_all_intermediate_values(intermediate_directory)

    # Save the intermediate values to CSV for further analysis
    csv_path = output_directory / "intermediate_bond_angle_improper_parameters.csv"
    all_data_df.to_csv(csv_path, index=False)
    logger.info(f"Saved intermediate parameter values to {csv_path}")

    json_path = output_directory / "intermediate_torsion_parameters.json"
    with open(json_path, "w") as f:
        json.dump(all_torsions, f, indent=4)
    logger.info(f"Saved intermediate torsion parameter values to {json_path}")

    # plot bonds and angles
    for parameter_type, attr in [
        ("Bonds", "k"),
        ("Bonds", "length"),
        ("Angles", "k"),
        ("Angles", "angle"),
        ("ImproperTorsions", "k")
    ]:
        df_subset = all_data_df[
            (all_data_df["parameter_type"] == parameter_type) &
            (all_data_df["attribute"] == attr)
        ]
        if attr == "angle":
            # convert from radians to degrees for better interpretability
            df_subset = pd.DataFrame(df_subset)  # make a copy to avoid SettingWithCopyWarning
            df_subset["value"] = np.degrees(df_subset["value"])
            df_subset["unit"] = "degrees"
        plot_all_highlight_highest(df_subset, n_top=n_top, output_directory=output_directory)
    
    # plot torsions as torsion profiles
    plot_torsion_profiles(all_torsions, output_directory=output_directory)


if __name__ == "__main__":
    main()
