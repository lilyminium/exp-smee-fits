"""Train a SMEE force field using gradient descent on QM reference data.

This script loads a pre-parameterized force field and molecular dataset,
then optimizes force field parameters (bonds, angles, torsions) to minimize
the loss between predicted and reference energies and forces.

Supports two training modes:
1. Standard: One optimizer step per epoch (default)
2. Mini-batch: Multiple optimizer steps per epoch for faster convergence

Workflow:
1. Load configuration, dataset, and pre-parameterized force field
2. Set up trainable parameters (bonds, angles, proper/improper torsions)
3. Train using Adam optimizer with batch processing on GPU
4. Log metrics to TensorBoard and save intermediate checkpoints
5. Export final optimized force field to OFFXML format

Inputs (via config file):
- input_force_field: Starting OFFXML force field
- input_dataset: HuggingFace dataset with QM energies/forces
- input_parameterized: Pickled SMEE force field and topology dict
- learning_rate, n_epochs, batch_size: Training hyperparameters
- minibatch_size (optional): Size of mini-batches for more frequent updates
- energy_weight, force_weight: Loss term weights
- parameter_configs: Which parameters to optimize and their constraints

Outputs:
- final-force-field.pt: Optimized SMEE force field (PyTorch)
- final-force-field.offxml: Optimized force field in OpenFF format
- metrics/: TensorBoard logs for loss curves
- intermediate/: Checkpoints saved periodically
- loss_plots.png: Training and validation loss curves
"""

import copy
from collections import defaultdict
import math
import pathlib
import pickle
import typing

import click
import tqdm
import yaml
import more_itertools

import datasets
import descent
import descent.train
import descent.targets
import descent.targets.energy
import torch
import numpy as np
from openff.toolkit import Molecule, ForceField, unit
from openff.interchange.models import PotentialKey
from matplotlib import pyplot as plt
import tensorboardX
import tbparse

from loguru import logger


def prepare_batch_for_device(
    batch: datasets.Dataset, device_str: typing.Literal["cpu", "cuda"]
) -> list[dict[str, torch.Tensor | typing.Any]]:
    """Prepare a batch for the target device (CPU or CUDA).

    Converts coordinate, energy, and force tensors to the specified device
    before passing to descent.targets.energy.predict().

    Parameters
    ----------
    batch : datasets.Dataset
        Batch of molecular data from HuggingFace dataset.
    device_str : str
        Target device string ("cpu" or "cuda").

    Returns
    -------
    list
        List of entries with tensors moved to the target device.
    """
    device_batch = []
    for entry in batch:
        entry_copy = {}
        for key, value in entry.items():
            if key in ["coords", "energy", "forces"]:
                if isinstance(value, torch.Tensor):
                    entry_copy[key] = value.to(device_str)
                else:
                    entry_copy[key] = torch.tensor(value, device=device_str)
            else:
                entry_copy[key] = value
        device_batch.append(entry_copy)
    return device_batch


def write_metrics(
    epoch: int,
    loss: torch.Tensor,
    loss_energy: torch.Tensor,
    loss_forces: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
    prefix: str,
) -> None:
    """Write training metrics to console and TensorBoard.

    Logs training progress including total loss, energy loss, force loss,
    and corresponding RMSE values to both console output and TensorBoard
    for monitoring and visualization.

    Parameters
    ----------
    epoch : int
        Current training epoch number.
    loss : torch.Tensor
        Total loss (energy + force loss) for the epoch.
    loss_energy : torch.Tensor
        Energy-specific loss component for the epoch.
    loss_forces : torch.Tensor
        Force-specific loss component for the epoch.
    writer : tensorboardX.SummaryWriter
        TensorBoard writer object for logging metrics.
    prefix : str
        Prefix for metric names (e.g., "train" or "validation").

    Returns
    -------
    None

    Notes
    -----
    TensorBoard metrics logged:
    - loss: Total combined loss
    - loss_energy: Energy component loss
    - loss_forces: Force component loss
    - rmse_energy: Square root of energy loss
    - rmse_forces: Square root of force loss

    Examples
    --------
    >>> with tensorboardX.SummaryWriter("logs") as writer:
    ...     write_metrics(10, epoch_loss, energy_loss, force_loss, writer, prefix="train")
    """
    logger.info(f"epoch={epoch} {prefix} loss={loss.detach().item():.6f}", flush=True)

    writer.add_scalar(f"{prefix}_loss", loss.detach().item(), epoch)
    writer.add_scalar(f"{prefix}_loss_energy", loss_energy.detach().item(), epoch)
    writer.add_scalar(f"{prefix}_loss_forces", loss_forces.detach().item(), epoch)

    writer.add_scalar(
        f"{prefix}_rmse_energy", math.sqrt(loss_energy.detach().item()), epoch
    )
    writer.add_scalar(
        f"{prefix}_rmse_forces", math.sqrt(loss_forces.detach().item()), epoch
    )
    writer.flush()


def write_hparams(writer: tensorboardX.SummaryWriter, config: dict) -> None:
    """Write hyperparameters to TensorBoard for run comparison.
    
    Parameters
    ----------
    writer : tensorboardX.SummaryWriter
        TensorBoard writer object.
    config : dict
        Configuration dictionary with training hyperparameters.
    """
    for v in tensorboardX.writer.hparams(
        {
            "lr": config["learning_rate"],
            "n_epochs": config["n_epochs"],
            "batch_size": config["batch_size"],
            "minibatch_size": config.get("minibatch_size", None),
            "energy_weight": config["energy_weight"],
            "force_weight": config["force_weight"],
            "energy_reference": config["energy_reference"],
            "input_force_field": config["input_force_field"],
            "input_dataset": config["input_dataset"],
            "input_parameterized": config["input_parameterized"],
        },
        {},
    ):
        writer.file_writer.add_summary(v)



def initialize_parameter_configs(
    initial_parameter_configs: dict,
    input_force_field: str,
) -> dict[str, descent.train.ParameterConfig]:
    """Initialize parameter configurations with automatic exclusions.
    
    Automatically excludes linear angles (180°) and zero-barrier torsions
    from optimization, as these are typically constrained by chemistry.
    Also converts angle limits from degrees to radians.

    Parameters
    ----------
    initial_parameter_configs : dict
        Initial parameter configurations from config file.
        Expected keys: Bonds, Angles, ProperTorsions, ImproperTorsions.
    input_force_field : str
        Path to input OFFXML force field file.

    Returns
    -------
    dict[str, descent.train.ParameterConfig]
        Initialized parameter configurations with exclusions applied.
    """

    initial_parameter_configs = copy.deepcopy(initial_parameter_configs)

    expected_keys = ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions"]
    for key in initial_parameter_configs:
        if key not in expected_keys:
            raise ValueError(f"Unexpected key in parameter_configs: {key}")

    ff = ForceField(input_force_field)

    # Automatically exclude parameters that should not be optimized.
    if "Angles" in initial_parameter_configs:
        angle_handler = ff.get_parameter_handler("Angles")
        # Don't optimize linear angles (180°) - these are chemically constrained.
        exclude_angles = [
            parameter.smirks
            for parameter in angle_handler.parameters
            if np.isclose(parameter.angle.m_as(unit.degrees), 180.0)
        ]
        logger.info(
            f"Excluding {len(exclude_angles)} linear angles: {', '.join(exclude_angles)}"
        )
        initial_parameter_configs["Angles"]["exclude"] = [
            PotentialKey(id=x)
            for x in exclude_angles
        ]

        if "limits" in initial_parameter_configs["Angles"] and "angle" in initial_parameter_configs["Angles"]["limits"]:
            initial_parameter_configs["Angles"]["limits"]["angle"] = [np.deg2rad(x) for x in initial_parameter_configs["Angles"]["limits"]["angle"]]

    if "ProperTorsions" in initial_parameter_configs:
        torsion_handler = ff.get_parameter_handler("ProperTorsions")
        # Exclude zero-barrier torsions (k=0, periodicity=1, phase=0).
        exclude_torsions = [
            parameter.smirks
            for parameter in torsion_handler.parameters
            if all(
                [
                    len(parameter.k) == 1,
                    parameter.periodicity1 == 1,
                    np.isclose(parameter.phase[0].m, 0.0),
                    np.isclose(parameter.k[0].m, 0.0),
                ]
            )
        ]
        logger.info(
            f"Excluding {len(exclude_torsions)} linear torsions: {', '.join(exclude_torsions)}"
        )
        initial_parameter_configs["ProperTorsions"]["exclude"] = [
            PotentialKey(id=x)
            for x in exclude_torsions
        ]

    return {
        key: descent.train.ParameterConfig(**values)
        for key, values in initial_parameter_configs.items()
    }



def dataset_batch_iterator(
    dataset: datasets.Dataset, batch_size: int, shuffle: bool = True
):
    """Yield batches of data from the dataset.
    
    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace dataset to batch.
    batch_size : int
        Number of samples per batch.
    shuffle : bool, optional
        Whether to shuffle before batching, by default True.
        
    Yields
    ------
    datasets.Dataset
        Batch of samples from the dataset.
    """
    if shuffle:
        dataset = dataset.shuffle()

    for batch_ids in more_itertools.batched(
        [i for i in range(len(dataset))], batch_size
    ):
        yield dataset.select(indices=batch_ids)


def batch_compute_loss_and_grad(
    x: torch.Tensor,
    dataset: datasets.Dataset,
    batch_size: int,
    trainable: descent.train.Trainable,
    topology_dict: dict,
    energy_reference: typing.Literal["min", "mean"],
    energy_weight: float,
    force_weight: float,
    compute_gradient: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute MSE loss (and gradient) for a dataset in batches.
    
    Processes the dataset in batches to fit in GPU memory. Computes force field
    predictions, compares to reference QM data, and accumulates loss. Optionally
    computes gradients with respect to force field parameters.
    
    Parameters
    ----------
    x : torch.Tensor
        Current force field parameter values to optimize.
    dataset : datasets.Dataset
        Dataset with reference energies and forces.
    batch_size : int
        Number of conformers per batch.
    trainable : descent.train.Trainable
        Trainable object to convert parameters to force field.
    topology_dict : dict
        Mapping from SMILES to SMEE topology objects.
    energy_reference : {"min", "mean"}
        Reference energy for each molecule (minimum or mean).
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.
    compute_gradient : bool, optional
        Whether to compute gradients for backpropagation, by default False.

    Returns
    -------
    energy_loss : torch.Tensor
        Mean squared error for energies.
    force_loss : torch.Tensor
        Mean squared error for forces.
    total_loss : torch.Tensor
        Weighted sum of energy and force loss.
    grad : torch.Tensor or None
        Gradient of total loss w.r.t. parameters (if compute_gradient=True).
    """
    ff = trainable.to_force_field(x).to("cuda")

    dataset_indices = list(range(len(dataset)))
    n_data_indices = len(dataset_indices)

    # Initialize accumulators for loss components.
    energy_loss = torch.zeros(size=(1,), device="cuda")
    force_loss = torch.zeros(size=(1,), device="cuda")
    total_loss = torch.zeros(size=(1,), device="cuda")
    grad = None

    for batch in tqdm.tqdm(
        dataset_batch_iterator(dataset, batch_size),
        total=math.ceil(len(dataset_indices) / batch_size),
        disable=True
    ):
        batch = prepare_batch_for_device(batch, "cuda")

        e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
            batch, ff, topology_dict, energy_reference
        )

        # Compute L2 (MSE) loss for this batch.
        batch_loss_energy = ((e_pred - e_ref) ** 2).sum() / n_data_indices
        batch_loss_force = ((f_pred - f_ref) ** 2).sum() / n_data_indices
        batch_loss = (energy_weight * batch_loss_energy) + (
            force_weight * batch_loss_force
        )
        # Accumulate loss (normalized by dataset size for true MSE).
        total_loss += batch_loss.detach()
        energy_loss += batch_loss_energy.detach()
        force_loss += batch_loss_force.detach()

        if compute_gradient:
            # Compute and accumulate gradients for optimizer step.
            (batch_grad,) = torch.autograd.grad(batch_loss, x, create_graph=True)
            batch_grad = batch_grad.detach()
            if grad is None:
                grad = batch_grad
            else:
                grad += batch_grad

    return energy_loss, force_loss, total_loss, grad


def write_new_offxml(
    offxml: pathlib.Path | str,
    optimized_ff_path: pathlib.Path | str,
    output_file: str,
) -> None:
    """Convert optimized SMEE force field parameters to OFFXML format.

    Loads the optimized force field from the specified path and writes the fitted
    parameters back to an OpenFF OFFXML file, preserving the original force field
    structure while updating only the fitted parameters.

    Parameters
    ----------
    offxml : pathlib.Path | str
        Path to the reference OFFXML file used for output structure.
        Must be the same OFFXML used during training.
    optimized_ff_path : pathlib.Path | str
        Path to the optimized SMEE force field .pt file.
    output_file : str
        Path to the output OFFXML file.
    

    Returns
    -------
    None

    Notes
    -----
    Side effects:
    - Loads optimized force field from optimized_ff_path (must exist)
    - Creates final-force-field.offxml in current working directory
    - Updates parameters for Bonds, Angles, ProperTorsions, and ImproperTorsions
    - Preserves original force field structure and non-fitted parameters

    Parameter handling by type:
    - Bonds/Angles: Updates k (force constant) and equilibrium values
    - ProperTorsions: Collects k values by periodicity for each SMIRKS pattern
    - ImproperTorsions: Updates only the k values (v2 terms)

    Examples
    --------
    >>> write_new_offxml("openff-2.2.1.offxml")
    >>> write_new_offxml("openff-2.2.1.offxml", "custom-dir/optimized-ff.pt")
    """

    # Load the optimized force field that was saved during training
    optimized_ff_path = pathlib.Path(optimized_ff_path)
    logger.info(f"Loading optimized force field from: {optimized_ff_path}")

    if not optimized_ff_path.exists():
        raise FileNotFoundError(
            f"Optimized force field not found at {optimized_ff_path}. "
            "Ensure training has completed successfully."
        )

    smee_force_field = torch.load(optimized_ff_path)

    offxml = pathlib.Path(offxml)
    logger.info("Writing optimized parameters to new OFFXML force field...")
    starting_ff = ForceField(str(offxml))

    for potential in smee_force_field.potentials:
        handler_name = potential.parameter_keys[0].associated_handler
        if handler_name is None:
            logger.warning("Skipping potential with no associated handler")
            continue

        try:
            handler = starting_ff.get_parameter_handler(handler_name)
        except Exception:
            logger.warning(f"Handler {handler_name} not found in force field, skipping")
            continue

        parameter_attrs = potential.parameter_cols
        parameter_units = potential.parameter_units

        logger.info(f"Updating {len(potential.parameters)} {handler_name} parameters")

        if handler_name in ["Bonds", "Angles"]:
            # Update force constants and equilibrium values directly
            for param_key, opt_parameters in zip(
                potential.parameter_keys, potential.parameters
            ):
                ff_parameter = handler[param_key.id]
                opt_parameters = opt_parameters.detach().cpu().numpy()

                for param_name, param_value, param_unit in zip(
                    parameter_attrs, opt_parameters, parameter_units
                ):
                    setattr(ff_parameter, param_name, param_value * param_unit)

        elif handler_name == "ProperTorsions":
            # Collect k values by periodicity for each SMIRKS pattern
            k_index = parameter_attrs.index("k")
            p_index = parameter_attrs.index("periodicity")
            collection_data: dict[str, dict[int, float]] = defaultdict(dict)

            for param_key, opt_parameters in zip(
                potential.parameter_keys, potential.parameters
            ):
                opt_parameters = opt_parameters.detach().cpu().numpy()
                k = opt_parameters[k_index] * parameter_units[k_index]
                periodicity = int(opt_parameters[p_index])
                collection_data[param_key.id][periodicity] = k

            # Update force field with collected k values
            for smirks, k_by_periodicity in collection_data.items():
                ff_parameter = handler[smirks]
                ff_parameter.k = [k_by_periodicity[p] for p in ff_parameter.periodicity]

        elif handler_name == "ImproperTorsions":
            # Only fit v2 terms for improper torsions
            k_index = parameter_attrs.index("k")
            for param_key, opt_parameters in zip(
                potential.parameter_keys, potential.parameters
            ):
                ff_parameter = handler[param_key.id]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                ff_parameter.k = [opt_parameters[k_index] * parameter_units[k_index]]

    starting_ff.to_file(output_file)


def plot_loss(directory, output_path: pathlib.Path) -> None:
    """Plot training and validation loss curves from TensorBoard logs.
    
    Parameters
    ----------
    directory : str or pathlib.Path
        Path to TensorBoard logs directory.
    output_path : pathlib.Path
        Path to save output PNG image.
    """
    reader = tbparse.SummaryReader(str(directory))
    df = reader.scalars

    # Create three side-by-side plots: total, force, and energy loss.
    with plt.style.context("ggplot"):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        scalar_names = {
            "Total Loss": {"Train": "train_loss", "Validation": "validation_loss"},
            "Force Loss": {"Train": "train_loss_forces", "Validation": "validation_loss_forces"},
            "Energy Loss": {"Train": "train_loss_energy", "Validation": "validation_loss_energy"},
        }

        for i, (title, scalars) in enumerate(scalar_names.items()):
            for label, scalar in scalars.items():
                df_filtered = df[df["tag"] == scalar]
                linestyle = "-" if label == "Train" else "--"
                axs[i].plot(
                    df_filtered["step"],
                    df_filtered["value"],
                    label=label,
                    alpha=0.8,
                    linestyle=linestyle,
                )

            axs[i].set_title(title)
            axs[i].set_xlabel("Batch")
            axs[i].set_ylabel("Loss")
            if i == 2:
                axs[i].legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))

        fig.savefig(str(output_path), dpi=900)
        logger.info(f"Saved loss plots to {output_path}")
        plt.close(fig)



@click.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Path to the YAML configuration file with training parameters."
)
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    help="Path to the output directory for checkpoints and results."
)
@click.option(
    "--exclude-smiles",
    "-x",
    "exclude_smiles_files",
    multiple=True,
    default=None,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Path to file(s) with SMILES to exclude (one per line)."
)
def main(
    config_file: str,
    output_directory: str,
    exclude_smiles_files: list[str] = [],
):
    """Train a SMEE force field using gradient descent.
    
    Loads configuration, datasets, and pre-parameterized force field,
    then optimizes parameters to minimize energy and force errors.
    """
    # Load and validate configuration.
    config = yaml.safe_load(open(config_file))

    input_force_field = config["input_force_field"]
    input_dataset = config["input_dataset"]

    input_dataset = pathlib.Path(input_dataset)
    assert input_dataset.exists(), f"Input dataset not found at {input_dataset}"
    assert input_dataset.is_dir(), f"Input dataset must be a directory containing a HuggingFace dataset dict, but {input_dataset} is not a directory."

    input_parameterized = config["input_parameterized"]
    input_parameterized = pathlib.Path(input_parameterized)
    assert input_parameterized.exists(), f"Input parameterized file not found at {input_parameterized}"
    assert input_parameterized.is_file(), f"Input parameterized must be a file containing the pickled SMEE force field and topology dict, but {input_parameterized} is not a file."

    learning_rate = config["learning_rate"]
    n_epochs = config["n_epochs"]
    batch_size = config["batch_size"]
    minibatch_size = config.get("minibatch_size", None)  # Optional: enables mini-batch training
    energy_weight = config["energy_weight"]
    force_weight = config["force_weight"]
    energy_reference = config["energy_reference"]

    PARAMETERS = initialize_parameter_configs(
        config["parameter_configs"], input_force_field
    )
    print(PARAMETERS)

    # Load SMILES to exclude from training/validation.
    if not exclude_smiles_files:
        exclude_smiles_files = []

    smiles_to_exclude = []
    for smiles_file in exclude_smiles_files:
        with open(smiles_file, "r") as f:
            smiles_to_exclude.extend([line.strip() for line in f.readlines()])

    logger.info(f"Loaded {len(smiles_to_exclude)} SMILES to exclude.")

    # Load dataset and parameterized force field.
    dataset = datasets.DatasetDict.load_from_disk(input_dataset)

    with open(input_parameterized, "rb") as f:
        smee_force_field, topology_dict = pickle.load(f)

    # Move force field and topologies to GPU.
    smee_force_field = smee_force_field.to("cuda")
    topology_dict = {
        smiles: topology.to("cuda") for smiles, topology in topology_dict.items()
    }

    # Validate that all dataset SMILES have corresponding topologies.
    missing_smiles = []
    for key in dataset:
        for smiles in dataset[key]["smiles"]:
            if smiles not in topology_dict:
                missing_smiles.append(smiles)

    logger.info(
        f"{len(missing_smiles)} missing SMILES in topology_dict: {missing_smiles}"
    )
    if missing_smiles:
        raise ValueError(
            "Some SMILES strings in the dataset are missing in the topology_dict."
        )
    logger.info(f"Loaded {len(topology_dict)} topologies into CUDA.")

    # Set up trainable wrapper for optimization.
    trainable = descent.train.Trainable(
        force_field=smee_force_field, parameters=PARAMETERS, attributes={}
    )

    training_set = dataset["train"]
    if smiles_to_exclude:
        training_set = training_set.filter(lambda x: x["smiles"] not in smiles_to_exclude)

    validation_set = dataset["validation"]
    if smiles_to_exclude:
        validation_set = validation_set.filter(lambda x: x["smiles"] not in smiles_to_exclude)

    # Create output directories.
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_directory = output_directory / "metrics"
    metrics_directory.mkdir(parents=True, exist_ok=True)
    intermediate_directory = output_directory / "intermediate"
    intermediate_directory.mkdir(parents=True, exist_ok=True)

    logger.info("Start training...")
    
    # Initialize parameter tensor and optimizer.
    x = trainable.to_values().to("cuda")
    
    # If minibatch_size not specified, use full dataset (one update per epoch).
    if minibatch_size is None:
        minibatch_size = len(training_set)
        checkpoint_freq = 10
        logger.info("Standard mode: one optimizer step per epoch")
    else:
        checkpoint_freq = 5
        logger.info(f"Mini-batch mode: {minibatch_size} samples per mini-batch")
    logger.info(f"Checkpoint frequency: every {checkpoint_freq} epochs")
    
    n_minibatches = math.ceil(len(training_set) / minibatch_size)
    
    # Common kwargs for batch_compute_loss_and_grad.
    batch_compute_kwargs = {
        "batch_size": batch_size,
        "trainable": trainable,
        "topology_dict": topology_dict,
        "energy_reference": energy_reference,
        "energy_weight": energy_weight,
        "force_weight": force_weight,
    }
    
    with tensorboardX.SummaryWriter(metrics_directory) as writer:
        write_hparams(writer, config)

        optimizer = torch.optim.Adam([x], lr=learning_rate, amsgrad=True)
        
        for i in tqdm.tqdm(range(n_epochs), total=n_epochs, desc="Training"):
            # Process training data in mini-batches (or full dataset if minibatch_size = dataset size).
            for j, minibatch in enumerate(
                dataset_batch_iterator(training_set, minibatch_size)
            ):
                # Compute training loss and gradients for this mini-batch.
                energy_loss, force_loss, total_loss, grad = batch_compute_loss_and_grad(
                    x, minibatch, **batch_compute_kwargs, compute_gradient=True
                )
                x.grad = grad

                # Calculate global step for logging (accounts for mini-batches).
                write_epoch = i * n_minibatches + j
                write_metrics(
                    epoch=write_epoch,
                    loss=total_loss,
                    loss_energy=energy_loss,
                    loss_forces=force_loss,
                    writer=writer,
                    prefix="train",
                )

                # Evaluate on validation set after each mini-batch (no gradients).
                val_energy, val_force, val_total, _ = batch_compute_loss_and_grad(
                    x, validation_set, **batch_compute_kwargs, compute_gradient=False
                )
                write_metrics(
                    epoch=write_epoch,
                    loss=val_total,
                    loss_energy=val_energy,
                    loss_forces=val_force,
                    writer=writer,
                    prefix="validation",
                )

                optimizer.step()
                optimizer.zero_grad()

            # Save checkpoints periodically (frequency depends on mode).
            if i % checkpoint_freq == 0:
                torch.save(
                    trainable.to_force_field(x),
                    intermediate_directory / f"force-field-epoch-{i:04d}.pt",
                )
                plot_loss(metrics_directory, output_directory / "loss_plots.png")
        
        # Save final optimized force field.
        output_pt_file = output_directory / "final-force-field.pt"
        torch.save(trainable.to_force_field(x), output_pt_file)
    
    logger.info(f"Plotting final loss curves from {metrics_directory}")
    plot_loss(metrics_directory, output_directory / "loss_plots.png")
    
    # Convert optimized force field to OFFXML format.
    output_file = output_directory / "final-force-field.offxml"
    write_new_offxml(input_force_field, output_pt_file, output_file)


if __name__ == "__main__":
    main()
