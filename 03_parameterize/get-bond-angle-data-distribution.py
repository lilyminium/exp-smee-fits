import pathlib
import multiprocessing as mp

import click
import tqdm

import datasets
from openff.toolkit import Molecule, ForceField, unit
import numpy as np
import MDAnalysis as mda
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds



def _process_row(task):
    subset, input_force_field, row = task

    molecule = Molecule.from_mapped_smiles(
        row["smiles"],
        allow_undefined_stereo=True,
    )
    coords = np.array(row["coords"]).reshape(
        (-1, molecule.n_atoms, 3)
    ).astype(float) * unit.angstrom

    force_field = ForceField(input_force_field)
    labels = force_field.label_molecules(molecule.to_topology())[0]
    bonds = labels["Bonds"]
    angles = labels["Angles"]

    local_rows = []

    for conf_index, coord in enumerate(coords):
        molecule._conformers = [coord]
        universe = mda.Universe(molecule.to_rdkit(), to_guess=["bonds", "angles"])

        for (i, j), bond in bonds.items():
            bond_length = universe.atoms[[i, j]].bond.value()
            local_rows.append({
                "subset": subset,
                "smiles": row["smiles"],
                "conformer_index": conf_index,
                "topology": "Bonds",
                "parameter_id": bond.id,
                "smirks": bond.smirks,
                "equilibrium_value": bond.length.m_as(unit.angstrom),
                "atom_indices": [i, j],
                "value": bond_length,
            })

        for (i, j, k), angle in angles.items():
            angle_value = universe.atoms[[i, j, k]].angle.value()
            local_rows.append({
                "subset": subset,
                "smiles": row["smiles"],
                "conformer_index": conf_index,
                "topology": "Angles",
                "parameter_id": angle.id,
                "smirks": angle.smirks,
                "equilibrium_value": angle.angle.m_as(unit.degree),
                "atom_indices": [i, j, k],
                "value": angle_value,
            })

    return local_rows


@click.command()
@click.option(
    "--input-force-field",
    "-ff",
    type=str,
    required=True,
    help="Path to the input force field file (e.g., OFFXML).",
)
@click.option(
    "--dataset-directory",
    "-i",
    type=str,
    required=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the dataset directory containing molecular data. Should be a DatasetDict format.",
)
@click.option(
    "--output-directory",
    "-o",
    type=str,
    required=True,
    help="Path to the output directory where results will be saved.",
)
@click.option(
    "--processes",
    "-np",
    type=int,
    default=None,
    show_default=True,
    help="Number of worker processes to use. Defaults to all available CPUs.",
)
def main(
    input_force_field: str,
    dataset_directory: str,
    output_directory: str,
    processes: int | None,
):
    dataset_dict = datasets.DatasetDict.load_from_disk(dataset_directory)

    rows = []

    worker_count = processes if processes is not None else mp.cpu_count()
    if worker_count < 1:
        raise click.BadParameter("--processes must be >= 1")

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)


    for key in dataset_dict:
        dataset = dataset_dict[key]
        rows = []
        tasks = ((key, input_force_field, row) for row in dataset)
        with mp.Pool(processes=worker_count) as pool:
            for row_rows in tqdm.tqdm(
                pool.imap_unordered(_process_row, tasks, chunksize=8),
                total=len(dataset),
                desc=f"Processing {key}",
            ):
                rows.extend(row_rows)
        table = pa.Table.from_pylist(rows)
        output_key_directory = output_directory / key
        n_rows = 10_000
        ds.write_dataset(table, output_key_directory, max_rows_per_file=n_rows, max_rows_per_group=n_rows, format="parquet")

    
if __name__ == "__main__":
    main()
