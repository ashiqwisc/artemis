import argparse
import random
import subprocess
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_ID = "piecewise_poisson_sample"
PARAMETER_COLUMNS = ("rho_1", "rho_2", "threshold")
RHO_MIN = 0.5
RHO_MAX = 2.0
THRESHOLD_MIN = -24
THRESHOLD_MAX = 24


def assemble_field(field, logical_locations):
    nblocks, block_ny, block_nx = field.shape
    nx = (int(logical_locations[:, 0].max()) + 1) * block_nx
    ny = (int(logical_locations[:, 1].max()) + 1) * block_ny
    assembled = np.empty((ny, nx), dtype=field.dtype)

    for block in range(nblocks):
        block_x = int(logical_locations[block, 0])
        block_y = int(logical_locations[block, 1])
        i0 = block_x * block_nx
        j0 = block_y * block_ny
        assembled[j0 : j0 + block_ny, i0 : i0 + block_nx] = field[block]

    return assembled


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a piecewise-Poisson dataset directly in HDF5."
    )
    parser.add_argument("--samples", type=int, default=65_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a compatible partial dataset instead of refusing to overwrite it.",
    )
    return parser.parse_args()


def generate_parameters(samples, seed):
    rng = random.Random(seed)
    parameters = np.empty((samples, len(PARAMETER_COLUMNS)), dtype=np.float64)
    for sample in range(samples):
        parameters[sample] = (
            rng.uniform(RHO_MIN, RHO_MAX),
            rng.uniform(RHO_MIN, RHO_MAX),
            rng.randint(THRESHOLD_MIN, THRESHOLD_MAX),
        )
    return parameters


def remove_native_outputs(scratch_dir):
    for output_file in scratch_dir.glob(f"{SAMPLE_ID}.*"):
        if output_file.is_file():
            output_file.unlink()


def run_sample(executable, input_file, scratch_dir, rho_1, rho_2, threshold):
    command = [
        "mpirun",
        "-n",
        "1",
        str(executable),
        "-i",
        str(input_file),
        f"problem/rho_1={rho_1:.17g}",
        f"problem/rho_2={rho_2:.17g}",
        f"problem/threshold={threshold}",
        f"parthenon/job/problem_id={SAMPLE_ID}",
    ]
    subprocess.run(
        command,
        cwd=scratch_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )


def load_sample(output_file):
    with h5py.File(output_file, "r") as data:
        logical_locations = data["LogicalLocations"][:]
        density_blocks = data["gas.prim.density_0"][:, 0, :, :]
        potential_blocks = data["grav.phi"][:, 0, :, :]

        density = assemble_field(density_blocks, logical_locations)
        potential = assemble_field(potential_blocks, logical_locations)

    return density, potential


def visualize_sample(figure_dir, sample, density, potential):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    density_image = axes[0].imshow(
        density,
        origin="lower",
        cmap="viridis",
        vmin=RHO_MIN,
        vmax=RHO_MAX,
    )
    axes[0].set_title("Density")
    figure.colorbar(density_image, ax=axes[0], label="rho")

    potential_image = axes[1].imshow(potential, origin="lower", cmap="viridis")
    axes[1].set_title("Potential")
    figure.colorbar(potential_image, ax=axes[1], label="phi")

    figure.tight_layout()
    figure.savefig(figure_dir / f"sample_{sample:05d}_rho_and_phi.png", dpi=200)
    plt.close(figure)


def create_dataset_file(dataset_file, parameters, seed):
    dataset = h5py.File(dataset_file, "x")
    parameters_dataset = dataset.create_dataset("parameters", data=parameters)
    parameters_dataset.attrs["columns"] = PARAMETER_COLUMNS
    completed_dataset = dataset.create_dataset(
        "completed", shape=(len(parameters),), dtype=bool
    )
    completed_dataset[:] = False
    dataset.attrs["seed"] = seed
    dataset.attrs["samples"] = len(parameters)
    dataset.attrs["status"] = "in_progress"
    dataset.flush()
    return dataset


def open_dataset_file(dataset_file, parameters, seed, resume):
    if dataset_file.exists() and resume:
        dataset = h5py.File(dataset_file, "r+")
        dataset.attrs["status"] = "in_progress"
        return dataset
    return create_dataset_file(dataset_file, parameters, seed)


def ensure_field_datasets(dataset, density, potential):
    if "rho" not in dataset:
        shape = (len(dataset["completed"]),) + density.shape
        chunks = (1,) + density.shape
        creation_options = {
            "shape": shape,
            "dtype": density.dtype,
            "chunks": chunks,
            "compression": "gzip",
            "compression_opts": 4,
            "shuffle": True,
            "fletcher32": True,
        }
        dataset.create_dataset("rho", **creation_options)
        dataset.create_dataset("phi", **creation_options)
        dataset.flush()


def main():
    args = parse_arguments()

    repo = Path(__file__).resolve().parents[1]
    executable = repo / "build" / "src" / "artemis"
    input_file = repo / "inputs" / "piecewise_poisson" / "piecewise.in"

    output_dir = args.output_dir or repo / "runs" / "piecewise_poisson_dataset"
    output_dir = output_dir.resolve()
    figure_dir = output_dir / "figures"
    scratch_dir = output_dir / "artemis_scratch"
    figure_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    parameters = generate_parameters(args.samples, args.seed)
    dataset_file = output_dir / "piecewise_poisson_dataset.h5"
    output_file = scratch_dir / f"{SAMPLE_ID}.out1.final.phdf"

    with open_dataset_file(
        dataset_file, parameters, args.seed, args.resume
    ) as dataset:
        completed = dataset["completed"]
        initial_completed = int(np.count_nonzero(completed[:]))
        completed_count = initial_completed
        if initial_completed:
            print(
                f"Resuming with {initial_completed}/{args.samples} samples complete",
                flush=True,
            )

        for sample, (rho_1, rho_2, threshold_value) in enumerate(parameters):
            if completed[sample]:
                continue

            threshold = int(threshold_value)
            remove_native_outputs(scratch_dir)
            run_sample(
                executable,
                input_file,
                scratch_dir,
                rho_1,
                rho_2,
                threshold,
            )
            density, potential = load_sample(output_file)
            ensure_field_datasets(dataset, density, potential)

            dataset["rho"][sample] = density
            dataset["phi"][sample] = potential
            if sample < 5:
                visualize_sample(figure_dir, sample, density, potential)
            dataset.flush()
            completed[sample] = True
            dataset.flush()
            remove_native_outputs(scratch_dir)

            completed_count += 1
            if completed_count % 100 == 0 or completed_count == args.samples:
                print(
                    f"Completed {completed_count}/{args.samples} samples",
                    flush=True,
                )

        dataset.attrs["status"] = "complete"
        dataset.flush()

    print(f"Saved {args.samples} samples to {dataset_file}")


if __name__ == "__main__":
    main()
