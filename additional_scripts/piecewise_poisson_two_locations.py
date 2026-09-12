import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np


SAMPLE_ID = "piecewise_poisson_two_locations_sample"
PARAMETER_COLUMNS = ("rho_1", "rho_2", "threshold")
DEFAULT_SAMPLES = 10_000
DEFAULT_RHO_1 = 1.0
DEFAULT_RHO_2 = 2.0
DEFAULT_THRESHOLDS = (-16.0, 16.0)
DEFAULT_SEED = 42
WRITE_BATCH_SIZE = 64


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced piecewise-Poisson HDF5 dataset with fixed "
            "coefficients and exactly two discontinuity locations."
        )
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--rho-1", type=float, default=DEFAULT_RHO_1)
    parser.add_argument("--rho-2", type=float, default=DEFAULT_RHO_2)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def validate_arguments(args):
    if args.samples <= 0 or args.samples % 2 != 0:
        raise ValueError("--samples must be a positive even integer")
    if args.rho_1 <= 0.0 or args.rho_2 <= 0.0:
        raise ValueError("--rho-1 and --rho-2 must be positive")
    left, right = sorted(args.thresholds)
    if left == right:
        raise ValueError("--thresholds must contain two distinct locations")
    if not (-32.0 < left < right < 32.0):
        raise ValueError("thresholds must lie strictly inside (-32, 32)")
    return np.asarray((left, right), dtype=np.float64)


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


def remove_native_outputs(scratch_dir):
    for output_file in scratch_dir.glob(f"{SAMPLE_ID}.*"):
        if output_file.is_file():
            output_file.unlink()


def solve_at_threshold(executable, input_file, scratch_dir, rho_1, rho_2, threshold):
    remove_native_outputs(scratch_dir)
    command = [
        "mpirun",
        "-n",
        "1",
        str(executable),
        "-i",
        str(input_file),
        f"problem/rho_1={rho_1:.17g}",
        f"problem/rho_2={rho_2:.17g}",
        f"problem/threshold={threshold:.17g}",
        f"parthenon/job/problem_id={SAMPLE_ID}",
    ]
    result = subprocess.run(
        command,
        cwd=scratch_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Artemis failed at threshold={threshold}:\n{result.stdout}"
        )

    output_file = scratch_dir / f"{SAMPLE_ID}.out1.final.phdf"
    with h5py.File(output_file, "r") as data:
        logical_locations = data["LogicalLocations"][:]
        density_blocks = data["gas.prim.density_0"][:, 0, :, :]
        potential_blocks = data["grav.phi"][:, 0, :, :]
        density = assemble_field(density_blocks, logical_locations)
        potential = assemble_field(potential_blocks, logical_locations)

    remove_native_outputs(scratch_dir)
    return density, potential


def make_location_ids(samples, seed):
    location_ids = np.repeat(np.arange(2, dtype=np.int8), samples // 2)
    np.random.default_rng(seed).shuffle(location_ids)
    return location_ids


def create_dataset(
    partial_file,
    samples,
    seed,
    rho_1,
    rho_2,
    thresholds,
    location_ids,
    density_bank,
    potential_bank,
):
    ny, nx = density_bank.shape[1:]
    parameters = np.empty((samples, len(PARAMETER_COLUMNS)), dtype=np.float64)
    parameters[:, 0] = rho_1
    parameters[:, 1] = rho_2
    parameters[:, 2] = thresholds[location_ids]

    creation_options = {
        "shape": (samples, ny, nx),
        "dtype": density_bank.dtype,
        "chunks": (1, ny, nx),
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
        "fletcher32": True,
    }

    with h5py.File(partial_file, "x") as dataset:
        parameter_dataset = dataset.create_dataset("parameters", data=parameters)
        parameter_dataset.attrs["columns"] = PARAMETER_COLUMNS
        dataset.create_dataset("location_id", data=location_ids)
        completed = dataset.create_dataset("completed", shape=(samples,), dtype=bool)
        completed[:] = False
        density_dataset = dataset.create_dataset("rho", **creation_options)
        potential_dataset = dataset.create_dataset("phi", **creation_options)

        dataset.attrs["samples"] = samples
        dataset.attrs["seed"] = seed
        dataset.attrs["status"] = "in_progress"
        dataset.attrs["varying_parameter"] = "threshold"
        dataset.attrs["rho_1"] = rho_1
        dataset.attrs["rho_2"] = rho_2
        dataset.attrs["threshold_values"] = thresholds
        dataset.attrs["samples_per_threshold"] = samples // 2
        dataset.attrs["mesh_shape"] = (ny, nx)
        dataset.flush()

        for start in range(0, samples, WRITE_BATCH_SIZE):
            stop = min(start + WRITE_BATCH_SIZE, samples)
            batch_ids = location_ids[start:stop]
            density_dataset[start:stop] = density_bank[batch_ids]
            potential_dataset[start:stop] = potential_bank[batch_ids]
            completed[start:stop] = True
            dataset.flush()
            print(f"Completed {stop:,}/{samples:,} samples", flush=True)

        dataset.attrs["status"] = "complete"
        dataset.flush()


def main():
    args = parse_arguments()
    thresholds = validate_arguments(args)

    repo = Path(__file__).resolve().parents[1]
    executable = repo / "build" / "src" / "artemis"
    input_file = repo / "inputs" / "piecewise_poisson" / "piecewise.in"
    output_dir = args.output_dir or repo / "runs" / "piecewise_poisson_two_locations_10k"
    output_dir = output_dir.resolve()
    scratch_dir = output_dir / "artemis_scratch"
    dataset_file = output_dir / "piecewise_poisson_two_locations_10k.h5"
    partial_file = output_dir / "piecewise_poisson_two_locations_10k.h5.partial"

    if not executable.is_file():
        raise FileNotFoundError(f"Artemis executable not found: {executable}")
    if dataset_file.exists() or partial_file.exists():
        raise FileExistsError(
            f"refusing to overwrite existing dataset or partial file in {output_dir}"
        )

    scratch_dir.mkdir(parents=True, exist_ok=True)
    location_ids = make_location_ids(args.samples, args.seed)

    try:
        density_fields = []
        potential_fields = []
        for threshold in thresholds:
            print(f"Solving unique threshold T={threshold:g}", flush=True)
            density, potential = solve_at_threshold(
                executable,
                input_file,
                scratch_dir,
                args.rho_1,
                args.rho_2,
                threshold,
            )
            density_fields.append(density)
            potential_fields.append(potential)

        density_bank = np.stack(density_fields)
        potential_bank = np.stack(potential_fields)
        create_dataset(
            partial_file,
            args.samples,
            args.seed,
            args.rho_1,
            args.rho_2,
            thresholds,
            location_ids,
            density_bank,
            potential_bank,
        )
        partial_file.replace(dataset_file)
    finally:
        remove_native_outputs(scratch_dir)
        try:
            scratch_dir.rmdir()
        except OSError:
            pass

    print(f"Saved {args.samples:,} samples to {dataset_file}", flush=True)


if __name__ == "__main__":
    main()
