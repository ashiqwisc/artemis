import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np


SAMPLE_ID = "piecewise_poisson_two_locations_1d_sample"
PARAMETER_COLUMNS = ("rho_1", "rho_2", "threshold")
DEFAULT_SAMPLES = 10_000
DEFAULT_NX = 128
DEFAULT_RHO_1 = 1.0
DEFAULT_RHO_2 = 2.0
DEFAULT_THRESHOLDS = (-2.0, 2.0)
DEFAULT_SEED = 42
BLOCK_NX = 16
WRITE_BATCH_SIZE = 256


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced 1D Poisson HDF5 dataset containing the "
            "piecewise input fields and their Artemis solutions."
        )
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
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
    if args.nx <= 0 or args.nx % BLOCK_NX != 0:
        raise ValueError(f"--nx must be a positive multiple of {BLOCK_NX}")
    if args.rho_1 <= 0.0 or args.rho_2 <= 0.0:
        raise ValueError("--rho-1 and --rho-2 must be positive")
    left, right = sorted(args.thresholds)
    if left == right:
        raise ValueError("--thresholds must contain two distinct locations")
    if not (-32.0 < left < right < 32.0):
        raise ValueError("thresholds must lie strictly inside (-32, 32)")
    return np.asarray((left, right), dtype=np.float64)


def assemble_1d(block_values, logical_locations):
    nblocks, block_nx = block_values.shape
    nx = (int(logical_locations[:, 0].max()) + 1) * block_nx
    assembled = np.empty(nx, dtype=block_values.dtype)
    for block in range(nblocks):
        block_x = int(logical_locations[block, 0])
        start = block_x * block_nx
        assembled[start : start + block_nx] = block_values[block]
    return assembled


def remove_native_outputs(scratch_dir):
    for output_file in scratch_dir.glob(f"{SAMPLE_ID}.*"):
        if output_file.is_file():
            output_file.unlink()


def solve_at_threshold(
    executable, input_file, scratch_dir, nx, rho_1, rho_2, threshold
):
    remove_native_outputs(scratch_dir)
    command = [
        "mpirun",
        "-n",
        "1",
        str(executable),
        "-i",
        str(input_file),
        f"parthenon/mesh/nx1={nx}",
        "parthenon/mesh/nx2=1",
        "parthenon/mesh/nx3=1",
        f"parthenon/meshblock/nx1={BLOCK_NX}",
        "parthenon/meshblock/nx2=1",
        "parthenon/meshblock/nx3=1",
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
        if int(data["Info"].attrs["NumDims"]) != 1:
            raise RuntimeError("Artemis did not construct a one-dimensional mesh")
        logical_locations = data["LogicalLocations"][:]
        x_blocks = data["VolumeLocations/x"][:]
        density_blocks = data["gas.prim.density_0"][:, 0, 0, :]
        potential_blocks = data["grav.phi"][:, 0, 0, :]
        x = assemble_1d(x_blocks, logical_locations)
        density = assemble_1d(density_blocks, logical_locations)
        potential = assemble_1d(potential_blocks, logical_locations)

    remove_native_outputs(scratch_dir)
    return x, density, potential


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
    x,
    density_bank,
    potential_bank,
):
    nx = len(x)
    parameters = np.empty((samples, len(PARAMETER_COLUMNS)), dtype=np.float64)
    parameters[:, 0] = rho_1
    parameters[:, 1] = rho_2
    parameters[:, 2] = thresholds[location_ids]
    creation_options = {
        "shape": (samples, nx),
        "dtype": density_bank.dtype,
        "chunks": (1, nx),
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
        "fletcher32": True,
    }

    with h5py.File(partial_file, "x") as dataset:
        parameter_dataset = dataset.create_dataset("parameters", data=parameters)
        parameter_dataset.attrs["columns"] = PARAMETER_COLUMNS
        dataset.create_dataset("location_id", data=location_ids)
        dataset.create_dataset("x", data=x)
        completed = dataset.create_dataset("completed", shape=(samples,), dtype=bool)
        completed[:] = False
        density_dataset = dataset.create_dataset("rho", **creation_options)
        density_dataset.attrs["role"] = "Poisson right-hand-side field"
        potential_dataset = dataset.create_dataset("phi", **creation_options)
        potential_dataset.attrs["role"] = "Poisson solution field"

        dataset.attrs["dimension"] = 1
        dataset.attrs["domain"] = (-32.0, 32.0)
        dataset.attrs["equation"] = "laplacian(phi) = rho"
        dataset.attrs["boundary_condition"] = "phi = 0 at x = -32 and x = 32"
        dataset.attrs["samples"] = samples
        dataset.attrs["seed"] = seed
        dataset.attrs["status"] = "in_progress"
        dataset.attrs["varying_parameter"] = "threshold"
        dataset.attrs["rho_1"] = rho_1
        dataset.attrs["rho_2"] = rho_2
        dataset.attrs["threshold_values"] = thresholds
        dataset.attrs["samples_per_threshold"] = samples // 2
        dataset.attrs["mesh_shape"] = (nx,)
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
    output_dir = (
        args.output_dir or repo / "runs" / "piecewise_poisson_two_locations_1d_10k"
    ).resolve()
    scratch_dir = output_dir / "artemis_scratch"
    dataset_file = output_dir / "piecewise_poisson_two_locations_1d_10k.h5"
    partial_file = output_dir / "piecewise_poisson_two_locations_1d_10k.h5.partial"

    if not executable.is_file():
        raise FileNotFoundError(f"Artemis executable not found: {executable}")
    if dataset_file.exists() or partial_file.exists():
        raise FileExistsError(
            f"refusing to overwrite existing dataset or partial file in {output_dir}"
        )

    scratch_dir.mkdir(parents=True, exist_ok=True)
    location_ids = make_location_ids(args.samples, args.seed)
    try:
        x_reference = None
        density_fields = []
        potential_fields = []
        for threshold in thresholds:
            print(f"Solving unique 1D threshold T={threshold:g}", flush=True)
            x, density, potential = solve_at_threshold(
                executable,
                input_file,
                scratch_dir,
                args.nx,
                args.rho_1,
                args.rho_2,
                threshold,
            )
            if x_reference is None:
                x_reference = x
            elif not np.array_equal(x, x_reference):
                raise RuntimeError("the two Artemis solves used different grids")
            density_fields.append(density)
            potential_fields.append(potential)

        create_dataset(
            partial_file,
            args.samples,
            args.seed,
            args.rho_1,
            args.rho_2,
            thresholds,
            location_ids,
            x_reference,
            np.stack(density_fields),
            np.stack(potential_fields),
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
