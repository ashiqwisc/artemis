import argparse
import subprocess
import textwrap
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def assemble_1d(block_values, logical_locations):
    block_nx = block_values.shape[1]
    nx = (int(logical_locations[:, 0].max()) + 1) * block_nx
    values = np.empty(nx, dtype=block_values.dtype)

    for block in range(block_values.shape[0]):
        i0 = int(logical_locations[block, 0]) * block_nx
        values[i0 : i0 + block_nx] = block_values[block]

    return values


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def write_input_file(input_file):
    input_file.write_text(
        textwrap.dedent(
            """
            <parthenon/job>
            problem_id = burgers_sample

            <parthenon/mesh>
            nghost = 4
            refinement = none

            nx1 = 256
            x1min = -0.5
            x1max = 0.5
            ix1_bc = periodic
            ox1_bc = periodic

            nx2 = 1
            x2min = -0.5
            x2max = 0.5
            ix2_bc = periodic
            ox2_bc = periodic

            nx3 = 1
            x3min = -0.5
            x3max = 0.5
            ix3_bc = periodic
            ox3_bc = periodic

            <parthenon/meshblock>
            nx1 = 64
            nx2 = 1
            nx3 = 1

            <parthenon/time>
            nlim = -1
            tlim = 0.4
            integrator = rk2
            ncycle_out = 100

            <parthenon/output0>
            file_type = hdf5
            dt = 0.010256410256410256
            variables = U

            <burgers>
            cfl = 0.4
            recon = weno5
            num_scalars = 1
            """
        ).strip()
        + "\n"
    )


def remove_native_outputs(scratch_dir):
    for output_file in scratch_dir.glob("burgers_sample.out0.*"):
        output_file.unlink()


def run_sample(executable, input_file, scratch_dir, amplitude, steepness, shift):
    command = [
        str(executable),
        "-i",
        str(input_file),
        f"burgers/amplitude={amplitude:.17g}",
        f"burgers/steepness={steepness:.17g}",
        f"burgers/shift={shift:.17g}",
    ]
    result = subprocess.run(
        command,
        cwd=scratch_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)


def load_sequence(scratch_dir):
    snapshots = []
    for output_file in scratch_dir.glob("burgers_sample.out0.*.phdf"):
        with h5py.File(output_file, "r") as data:
            time = float(data["Info"].attrs["Time"])
            logical_locations = data["LogicalLocations"][:]
            u_blocks = data["U"][:, 0, 0, 0, :]
            x_blocks = data["VolumeLocations/x"][:]
            u = assemble_1d(u_blocks, logical_locations)
            x = assemble_1d(x_blocks, logical_locations)
            snapshots.append((time, x, u))

    snapshots.sort(key=lambda snapshot: snapshot[0])
    if len(snapshots) != 40:
        raise ValueError(f"Expected 40 snapshots, found {len(snapshots)}")

    times = np.asarray([snapshot[0] for snapshot in snapshots])
    x = snapshots[0][1]
    u = np.stack([snapshot[2] for snapshot in snapshots])
    return times, x, u


def initial_condition(x, amplitude, steepness, shift):
    x_shifted = x - shift
    x_periodic = x_shifted - np.floor(x_shifted + 0.5)
    return 1.0 + amplitude * np.tanh(-steepness * x_periodic) * np.cos(
        np.pi * x_periodic
    )


def visualize_sample(figure_dir, sample, parameters, times, x, u):
    amplitude, steepness, shift = parameters
    snapshot_indices = np.linspace(0, len(times) - 1, 5, dtype=int)
    colors = ["#084081", "#2b8cbe", "#756bb1", "#e34a33", "#99000d"]

    fig, ax = plt.subplots(figsize=(7, 4))
    for index, color in zip(snapshot_indices, colors):
        ax.plot(x, u[index], color=color, label=rf"$t={times[index]:.3f}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")
    ax.set_ylim(0.0, 2.0)
    ax.set_title(
        rf"Sample {sample}: $A={amplitude:.3f}$, $s={steepness:.2f}$, "
        rf"$\xi={shift:.3f}$"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / f"sample_{sample:05d}_snapshots.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.pcolormesh(
        x,
        times,
        u,
        shading="auto",
        cmap="RdBu_r",
        vmin=0.0,
        vmax=2.0,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(rf"Sample {sample}: space-time evolution of $u(x,t)$")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$u(x,t)$")
    fig.tight_layout()
    fig.savefig(figure_dir / f"sample_{sample:05d}_spacetime.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_arguments()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    repo = Path(__file__).resolve().parents[1]
    executable = (
        repo
        / "build-parthenon-burgers"
        / "benchmarks"
        / "burgers"
        / "burgers-benchmark"
    )
    if not executable.is_file():
        raise FileNotFoundError(
            "Build the Parthenon burgers-benchmark in build-parthenon-burgers first."
        )

    output_dir = args.output_dir or repo / "runs" / "inviscid_burgers_dataset"
    output_dir = output_dir.resolve()
    figure_dir = output_dir / "figures"
    scratch_dir = output_dir / "parthenon_scratch"
    figure_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    input_file = scratch_dir / "burgers_1d.pin"
    write_input_file(input_file)

    rng = np.random.default_rng(args.seed)
    parameters = np.column_stack(
        (
            rng.uniform(0.75, 1.0, args.samples),
            rng.uniform(10.0, 30.0, args.samples),
            rng.uniform(-0.5, 0.5, args.samples),
        )
    )
    dataset_file = output_dir / "inviscid_burgers_dataset.h5"
    with h5py.File(dataset_file, "w") as dataset:
        parameters_dataset = dataset.create_dataset("parameters", data=parameters)
        parameters_dataset.attrs["columns"] = ["amplitude", "steepness", "shift"]
        dataset.attrs["seed"] = args.seed
        x_dataset = dataset.create_dataset("x", shape=(256,), dtype=np.float64)
        time_dataset = dataset.create_dataset(
            "time", shape=(args.samples, 40), dtype=np.float64
        )
        u_dataset = dataset.create_dataset(
            "u",
            shape=(args.samples, 40, 256),
            dtype=np.float32,
            chunks=(1, 40, 256),
        )
        completed_dataset = dataset.create_dataset(
            "completed", shape=(args.samples,), dtype=bool
        )
        completed_dataset[:] = False

        x_reference = None
        for sample, (amplitude, steepness, shift) in enumerate(parameters):
            remove_native_outputs(scratch_dir)
            run_sample(
                executable,
                input_file,
                scratch_dir,
                amplitude,
                steepness,
                shift,
            )
            times, x, u = load_sequence(scratch_dir)

            if not np.isfinite(u).all():
                raise ValueError(f"Sample {sample} contains non-finite values")
            if not np.all(np.diff(times) > 0.0):
                raise ValueError(f"Sample {sample} has non-increasing output times")
            expected_initial = initial_condition(x, amplitude, steepness, shift)
            if not np.allclose(u[0], expected_initial, rtol=1.0e-6, atol=1.0e-7):
                raise ValueError(
                    f"Sample {sample} failed initial-condition verification"
                )

            if x_reference is None:
                x_reference = x
                x_dataset[:] = x_reference
            elif not np.array_equal(x, x_reference):
                raise ValueError(f"Sample {sample} uses a different spatial mesh")

            u_dataset[sample] = u
            time_dataset[sample] = times
            completed_dataset[sample] = True
            if sample < 5:
                visualize_sample(
                    figure_dir,
                    sample,
                    parameters[sample],
                    times,
                    x,
                    u,
                )

            remove_native_outputs(scratch_dir)
            if (sample + 1) % 100 == 0 or sample + 1 == args.samples:
                dataset.flush()
                print(f"Completed {sample + 1}/{args.samples} samples", flush=True)

    print(f"Saved dataset to {dataset_file}")


if __name__ == "__main__":
    main()
