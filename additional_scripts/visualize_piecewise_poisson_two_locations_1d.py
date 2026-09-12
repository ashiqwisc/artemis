import argparse
import os
from pathlib import Path

import h5py
import numpy as np

PLOT_CACHE = Path("/private/tmp/artemis-plot-cache")
PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_arguments():
    repo = Path(__file__).resolve().parents[1]
    default_dataset = (
        repo
        / "runs"
        / "piecewise_poisson_two_locations_1d_10k"
        / "piecewise_poisson_two_locations_1d_10k.h5"
    )
    parser = argparse.ArgumentParser(
        description="Plot input fields and solutions for five 1D Poisson samples."
    )
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def select_alternating_indices(location_ids, count):
    groups = [np.flatnonzero(location_ids == value) for value in np.unique(location_ids)]
    if len(groups) != 2:
        raise ValueError("expected exactly two location IDs")

    selected = []
    offsets = [0, 0]
    for row in range(count):
        group = row % 2
        if offsets[group] >= len(groups[group]):
            raise ValueError("not enough samples to construct the requested plot")
        selected.append(int(groups[group][offsets[group]]))
        offsets[group] += 1
    return selected


def main():
    args = parse_arguments()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    output = args.output or args.dataset.with_name("five_1d_instances.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset, "r") as dataset:
        required = {"location_id", "parameters", "x", "rho", "phi"}
        missing = required.difference(dataset.keys())
        if missing:
            raise KeyError(f"dataset is missing required fields: {sorted(missing)}")

        indices = select_alternating_indices(dataset["location_id"][:], args.count)
        parameters = np.stack([dataset["parameters"][index] for index in indices])
        x = dataset["x"][:]
        density = np.stack([dataset["rho"][index] for index in indices])
        potential = np.stack([dataset["phi"][index] for index in indices])
        domain = tuple(float(value) for value in dataset.attrs["domain"])

    figure, axes = plt.subplots(
        args.count,
        2,
        figsize=(11, 2.6 * args.count),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    potential_with_boundary = np.column_stack(
        [np.zeros(args.count), potential, np.zeros(args.count)]
    )
    x_with_boundary = np.concatenate(([domain[0]], x, [domain[1]]))
    phi_min = float(potential.min())

    for row, sample_index in enumerate(indices):
        threshold = parameters[row, 2]
        axes[row, 0].plot(
            x,
            density[row],
            color="tab:blue",
            linewidth=2.0,
            drawstyle="steps-mid",
        )
        axes[row, 1].plot(
            x_with_boundary,
            potential_with_boundary[row],
            color="tab:red",
            linewidth=2.0,
        )
        for axis in axes[row]:
            axis.axvline(threshold, color="0.25", linewidth=1.2, linestyle="--")
            axis.grid(True, color="0.88", linewidth=0.7)
            axis.set_xlim(domain)
        axes[row, 0].set_ylim(0.85, 2.15)
        axes[row, 1].set_ylim(1.06 * phi_min, 25.0)
        axes[row, 0].set_ylabel(r"$\rho$")
        axes[row, 1].set_ylabel(r"$\phi$")
        axes[row, 0].text(
            0.03,
            0.90,
            f"sample {sample_index}, $T^{{(m)}}={threshold:g}$",
            transform=axes[row, 0].transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )

    axes[0, 0].set_title(r"Input field $\rho^{(m)}(x)$")
    axes[0, 1].set_title(r"Solution $\phi^{(m)}(x)$")
    axes[-1, 0].set_xlabel(r"$x$")
    axes[-1, 1].set_xlabel(r"$x$")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved visualization to {output}")


if __name__ == "__main__":
    main()
