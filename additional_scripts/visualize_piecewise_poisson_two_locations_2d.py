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
        / "piecewise_poisson_two_locations_10k"
        / "piecewise_poisson_two_locations_10k.h5"
    )
    parser = argparse.ArgumentParser(
        description="Plot coefficient/source fields and solutions for five 2D samples."
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

    output = args.output or args.dataset.with_name("five_2d_instances.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset, "r") as dataset:
        required = {"location_id", "parameters", "rho", "phi"}
        missing = required.difference(dataset.keys())
        if missing:
            raise KeyError(f"dataset is missing required fields: {sorted(missing)}")

        indices = select_alternating_indices(dataset["location_id"][:], args.count)
        parameters = np.stack([dataset["parameters"][index] for index in indices])
        density = np.stack([dataset["rho"][index] for index in indices])
        potential = np.stack([dataset["phi"][index] for index in indices])

    extent = (-32.0, 32.0, -32.0, 32.0)
    phi_min = float(potential.min())
    phi_max = float(potential.max())
    figure, axes = plt.subplots(
        args.count,
        2,
        figsize=(10, 3.25 * args.count),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for row, sample_index in enumerate(indices):
        threshold = parameters[row, 2]
        rho_image = axes[row, 0].imshow(
            density[row],
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=float(density.min()),
            vmax=float(density.max()),
        )
        phi_image = axes[row, 1].imshow(
            potential[row],
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=phi_min,
            vmax=phi_max,
        )
        for axis in axes[row]:
            axis.axvline(threshold, color="white", linewidth=1.0, linestyle="--")
            axis.set_aspect("equal")
        axes[row, 0].set_ylabel(r"$x_2$")
        axes[row, 0].text(
            0.03,
            0.95,
            f"sample {sample_index}, $T^{{(m)}}={threshold:g}$",
            transform=axes[row, 0].transAxes,
            va="top",
            ha="left",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
        )

    axes[0, 0].set_title(r"Input field $\rho^{(m)}(x_1,x_2)$")
    axes[0, 1].set_title(r"Solution $\phi^{(m)}(x_1,x_2)$")
    axes[-1, 0].set_xlabel(r"$x_1$")
    axes[-1, 1].set_xlabel(r"$x_1$")
    figure.colorbar(rho_image, ax=axes[:, 0], shrink=0.75, label=r"$\rho$")
    figure.colorbar(phi_image, ax=axes[:, 1], shrink=0.75, label=r"$\phi$")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved visualization to {output}")


if __name__ == "__main__":
    main()
