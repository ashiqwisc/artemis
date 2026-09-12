import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigh_tridiagonal


DEFAULT_NX = 128
DEFAULT_DOMAIN = (-32.0, 32.0)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Export a compact spectral representation of the discrete Poisson "
            "solution operator G=M^{-1}."
        )
    )
    parser.add_argument("--dimension", type=int, choices=(1, 2), required=True)
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
    parser.add_argument(
        "--domain",
        type=float,
        nargs=2,
        metavar=("LOWER", "UPPER"),
        default=DEFAULT_DOMAIN,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing output after the new file is complete",
    )
    return parser.parse_args()


def default_output(repo, dimension):
    if dimension == 1:
        folder = "piecewise_poisson_two_locations_1d_10k"
    else:
        folder = "piecewise_poisson_two_locations_10k"
    return repo / "runs" / folder / "poisson_solution_operator_G.h5"


def build_1d_laplacian(nx, spacing):
    inverse_h2 = 1.0 / spacing**2
    diagonal = np.full(nx, -2.0 * inverse_h2)
    diagonal[[0, -1]] = -3.0 * inverse_h2
    off_diagonal = np.full(nx - 1, inverse_h2)
    return diagonal, off_diagonal


def tridiagonal_matrix(diagonal, off_diagonal):
    matrix = np.diag(diagonal)
    matrix += np.diag(off_diagonal, k=1)
    matrix += np.diag(off_diagonal, k=-1)
    return matrix


def export_operator(output, dimension, nx, domain, overwrite=False):
    if nx < 2:
        raise ValueError("--nx must be at least 2")
    lower, upper = domain
    if not lower < upper:
        raise ValueError("--domain must satisfy LOWER < UPPER")

    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite partial file {partial}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")

    spacing = (upper - lower) / nx
    diagonal, off_diagonal = build_1d_laplacian(nx, spacing)
    eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, off_diagonal)
    if dimension == 1:
        inverse_spectrum = 1.0 / eigenvalues
        inverse_residual = np.max(
            np.abs(eigenvalues * inverse_spectrum - 1.0)
        )
        matrix_order = nx
    else:
        spectrum = eigenvalues[:, np.newaxis] + eigenvalues[np.newaxis, :]
        inverse_spectrum = 1.0 / spectrum
        inverse_residual = np.max(np.abs(spectrum * inverse_spectrum - 1.0))
        matrix_order = nx**2

    matrix_1d = tridiagonal_matrix(diagonal, off_diagonal)
    orthogonality_residual = np.max(
        np.abs(eigenvectors.T @ eigenvectors - np.eye(nx))
    )
    eigendecomposition_residual = np.max(
        np.abs(matrix_1d @ eigenvectors - eigenvectors * eigenvalues)
    )

    x = lower + (np.arange(nx, dtype=np.float64) + 0.5) * spacing
    try:
        with h5py.File(partial, "x") as data:
            data.attrs["status"] = "in_progress"
            data.attrs["representation"] = "spectral_factorization"
            data.attrs["dimension"] = dimension
            data.attrs["nx"] = nx
            data.attrs["ny"] = nx if dimension == 2 else 1
            data.attrs["domain"] = (lower, upper)
            data.attrs["spacing"] = spacing
            data.attrs["equation"] = "laplacian(u) = rho"
            data.attrs["boundary_condition"] = "u = 0 on boundary faces"
            data.attrs["operator_definition"] = (
                "G=M^{-1}, stored as a separable spectral factorization"
            )
            data.attrs["flatten_order"] = "C"
            data.attrs["flat_index"] = "i + nx*j (2D); i (1D)"
            data.attrs["dtype"] = "float64"
            data.attrs["matrix_order"] = matrix_order

            grid = data.create_group("grid")
            grid.create_dataset("x", data=x)
            if dimension == 2:
                grid.create_dataset("y", data=x)

            spectral = data.create_group("spectral_factorization")
            spectral.attrs["definition_1d"] = (
                "u = Q @ (inverse_spectrum * (Q.T @ rho))"
            )
            spectral.attrs["definition_2d"] = (
                "Rhat=Q.T@rho@Q; Uhat=inverse_spectrum*Rhat; u=Q@Uhat@Q.T"
            )
            spectral.attrs["max_orthogonality_residual"] = (
                orthogonality_residual
            )
            spectral.attrs["max_eigendecomposition_residual"] = (
                eigendecomposition_residual
            )
            spectral.attrs["max_inverse_spectrum_residual"] = inverse_residual
            spectral.create_dataset("Q", data=eigenvectors)
            spectral.create_dataset("eigenvalues_1d", data=eigenvalues)
            spectral.create_dataset("inverse_spectrum", data=inverse_spectrum)
            data.attrs["status"] = "complete"
            data.flush()
        partial.replace(output)
    except BaseException:
        if partial.exists():
            try:
                with h5py.File(partial, "r+") as data:
                    data.attrs["status"] = "failed"
                    data.flush()
            except OSError:
                pass
        raise

    print(
        f"Saved compact {dimension}D Poisson spectral operator to {output}",
        flush=True,
    )


def main():
    args = parse_arguments()
    repo = Path(__file__).resolve().parents[1]
    output = args.output or default_output(repo, args.dimension)
    export_operator(
        output,
        args.dimension,
        args.nx,
        args.domain,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
