import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu


DEFAULT_NX = 128
DEFAULT_DOMAIN = (-32.0, 32.0)
DEFAULT_BATCH_SIZE = 64


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Export the discrete Poisson matrix A, its dense inverse G=A^{-1}, "
            "and an equivalent compact spectral factorization."
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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output", type=Path)
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
    matrix = diags(
        (off_diagonal, diagonal, off_diagonal),
        offsets=(-1, 0, 1),
        format="csc",
    )
    return matrix, diagonal, off_diagonal


def build_laplacian(dimension, matrix_1d):
    if dimension == 1:
        return matrix_1d
    identity = eye(matrix_1d.shape[0], format="csc")
    return kron(identity, matrix_1d, format="csc") + kron(
        matrix_1d, identity, format="csc"
    )


def write_csr(group, matrix):
    matrix = matrix.tocsr()
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices)
    group.create_dataset("indptr", data=matrix.indptr)
    group.attrs["shape"] = matrix.shape
    group.attrs["format"] = "csr"


def write_dense_inverse(dataset, matrix, batch_size):
    order = matrix.shape[0]
    solver = splu(csc_matrix(matrix))
    max_residual = 0.0

    for start in range(0, order, batch_size):
        stop = min(start + batch_size, order)
        width = stop - start
        right_hand_sides = np.zeros((order, width), dtype=np.float64)
        right_hand_sides[np.arange(start, stop), np.arange(width)] = 1.0
        columns = solver.solve(right_hand_sides)
        residual = matrix @ columns - right_hand_sides
        max_residual = max(max_residual, float(np.max(np.abs(residual))))

        # A is symmetric, so columns start:stop of A^{-1} are the corresponding rows.
        dataset[start:stop, :] = columns.T
        dataset.file.flush()
        print(f"Materialized rows {stop:,}/{order:,}", flush=True)

    return max_residual


def export_operator(output, dimension, nx, domain, batch_size):
    if nx < 2:
        raise ValueError("--nx must be at least 2")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    lower, upper = domain
    if not lower < upper:
        raise ValueError("--domain must satisfy LOWER < UPPER")

    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")

    spacing = (upper - lower) / nx
    matrix_1d, diagonal, off_diagonal = build_1d_laplacian(nx, spacing)
    matrix = build_laplacian(dimension, matrix_1d)
    order = matrix.shape[0]
    dense_bytes = order * order * np.dtype(np.float64).itemsize
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < dense_bytes + 256 * 1024**2:
        raise OSError(
            f"insufficient free space: need at least {dense_bytes + 256 * 1024**2:,} "
            f"bytes, found {free_bytes:,}"
        )

    eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, off_diagonal)
    if dimension == 1:
        inverse_spectrum = 1.0 / eigenvalues
    else:
        inverse_spectrum = 1.0 / (
            eigenvalues[:, np.newaxis] + eigenvalues[np.newaxis, :]
        )

    x = lower + (np.arange(nx, dtype=np.float64) + 0.5) * spacing
    try:
        with h5py.File(partial, "x") as data:
            data.attrs["status"] = "in_progress"
            data.attrs["dimension"] = dimension
            data.attrs["nx"] = nx
            data.attrs["ny"] = nx if dimension == 2 else 1
            data.attrs["domain"] = (lower, upper)
            data.attrs["spacing"] = spacing
            data.attrs["equation"] = "laplacian(u) = rho"
            data.attrs["boundary_condition"] = "u = 0 on boundary faces"
            data.attrs["operator_definition"] = "u_flat = G @ rho_flat; G = A^{-1}"
            data.attrs["flatten_order"] = "C"
            data.attrs["flat_index"] = "i + nx*j (2D); i (1D)"
            data.attrs["dtype"] = "float64"
            data.attrs["matrix_order"] = order
            data.attrs["dense_G_bytes"] = dense_bytes

            grid = data.create_group("grid")
            grid.create_dataset("x", data=x)
            if dimension == 2:
                grid.create_dataset("y", data=x)

            write_csr(data.create_group("A_csr"), matrix)

            spectral = data.create_group("spectral_factorization")
            spectral.attrs["definition_1d"] = (
                "G = Q @ diag(inverse_spectrum) @ Q.T"
            )
            spectral.attrs["definition_2d"] = (
                "Rhat=Q.T@rho@Q; Uhat=inverse_spectrum*Rhat; u=Q@Uhat@Q.T"
            )
            spectral.create_dataset("Q", data=eigenvectors)
            spectral.create_dataset("eigenvalues_1d", data=eigenvalues)
            spectral.create_dataset("inverse_spectrum", data=inverse_spectrum)

            chunk_rows = min(batch_size, order)
            dense_inverse = data.create_dataset(
                "G",
                shape=(order, order),
                dtype=np.float64,
                chunks=(chunk_rows, order),
                fletcher32=True,
            )
            dense_inverse.attrs["definition"] = "dense inverse of A"
            dense_inverse.attrs["application"] = "u_flat = G @ rho_flat"
            max_residual = write_dense_inverse(dense_inverse, matrix, batch_size)
            data.attrs["max_column_inverse_residual"] = max_residual
            data.attrs["status"] = "complete"
            data.flush()
        partial.replace(output)
    except BaseException:
        if partial.exists():
            with h5py.File(partial, "r+") as data:
                data.attrs["status"] = "failed"
                data.flush()
        raise

    print(f"Saved {dimension}D Poisson solution operator to {output}", flush=True)


def main():
    args = parse_arguments()
    repo = Path(__file__).resolve().parents[1]
    output = args.output or default_output(repo, args.dimension)
    export_operator(output, args.dimension, args.nx, args.domain, args.batch_size)


if __name__ == "__main__":
    main()
