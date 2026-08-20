import random
import subprocess
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def main():
    random.seed(42)
    M = 5
    rho_min = 0.5
    rho_max = 2.0

    repo = Path(__file__).resolve().parents[1]
    executable = repo / "build" / "src" / "artemis"
    input_file = repo / "inputs" / "piecewise_poisson" / "piecewise.in"
    output_root = repo / "runs" / "piecewise_poisson_dataset"
    output_root.mkdir(parents=True, exist_ok=True)

    parameters = []
    density_fields = []
    potential_fields = []

    for i in range(M):
        # Sample rho_1^{(m)}, rho_2^{(m)}, and T_m.
        rho_1 = random.uniform(rho_min, rho_max)
        rho_2 = random.uniform(rho_min, rho_max)
        threshold = random.randint(-24, 24)
        parameters.append((rho_1, rho_2, threshold))

        # Run the Poisson solver in a separate directory for this sample.
        sample_id = f"sample_{i:05d}"
        sample_dir = output_root / sample_id
        sample_dir.mkdir(exist_ok=True)
        subprocess.run(
            [
                "mpirun",
                "-n",
                "1",
                str(executable),
                "-i",
                str(input_file),
                f"problem/rho_1={rho_1}",
                f"problem/rho_2={rho_2}",
                f"problem/threshold={threshold}",
                f"parthenon/job/problem_id={sample_id}",
            ],
            cwd=sample_dir,
            check=True,
        )

        # Verify and load the HDF5 output.
        output_file = sample_dir / f"{sample_id}.out1.final.phdf"
        if not output_file.is_file():
            raise FileNotFoundError(f"Missing Artemis output: {output_file}")

        with h5py.File(output_file, "r") as data:
            logical_locations = data["LogicalLocations"][:]
            density_blocks = data["gas.prim.density_0"][:, 0, :, :]
            potential_blocks = data["grav.phi"][:, 0, :, :]
            rhs_blocks = data["grav.rhs"][:, 0, :, :]
            x_centers = data["VolumeLocations/x"][:]

            expected_density = np.where(
                x_centers[:, np.newaxis, :] < threshold, rho_1, rho_2
            )
            if not np.allclose(density_blocks, expected_density):
                raise ValueError(f"Density verification failed for {sample_id}")
            if not np.allclose(rhs_blocks, density_blocks):
                raise ValueError(f"Poisson RHS verification failed for {sample_id}")
            if not np.isfinite(potential_blocks).all():
                raise ValueError(f"Non-finite potential found for {sample_id}")

            density = assemble_field(density_blocks, logical_locations)
            potential = assemble_field(potential_blocks, logical_locations)
            density_fields.append(density)
            potential_fields.append(potential)

            if i < 5:
                figure, axes = plt.subplots(1, 2, figsize=(10, 4))
                density_image = axes[0].imshow(
                    density,
                    origin="lower",
                    cmap="viridis",
                    vmin=rho_min,
                    vmax=rho_max,
                )
                axes[0].set_title("Density")
                figure.colorbar(density_image, ax=axes[0], label="rho")

                potential_image = axes[1].imshow(potential, origin="lower", cmap="viridis")
                axes[1].set_title("Potential")
                figure.colorbar(potential_image, ax=axes[1], label="phi")

                figure.tight_layout()
                figure.savefig(sample_dir / "rho_and_phi.png", dpi=200)
                plt.close(figure)

    # Save NumPy-compatible arrays that can later be loaded by PyTorch.
    dataset_file = output_root / "piecewise_poisson_dataset.h5"
    with h5py.File(dataset_file, "w") as data:
        data.create_dataset("parameters", data=np.asarray(parameters))
        data.create_dataset("rho", data=np.stack(density_fields))
        data.create_dataset("phi", data=np.stack(potential_fields))

    print(f"Saved {M} samples to {dataset_file}")


if __name__ == "__main__":
    main()
