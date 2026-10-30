"""Voxelized pipe geometry, smoothing, and material assignment."""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import epsilon_0, mu_0

from wakis import GridFIT3D, SolverFIT3D


class TestVoxelize:
    # Physical mid-z cross-sections sampled every third x/y cell, factor=3,
    # threshold=0.3. These weights protect smoothing, not wake postprocessing.
    # fmt: off
    smoothed_samples = {
        "square": np.array([
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.351163, 0.351163, 0.351163, 0.351163, 0.328680, 0.000000],
            [0.000000, 0.797631, 0.995821, 0.982040, 0.982039, 0.992808, 0.964253, 0.000000],
            [0.000000, 0.797674, 0.803882, 0.000000, 0.000000, 0.659668, 0.964297, 0.000000],
            [0.000000, 0.797674, 0.797674, 0.000000, 0.000000, 0.648837, 0.964297, 0.000000],
            [0.000000, 0.797674, 0.806975, 0.000000, 0.000000, 0.665064, 0.964297, 0.000000],
            [0.000000, 0.797589, 0.999882, 0.999780, 0.999780, 0.999859, 0.964209, 0.000000],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        ]),
        "circular": np.array([
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.920919, 0.999209, 0.998570, 0.966046, 0.000000, 0.000000],
            [0.000000, 0.642883, 0.987913, 0.000000, 0.000000, 0.965086, 0.842996, 0.000000],
            [0.000000, 0.797657, 0.809845, 0.000000, 0.000000, 0.660509, 0.962428, 0.000000],
            [0.000000, 0.637860, 0.990927, 0.000000, 0.000000, 0.971817, 0.795045, 0.000000],
            [0.000000, 0.000000, 0.899013, 0.999468, 0.999824, 0.945347, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        ]),
    }
    # fmt: on

    @pytest.mark.parametrize(
        "smoothing,make_bool", [(False, True), (True, True), (True, False)]
    )
    def test_pipe(self, pipe_surface, tmp_path, request, smoothing, make_bool):
        """Offset square/circular pipes preserve placement and material weights.

        Raw masks match analytical cell-centre classification. Smoothing has
        compact static references; all modes preserve translation/scaling and
        supply full-size material tensors, including terminal storage.
        """
        shape, surface, path = pipe_surface
        center = (0.007, -0.004)
        inner, outer = 0.023, 0.04
        grid = GridFIT3D(
            xmin=-0.055, xmax=0.065,
            ymin=-0.06, ymax=0.05,
            zmin=-0.02, zmax=0.03,
            Nx=25, Ny=23, Nz=7,
            stl_solids={"pipe": str(path)},
            stl_materials={"pipe": [2.0, 3.0, 0.1]},
            stl_method="voxelize_rectilinear",
            subpixel_smoothing=smoothing,
            subpixel_smoothing_factor=3,
            subpixel_smoothing_threshold=0.3,
            subpixel_smoothing_bool=make_bool,
            verbose=0,
        )  # fmt: skip
        mask = grid.grid.cell_data["pipe"].reshape((grid.Nx, grid.Ny, grid.Nz))
        x = (grid.x[:-1] + grid.x[1:]) / 2
        y = (grid.y[:-1] + grid.y[1:]) / 2
        xx, yy = np.meshgrid(x - center[0], y - center[1], indexing="ij")
        distance = (
            np.maximum(np.abs(xx), np.abs(yy))
            if shape == "square"
            else np.hypot(xx, yy)
        )
        expected = (distance >= inner) & (distance <= outer)
        section = mask[:-1, :-1, (grid.Nz - 1) // 2]

        # Plot before assertions so a failing comparison can be inspected.
        if request.config.getoption("--debug-plots"):
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, Rectangle

            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            image = axes[0].pcolormesh(grid.x, grid.y, section.T, vmin=0, vmax=1)
            difference = axes[1].pcolormesh(
                grid.x,
                grid.y,
                (section.astype(float) - expected).T,
                vmin=-1,
                vmax=1,
                cmap="RdBu_r",
            )
            for ax in axes:
                for radius in (inner, outer):
                    if shape == "square":
                        outline = Rectangle(
                            (center[0] - radius, center[1] - radius),
                            2 * radius,
                            2 * radius,
                        )
                    else:
                        outline = Circle(center, radius)
                    outline.set(fill=False, edgecolor="black", linewidth=0.8)
                    ax.add_patch(outline)
                ax.set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
            axes[0].set_title("Physical mid-z mask")
            axes[1].set_title("Mask − analytical centre classification")
            fig.colorbar(image, ax=axes[0], label="Material weight")
            fig.colorbar(difference, ax=axes[1], label="Difference")
            mode = "raw" if not smoothing else "boolean" if make_bool else "fractional"
            fig.suptitle(f"{shape.capitalize()} pipe: {mode}")
            plt.show()
            plt.close(fig)

        assert np.all((mask >= 0) & (mask <= 1))
        if smoothing:
            reference = self.smoothed_samples[shape]
            if make_bool:
                np.testing.assert_array_equal(section[::3, ::3], reference > 0.3)
            else:
                # Six-decimal reference rounding bounds the absolute error.
                np.testing.assert_allclose(
                    section[::3, ::3], reference, rtol=0, atol=5e-7
                )
        else:
            np.testing.assert_array_equal(section, expected)
            area = (4 if shape == "square" else np.pi) * (outer**2 - inner**2)
            volume = np.sum(
                mask[:-1, :-1, :-1]
                * grid.dx[:, None, None]
                * grid.dy[None, :, None]
                * grid.dz[None, None, :]
            )
            # Centre classification can move each wall by half a cell diagonal.
            perimeter = (8 if shape == "square" else 2 * np.pi) * (inner + outer)
            area_error = perimeter * np.hypot(grid.dx[0], grid.dy[0]) / 2
            assert abs(volume - area * (grid.z[-1] - grid.z[0])) <= area_error * (
                grid.z[-1] - grid.z[0]
            )
        np.testing.assert_allclose(
            mask[:-1, :-1, :-1],
            np.broadcast_to(
                section[:, :, None], (grid.Nx - 1, grid.Ny - 1, grid.Nz - 1)
            ),
        )
        for axis in range(3):
            np.testing.assert_array_equal(
                np.take(mask, -1, axis=axis), np.take(mask, -2, axis=axis)
            )

        solver = SolverFIT3D(
            grid, use_stl=True, bc_low=["pmc"] * 3, bc_high=["pmc"] * 3, verbose=0
        )
        for component in "xyz":
            np.testing.assert_allclose(
                1 / (solver.ieps.to_matrix(component) * epsilon_0),
                1 + mask.astype(float),
            )
            np.testing.assert_allclose(
                1 / (solver.imu.to_matrix(component) * mu_0), 1 + 2 * mask.astype(float)
            )
            np.testing.assert_allclose(
                solver.sigma.to_matrix(component), 0.1 * mask, atol=0
            )

        for scale, offset in ((1.0, (0.2, -0.3, 0.4)), (0.01, (0, 0, 0))):
            transformed_path = tmp_path / "transformed.stl"
            surface.scale(scale).translate(offset).save(transformed_path)
            coordinates = {
                axis: scale * getattr(grid, axis) + shift
                for axis, shift in zip("xyz", offset)
            }
            transformed = GridFIT3D(
                **coordinates,
                stl_solids={"pipe": str(transformed_path)},
                stl_materials={"pipe": [2.0, 3.0, 0.1]},
                stl_method="voxelize_rectilinear",
                subpixel_smoothing=smoothing,
                subpixel_smoothing_factor=3,
                subpixel_smoothing_threshold=0.3,
                subpixel_smoothing_bool=make_bool,
                verbose=0,
            )
            np.testing.assert_allclose(
                transformed.grid.cell_data["pipe"], mask.ravel(), rtol=1e-12, atol=0
            )


@pytest.fixture(params=["square", "circular"])
def pipe_surface(request, tmp_path):
    """Closed notebook pipe shells, with end caps outside the physical domain."""
    shape = request.param
    segments = 4 if shape == "square" else 256
    if shape == "square":
        ring = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    else:
        angle = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        ring = np.column_stack((np.cos(angle), np.sin(angle)))
    points = []
    for z in (-0.05, 0.06):
        for radius in (0.023, 0.04):
            xy = np.array([0.007, -0.004]) + radius * ring
            points.extend(np.column_stack((xy, np.full(segments, z))))
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        for quad in (
            (i, j, segments + j, segments + i),
            (2 * segments + i, 3 * segments + i, 3 * segments + j, 2 * segments + j),
            (i, 2 * segments + i, 2 * segments + j, j),
            (segments + i, segments + j, 3 * segments + j, 3 * segments + i),
        ):
            faces.extend((4, *quad))
    surface = pv.PolyData(np.asarray(points), faces).triangulate().clean()
    surface = surface.compute_normals(auto_orient_normals=True, consistent_normals=True)
    path = tmp_path / f"{shape}_pipe.stl"
    surface.save(path)
    return shape, surface, path


@pytest.mark.parametrize("smoothing", [False, True])
@pytest.mark.parametrize("xmin", [0.7, 0.99])
def test_voxel_terminal_material(tmp_path, smoothing, xmin):
    """Terminal labels extend the last resolved cell, including after smoothing."""
    path = tmp_path / "boundary.stl"
    pv.Box(bounds=(xmin, 1.01, -1, 2, -1, 2)).triangulate().save(path)
    grid = GridFIT3D(
        xmin=0, xmax=1,
        ymin=0, ymax=1,
        zmin=0, zmax=1,
        Nx=5, Ny=4, Nz=2,
        stl_solids={"solid": str(path)},
        stl_materials={"solid": [2.0, 1.0, 0.0]},
        stl_method="voxelize_rectilinear",
        subpixel_smoothing=smoothing,
        verbose=0,
    )  # fmt: skip
    mask = grid.grid.cell_data["solid"].reshape((grid.Nx, grid.Ny, grid.Nz))
    assert not mask[0].any()
    assert mask[-1].all() == (xmin < 0.875)
    np.testing.assert_array_equal(mask[-1], mask[-2])
