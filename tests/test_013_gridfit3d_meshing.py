"""Voxelized pipe geometry, smoothing, and material assignment."""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import epsilon_0, mu_0

from wakis import GridFIT3D, SolverFIT3D


def test_grid_h5_roundtrip_without_solids(tmp_path):
    """Nonuniform node axes rebuild the symmetric FIT metrics after loading."""
    grid = GridFIT3D(
        x=np.array([0.0, 0.2, 0.5, 1.0]),
        y=np.array([-1.0, -0.4, 0.5]),
        z=np.array([0.0, 0.1, 0.4, 1.0]),
        verbose=0,
    )
    path = str(tmp_path / "grid.h5")
    grid.save_to_h5(path)
    loaded = GridFIT3D(load_from_h5=path, verbose=0)
    with pytest.raises(NotImplementedError, match="MPI decomposition"):
        GridFIT3D(load_from_h5=path, use_mpi=True, verbose=0)

    for attr in ("x", "y", "z", "dx", "dy", "dz", "tx", "ty", "tz"):
        np.testing.assert_array_equal(getattr(loaded, attr), getattr(grid, attr))
    for attr in ("L", "iA", "tL", "itA"):
        np.testing.assert_array_equal(
            getattr(loaded, attr).toarray(), getattr(grid, attr).toarray()
        )
    for component in "xyz":
        for axis, nodes in zip("xyz", (grid.x, grid.y, grid.z)):
            midpoint = (nodes[:-1] + nodes[1:]) / 2
            index = "xyz".index(axis)
            for obj in (grid, loaded):
                np.testing.assert_array_equal(
                    obj.primal_coords[component][index],
                    midpoint if axis == component else nodes,
                )
                np.testing.assert_array_equal(
                    obj.dual_coords[component][index],
                    nodes if axis == component else midpoint,
                )
    assert loaded.grid.n_cells == grid.Nx * grid.Ny * grid.Nz
    assert not loaded.grid.cell_data


def test_grid_h5_roundtrip_with_solid(tmp_path):
    """A saved STL mask remains aligned with the rebuilt computational grid."""
    solid = tmp_path / "solid.stl"
    pv.Cube(bounds=(0.2, 0.8, 0.2, 0.8, 0.2, 0.8)).triangulate().save(solid)
    grid = GridFIT3D(
        x=np.array([0.0, 0.25, 0.6, 1.0]),
        y=np.array([0.0, 0.3, 0.7, 1.0]),
        z=np.array([0.0, 0.2, 0.5, 1.0]),
        stl_solids={"solid": str(solid)},
        stl_materials={"solid": [2.0, 1.0, 0.0]},
        stl_method="legacy",
        verbose=0,
    )
    path = str(tmp_path / "grid.h5")
    grid.save_to_h5(path)
    loaded = GridFIT3D(load_from_h5=path, verbose=0)

    np.testing.assert_array_equal(
        loaded.primal_masks["solid"].array, grid.primal_masks["solid"].array
    )
    for attr in ("L", "iA", "tL", "itA"):
        np.testing.assert_array_equal(
            getattr(loaded, attr).toarray(), getattr(grid, attr).toarray()
        )
    assert loaded.stl_materials["solid"] == pytest.approx([2.0, 1.0, 0.0])


@pytest.mark.parametrize(
    "method,nonuniform",
    [("voxelize_rectilinear", False), ("legacy", False), ("legacy", True)],
)
def test_staggered_masks(tmp_path, method, nonuniform):
    """Translated solids classify electric and magnetic samples and survive HDF5.

    Include vacuum and a dielectric. Bounds lie
    away from sample locations so surface classification is unambiguous.
    """
    solids = {}
    bounds = {
        "vacuum": (-0.023, 0.037, -0.033, 0.027, -0.017, 0.027),
        "dielectric": (-0.031, 0.016, -0.034, 0.018, -0.037, 0.037),
    }
    for key, box in bounds.items():
        path = tmp_path / f"{key}.stl"
        pv.Cube(bounds=box).triangulate().save(path)
        solids[key] = str(path)
    axes = [np.linspace(-0.06, 0.06, n) for n in (9, 9, 7)]
    if nonuniform:
        axes = [0.06 * np.sign(a) * (np.abs(a) / 0.06) ** 1.3 for a in axes]
    grid = GridFIT3D(
        x=axes[0],
        y=axes[1],
        z=axes[2],
        stl_solids=solids,
        stl_materials={"vacuum": "vacuum", "dielectric": [2.0, 2.0, 0.01]},
        stl_method=method,
        subpixel_smoothing=False,
        subpixel_smoothing_factor=2,
        verbose=0,
    )
    dimensions = (grid.Nx, grid.Ny, grid.Nz)
    solver = SolverFIT3D(
        grid,
        use_stl=True,
        bg=[3.0, 1.5, 0.02],
        bc_low=["pmc"] * 3,
        bc_high=["pmc"] * 3,
        verbose=0,
    )
    expected_ieps = np.full((3, *dimensions), 1 / (3 * epsilon_0))
    expected_sigma = np.full((3, *dimensions), 0.02)
    expected_imu = np.full((3, *dimensions), 1 / (1.5 * mu_0))
    for field in (solver.E, solver.J, solver.ieps, solver.sigma):
        assert field.coordinates is grid.primal_coords
    for field in (solver.H, solver.imu):
        assert field.coordinates is grid.dual_coords
    assert not grid.grid.cell_data
    for key, box in bounds.items():
        material_mask = grid.primal_masks[key]
        assert material_mask.coordinates is grid.primal_coords
        packed = material_mask.array
        assert packed.dtype == bool
        assert packed.shape == (3 * np.prod(dimensions),)
        for component, block in enumerate(packed.reshape(3, -1)):
            positions = [a.copy() for a in axes]
            a = axes[component]
            positions[component] = (a[:-1] + a[1:]) / 2
            xyz = np.meshgrid(*positions, indexing="ij")
            expected = np.ones(xyz[0].shape, dtype=bool)
            for axis, values in enumerate(xyz):
                expected &= (values > box[2 * axis]) & (values < box[2 * axis + 1])
            mask = block.reshape(dimensions, order="F")
            physical = tuple(slice(0, n) for n in expected.shape)
            np.testing.assert_array_equal(mask[physical], expected)
            occupancy = np.zeros(dimensions, dtype=bool)
            occupancy[physical] = expected
            eps, _, conductivity = grid.stl_materials[key]
            expected_ieps[component][occupancy] = 1 / (eps * epsilon_0)
            expected_sigma[component][occupancy] = conductivity
            terminal = [slice(None)] * 3
            terminal[component] = -1
            assert not mask[tuple(terminal)].any()

    for key, box in bounds.items():
        dual_mask = grid.dual_masks[key]
        assert dual_mask.coordinates is grid.dual_coords
        for index, component in enumerate("xyz"):
            xyz = np.meshgrid(*grid.dual_coords[component], indexing="ij")
            expected = np.ones(xyz[0].shape, dtype=bool)
            for axis, values in enumerate(xyz):
                expected &= (values > box[2 * axis]) & (values < box[2 * axis + 1])
            occupancy = np.zeros(dimensions, dtype=bool)
            physical = tuple(slice(0, n) for n in expected.shape)
            occupancy[physical] = expected
            np.testing.assert_array_equal(dual_mask.to_matrix(component), occupancy)
            expected_imu[index][occupancy] = 1 / (grid.stl_materials[key][1] * mu_0)

    for index, component in enumerate("xyz"):
        np.testing.assert_allclose(
            solver.ieps.to_matrix(component), expected_ieps[index]
        )
        np.testing.assert_allclose(solver.imu.to_matrix(component), expected_imu[index])
        np.testing.assert_allclose(
            solver.sigma.to_matrix(component), expected_sigma[index]
        )
    solver.one_step()
    assert solver.E.coordinates is grid.primal_coords
    assert solver.H.coordinates is grid.dual_coords

    path = str(tmp_path / "grid.h5")
    grid.save_to_h5(path)
    loaded = GridFIT3D(load_from_h5=path, verbose=0)
    for key in solids:
        assert loaded.dual_masks[key].coordinates is loaded.dual_coords
        np.testing.assert_array_equal(
            loaded.dual_masks[key].array, grid.dual_masks[key].array
        )
        assert loaded.primal_masks[key].coordinates is loaded.primal_coords
        np.testing.assert_array_equal(
            loaded.primal_masks[key].array, grid.primal_masks[key].array
        )


class TestVoxelize:
    def test_pipe(self, pipe_surface, tmp_path):
        """Electric samples match translated analytical pipe walls and material values."""
        shape, surface, path = pipe_surface
        grid = GridFIT3D(
            xmin=-0.055,
            xmax=0.065,
            ymin=-0.06,
            ymax=0.05,
            zmin=-0.02,
            zmax=0.03,
            Nx=25,
            Ny=23,
            Nz=7,
            stl_solids={"pipe": str(path)},
            stl_materials={"pipe": [2.0, 1.0, 0.1]},
            stl_method="voxelize_rectilinear",
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            use_stl=True,
            bc_low=["pmc"] * 3,
            bc_high=["pmc"] * 3,
            verbose=0,
        )
        for component in "xyz":
            x, y, z = np.meshgrid(*grid.primal_coords[component], indexing="ij")
            distance = (
                np.maximum(abs(x - 0.007), abs(y + 0.004))
                if shape == "square"
                else np.hypot(x - 0.007, y + 0.004)
            )
            expected = (distance >= 0.023) & (distance <= 0.04)
            physical = tuple(slice(0, n) for n in expected.shape)
            actual = grid.primal_masks["pipe"].to_matrix(component)[physical]
            # Polygonal STL walls differ from a circle by at most the sagitta.
            margin = 0.04 * (1 - np.cos(np.pi / 256)) + 1e-8
            away = (abs(distance - 0.023) > margin) & (abs(distance - 0.04) > margin)
            np.testing.assert_array_equal(actual[away], expected[away])
            expected[~away] = actual[~away]
            np.testing.assert_allclose(
                solver.ieps.to_matrix(component)[physical] * epsilon_0,
                np.where(expected, 0.5, 1.0),
            )
            np.testing.assert_allclose(
                solver.sigma.to_matrix(component)[physical], 0.1 * expected, atol=0
            )
            np.testing.assert_allclose(solver.imu.to_matrix(component) * mu_0, 1.0)

        for scale, offset in ((1.0, (0.2, -0.3, 0.4)), (0.01, (0.0, 0.0, 0.0))):
            transformed_path = tmp_path / "transformed.stl"
            surface.scale(scale).translate(offset).save(transformed_path)
            coordinates = {
                axis: scale * getattr(grid, axis) + shift
                for axis, shift in zip("xyz", offset)
            }
            transformed = GridFIT3D(
                **coordinates,
                stl_solids={"pipe": str(transformed_path)},
                stl_materials={"pipe": [2.0, 1.0, 0.1]},
                stl_method="voxelize_rectilinear",
                verbose=0,
            )
            np.testing.assert_array_equal(
                transformed.primal_masks["pipe"].array,
                grid.primal_masks["pipe"].array,
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


@pytest.mark.parametrize("xmin", [0.7, 0.99])
def test_voxel_terminal_material(tmp_path, xmin):
    """Boundary node samples resolve material even beyond the final edge midpoint."""
    path = tmp_path / "boundary.stl"
    pv.Box(bounds=(xmin, 1.01, -1, 2, -1, 2)).triangulate().save(path)
    grid = GridFIT3D(
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=1,
        zmin=0,
        zmax=1,
        Nx=5,
        Ny=4,
        Nz=2,
        stl_solids={"solid": str(path)},
        stl_materials={"solid": [2.0, 1.0, 0.1]},
        stl_method="voxelize_rectilinear",
        verbose=0,
    )
    solver = SolverFIT3D(
        grid, use_stl=True, bc_low=["pmc"] * 3, bc_high=["pmc"] * 3, verbose=0
    )
    for component in "xyz":
        x, y, z = np.meshgrid(*grid.primal_coords[component], indexing="ij")
        expected = x > xmin
        physical = tuple(slice(0, n) for n in expected.shape)
        np.testing.assert_array_equal(
            grid.primal_masks["solid"].to_matrix(component)[physical], expected
        )
        np.testing.assert_allclose(
            solver.ieps.to_matrix(component)[physical] * epsilon_0,
            np.where(expected, 0.5, 1.0),
        )


@pytest.mark.parametrize("method", ["implicit_distance", "interior_points"])
def test_unsupported_stl_method(tmp_path, method):
    """Removed classification methods fail explicitly rather than producing scalar masks."""
    path = tmp_path / "solid.stl"
    pv.Cube().triangulate().save(path)
    with pytest.raises(ValueError):
        GridFIT3D(
            x=[-1.0, 0.0, 1.0],
            y=[-1.0, 0.0, 1.0],
            z=[-1.0, 0.0, 1.0],
            stl_solids={"solid": str(path)},
            stl_materials={"solid": "vacuum"},
            stl_method=method,
            verbose=0,
        )


def test_unsupported_sibc(tmp_path):
    """Volume masks cannot substitute for SIBC geometry."""
    path = tmp_path / "solid.stl"
    pv.Cube().triangulate().save(path)
    grid = GridFIT3D(
        x=[-1.0, 0.0, 1.0],
        y=[-1.0, 0.0, 1.0],
        z=[-1.0, 0.0, 1.0],
        stl_solids={"solid": str(path)},
        stl_materials={"solid": [2.0, 1.0, 0.1]},
        verbose=0,
    )
    with pytest.raises(NotImplementedError):
        SolverFIT3D(grid, use_stl=True, use_sibc=True, verbose=0)


@pytest.mark.parametrize("method", ["legacy", "voxelize_rectilinear"])
def test_mirrored_beam_wakes(tmp_path, monkeypatch, method):
    """A symmetric PEC pipe/cavity gives opposite on-axis kicks for mirrored beams.

    The step excites a nonzero wake. Both runs include TF/SF injection, CPML,
    saved longitudinal edges, and Panofsky-Wenzel transverse integration.
    """
    from wakis import WakeSolver

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "vacuum.stl"
    # Build a closed stepped square pipe directly, avoiding boolean-surface
    # intersection artifacts in the legacy point classifier.
    ring = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    sections = [
        (-0.16, 0.0415),
        (-0.05, 0.0415),
        (-0.05, 0.078),
        (0.05, 0.078),
        (0.05, 0.0415),
        (0.16, 0.0415),
    ]
    points = np.vstack(
        [np.column_stack((radius * ring, np.full(4, z))) for z, radius in sections]
    )
    faces = [[4, 3, 2, 1, 0]]
    for section in range(len(sections) - 1):
        for i in range(4):
            j = (i + 1) % 4
            faces.append(
                [
                    4,
                    4 * section + i,
                    4 * section + j,
                    4 * (section + 1) + j,
                    4 * (section + 1) + i,
                ]
            )
    faces.append([4, 20, 21, 22, 23])
    surface = pv.PolyData(points, np.array(faces).ravel()).triangulate()
    surface.compute_normals(auto_orient_normals=True, consistent_normals=True).save(
        path
    )
    grid = GridFIT3D(
        xmin=-0.09,
        xmax=0.09,
        ymin=-0.09,
        ymax=0.09,
        zmin=-0.15,
        zmax=0.15,
        Nx=21,
        Ny=21,
        Nz=41,
        stl_solids={"vacuum": str(path)},
        stl_materials={"vacuum": "vacuum"},
        stl_method=method,
        verbose=0,
    )
    results = []
    for index, offset in enumerate((0.009, -0.009)):
        wake = WakeSolver(
            xsource=offset,
            xtest=0.0,
            ysource=0.0,
            ytest=0.0,
            sigmaz=0.015,
            skip_cells=7,
            save=False,
            verbose=0,
            results_folder=str(tmp_path / f"run{index}"),
            Ez_file=str(tmp_path / f"Ez{index}.h5"),
        )
        solver = SolverFIT3D(
            grid,
            wake,
            use_stl=True,
            bg="pec",
            bc_low=["pec", "pec", "cpml"],
            bc_high=["pec", "pec", "cpml"],
            n_pml=4,
            verbose=0,
        )
        assert solver.E_trans.coordinates is grid.primal_coords
        assert solver.H_trans.coordinates is grid.dual_coords
        solver.wakesolve(wakelength=0.08)
        results.append(
            (wake.WP.copy(), wake.WPx.copy(), solver.E.to_matrix("z").copy())
        )
    plus, minus = results
    assert np.max(abs(plus[1])) > 1e-10
    for left, right in (
        (plus[0], minus[0]),
        (plus[1], -minus[1]),
        (plus[2], minus[2][::-1, :, :]),
    ):
        scale = max(np.max(abs(left)), np.max(abs(right)))
        np.testing.assert_allclose(left, right, rtol=0, atol=5e-11 * scale)
