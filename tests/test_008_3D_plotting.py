import os
import sys

import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest
from scipy.constants import c
from tqdm import tqdm

from wakis import GridFIT3D, SolverFIT3D, WakeSolver
from wakis.sources import Beam


@pytest.mark.slow
class Test3Dplotting:
    img_folder = "tests/008_img/"

    def test_simulation(self, use_gpu, flag_offscreen):
        # ---------- Domain setup ---------

        # Geometry & Materials
        solid_1 = "tests/stl/007_vacuum_cavity.stl"
        solid_2 = "tests/stl/007_lossymetal_shell.stl"

        stl_solids = {"cavity": solid_1, "shell": solid_2}

        stl_materials = {
            "cavity": "vacuum",
            "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
        }

        stl_colors = {"cavity": "tab:blue", "shell": "silver"}

        # Extract domain bounds from geometry
        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, ZMIN, ZMAX = solids.bounds

        # Number of mesh cells
        Nx = 60
        Ny = 60
        NZ = 140

        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            ZMIN,
            ZMAX,
            Nx,
            Ny,
            NZ,
            use_mpi=False,  # Enables MPI subdivision of the domain
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_colors=stl_colors,
            stl_scale=1.0,
            stl_rotate=[0, 0, 0],
            stl_translate=[0, 0, 0],
            verbose=1,
        )

        # ------------ Beam source & Wake ----------------
        # Beam parameters
        sigmaz = 10e-2  # [m] -> 2 GHz
        q = 1e-9  # [C]
        beta = 1.0  # beam beta
        xs = 0.0  # x source position [m]
        ys = 1e-2  # y source position [m]
        ti = 3 * sigmaz / c  # injection time [s]

        beam = Beam(q=q, sigmaz=sigmaz, beta=beta, xsource=xs, ysource=ys, ti=ti)

        wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta, xsource=xs, ysource=ys, ti=ti)

        # ----------- Solver & Simulation ----------
        # boundary conditions
        bc_low = ["pec", "pec", "pec"]
        bc_high = ["pec", "pec", "pec"]

        # Solver setup
        global solver
        solver = SolverFIT3D(
            grid,
            wake,
            bc_low=bc_low,
            bc_high=bc_high,
            use_stl=True,
            use_mpi=False,  # Activate MPI
            bg="pec",  # Background material
            dtype=np.float32,
            use_gpu=use_gpu,
        )

        # -------------- Output folder ---------------------
        if not os.path.exists(self.img_folder):
            os.mkdir(self.img_folder)

        # -------------- Custom time loop  -----------------
        Nt = 1000
        for n in tqdm(range(Nt)):
            beam.update(solver, n * solver.dt)
            solver.one_step()

    def test_grid_inspect(self, flag_offscreen):
        # Plot grid and imported solids
        global solver
        pl = solver.grid.inspect(
            add_stl=["cavity", "shell"],
            stl_opacity=0.1,
            off_screen=flag_offscreen,
            anti_aliasing="ssaa",
        )
        if flag_offscreen:
            pl.screenshot(self.img_folder + "grid_inspect.png")
            # pl.export_html(self.img_folder + "grid_inspect.html") # requires trame

    def test_grid_plot_solids(self, flag_offscreen):
        # Plot only imported solids
        global solver
        solver.grid.plot_solids(
            bounding_box=True,
            show_grid=False,
            opacity=1,
            specular=0.5,
            smooth_shading=False,
            off_screen=flag_offscreen,
        )

    def test_grid_stl_mask(self, flag_offscreen):
        # Plot STL solid masks in the grid
        global solver
        pl = solver.grid.plot_stl_mask(
            stl_solid="cavity",
            cmap="viridis",
            bounding_box=True,
            add_stl=True,
            stl_opacity=0.1,
            stl_colors=None,
            anti_aliasing="ssaa",
            smooth_shading=False,
            off_screen=flag_offscreen,
        )
        if flag_offscreen:
            pl.close()

    def test_grid_stl_mask_slice(self, flag_offscreen):
        # Plot STL solid masks in the grid
        global solver
        solver.grid.plot_stl_mask_slice(
            stl_solid="cavity",
            plane="ZX",
            position=None,
            cmap="bwr",
            bounding_box=False,
            show_grid=True,
            add_stl=False,
            off_screen=flag_offscreen,
        )

    def test_solver_inspect(self, flag_offscreen):
        # Plot imported solids and beam source and integraiton path
        global solver
        pl = solver.inspect(
            window_size=(1200, 800),
            off_screen=flag_offscreen,
            specular=0.0,
            opacity=1,
            inactive_opacity=0.1,
            add_silhouette=True,
        )
        if flag_offscreen:
            pl.screenshot(self.img_folder + "solver_inspect.png")
            # pl.export_html(self.img_folder + "solver_inspect.html") # requires trame

    def test_field_inspect3d(self, flag_offscreen):
        global solver
        pl = solver.sigma.inspect3D(
            field="x",
            backend="pyvista",  # matplotlib not covered
            grid=solver.grid,
            ymax=0.0,
            show_grid=False,
            off_screen=flag_offscreen,
        )

        if flag_offscreen:
            pl.screenshot(self.img_folder + "sigma_inspect3d_x.png")
        pl.close()

    def test_field_inspect_pyvista(self, flag_offscreen):
        global solver
        pl = solver.E.inspect(
            plane="YZ",
            backend="pyvista",
            component="z",
            grid=solver.grid,
            off_screen=flag_offscreen,
        )
        if flag_offscreen:
            assert pl.off_screen
            pl.screenshot(self.img_folder + "E_inspect_yz.png")
            pl.close()

    def test_plot3D(self, flag_offscreen):
        # Plot Abs Electric field on domain
        global solver
        solver.plot3D(
            "E",
            component="z",
            cmap="rainbow",
            clim=[0, 500],
            add_stl=["cavity", "shell"],
            stl_opacity=0.1,
            clip_interactive=True,
            clip_normal="-y",
            title=self.img_folder + "Ez3d",
            off_screen=flag_offscreen,
        )

    def test_plot3DonSTL(self, flag_offscreen):
        # Plot Abs Electric field on STL solid `cavity`
        global solver
        solver.plot3DonSTL(
            "E",
            component="Abs",
            cmap="rainbow",
            clim=[0, 500],
            stl_with_field="cavity",
            field_opacity=1.0,
            stl_transparent="shell",
            stl_opacity=0.1,
            stl_colors="white",
            clip_plane=True,
            clip_normal="-y",
            clip_origin=[0, 0, 0],
            off_screen=flag_offscreen,
            zoom=1.2,
            title=self.img_folder + "EAbs3donSTL",
        )


@pytest.mark.parametrize("location", ["primal_coords", "dual_coords", None])
@pytest.mark.parametrize("plane", ["XY", "YX", "XZ", "ZX", "YZ", "ZY"])
@pytest.mark.parametrize("backend", ["matplotlib", "pyvista"])
def test_field_inspect_coordinates(location, plane, backend):
    """Nonuniform staggered samples plot in metres without terminal storage.

    A linear field checks both sample selection and PyVista slice interpolation.
    Fields without coordinates continue to support index-based inspection.
    """
    import matplotlib.pyplot as plt

    from wakis import Field

    grid = GridFIT3D(
        x=np.array([-0.08, -0.03, 0.01, 0.07]),
        y=np.array([-0.04, -0.01, 0.02, 0.09]),
        z=np.array([0.10, 0.12, 0.17, 0.21, 0.30]),
        verbose=0,
    )
    coords = getattr(grid, location) if location is not None else None
    field = Field(grid.Nx, grid.Ny, grid.Nz, coordinates=coords)
    field.array[:] = -999
    if coords is not None:
        for component in "xyz":
            xyz = np.meshgrid(*coords[component], indexing="ij")
            physical = tuple(slice(0, len(axis)) for axis in coords[component])
            field.to_matrix(component)[physical] = xyz[0] + 2 * xyz[1] + 3 * xyz[2]
    normal = next(i for i, axis in enumerate("XYZ") if axis not in plane)
    if backend == "matplotlib":
        fig, axs = field.inspect(plane=plane, off_screen=True)
        try:
            for component, ax in zip("xyz", axs):
                if coords is None:
                    assert ax.images
                    continue
                axes = coords[component]
                pos = (axes[normal][0] + axes[normal][-1]) / 2
                index = np.argmin(abs(axes[normal] - pos))
                selection = [slice(0, len(axis)) for axis in axes]
                selection[normal] = index
                expected = field.to_matrix(component)[tuple(selection)]
                if plane[0] < plane[1]:
                    expected = expected.T
                np.testing.assert_allclose(
                    ax.collections[0].get_array().reshape(expected.shape), expected
                )
                # The plotted bounds must come from the physical axes, not N.
                horizontal = "XYZ".index(plane[0])
                axis = axes[horizontal]
                expected_limits = (
                    axis[0] - (axis[1] - axis[0]) / 2,
                    axis[-1] + (axis[-1] - axis[-2]) / 2,
                )
                np.testing.assert_allclose(ax.get_xlim(), expected_limits)
        finally:
            plt.close(fig)
    else:
        plotter = field.inspect(
            plane=plane,
            component="z",
            backend="pyvista",
            off_screen=True,
        )
        try:
            mesh = plotter.actors["slice"].mapper.dataset
            assert mesh.n_points > 0
            axes = {
                "X": np.array([1.0, 0.0, 0.0]),
                "Y": np.array([0.0, 1.0, 0.0]),
                "Z": np.array([0.0, 0.0, 1.0]),
            }
            np.testing.assert_allclose(plotter.camera.up, axes[plane[1]], atol=1e-12)
            direction = np.array(plotter.camera.position) - plotter.camera.focal_point
            direction /= np.linalg.norm(direction)
            np.testing.assert_allclose(
                direction, np.cross(axes[plane[0]], axes[plane[1]]), atol=1e-12
            )
            if coords is not None:
                points = mesh.points
                np.testing.assert_allclose(
                    mesh["Field z"],
                    points[:, 0] + 2 * points[:, 1] + 3 * points[:, 2],
                    rtol=1e-6,
                    atol=1e-8,
                )
        finally:
            plotter.close()


@pytest.mark.parametrize("backend", ["matplotlib", "pyvista"])
def test_field_inspect_one_cell(backend):
    """A single longitudinal edge remains inspectable at its midpoint."""
    import matplotlib.pyplot as plt

    from wakis import Field

    grid = GridFIT3D(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0], z=[0.1, 0.3], verbose=0)
    field = Field(3, 3, 2, coordinates=grid.primal_coords, use_ones=True)
    result = field.inspect(plane="XY", backend=backend, off_screen=True)
    if backend == "matplotlib":
        plt.close(result[0])
    else:
        try:
            mesh = result.actors["slice"].mapper.dataset
            np.testing.assert_allclose(mesh.points[:, 2], 0.2)
            assert mesh.n_points == 9
        finally:
            result.close()


@pytest.mark.parametrize("coordinates", ["primal_coords", "dual_coords", None])
@pytest.mark.parametrize("component", ["x", "y", "z", "all"])
@pytest.mark.parametrize("backend", ["matplotlib", "pyvista"])
def test_field_inspect3d_coordinates(tmp_path, coordinates, component, backend):
    """3D inspection uses staggered positions with optional STL context.

    A linear field makes interpolated clip values independently verifiable;
    sentinel terminal values must be absent from the physical plot.
    """
    import matplotlib.pyplot as plt

    from wakis import Field

    surface = tmp_path / "solid.stl"
    pv.Cube(bounds=(-0.02, 0.03, -0.02, 0.03, 0.13, 0.2)).triangulate().save(surface)
    grid = GridFIT3D(
        x=np.array([-0.08, -0.03, 0.01, 0.07]),
        y=np.array([-0.04, -0.01, 0.02, 0.09]),
        z=np.array([0.10, 0.12, 0.17, 0.21, 0.30]),
        verbose=0,
    )
    # STL context is optional and independent of the field's physical axes.
    grid.stl_solids = {"solid": str(surface)}
    grid.stl_colors = {"solid": "tab:blue"}
    grid.stl_scale = {"solid": 1.0}
    grid.stl_rotate = {"solid": [0.0, 0.0, 0.0]}
    grid.stl_translate = {"solid": [0.0, 0.0, 0.0]}
    coords = getattr(grid, coordinates) if coordinates is not None else None
    field = Field(grid.Nx, grid.Ny, grid.Nz, coordinates=coords, use_ones=True)
    if coords is not None:
        field.array[:] = -999
        for name in "xyz":
            xyz = np.meshgrid(*coords[name], indexing="ij")
            physical = tuple(slice(0, len(axis)) for axis in coords[name])
            field.to_matrix(name)[physical] = xyz[0] + 2 * xyz[1] + 3 * xyz[2]
    # Coordinate-free cases exercise the index fallback without physical geometry.
    context = grid if coords is not None else None
    result = field.inspect3D(
        field=component,
        backend=backend,
        grid=context,
        off_screen=True,
        stl_opacity=0.15,
    )
    names = "xyz" if component == "all" else component
    if backend == "matplotlib":
        fig, axs = result
        try:
            assert len(axs) == len(names)
            for ax in axs:
                assert ax.collections
                assert ax.get_xlabel() == ("x [m]" if coords is not None else "x index")
        finally:
            plt.close(fig)
    else:
        try:
            for panel, name in enumerate(names):
                result.subplot(0, panel)
                mesh = result.actors["clip"].mapper.dataset
                assert mesh.n_points > 0
                if coords is not None:
                    assert "stl_solid" in result.actors
                    xyz = mesh.points
                    np.testing.assert_allclose(
                        mesh[f"Field {name}"],
                        xyz[:, 0] + 2 * xyz[:, 1] + 3 * xyz[:, 2],
                        atol=1e-7,
                    )
                    np.testing.assert_allclose(
                        mesh.bounds,
                        [v for axis in coords[name] for v in (axis[0], axis[-1])],
                        atol=1e-7,
                    )
        finally:
            result.close()


def test_field_inspect3d_clipping():
    """Changing one panel's clipping slider leaves the other components intact."""
    from wakis import Field

    grid = GridFIT3D(
        x=[-0.08, -0.03, 0.01, 0.07],
        y=[-0.04, -0.01, 0.02, 0.09],
        z=[0.10, 0.12, 0.17, 0.21, 0.30],
        verbose=0,
    )
    field = Field(4, 4, 5, coordinates=grid.primal_coords, use_ones=True)
    plotter = field.inspect3D(off_screen=True, show_grid=False)
    try:
        plotter.subplot(0, 1)
        before = tuple(plotter.actors["clip"].mapper.dataset.bounds)
        widget = plotter.widgets.slider_widgets[0]
        widget.GetRepresentation().SetValue(-0.01)
        widget.InvokeEvent("EndInteractionEvent")
        plotter.subplot(0, 0)
        assert plotter.actors["clip"].mapper.dataset.bounds[1] == pytest.approx(-0.01)
        plotter.subplot(0, 1)
        np.testing.assert_allclose(plotter.actors["clip"].mapper.dataset.bounds, before)
    finally:
        plotter.close()


@pytest.mark.parametrize("backend", ["matplotlib", "pyvista"])
def test_field_inspect3d_one_cell(backend):
    """Boolean masks with a single physical edge render without invented widths."""
    import matplotlib.pyplot as plt

    from wakis import Field

    grid = GridFIT3D(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0], z=[0.1, 0.3], verbose=0)
    mask = Field(3, 3, 2, coordinates=grid.primal_coords, dtype=bool, use_ones=True)
    result = mask.inspect3D(field="z", backend=backend, off_screen=True)
    if backend == "matplotlib":
        plt.close(result[0])
    else:
        try:
            mesh = result.actors["clip"].mapper.dataset
            assert mesh.n_points == 9
            np.testing.assert_allclose(mesh.points[:, 2], 0.2)
        finally:
            result.close()


@pytest.mark.parametrize(
    "method,backend",
    [("inspect", "pyvista"), ("inspect3D", "pyvista"), ("inspect3D", "matplotlib")],
)
@pytest.mark.parametrize("location", ["primal_coords", "dual_coords"])
def test_inspect_absolute(location, method, backend):
    """Keep the existing index norm available while interpolation is deferred."""
    import matplotlib.pyplot as plt

    from wakis import Field

    grid = GridFIT3D(
        x=[0.0, 0.1, 0.4], y=[-0.2, 0.0, 0.3], z=[0.0, 0.2, 0.5], verbose=0
    )
    field = Field(3, 3, 3, coordinates=getattr(grid, location))
    field.field_x = np.ones(field.N)
    field.field_y = np.full(field.N, 2.0)
    field.field_z = np.full(field.N, 3.0)
    if method == "inspect":
        result = field.inspect(component="abs", backend=backend, off_screen=True)
    else:
        result = field.inspect3D(field="abs", backend=backend, off_screen=True)
    if backend == "matplotlib":
        plt.close(result[0])
    else:
        try:
            actor = "slice" if method == "inspect" else "clip"
            scalars = "Field Abs" if method == "inspect" else "Field abs"
            mesh = result.actors[actor].mapper.dataset
            assert mesh.n_points > 0
            np.testing.assert_allclose(mesh[scalars], np.sqrt(14.0))
        finally:
            result.close()


@pytest.fixture(params=[False, True], ids=["uniform", "nonuniform"])
def stl_mask_grid(tmp_path, request):
    """Small overlapping solids classified at staggered component locations."""
    solids = {}
    for key, bounds in {
        "solid": (-0.025, 0.035, -0.035, 0.025, -0.025, 0.035),
        "vacuum": (-0.015, 0.015, -0.015, 0.015, -0.015, 0.015),
    }.items():
        path = tmp_path / f"{key}.stl"
        pv.Cube(bounds=bounds).triangulate().save(path)
        solids[key] = str(path)
    axes = [np.linspace(-0.06, 0.06, n) for n in (7, 8, 9)]
    if request.param:
        axes = [0.06 * np.sign(a) * (abs(a) / 0.06) ** 1.3 for a in axes]
    return GridFIT3D(
        x=axes[0],
        y=axes[1],
        z=axes[2],
        stl_solids=solids,
        stl_materials={"solid": [2.0, 1.0, 0.0], "vacuum": "vacuum"},
        stl_colors={"solid": "orange", "vacuum": "blue"},
        stl_method="legacy" if request.param else "voxelize_rectilinear",
        verbose=0,
    )


@pytest.mark.parametrize("method", ["plot_stl_mask", "plot_stl_mask_slice"])
@pytest.mark.parametrize("grid_type", ["primal", "dual"])
@pytest.mark.parametrize("component", list("xyz"))
def test_mask_components(stl_mask_grid, method, grid_type, component):
    """Masks render at physical component locations without scalar grid labels."""
    grid = stl_mask_grid
    fields = grid.primal_masks if grid_type == "primal" else grid.dual_masks
    before = fields["solid"].array.copy()
    assert not grid.grid.cell_data
    kwargs = dict(
        component=component,
        grid_type=grid_type,
        off_screen=True,
        add_stl={"x": True, "y": ["solid", "vacuum"], "z": "all"}[component],
        stl_colors=["orange", "blue"] if grid_type == "dual" else None,
    )
    if method == "plot_stl_mask":
        kwargs["anti_aliasing"] = None
    else:
        kwargs["plane"] = {
            "primal": {"x": "XY", "y": "ZX", "z": "YZ"},
            "dual": {"x": "YX", "y": "XZ", "z": "ZY"},
        }[grid_type][component]
    plotter = getattr(grid, method)("solid", **kwargs)
    try:
        if method == "plot_stl_mask":
            clipped = plotter.actors["mask"].mapper.dataset
            assert clipped.n_cells > 0
            assert np.all(clipped.cell_data["solid"] == 1)
            initial_count = clipped.n_cells
            bounds = list(clipped.bounds)
            bounds[1] = (bounds[0] + bounds[1]) / 2
            widget = plotter.widgets.box_widgets[0]
            widget.PlaceWidget(bounds)
            widget.InvokeEvent("EndInteractionEvent")
            clipped = plotter.actors["mask"].mapper.dataset
            assert 0 < clipped.n_cells < initial_count
            assert np.all(clipped.cell_data["solid"] == 1)
            assert set(clipped.celltypes) <= {pv.CellType.VOXEL, pv.CellType.HEXAHEDRON}
            widget.PlaceWidget(np.asarray(bounds) + 1)
            widget.InvokeEvent("EndInteractionEvent")
            assert "mask" not in plotter.actors
        else:
            plane = kwargs["plane"]
            normal = next(i for i, axis in enumerate("XYZ") if axis not in plane)
            coords = fields["solid"].coordinates[component]
            widget = plotter.widgets.slider_widgets[0]
            widget.GetRepresentation().SetValue(float(coords[normal][-1]))
            widget.InvokeEvent("EndInteractionEvent")
            section = plotter.actors["slice"].mapper.dataset
            assert section.bounds[2 * normal] == pytest.approx(coords[normal][-1])
            physical = tuple(slice(0, len(a)) for a in coords)
            values = fields["solid"][:, :, :, component][physical]
            expected = np.take(values, -1, axis=normal)
            np.testing.assert_array_equal(
                section.cell_data["solid"], expected.ravel(order="F")
            )
        assert not grid.grid.cell_data
        np.testing.assert_array_equal(fields["solid"].array, before)
    finally:
        plotter.close()


@pytest.mark.parametrize("field,component", [("E", "x"), ("H", "y"), ("E", "Abs")])
def test_mask_patch(stl_mask_grid, field, component):
    """Solver patch overlays consume the selected primal/dual component masks."""
    import matplotlib.pyplot as plt

    solver = SolverFIT3D(stl_mask_grid, use_stl=True, verbose=0)
    solver.E.array.fill(1.0)
    solver.H.array.fill(1.0)
    fig, _ = solver.plot2D(
        field=field,
        component=component,
        plane="XY",
        pos=0.5,
        add_patch=["solid", "vacuum"],
        return_handles=True,
    )
    plt.close(fig)


@pytest.mark.parametrize("method", ["plot_stl_mask", "plot_stl_mask_slice"])
@pytest.mark.parametrize("grid_type", ["primal", "dual"])
@pytest.mark.parametrize("empty", [False, True])
def test_mask_single_cell(tmp_path, method, grid_type, empty):
    """Empty masks and singleton sample axes render without clipping failures."""
    path = tmp_path / "solid.stl"
    pv.Cube(bounds=(-0.03, 0.03, -0.03, 0.03, -0.03, 0.03)).triangulate().save(path)
    grid = GridFIT3D(
        x=[-0.06, 0.06],
        y=[-0.06, 0.06],
        z=[-0.06, 0.06],
        stl_solids={"solid": str(path)},
        stl_materials={"solid": "vacuum"},
        stl_method="voxelize_rectilinear",
        verbose=0,
    )
    fields = grid.primal_masks if grid_type == "primal" else grid.dual_masks
    fields["solid"].array.fill(not empty)
    kwargs = dict(component="x", grid_type=grid_type, off_screen=True, add_stl=False)
    if method == "plot_stl_mask":
        kwargs["anti_aliasing"] = None
    else:
        kwargs["plane"] = "YZ" if grid_type == "primal" else "XY"
    plotter = getattr(grid, method)("solid", **kwargs)
    plotter.close()


def test_grid_geometry_inspection(stl_mask_grid):
    """Grid geometry plots work without scalar masks or dense coordinate grids."""
    for method in ("inspect", "plot_solids"):
        plotter = getattr(stl_mask_grid, method)(off_screen=True, anti_aliasing=None)
        plotter.close()
