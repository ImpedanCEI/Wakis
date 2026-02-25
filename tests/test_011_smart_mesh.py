import os
import sys

import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


@pytest.mark.slow
class TestSmartMesh:
    zedges = [-0.05, -0.0075, 0.0075, 0.05]
    xedges = [-0.025, -0.005, 0.005, 0.025]
    yedges = [-0.025, -0.005, 0.005, 0.025]

    print("\n---------- Initializing simulation ------------------")
    # Number of mesh cells
    Nx = 50
    Ny = 50
    Nz = 80

    # Embedded boundaries
    stl_file = "tests/stl/001_cubic_cavity.stl"
    surf = pv.read(stl_file)

    stl_solids = {"cavity": stl_file}
    stl_materials = {"cavity": "vacuum"}

    # Domain bounds
    xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds

    refinement_tol = 1e-8
    snap_tol = 1e-2
    # set grid and geometry
    global grid
    grid = GridFIT3D(
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        Nx,
        Ny,
        Nz,
        stl_solids=stl_solids,
        stl_materials=stl_materials,
        use_mesh_refinement=True,
        snap_tol=snap_tol,
        refinement_tol=refinement_tol,
    )

    # Beam parameters
    beta = 1.0  # beam beta
    sigmaz = 18.5e-3 * beta  # [m]
    q = 1e-9  # [C]
    xs = 0.0  # x source position [m]
    ys = 0.0  # y source position [m]
    xt = 0.0  # x test position [m]
    yt = 0.0  # y test position [m]

    global wake
    skip_cells = 12  # no. cells to skip in WP integration
    wake = WakeSolver(
        q=q,
        sigmaz=sigmaz,
        beta=beta,
        xsource=xs,
        ysource=ys,
        xtest=xt,
        ytest=yt,
        save=False,
        Ez_file="tests/011_Ez.h5",
        skip_cells=skip_cells,
    )

    # boundary conditions
    bc_low = ["pec", "pec", "pec"]
    bc_high = ["pec", "pec", "pec"]

    # set Solver object
    solver = SolverFIT3D(
        grid,
        wake,
        bc_low=bc_low,
        bc_high=bc_high,
        use_stl=True,
        bg="pec",
        dtype=np.float32,
    )

    wakelength = 1.0  # [m]
    solver.wakesolve(wakelength=wakelength, save_J=False)
    os.remove("tests/011_Ez.h5")

    def test_grid_generation(self):
        global grid
        for edg in self.zedges:
            diff = np.min(np.abs(grid.z - edg))
            assert diff <= 1e-8 + 1e-2, "Mesh not inside the tolerance at the z edges"

        for edg in self.yedges:
            diff = np.min(np.abs(grid.y - edg))
            assert diff <= 1e-8 + 1e-2, "Mesh not inside the tolerance at the y edges"

        for edg in self.xedges:
            diff = np.min(np.abs(grid.x - edg))
            assert diff <= 1e-8 + 1e-2, "Mesh not inside the tolerance at the x edges"

        zdiff = np.abs(np.diff(grid.dz))
        for cell in range(len(grid.dz) - 1):
            assert zdiff[cell] <= 0.1 * grid.dz[cell], "Mesh variance in z is too big"

        ydiff = np.abs(np.diff(grid.dy))
        for cell in range(len(grid.dy) - 1):
            assert ydiff[cell] <= 0.1 * grid.dy[cell], "Mesh variance in y is too big"

        xdiff = np.abs(np.diff(grid.dx))
        for cell in range(len(grid.dx) - 1):
            assert xdiff[cell] <= 0.1 * grid.dx[cell], "Mesh variance in x is too big"

        assert np.min(grid.dz) > 0.5 * (grid.zmax - grid.zmin) / grid.Nz, (
            "Smallest z difference is too small"
        )
        assert np.min(grid.dy) > 0.5 * (grid.ymax - grid.ymin) / grid.Ny, (
            "Smallest y difference is too small"
        )
        assert np.min(grid.dx) > 0.5 * (grid.xmax - grid.xmin) / grid.Nx, (
            "Smallest x difference is too small"
        )

    def test_nonuniform_simulation(self):
        global wake
        assert np.abs(wake.Z[0]) < 0.01 * np.abs(np.max(wake.Z)), (
            "Charge Accumulation - DC component"
        )
