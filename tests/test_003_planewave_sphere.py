import os
import sys

import numpy as np
import pyvista as pv
from scipy.constants import c

sys.path.append("../wakis")

import pytest

from wakis import GridFIT3D, SolverFIT3D
from wakis.sources import PlaneWave

# Turn False when running local
flag_offscreen = True


class TestPlanewave:
    img_folder = "tests/003_img/"

    def test_simulation(self):
        print("\n---------- Initializing simulation ------------------")
        # Number of mesh cells
        Nx = 60
        Ny = 60
        Nz = 120

        # Embedded boundaries
        stl_file = "tests/stl/003_sphere.stl"
        surf = pv.read(stl_file)

        stl_solids = {"Sphere": stl_file}
        stl_materials = {"Sphere": [10.0, 1.0]}  # dielectric [eps_r, mu_r]
        stl_rotate = [0, 0, 0]
        stl_scale = 1e-3

        surf = surf.rotate_x(stl_rotate[0])
        surf = surf.rotate_y(stl_rotate[1])
        surf = surf.rotate_z(stl_rotate[2])
        surf = surf.scale(stl_scale)

        # Domain bounds and grid
        xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
        padx, pady, padz = (xmax - xmin) * 0.2, (ymax - ymin) * 0.2, (zmax - zmin) * 1.0

        xmin, ymin, zmin = (xmin - padx), (ymin - pady), (zmin - padz)
        xmax, ymax, zmax = (xmax + padx), (ymax + pady), (zmax + padz)

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
            stl_rotate=stl_rotate,
            stl_scale=stl_scale,
            stl_materials=stl_materials,
        )

        # Boundary conditions and
        bc_low = ["periodic", "periodic", "pec"]
        bc_high = ["periodic", "periodic", "pml"]

        # -------------- Output folder ---------------------
        if not os.path.exists(self.img_folder):
            os.mkdir(self.img_folder)

        # simulation
        global solver
        solver = SolverFIT3D(
            grid, use_stl=True, bc_low=bc_low, bc_high=bc_high, dtype=np.float32
        )

        # source
        f = 15 / ((solver.z.max() - solver.z.min()) / c)
        source = PlaneWave(
            xs=slice(1, Nx - 1), ys=slice(1, Ny - 1), zs=1, f=f, beta=1.0
        )

        Nt = int(1.0 * (solver.z.max() - solver.z.min()) / c / solver.dt)
        solver.emsolve(Nt, source)

    def test_field_inspect(self):
        global solver
        # Inspect plane
        solver.E.inspect(plane="YZ", cmap="bwr", dpi=100, figsize=[8, 6])
        # Inspect custom slice
        solver.ieps.inspect(x=slice(20, 60), y=slice(20, 60), z=int(grid.Nz / 2))
        # Inspect with handles
        _, _ = solver.H.inspect(plane="XY", cmap="bwr", handles=True)

    def test_plot1D(self):
        global solver
        solver.plot1D(
            "Ex",
            line="z",
            pos=[0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            xscale="linear",
            yscale="linear",
            off_screen=flag_offscreen,
            n=solver.Nt,
            colors=["#5ccfe6", "#fdb6d0", "#ffae57", "#bae67e", "#ffd580", "#a2aabc"],
            title=self.img_folder + "1Dplot_Ex",
        )

    def test_plot2D(self):
        global solver
        solver.plot2D(
            "Hy",
            plane="ZY",
            pos=0.5,
            cmap="bwr",
            add_patch="Sphere",
            patch_alpha=0.1,
            interpolation="spline36",
            off_screen=flag_offscreen,
            n=solver.Nt,
            title=self.img_folder + "2Dplot_Hy",
        )

    @pytest.mark.skipif(flag_offscreen, reason="Requires interactive plotting")
    def test_plot3D_interactive(self):
        global solver
        solver.plot3D(
            field="E",
            component="x",
            cmap="jet",
            add_stl="Sphere",
            stl_opacity=0.1,
            stl_colors="white",
            clip_interactive=True,
            clip_normal="-y",
            off_screen=False,
        )

    def test_plot3D_offscreen(self):
        global solver
        solver.plot3D(
            field="H",
            component="y",
            cmap="bwr",
            add_stl="Sphere",
            stl_opacity=0.1,
            stl_colors="white",
            clip_box=True,
            clip_bounds=None,
            off_screen=flag_offscreen,
            title=self.img_folder + "3Dplot_Hy",
        )

    @pytest.mark.skipif(flag_offscreen, reason="Requires interactive plotting")
    def test_plot3DonSTL_interactive(self):
        global solver
        solver.plot3DonSTL(
            "Ex",
            cmap="jet",
            stl_with_field="Sphere",
            field_opacity=1.0,
            stl_transparent="Sphere",
            stl_opacity=0.3,
            stl_colors="white",
            clip_interactive=True,
            clip_normal="-y",
            off_screen=False,
            zoom=1.0,
        )

    def test_plot3DonSTL_offscreen(self):
        global solver
        solver.plot3DonSTL(
            "Ex",
            cmap="jet",
            stl_with_field="Sphere",
            field_opacity=1.0,
            stl_transparent=None,
            stl_opacity=0.0,
            stl_colors=None,
            off_screen=flag_offscreen,
            zoom=1.0,
            title=self.img_folder + "3DplotOnSTL_Hy",
        )
