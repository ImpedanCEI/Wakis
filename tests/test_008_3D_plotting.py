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

# Turn False when running local
flag_offscreen = True


@pytest.mark.slow
class Test3Dplotting:
    img_folder = "tests/008_img/"

    def test_simulation(self):
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
        )

        # -------------- Output folder ---------------------
        if not os.path.exists(self.img_folder):
            os.mkdir(self.img_folder)

        # -------------- Custom time loop  -----------------
        Nt = 1000
        for n in tqdm(range(Nt)):
            beam.update(solver, n * solver.dt)
            solver.one_step()

    def test_grid_inspect(self):
        # Plot grid and imported solids
        global solver
        pl = solver.grid.inspect(
            add_stl=["cavity", "shell"],
            stl_opacity=0.1,
            off_screen=flag_offscreen,
            anti_aliasing="ssaa",
        )
        if flag_offscreen:
            # pl.screenshot(self.img_folder+'grid_inspect.png')
            pl.export_html(self.img_folder + "grid_inspect.html")

    def test_grid_plot_solids(self):
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

    def test_grid_stl_mask(self):
        # Plot STL solid masks in the grid
        global solver
        solver.grid.plot_stl_mask(
            stl_solid="cavity",
            cmap="viridis",
            add_stl="all",
            stl_opacity=0.5,
            smooth_shading=False,
            anti_aliasing="ssaa",
            ymax=0.0,
            off_screen=flag_offscreen,
        )

    def test_solver_inspect(self):
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
            # pl.screenshot(self.img_folder+'solver_inspect.png')
            pl.export_html(self.img_folder + "solver_inspect.html")

    def test_plot3D(self):
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

    def test_plot3DonSTL(self):
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
