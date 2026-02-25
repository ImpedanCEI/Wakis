# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# Run with:
# mpiexec -n 2 python 004_MPI+GPU_wakefield_simulation.py
# where 2 is the number of GPU devices

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("../")

# ---------- MPI setup ------------
# can be skipped since it is handled inside GridFIT3D
from mpi4py import MPI
from tqdm import tqdm

from wakis import GridFIT3D, SolverFIT3D, WakeSolver

comm = MPI.COMM_WORLD  # Get MPI communicator
rank = comm.Get_rank()  # Process ID
size = comm.Get_size()  # Total number of MPI processes

# ------- GPU setup --------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank + 2)

# ---------- Domain setup ---------

# Geometry & Materials
solid_1 = "data/001_vacuum_cavity.stl"
solid_2 = "data/001_lossymetal_shell.stl"

stl_solids = {"cavity": solid_1, "shell": solid_2}

stl_materials = {
    "cavity": "vacuum",
    "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
}

# Extract domain bounds from geometry
solids = pv.read(solid_1) + pv.read(solid_2)
xmin, xmax, ymin, ymax, ZMIN, ZMAX = solids.bounds

# Number of mesh cells
Nx = 80
Ny = 80
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
    use_mpi=True,  # Enables MPI subdivision of the domain
    stl_solids=stl_solids,
    stl_materials=stl_materials,
    stl_scale=1.0,
    stl_rotate=[0, 0, 0],
    stl_translate=[0, 0, 0],
    verbose=1,
)

print(
    f"Process {rank}: Handling Z range {grid.zmin} to {grid.zmax} with {grid.Nz} cells"
)

# ------------ Beam source & Wake ----------------
# Beam parameters
sigmaz = 10e-2  # [m] -> 2 GHz
q = 1e-9  # [C]
beta = 1.0  # beam beta
xs = 0.0  # x source position [m]
ys = 0.0  # y source position [m]
xt = 0.0  # x test position [m]
yt = 0.0  # y test position [m]
# [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s]

from scipy.constants import c

from wakis.sources import Beam

beam = Beam(q=q, sigmaz=sigmaz, beta=beta, xsource=xs, ysource=ys, ti=3 * sigmaz / c)

# ----------- Solver & Simulation ----------
# boundary conditions
bc_low = ["pec", "pec", "pec"]
bc_high = ["pec", "pec", "pec"]

# Solver setup
solver = SolverFIT3D(
    grid,
    bc_low=bc_low,
    bc_high=bc_high,
    use_stl=True,
    use_mpi=True,  # Activate MPI
    use_gpu=True,
    bg="pec",  # Background material
)

img_folder = "004_img/"
if not os.path.exists(img_folder) and rank == 0:
    os.mkdir(img_folder)

# -------------- Custom time loop  -----------------
run_timeloop = False
if run_timeloop:
    Nt = 400
    # Plot beam current vs time
    # beam.plot(np.linspace(0, solver.dt*Nt, Nt+1))

    plot_inspect = True
    plot_2D = False
    plot_1D = False

    for n in tqdm(range(Nt)):
        beam.update(solver, n * solver.dt)

        solver.mpi_one_step()

        # Plot inspect every 20 timesteps
        if n % 20 == 0 and plot_inspect:
            E = solver.mpi_gather_asField("E")
            if rank == 0:
                fig, ax = E.inspect(
                    figsize=[20, 6], plane="YZ", show=False, handles=True
                )
                fig.savefig(img_folder + "Einspect_" + str(n).zfill(4) + ".png")
                plt.close(fig)

        # Plot E abs in 2D every 20 timesteps
        if n % 20 == 0 and plot_2D:
            solver.plot2D(
                field="E",
                component="Abs",
                plane="YZ",
                pos=0.5,
                cmap="rainbow",
                vmin=0,
                vmax=500.0,
                interpolation="hanning",
                off_screen=True,
                title=img_folder + "Ez2d",
                n=n,
            )

        # Plot E z in 1D at diferent transverse positions `pos` every 20 timesteps
        if n % 20 == 0 and plot_1D:
            solver.plot1D(
                field="E",
                component="z",
                line="z",
                pos=[0.45, 0.5, 0.55],
                xscale="linear",
                yscale="linear",
                off_screen=True,
                title=img_folder + "Ez1d",
                n=n,
            )

# -------------- using Wakefield routine  -----------------
run_wakefield = True
if run_wakefield:
    # ------------ Beam source ----------------
    # Beam parameters
    sigmaz = 10e-2  # [m] -> 2 GHz
    q = 1e-9  # [C]
    beta = 1.0  # beam beta
    xs = 0.0  # x source position [m]
    ys = 0.0  # y source position [m]
    xt = 0.0  # x test position [m]
    yt = 0.0  # y test position [m]
    # [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s]

    # ----------- Wake Solver  setup  ----------
    # Wakefield post-processor
    wakelength = 10.0  # [m] -> Partially decayed
    skip_cells = 10  # no. cells to skip at zlo/zhi for wake integration
    results_folder = "004_results/"
    wake = WakeSolver(
        q=q,
        sigmaz=sigmaz,
        beta=beta,
        xsource=xs,
        ysource=ys,
        xtest=xt,
        ytest=yt,
        skip_cells=10,
        results_folder=results_folder,
        Ez_file=results_folder + "Ez.h5",
    )

    # Reset fields
    solver.reset_fields()

    # Solver run
    plotkw2D = {
        "title": img_folder + "Ez",
        "add_patch": "cavity",
        "patch_alpha": 1.0,
        "patch_reverse": True,  # patch logical_not('cavity')
        "vmin": -1e3,
        "vmax": 1e3,  # colormap limits
        "cmap": "rainbow",
        "plane": [
            int(Nx / 2),  # x
            slice(0, Ny),  # y
            slice(skip_cells, -skip_cells),
        ],
    }  # z

    solver.wakesolve(
        wakelength=wakelength,
        wake=wake,
        plot=False,  # turn False for speedup
        plot_every=30,
        plot_until=3000,
        **plotkw2D,
    )

    if solver.rank == 0:
        # ----------- 1d plot results --------------------
        # Plot longitudinal wake potential and impedance
        fig1, ax = plt.subplots(1, 2, figsize=[12, 4], dpi=150)
        ax[0].plot(wake.s * 1e2, wake.WP, c="r", lw=1.5, label="Wakis")
        ax[0].set_xlabel("s [cm]")
        ax[0].set_ylabel("Longitudinal wake potential [V/pC]", color="r")
        ax[0].legend()
        ax[0].set_xlim(xmax=wakelength * 1e2)

        ax[1].plot(wake.f * 1e-9, np.abs(wake.Z), c="b", lw=1.5, label="Wakis")
        ax[1].set_xlabel("f [GHz]")
        ax[1].set_ylabel("Longitudinal impedance [Abs][$\Omega$]", color="b")
        ax[1].legend()

        fig1.tight_layout()
        fig1.savefig("004_results/longitudinal.png")
        # plt.show()

        # Plot transverse x wake potential and impedance
        fig2, ax = plt.subplots(1, 2, figsize=[12, 4], dpi=150)
        ax[0].plot(wake.s * 1e2, wake.WPx, c="r", lw=1.5, label="Wakis")
        ax[0].set_xlabel("s [cm]")
        ax[0].set_ylabel("Transverse wake potential X [V/pC]", color="r")
        ax[0].legend()
        ax[0].set_xlim(xmax=wakelength * 1e2)

        ax[1].plot(wake.f * 1e-9, np.abs(wake.Zx), c="b", lw=1.5, label="Wakis")
        ax[1].set_xlabel("f [GHz]")
        ax[1].set_ylabel("Transverse impedance X [Abs][$\Omega$]", color="b")
        ax[1].legend()

        fig2.tight_layout()
        fig2.savefig("004_results/transverse_x.png")
        # plt.show()

        # Plot transverse y wake potential and impedance
        fig3, ax = plt.subplots(1, 2, figsize=[12, 4], dpi=150)
        ax[0].plot(wake.s * 1e2, wake.WPy, c="r", lw=1.5, label="Wakis")
        ax[0].set_xlabel("s [cm]")
        ax[0].set_ylabel("Transverse wake potential Y [V/pC]", color="r")
        ax[0].legend()
        ax[0].set_xlim(xmax=wakelength * 1e2)

        ax[1].plot(wake.f * 1e-9, np.abs(wake.Zy), c="b", lw=1.5, label="Wakis")
        ax[1].set_xlabel("f [GHz]")
        ax[1].set_ylabel("Transverse impedance Y [Abs][$\Omega$]", color="b")
        ax[1].legend()

        fig3.tight_layout()
        fig3.savefig("004_results/transverse_y.png")
        # plt.show()

    # Plot Electric field component in 1D for several transverse positions
    solver.plot1D(
        field="E",
        component="z",
        line="z",
        pos=[0.4, 0.5, 0.6],
        xscale="linear",
        yscale="linear",
        off_screen=True,
        title=img_folder + "Ez1d",
    )
    # plt.show()

    # ----------- 2d plots results --------------------
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "name", plt.cm.jet(np.linspace(0.1, 0.9))
    )  # CST's colormap

    # Plot Electric field component in 2D using imshow
    solver.plot2D(
        field="E",
        component="z",
        plane="XY",
        pos=0.5,
        cmap=cmap,
        vmin=-500,
        vmax=500.0,
        interpolation="hanning",
        add_patch="cavity",
        patch_reverse=True,
        patch_alpha=0.8,
        off_screen=True,
        title=img_folder + "Ez2d",
    )
    # plt.show()

    # BONUS: Generate an animation from the plots generated during the simulation
    #        Needs imagemagick package -> `apt install imagemagick`
    # os.system('convert -loop 0 -delay 5 001_img/Ez_*.png 001_img/Ez_sim.gif')

    # ----------- 3d plots results --------------------
    # Plot Electric field component in 3D using pyvista.plotter
    solver.plot3D(
        "E",
        component="Abs",
        cmap="rainbow",
        clim=[0, 500],
        add_stl=["cavity", "shell"],
        stl_opacity=0.1,
        clip_interactive=True,
        clip_normal="-y",
    )

    # Plot Abs Electric field on STL solid `cavity`
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
        # clip_plane=True, clip_normal='-y', clip_origin=[0,0,0], #coming in v0.5.0
        off_screen=False,
        zoom=1.2,
        title="004_img/Ez3d",
    )


# --------------------- Extra -----------------------
# # Check global material tensors
if False:
    ieps = solver.mpi_gather_asField(solver.ieps)
    sigma = solver.mpi_gather(solver.sigma, "z", x=Nx // 2)
    if rank == 0:
        # Plot eps^-1 tensor
        fig, ax = ieps.inspect(figsize=[20, 6], plane="YZ", show=False, handles=True)
        fig.savefig(img_folder + "ieps_inspect.png")
        plt.close(fig)

        # Plot
        plot2D_field(sigma, "sigmaz_", results_folder=img_folder)


# Callback plotting functions for debugging
# Need to gather field before, e.g.: Ez = solver.mpi_gather('Ez', x=Nx//2)
# Need to be plotted in rank 0 only e.g.: if rank == 0: plot1D_field(Ez, 'Ez', n=n, results_folder=img_folder, vmin=-800, vmax=800)
if False:

    def plot1D_field(
        field,
        name="E",
        y=Ny // 2,
        n=None,
        results_folder="img/",
        vmin=None,
        vmax=None,
    ):
        if vmin is None:
            vmin = -field.max()
        if ymax is None:
            vmax = field.max()

        fig, ax = plt.subplots()
        ax.plot(field[y, :], c="g")
        ax.set_title(f"{name} at timestep={n}")
        ax.set_xlabel("z [m]")
        ax.set_ylabel("Field amplitude")
        ax.set_ylim((vmin, vmax))

        # plot vertical lines at subdomain borders
        for r in range(size):
            if r == 0:
                ax.axvline(NZ // size * (r + 1) + grid.n_ghosts, c="red", alpha=0.5)
                ax.axvline(NZ // size * (r), c="blue", alpha=0.5)
            elif r == (size - 1):
                ax.axvline(NZ // size * (r + 1), c="red", alpha=0.5)
                ax.axvline(NZ // size * (r) - grid.n_ghosts, c="blue", alpha=0.5)
            else:  # outside subdomains
                ax.axvline(NZ // size * (r + 1) + grid.n_ghosts, c="red", alpha=0.5)
                ax.axvline(NZ // size * (r) - grid.n_ghosts, c="blue", alpha=0.5)

        fig.tight_layout()
        fig.savefig(results_folder + name + "1d_" + str(n).zfill(4) + ".png")

        plt.clf()
        plt.close(fig)

    def plot2D_field(
        field,
        name="E",
        n=None,
        results_folder="img/",
        vmin=None,
        vmax=None,
    ):
        extent = (0, NZ, 0, Ny)
        if vmin is None:
            vmin = -field.max()
        if vmax is None:
            vmax = field.max()

        fig, ax = plt.subplots()
        im = ax.imshow(field, cmap="rainbow", extent=extent, vmin=vmin, vmax=vmax)
        fig.colorbar(
            im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        )
        ax.set_title(f"{name} at timestep={n}")
        ax.set_xlabel("z [id]")
        ax.set_ylabel("y [id]")

        # plot vertical lines at subdomain borders
        if True:
            for r in range(size):
                if r == 0:
                    ax.axvline(NZ // size * (r + 1) + grid.n_ghosts, c="red", alpha=0.5)
                    ax.axvline(NZ // size * (r), c="blue", alpha=0.5)
                elif r == (size - 1):
                    ax.axvline(NZ // size * (r + 1), c="red", alpha=0.5)
                    ax.axvline(NZ // size * (r) - grid.n_ghosts, c="blue", alpha=0.5)
                else:  # outside subdomains
                    ax.axvline(NZ // size * (r + 1) + grid.n_ghosts, c="red", alpha=0.5)
                    ax.axvline(NZ // size * (r) - grid.n_ghosts, c="blue", alpha=0.5)

        # plot fill rectangles for each subdomain
        if False:
            for r in range(size):
                if r == 0:
                    ax.axvspan(
                        NZ // size * (r),
                        NZ // size * (r + 1) + grid.n_ghosts,
                        color="darkorange",
                        alpha=0.1,
                    )
                elif r == (size - 1):
                    ax.axvspan(
                        NZ // size * (r) - grid.n_ghosts,
                        NZ // size * (r + 1),
                        color="darkorange",
                        alpha=0.1,
                    )
                else:  # outside subdomains
                    ax.axvspan(
                        NZ // size * (r) - grid.n_ghosts,
                        NZ // size * (r + 1) + grid.n_ghosts,
                        color="green",
                        alpha=0.1,
                    )

        fig.tight_layout(h_pad=0.3)
        fig.savefig(results_folder + name + "2d_" + str(n).zfill(4) + ".png")

        plt.clf()
        plt.close(fig)
