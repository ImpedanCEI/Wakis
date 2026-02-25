# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import os

import numpy as np
import pyvista as pv

from wakis import GridFIT3D, SolverFIT3D, WakeSolver
from wakis.field_monitors import FieldMonitor

# ---------- Domain setup ---------

# Geometry & Materials
solid_1 = "../examples/data/001_vacuum_cavity.stl"
solid_2 = "../examples/data/001_lossymetal_shell.stl"

stl_solids = {"cavity": solid_1, "shell": solid_2}

stl_materials = {
    "cavity": "vacuum",
    "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
}


# Extract domain bounds from geometry
solids = pv.read(solid_1) + pv.read(solid_2)
xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

# Number of mesh cells
Nx = 40
Ny = 40
Nz = 141

# set grid and geometry
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
ys = 0.0  # y source position [m]
xt = 0.0  # x test position [m]
yt = 0.0  # y test position [m]
# [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s]

# Simualtion
wakelength = 10.0  # [m]
add_space = 10  # no. cells to skip from boundaries - removes BC artifacts

wake = WakeSolver(
    q=q,
    sigmaz=sigmaz,
    beta=beta,
    xsource=xs,
    ysource=ys,
    xtest=xt,
    ytest=yt,
    add_space=add_space,
    results_folder="005_results/",
    Ez_file="005_results/005_Ez.h5",
)

# ----------- Solver & Simulation ----------
# boundary conditions``
bc_low = ["pec", "pec", "pec"]
bc_high = ["pec", "pec", "pec"]


# Solver setup
solver = SolverFIT3D(
    grid,
    wake,
    bc_low=bc_low,
    bc_high=bc_high,
    use_stl=True,
    bg="pec",  # Background material
)

# CREATE field monitor
monitor = FieldMonitor(
    frequencies=[0.549e9]
)  # we assume we already found the fundamental mode frequency

# Solver run
solver.wakesolve(
    wakelength=wakelength,
    add_space=add_space,
    plot=True,  # turn False for speedup
    plot_every=30,
    plot_until=3000,
    use_field_monitor=True,  # argument for if to use field monitor
    field_monitor=monitor,  # add field monitor
)

results_folder = "005_results/"
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

freq_field = monitor.get_components()
np.savez(
    "./005_results/field_at_frequencies.npz",
    Ex=freq_field["Ex"],
    Ey=freq_field["Ey"],
    Ez=freq_field["Ez"],
)

Ez = freq_field["Ez"][0]  # Z field at the first frequency (only frequency in our case)
Ez_flat = np.reshape(Ez, solver.N)  #


solver.grid.grid.cell_data["Ez_mag"] = np.real(Ez_flat)
# interactive slice
pl = pv.Plotter()
pl.add_mesh_clip_plane(
    solver.grid.grid, scalars="Ez_mag", cmap="inferno", show_scalar_bar=True
)
pl.show()
