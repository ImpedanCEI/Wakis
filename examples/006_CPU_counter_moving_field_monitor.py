# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import numpy as np
import pyvista as pv
import trimesh

from wakis import GridFIT3D, SolverFIT3D, WakeSolver
from wakis.field_monitors import FieldMonitor

# ---------- Geometry Parameters (in mm) ----------
rc = 100e-3  # Cavity radius
rp = 10e-3  # Beam pipe radius
L = 60e-3  # Cavity lengt# #
l = 10e-3  # Beam pipe length
shell_thickness = 1e-3  # Shell thickness (e.g., 5 mm metal wall)
extra_shell = 1e-3

cavity = trimesh.creation.cylinder(radius=rc, height=L, sections=128)


pipe_left = trimesh.creation.cylinder(radius=rp, height=l, sections=128)
pipe_left.apply_translation([0, 0, -L / 2 - l / 2])

pipe_right = trimesh.creation.cylinder(radius=rp, height=l, sections=128)
pipe_right.apply_translation([0, 0, L / 2 + l / 2])

# Combine into one vacuum structure
solid_cavity = trimesh.util.concatenate([pipe_left, cavity, pipe_right])

# ---------- Outer Shell with Hollow Pipe Ends ----------
# Create outer shell for the cavity
rc_outer = rc + shell_thickness
rp_outer = rp + shell_thickness

shell_cavity = trimesh.creation.cylinder(
    radius=rc_outer, height=L + shell_thickness * 2, sections=128
)
shell_cavity = trimesh.boolean.difference(
    [shell_cavity, cavity], engine="manifold"
)
shell_hole = trimesh.creation.cylinder(
    radius=rp_outer, height=L + shell_thickness * 2, sections=128
)
shell_cavity = trimesh.boolean.difference(
    [shell_cavity, shell_hole], engine="manifold"
)


# Hollow beam pipes (outer shell - inner hole)
outer_pipe_left = trimesh.creation.cylinder(
    radius=rp_outer, height=l, sections=128
)
inner_pipe_left = trimesh.creation.cylinder(
    radius=rp, height=l + extra_shell, sections=128
)

hollow_pipe_left = trimesh.boolean.difference(
    [outer_pipe_left, inner_pipe_left], engine="manifold"
)
hollow_pipe_left.apply_translation([0, 0, -L / 2 - l / 2])

outer_pipe_right = trimesh.creation.cylinder(
    radius=rp_outer, height=l, sections=128
)
inner_pipe_right = trimesh.creation.cylinder(
    radius=rp, height=l + extra_shell, sections=128
)

hollow_pipe_right = trimesh.boolean.difference(
    [outer_pipe_right, inner_pipe_right], engine="manifold"
)
hollow_pipe_right.apply_translation([0, 0, L / 2 + l / 2])

# Combine into one outer shell structure
solid_shell = trimesh.util.concatenate(
    [hollow_pipe_left, shell_cavity, hollow_pipe_right]
)


# Save meshes as STL
solid_cavity.export("./data/hollow_cavity.stl")
solid_shell.export("./data/hollow_shell.stl")

# Geometry & Materials
solid_1 = "./data/hollow_cavity.stl"
solid_2 = "./data/hollow_shell.stl"
stl_solids = {"cavity": solid_1, "shell": solid_2}

stl_materials = {
    "cavity": "vacuum",
    "shell": [1e3, 1.0, 1e3],  # [eps_r, mu_r, sigma[S/m]]
}

# Extract domain bounds from geometry
solids = pv.read(solid_1) + pv.read(solid_2)
xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

# Number of mesh cells
Nx = 107
Ny = 107
Nz = 41

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
    stl_scale=1,
    stl_rotate=[0, 0, 0],
    stl_translate=[0, 0, 0],
    verbose=1,
    stl_tol=1e-4,
)

# ------------ Beam source & Wake ----------------
# Beam parameters
sigmaz = 20e-3  # [m] -> 2 GHz
q = 4.5e-8  # [C]
beta = 1.0  # beam beta
xs = 0.0  # x source position [m]
ys = 0.0  # y source position [m]
xt = 0.0  # x test position [m]
yt = 0.0  # y test position [m]

# Simualtion
wakelength = 100.0  # [m]
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
    counter_moving=True,
    results_folder="006_results/",
    Ez_file="006_results/006_Ez.h5",
)

# ----------- Solver & Simulation ----------
# boundary conditions``
bc_low = ["pec", "pec", "pec"]
bc_high = ["pec", "pec", "pec"]


frequencies = [
    1.14325797e09,
    2.60602907e09,
    3.55458372e09,
    4.06380779e09,
    4.70782647e09,
]


monitor = FieldMonitor(
    frequencies=frequencies
)  # we assume we already found the fundamental mode frequency


# Solver setup
solver = SolverFIT3D(
    grid,
    wake,
    bc_low=bc_low,
    bc_high=bc_high,
    use_stl=True,
    bg="pec",  # Background material
)
solver.ieps.inspect()

# Solver run
solver.wakesolve(
    wakelength=wakelength,
    add_space=add_space,
    plot=False,  # turn False for speedup
    plot_every=30,
    plot_until=3000,
    use_field_monitor=True,
    field_monitor=monitor,
)
solver.ieps.inspect()

freq_field = monitor.get_components()
np.savez(
    "./006_results/field_at_frequencies.npz",
    Ex=freq_field["Ex"],
    Ey=freq_field["Ey"],
    Ez=freq_field["Ez"],
)

Ez = freq_field["Ez"][
    0
]  # Z field at the first frequency (only frequency in our case)
Ez_flat = np.reshape(Ez, solver.N)

solver.grid.grid.cell_data["Ez_mag"] = np.real(Ez_flat)
# interactive slice
pl = pv.Plotter()
pl.add_mesh_clip_plane(
    solver.grid.grid, scalars="Ez_mag", cmap="inferno", show_scalar_bar=True
)
pl.show()

Ez = freq_field["Ez"][
    1
]  # Z field at the first frequency (only frequency in our case)
Ez_flat = np.reshape(Ez, solver.N)

solver.grid.grid.cell_data["Ez_mag"] = np.real(Ez_flat)
# interactive slice
pl = pv.Plotter()
pl.add_mesh_clip_plane(
    solver.grid.grid, scalars="Ez_mag", cmap="inferno", show_scalar_bar=True
)
pl.show()

Ez = freq_field["Ez"][
    2
]  # Z field at the first frequency (only frequency in our case)
Ez_flat = np.reshape(Ez, solver.N)

solver.grid.grid.cell_data["Ez_mag"] = np.real(Ez_flat)
# interactive slice
pl = pv.Plotter()
pl.add_mesh_clip_plane(
    solver.grid.grid, scalars="Ez_mag", cmap="inferno", show_scalar_bar=True
)
pl.show()

Ez = freq_field["Ez"][
    3
]  # Z field at the first frequency (only frequency in our case)
Ez_flat = np.reshape(Ez, solver.N)

solver.grid.grid.cell_data["Ez_mag"] = np.real(Ez_flat)
# interactive slice
pl = pv.Plotter()
pl.add_mesh_clip_plane(
    solver.grid.grid, scalars="Ez_mag", cmap="inferno", show_scalar_bar=True
)
pl.show()

Ez = freq_field["Ez"][
    4
]  # Z field at the first frequency (only frequency in our case)
Ez_flat = np.reshape(Ez, solver.N)

solver.grid.grid.cell_data["Ez_mag"] = np.real(Ez_flat)
# interactive slice
pl = pv.Plotter()
pl.add_mesh_clip_plane(
    solver.grid.grid, scalars="Ez_mag", cmap="inferno", show_scalar_bar=True
)
pl.show()
