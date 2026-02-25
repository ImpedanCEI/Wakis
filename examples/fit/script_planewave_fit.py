import sys

import numpy as np
import pyvista as pv
from scipy.constants import c as c_light
from scipy.constants import mu_0 as mu_0
from tqdm import tqdm

sys.path.append("../")

from gridFIT3D import GridFIT3D
from solverFIT3D import SolverFIT3D

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 80
Ny = 80
Nz = 160

# Embedded boundaries
stl_file = "stl/sphere.stl"
surf = pv.read(stl_file)

stl_solids = {"Solid 1": stl_file}
stl_materials = {"Solid 1": "pec"}
stl_rotate = [0, 0, 0]
stl_scale = 1e-3

surf = surf.rotate_x(stl_rotate[0])
surf = surf.rotate_y(stl_rotate[1])
surf = surf.rotate_z(stl_rotate[2])
surf = surf.scale(stl_scale)
# surf.plot()

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
padx, pady, padz = (xmax - xmin) * 0.2, (ymax - ymin) * 0.2, (zmax - zmin) * 1.0

xmin, ymin, zmin = (xmin - padx), (ymin - pady), (zmin - padz)
xmax, ymax, zmax = (xmax + padx), (ymax + pady), (zmax + padz)

Lx, Ly, Lz = (xmax - xmin), (ymax - ymin), (zmax - zmin)

# boundary conditions
bc_low = ["periodic", "periodic", "pec"]
bc_high = ["periodic", "periodic", "pec"]

# set FIT solver
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

solver = SolverFIT3D(grid, bc_low=bc_low, bc_high=bc_high, use_stl=False)

# ------------ Time loop ----------------


# Initial conditions
def plane_wave(solver, t, nodes=15, f=None, beta=1.0):
    if f is None:
        T = (solver.z.max() - solver.z.min()) / c_light
        f = nodes / T

    vp = beta * c_light  # wavefront velocity beta*c
    w = 2 * np.pi * f  # ang. frequency
    kz = w / c_light  # wave number

    solver.H[:, :, 0, "y"] = -1.0 * np.cos(w * t)
    solver.E[:, :, 0, "x"] = 1.0 * np.cos(w * t) / (kz / (mu_0 * vp))


# Nt = 300
Nt = int((zmax - zmin) / (solver.dt * c_light)) + 300
plot2D = True
plot1D = False
for n in tqdm(range(Nt)):
    # Initial condition
    if n < int((zmax - zmin) / (solver.dt * c_light)):
        plane_wave(solver, n * solver.dt, nodes=15)

    # Advance
    solver.one_step()

    # Plot
    if plot2D and n % 10 == 0:
        solver.plot2D(
            field="H",
            component="y",
            plane="ZX",
            pos=0.5,
            norm=None,
            vmin=-1,
            vmax=1,
            figsize=[8, 4],
            cmap="rainbow",
            patch_alpha=0.1,
            add_patch=None,
            title="imgPwH/Hy",
            off_screen=True,
            n=n,
            interpolation="spline36",
        )
    if plot1D and n % 15 == 0:
        solver.plot1D(
            field="E",
            component="x",
            line="z",
            pos=0.5,
            xscale="linear",
            yscale="linear",
            xlim=None,
            ylim=(-2.0 * 500, 2.0 * 500),
            figsize=[8, 4],
            title="imgPwH/Ex1d",
            off_screen=True,
            n=n,
        )

# Plot 3D built-in
"""
solver.plot3D(field='H', component='z', clim=None,  hide_solids=None, show_solids=None,
               add_stl=None, stl_opacity=0.1, stl_colors='white',
               title=None, cmap='rainbow', clip_volume=False, clip_normal='-y',
               clip_box=True, clip_bounds=None, off_screen=False, zoom=2.0, n=n)

solver.plot2D(field='E', component='x', plane='ZY', pos=0.5, norm=None,
               vmin=None, vmax=None, figsize=[8,4], cmap='jet', patch_alpha=0.1,
               add_patch=False, title=None, off_screen=True, n=n, interpolation='spline36')
"""
