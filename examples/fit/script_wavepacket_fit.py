import sys

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import mu_0 as mu_0
from tqdm import tqdm

sys.path.append("../")

from gridFIT3D import GridFIT3D
from solverFIT3D import SolverFIT3D

# ---------- Domain setup ---------

# Number of mesh cells
Nx = 50
Ny = 50
Nz = 150

# Domain bounds: box 10cmx10cmx30cm
xmin, xmax, ymin, ymax, zmin, zmax = -1.0e-1, 1.0e-1, -1.0e-1, 1.0e-1, -2.0e-1, 2.0e-1
Lx, Ly, Lz = (xmax - xmin), (ymax - ymin), (zmax - zmin)

# boundary conditions
bc_low = ["periodic", "periodic", "pec"]
bc_high = ["periodic", "periodic", "pec"]

# set FIT solver
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

solver = SolverFIT3D(grid, bc_low=bc_low, bc_high=bc_high)

# ------------ Source ----------------


def wave_packet(solver, t, wavelength=10, sigma_z=10, sigma_x=5, beta=1.0, f=None):
    """2d gaussian wave packet

    * Note: params `sigma_z` and `wavelength` are expressed as
    multiples of `solver.dz`.
    ** Note: Param `sigma_x` is expressed as
    multiple of `solver.dx`.
    """
    wavelength = wavelength * solver.dz
    sigma_z = sigma_z * solver.dz
    sigma_x = sigma_x * solver.dx

    if f is None:
        f = c_light / wavelength

    w = 2 * np.pi * f  # ang. frequency
    s0 = solver.z.min() - 6 * sigma_z
    s = solver.z.min() - beta * c_light * t

    X, Y = np.meshgrid(solver.x, solver.y)
    gaussxy = np.exp(-(X**2 + Y**2) / (2 * sigma_x**2))
    gausst = np.exp(-((s - s0) ** 2) / (2 * sigma_z**2))

    solver.H[:, :, 0, "y"] = -1.0 * np.cos(w * t) * gaussxy * gausst
    solver.E[:, :, 0, "x"] = 1.0 * mu_0 * c_light * np.cos(w * t) * gaussxy * gausst


# ------------ Time loop ----------------

Nt = int((zmax - zmin) / (solver.dt * c_light)) * 2
plot2D = False
plot1D = True
for n in tqdm(range(Nt)):
    # Initial condition
    if n < int((zmax - zmin) / (solver.dt * c_light)):
        wave_packet(solver, n * solver.dt)

    # Advance
    solver.one_step()

    # Plot
    if plot2D and n % 10 == 0:
        solver.plot2D(
            field="H",
            component="y",
            plane="ZY",
            pos=0.5,
            norm="symlog",
            vmin=-1,
            vmax=1,
            figsize=[8, 4],
            cmap="RdBu",
            patch_alpha=0.1,
            add_patch=None,
            title="imgGWP/Hy",
            off_screen=True,
            n=n,
            interpolation="spline36",
        )
    if plot1D and n % 15 == 0:
        solver.plot1D(
            field="H",
            component="y",
            line="z",
            pos=0.5,
            xscale="linear",
            yscale="linear",
            xlim=None,
            ylim=(-1, 1),
            figsize=[8, 4],
            title="imgGWP/Hy1d",
            off_screen=True,
            n=n,
            c="b",
        )

# Plot 3D built-in
"""
solver.plot3D(field='H', component='z', clim=None,  hide_solids=None, show_solids=None,
               add_stl=None, stl_opacity=0.1, stl_colors='white',
               title=None, cmap='rainbow', clip_volume=True, clip_normal='-y',
               clip_box=False, clip_bounds=None, off_screen=False, zoom=2.0, n=n)

solver.plot2D(field='E', component='x', plane='ZY', pos=0.5, norm=None,
               vmin=None, vmax=None, figsize=[8,4], cmap='jet', patch_alpha=0.1,
               add_patch=False, title=None, off_screen=True, n=n, interpolation='spline36')
"""
