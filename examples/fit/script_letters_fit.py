import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c as c_light
from tqdm import tqdm

sys.path.append("../")

from gridFIT3D import GridFIT3D
from solverFIT3D import SolverFIT3D

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 30
Ny = 150
Nz = 200

# Embedded boundaries
stl_file = "stl/ceiheart.stl"
surf = pv.read(stl_file)

stl_solids = {"Solid 1": stl_file}
stl_materials = {"Solid 1": "PEC"}
stl_rotate = [0, -90, 0]
stl_scale = 1e-3

surf = surf.rotate_x(stl_rotate[0])
surf = surf.rotate_y(stl_rotate[1])
surf = surf.rotate_z(stl_rotate[2])
surf = surf.scale(stl_scale)

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
padx, pady, padz = (xmax - xmin) * 0.1, (ymax - ymin) * 0.3, (zmax - zmin) * 0.1

xmin, ymin, zmin = (xmin - padx), (ymin - pady), (zmin - padz)
xmax, ymax, zmax = (xmax + padx), (ymax + pady), (zmax + padz)

# boundary conditions
bc_low = ["pec", "pec", "pec"]
bc_high = ["pec", "pec", "pec"]

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

solver = SolverFIT3D(grid, bc_low=bc_low, bc_high=bc_high, use_stl=True)

# ------------ Plot functions ---------------
plane = "YZ"

if plane == "XY":
    x, y, z = slice(0, Nx), slice(0, Ny), int(Nz // 2)  # plane XY
    title = "(x,y,Nz/2)"
    xax, yax = "y", "x"
    extent = [0, Ny, 0, Nx]

if plane == "YZ":
    x, y, z = int(Nx // 2), slice(0, Ny), slice(0, Nz)  # plane YZ
    title = "(Nx/2,y,z)"
    xax, yax = "z", "y"
    extent = [0, Nz, 0, Ny]


def plot_E_field(solver, n):
    fig, ax = plt.subplots(1, 1, figsize=[8, 4])
    vmin, vmax = 0.0, 1.0e6

    im = ax.imshow(
        solver.E.get_abs()[x, y, z],
        cmap="jet",
        extent=extent,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FIT Abs(E){title}, timestep={n}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    # Patch stl
    mask = np.reshape(grid.grid["Solid 1"], (Nx, Ny, Nz))
    patch = np.ones((Nx, Ny, Nz))
    patch[np.logical_not(mask)] = np.nan
    ax.imshow(patch[x, y, z], cmap="Greys", extent=extent, origin="lower", alpha=0.2)

    # fig.suptitle(f'Abs(E) field, timestep={n}')
    fig.tight_layout()
    fig.savefig("imgAbsE/" + str(n).zfill(4) + ".png")
    plt.clf()
    plt.close(fig)


def plot_H_field(solver, n):
    fig, ax = plt.subplots(1, 1, figsize=[8, 4])
    vmin, vmax = 0, 2.0e3

    im = ax.imshow(
        solver.H.get_abs()[x, y, z],
        cmap="jet",
        extent=extent,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FIT Abs(H){title}, timestep={n}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    # Patch stl
    """
    mask = np.reshape(grid.grid['Solid 1'], (Nx, Ny, Nz))
    patch = np.ones((Nx, Ny, Nz))
    patch[np.logical_not(mask)] = np.nan
    ax.imshow(patch[x,y,z], cmap='Greys', extent=extent, origin='lower', alpha=0.2)
    """

    # fig.suptitle(f'Abs(H) field, timestep={n}')
    fig.tight_layout()
    fig.savefig("imgAbsH/" + str(n).zfill(4) + ".png")
    plt.clf()
    plt.close(fig)


# ------------ Time loop ----------------

# solver.J[int(Nx/2), Ny-1, 1, 'z'] = 1.0*c_light

Nt = 800
for n in tqdm(range(Nt)):
    # Initial conditions
    zpos = int(Nz * n / (200))
    if zpos < Nz:
        solver.J[int(Nx / 2), Ny - 10, zpos, "z"] = 1.0 * c_light
        solver.J[int(Nx / 2), 10, Nz - 1 - zpos, "z"] = 1.0 * c_light

    # Advance
    solver.one_step()

    if zpos < Nz:
        solver.J[int(Nx / 2), Ny - 10, zpos, "z"] = 0.0
        solver.J[int(Nx / 2), 10, Nz - 1 - zpos, "z"] = 0.0

    # Plot
    if n % 10 == 0:
        # plot_E_field(solver, n)
        plot_H_field(solver, n)
