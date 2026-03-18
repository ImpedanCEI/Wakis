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
stl_file1 = "stl/rect.stl"
stl_file2 = "stl/wakis.stl"
stl_solids = {"Solid 1": stl_file1, "Solid 2": stl_file2}
stl_rotate = [0, -90, 0]
stl_translate = {"Solid 1": [0, 0, 0], "Solid 2": [-5, 0, 0]}
stl_scale = 1e-3

# Materials: [eps_r, mu_r]
stl_materials = {"Solid 1": [5.0, 1.0], "Solid 2": "PEC"}

# Domain bounds (from stl_file1)
surf1 = pv.read(stl_file1)
surf1 = surf1.rotate_x(stl_rotate[0]).rotate_y(stl_rotate[1]).rotate_z(stl_rotate[2])
surf1 = surf1.translate(stl_translate["Solid 1"])
surf1 = surf1.scale(stl_scale)

surf2 = pv.read(stl_file2)
surf2 = surf2.rotate_x(stl_rotate[0]).rotate_y(stl_rotate[1]).rotate_z(stl_rotate[2])
surf2 = surf2.translate(stl_translate["Solid 2"])
surf2 = surf2.scale(stl_scale)

xmin, xmax, ymin, ymax, zmin, zmax = surf1.bounds
padx, pady, padz = (xmax - xmin) * 0.1, (ymax - ymin) * 0.2, (zmax - zmin) * 0.1

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

# Plot geometry
plot_geometry = False
if plot_geometry:
    pl = pv.Plotter()
    pl.add_mesh(grid.grid, show_edges=True, style="wireframe", color="w", opacity=0.15)
    pl.add_mesh(surf1, show_edges=False, color="b", opacity=0.25)
    pl.add_mesh(surf2, show_edges=False, color="r", opacity=0.55)
    pl.show()

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


def plot_E_field(solver, n, patch_stl=False):
    fig, ax = plt.subplots(1, 1, figsize=[8, 4], dpi=150)
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
    if patch_stl:
        mask = np.reshape(grid.grid["Solid 2"], (Nx, Ny, Nz))
        patch = np.ones((Nx, Ny, Nz))
        patch[np.logical_not(mask)] = np.nan
        ax.imshow(
            patch[x, y, z], cmap="Greys", extent=extent, origin="lower", alpha=0.2
        )

    # fig.suptitle(f'Abs(E) field, timestep={n}')
    fig.tight_layout()
    fig.savefig("imgAbsE/" + str(n).zfill(4) + ".png")
    plt.clf()
    plt.close(fig)


def plot_H_field(solver, n, patch_stl=False):
    fig, ax = plt.subplots(1, 1, figsize=[8, 4])
    vmin, vmax = 0, 1.0e3

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
    if patch_stl:
        mask = np.reshape(grid.grid["Solid 2"], (Nx, Ny, Nz))
        patch = np.ones((Nx, Ny, Nz))
        patch[np.logical_not(mask)] = np.nan
        ax.imshow(
            patch[x, y, z], cmap="Greys", extent=extent, origin="lower", alpha=0.2
        )

    # fig.suptitle(f'Abs(H) field, timestep={n}')
    fig.tight_layout()
    fig.savefig("imgAbsH/" + str(n).zfill(4) + ".png")
    plt.clf()
    plt.close(fig)


# ------------ Time loop ----------------

solver.J[int(Nx / 2), int(Ny / 2), 0, "z"] = 1.0 * c_light
solver.J[int(Nx / 2), int(Ny / 2), Nz - 2, "z"] = 1.0 * c_light
solver.J[int(Nx / 2), 1, int(Nz / 2), "z"] = 1.0 * c_light
solver.J[int(Nx / 2), Ny - 2, int(Nz / 2), "z"] = 1.0 * c_light

Nt = 1000
for n in tqdm(range(Nt)):
    # Advance
    solver.one_step()

    # Plot
    if n % 10 == 0:
        plot_E_field(solver, n)
        plot_H_field(solver, n)
