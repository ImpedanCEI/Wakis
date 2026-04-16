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

# ----- Funtions -----#
m = 1
n = 0
p = 1
theta = 0  # np.pi/8

# Analytic solution of cubic resonator
# Ref: http://faculty.pccu.edu.tw/~meng/new%20EM6.pdf pp.20/24


def analytic_sol_Hz(x, y, z, t):
    Rm = np.array(
        [
            [np.cos(-theta), -np.sin(-theta), 0],
            [np.sin(-theta), np.cos(-theta), 0],
            [0, 0, 1],
        ]
    )
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))

    return (
        np.cos(m * np.pi / Lx * (x_0 - Lx / 2))
        * np.cos(n * np.pi / Ly * (y_0 - Ly / 2))
        * np.sin(p * np.pi / Lz * (z_0 - Lz / 2))
        * np.cos(np.sqrt(2) * np.pi / Lx * c_light * t)
    )


def analytic_sol_Hy(x, y, z, t):
    Rm = np.array(
        [
            [np.cos(-theta), -np.sin(-theta), 0],
            [np.sin(-theta), np.cos(-theta), 0],
            [0, 0, 1],
        ]
    )
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))
    h_2 = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2 + (p * np.pi / Lz) ** 2

    return (
        -2
        / h_2
        * (n * np.pi / Ly)
        * (p * np.pi / Lz)
        * np.cos(m * np.pi / Lx * (x_0 - Lx / 2))
        * np.sin(n * np.pi / Ly * (y_0 - Ly / 2))
        * np.cos(p * np.pi / Lz * (z_0 - Lz / 2))
        * np.cos(np.sqrt(2) * np.pi / Lx * c_light * t)
    )


def analytic_sol_Hx(x, y, z, t):
    Rm = np.array(
        [
            [np.cos(-theta), -np.sin(-theta), 0],
            [np.sin(-theta), np.cos(-theta), 0],
            [0, 0, 1],
        ]
    )
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))
    h_2 = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2 + (p * np.pi / Lz) ** 2

    return (
        -2
        / h_2
        * (m * np.pi / Lx)
        * (p * np.pi / Lz)
        * np.sin(m * np.pi / Lx * (x_0 - Lx / 2))
        * np.cos(n * np.pi / Ly * (y_0 - Ly / 2))
        * np.cos(p * np.pi / Lz * (z_0 - Lz / 2))
        * np.cos(np.sqrt(2) * np.pi / Lx * c_light * t)
    )


# ---- Domain definition ----#

# Number of mesh cells
Nx = 50
Ny = 50
Nz = 50

# Embedded boundaries
stl_file = "stl/cube.stl"
surf = pv.read(stl_file)

stl_solids = {"Solid 1": stl_file}
stl_materials = {"Solid 1": "vacuum"}
stl_rotate = [0, 0, 0]
stl_translate = -np.array(surf.center)
stl_scale = 1.0

surf = surf.rotate_x(stl_rotate[0])
surf = surf.rotate_y(stl_rotate[1])
surf = surf.rotate_z(stl_rotate[2])
surf = surf.translate(stl_translate)
surf = surf.scale(stl_scale)

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
padx, pady, padz = Lx * 0.2, Ly * 0.2, Lz * 0.2

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
    stl_translate=stl_translate,
    stl_materials=stl_materials,
)

solver = SolverFIT3D(
    grid, bc_low=bc_low, bc_high=bc_high, use_stl=True, bg=[np.inf, 1.0]
)  # PEC backgroung

analytic = SolverFIT3D(grid, bc_low=bc_low, bc_high=bc_high, use_conductors=False)

dx, dy, dz = solver.dx, solver.dy, solver.dz

# Plotting functions
folder = "imgRes"
plane = "ZX"

if plane == "XY":
    xx, yy, zz = slice(0, Nx), slice(0, Ny), int(Nz // 2)  # plane XY
    title = "(x,y,Nz/2)"
    xax, yax = "ny", "nx"

if plane == "ZY":
    xx, yy, zz = int(Nx // 2), slice(0, Ny), slice(0, Nz)  # plane YZ
    title = "(Nx/2,y,z)"
    xax, yax = "nz", "ny"

if plane == "ZX":
    xx, yy, zz = slice(0, Nx), int(Ny // 2), slice(0, Nz)  # plane ZX
    title = "(x,Ny/2,z)"
    xax, yax = "nz", "nx"


def get_analytic_H(analytic, n):
    dx, dy, dz = analytic.dx, analytic.dy, analytic.dz
    xmin, ymin, zmin = analytic.grid.xmin, analytic.grid.ymin, analytic.grid.zmin
    Nx, Ny, Nz = analytic.Nx, analytic.Ny, analytic.Nz

    for ii in range(Nx):
        for jj in range(Ny):
            for kk in range(Nz):
                if mask[ii, jj, kk]:
                    x = (ii + 0.5) * dx + xmin
                    y = (jj + 0.5) * dy + ymin
                    z = (kk + 0.5) * dz + zmin
                    analytic.H[ii, jj, kk, "z"] = analytic_sol_Hz(
                        x, y, z, (n - 0.5) * solver.dt
                    )
                    analytic.H[ii, jj, kk, "y"] = analytic_sol_Hy(
                        x, y, z, (n - 0.5) * solver.dt
                    )
                    analytic.H[ii, jj, kk, "x"] = analytic_sol_Hx(
                        x, y, z, (n - 0.5) * solver.dt
                    )


def plot_E_field(solver, n):
    fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=[8, 6])
    dims = {0: "x", 1: "y", 2: "z"}
    lims = {0: 50, 1: 50, 2: 50}

    # FIT
    for i, ax in enumerate(axs[:]):
        vmin, vmax = -lims[i], lims[i]
        # vmin, vmax = -np.max(np.abs(solver.E[:,:,int(Nz/2), dims[i]])), np.max(np.abs(solver.E[:,:,int(Nz/2), dims[i]]))
        im = ax.imshow(
            solver.E[xx, yy, zz, dims[i]], cmap="rainbow", vmin=vmin, vmax=vmax
        )
        fig.colorbar(
            im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        )
        ax.set_title(f"FIT E{dims[i]}{title}")
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

    fig.suptitle(f"E field, timestep={n}")
    fig.savefig(f"{folder}E/" + str(n).zfill(4) + ".png")
    plt.clf()
    plt.close(fig)


def plot_H_field(solver, analytic, n):
    fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=[8, 6])
    dims = {0: "x", 1: "y", 2: "z"}
    lims = {0: 1.0, 1: 0.1, 2: 1}

    # FIT
    for i, ax in enumerate(axs[0, :]):
        vmin, vmax = -lims[i], lims[i]
        # vmin, vmax = -np.max(np.abs(solver.H[xx, yy, zz, dims[i]])), np.max(np.abs(solver.H[xx, yy, zz, dims[i]]))
        im = ax.imshow(
            solver.H[xx, yy, zz, dims[i]], cmap="rainbow", vmin=vmin, vmax=vmax
        )
        fig.colorbar(
            im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        )
        ax.set_title(f"FIT H{dims[i]}{title}")
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

    # Analytic
    get_analytic_H(analytic, n)
    for i, ax in enumerate(axs[1, :]):
        vmin, vmax = -lims[i], lims[i]
        im = ax.imshow(
            analytic.H[xx, yy, zz, dims[i]], cmap="rainbow", vmin=vmin, vmax=vmax
        )
        fig.colorbar(
            im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        )
        ax.set_title(f"Analytic H{dims[i]}{title}")
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

    fig.suptitle(f"H field, timestep={n}")
    fig.savefig(f"{folder}H/" + str(n).zfill(4) + ".png")
    plt.clf()
    plt.close(fig)


# ---- Initial conditions -----#
mask = np.reshape(solver.grid.grid["Solid 1"], (solver.Nx, solver.Ny, solver.Nz))
# mask_x = mask
# mask_y = mask
# mask_z = mask

# mask_x = np.roll(mask, [-1, -1], axis=[1,2] )
# mask_y = np.roll(mask, [-1, -1], axis=[0,2] )
# mask_z = np.roll(mask, [-1, -1], axis=[0,1] )

# mask_x = np.roll(mask, [-1], axis=[0] )
# mask_y = np.roll(mask, [-1], axis=[1] )
# mask_z = np.roll(mask, [-1], axis=[2] )

# No zero waves for E IC
mask_x = np.roll(mask, [-1], axis=[1]) + np.roll(mask, [+1], axis=[1])
mask_y = np.roll(mask, [-1], axis=[0]) + np.roll(mask, [+1], axis=[0])
mask_z = mask_x + mask_y

#
# mask_x = np.roll(mask, [-1], axis=[1])+np.roll(mask, [+1], axis=[1])
# mask_y = np.roll(mask, [-1], axis=[0])+np.roll(mask, [+1], axis=[0])
# mask_z = mask_x+mask_y


for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):
            x = (ii + 0.5) * dx + xmin
            y = (jj + 0.5) * dy + ymin
            z = (kk + 0.5) * dz + zmin

            if mask_x[ii, jj, kk]:
                solver.H[ii, jj, kk, "x"] = analytic_sol_Hx(x, y, z, -0.5 * solver.dt)
            if mask_y[ii, jj, kk]:
                solver.H[ii, jj, kk, "y"] = analytic_sol_Hy(x, y, z, -0.5 * solver.dt)
            if mask_z[ii, jj, kk]:
                solver.H[ii, jj, kk, "z"] = analytic_sol_Hz(x, y, z, -0.5 * solver.dt)

# ----- Time loop -----#
Nt = 100
for nt in tqdm(range(Nt)):
    solver.one_step()

    if nt % 2:
        plot_E_field(solver, nt)
        plot_H_field(solver, analytic, nt)

# convert -delay 5 -loop 0 imgRes/*.png imgRes/.gif
"""
solver.plot3D(field='H', component='z', clim=None,  hide_solids=None, show_solids=None,
               add_stl='Solid 1', stl_opacity=0.1, stl_colors='white',
               title=None, cmap='rainbow', clip_volume=True, clip_normal='-z',
               clip_box=False, clip_bounds=None, off_screen=False, zoom=0.4, n=nt)
"""
