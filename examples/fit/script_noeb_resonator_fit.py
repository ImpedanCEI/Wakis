import sys

import numpy as np

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0
from tqdm import tqdm

sys.path.append("../")

from conductors3d import noConductor
from grid3D import Grid3D
from solver3D import EMSolver3D
from solverFIT3D import SolverFIT3D

# ----- TE Funtions -----#
m = 1
n = 1
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

Z0 = np.sqrt(mu_0 / eps_0)

L = 1.0  # Domain length
N = 30  # Number of mesh cells

Nx = N
Ny = N
Nz = N
Lx = L
Ly = L
Lz = L
dx = L / Nx
dy = L / Ny
dz = L / Nz

xmin = -Lx / 2 + dx / 2
xmax = Lx / 2 + dx / 2
ymin = -Ly / 2 + dy / 2
ymax = Ly / 2 + dy / 2
zmin = -Lz / 2 + dz / 2
zmax = Lz / 2 + dz / 2

# ---- Plotting functions ----#
folder = "imgRes"
plane = "XY"

if plane == "XY":
    xx, yy, zz = slice(0, Nx), slice(0, Ny), int(Nz // 2)  # plane XY
    title = "(x,y,Nz/2)"
    xax, yax = "y", "x"

if plane == "YZ":
    xx, yy, zz = int(Nx // 2), slice(0, Ny), slice(0, Nz)  # plane YZ
    title = "(Nx/2,y,z)"
    xax, yax = "z", "y"


def plot_E_field(solverFIT, solverFDTD, n):
    fig, axs = plt.subplots(2, 3, figsize=[8, 6])
    dims = {0: "x", 1: "y", 2: "z"}
    vmin, vmax = -500.0, 500.0

    # FIT
    extent = (0, N, 0, N)
    for i, ax in enumerate(axs[0, :]):
        # vmin, vmax = -np.max(np.abs(solverFIT.E[xx, yy, zz, dims[i]])), np.max(np.abs(solverFIT.E[xx, yy, zz, dims[i]]))
        im = ax.imshow(
            solverFIT.E[xx, yy, zz, dims[i]],
            cmap="rainbow",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        fig.colorbar(
            im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        )
        ax.set_title(f"FIT E{dims[i]}{title}")
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

    # FDTD
    ax = axs[1, 0]
    extent = (0, N, 0, N)
    # vmin, vmax = -np.max(np.abs(solverFDTD.Ex[xx, yy, zz])), np.max(np.abs(solverFDTD.Ex[xx, yy, zz]))
    im = ax.imshow(
        solverFDTD.Ex[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax, extent=extent
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Ex{title}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    ax = axs[1, 1]
    extent = (0, N, 0, N)
    # vmin, vmax = -np.max(np.abs(solverFDTD.Ey[xx, yy, zz])), np.max(np.abs(solverFDTD.Ey[xx, yy, zz]))
    im = ax.imshow(
        solverFDTD.Ey[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax, extent=extent
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Ey{title}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    ax = axs[1, 2]
    extent = (0, N, 0, N)
    # vmin, vmax = -np.max(np.abs(solverFDTD.Ez[xx, yy, zz])), np.max(np.abs(solverFDTD.Ez[xx, yy, zz]))
    im = ax.imshow(
        solverFDTD.Ez[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax, extent=extent
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Ez{title}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    fig.tight_layout(h_pad=0.3)
    fig.suptitle(f"E field, timestep={n}")
    fig.savefig(f"{folder}E/" + str(n).zfill(4) + ".png")
    plt.clf()
    plt.close(fig)


def plot_H_field(solverFIT, solverFDTD, n):
    fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=[8, 6])
    dims = {0: "x", 1: "y", 2: "z"}
    vmin, vmax = -1.0, 1.0
    # FIT
    for i, ax in enumerate(axs[0, :]):
        # vmin, vmax = -np.max(np.abs(solverFIT.H[xx, yy, zz, dims[i]])), np.max(np.abs(solverFIT.H[xx, yy, zz, dims[i]]))
        im = ax.imshow(
            solverFIT.H[xx, yy, zz, dims[i]], cmap="rainbow", vmin=vmin, vmax=vmax
        )
        fig.colorbar(
            im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        )
        ax.set_title(f"FIT H{dims[i]}{title}")
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

    # FDTD
    ax = axs[1, 0]
    # vmin, vmax = -np.max(np.abs(solverFDTD.Hx[xx, yy, zz])), np.max(np.abs(solverFDTD.Hx[xx, yy, zz]))
    im = ax.imshow(solverFDTD.Hx[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Hx{title}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    ax = axs[1, 1]
    # vmin, vmax = -np.max(np.abs(solverFDTD.Hy[xx, yy, zz])), np.max(np.abs(solverFDTD.Hy[xx, yy, zz]))
    im = ax.imshow(solverFDTD.Hy[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Hy{title}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    ax = axs[1, 2]
    # vmin, vmax = -np.max(np.abs(solverFDTD.Hz[xx, yy, zz])), np.max(np.abs(solverFDTD.Hz[xx, yy, zz]))
    im = ax.imshow(solverFDTD.Hz[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Hz{title}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    fig.suptitle(f"H field, timestep={n}")
    fig.savefig(f"{folder}H/" + str(n).zfill(4) + ".png")
    plt.clf()
    plt.close(fig)


# ---- Solver definitions ---------#

conductors = noConductor()
bc_low = ["pec", "pec", "pec"]
bc_high = ["pec", "pec", "pec"]

NCFL = 1.0

gridFIT = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, "FIT")
tgridFIT = Grid3D(
    xmin + dx / 2,
    xmax + dx / 2,
    ymin + dy / 2,
    ymax + dy / 2,
    zmin + dz / 2,
    zmax + dz / 2,
    Nx,
    Ny,
    Nz,
    conductors,
    "FIT",
)
# solverFIT = SolverFIT3D(gridFIT, bc_low=bc_low, bc_high=bc_high)
solverFIT = SolverFIT3D(gridFIT, tgridFIT, bc_low=bc_low, bc_high=bc_high)

gridFDTD = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, "FDTD")
solverFDTD = EMSolver3D(gridFDTD, "FDTD", NCFL)

# ---- Initial conditions ------------#

for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):
            x = (ii + 0.5) * dx + xmin
            y = (jj + 0.5) * dy + ymin
            z = kk * dz + zmin
            solverFDTD.Hz[ii, jj, kk] = analytic_sol_Hz(x, y, z, -0.5 * solverFDTD.dt)

            x = ii * dx + xmin
            y = jj * dy + ymin
            z = kk * dz + zmin
            solverFIT.H[ii, jj, kk, "z"] = analytic_sol_Hz(x, y, z, -0.5 * solverFIT.dt)

            x = (ii + 0.5) * dx + xmin
            y = jj * dy + ymin
            z = (kk + 0.5) * dz + zmin
            solverFDTD.Hy[ii, jj, kk] = analytic_sol_Hy(x, y, z, -0.5 * solverFDTD.dt)

            x = ii * dx + xmin
            y = jj * dy + ymin
            z = kk * dz + zmin
            solverFIT.H[ii, jj, kk, "y"] = analytic_sol_Hy(x, y, z, -0.5 * solverFIT.dt)

            x = ii * dx + xmin
            y = (jj + 0.5) * dy + ymin
            z = (kk + 0.5) * dz + zmin
            solverFDTD.Hx[ii, jj, kk] = analytic_sol_Hx(x, y, z, -0.5 * solverFDTD.dt)

            x = ii * dx + xmin
            y = jj * dy + ymin
            z = kk * dz + zmin
            solverFIT.H[ii, jj, kk, "x"] = analytic_sol_Hx(x, y, z, -0.5 * solverFIT.dt)

# ----- Time loop -----#

Nt = 50
for nt in tqdm(range(Nt)):
    solverFIT.one_step()
    solverFDTD.one_step()

    plot_E_field(solverFIT, solverFDTD, nt)
    plot_H_field(solverFIT, solverFDTD, nt)

# ----- Compare -----#

analytic = EMSolver3D(gridFDTD, "FDTD", NCFL)

for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):
            x = (ii + 0.5) * dx + xmin
            y = (jj + 0.5) * dy + ymin
            z = kk * dz + zmin
            analytic.Hz[ii, jj, kk] = analytic_sol_Hz(x, y, z, (Nt - 0.5) * analytic.dt)

            x = (ii + 0.5) * dx + xmin
            y = jj * dy + ymin
            z = (kk + 0.5) * dz + zmin
            analytic.Hy[ii, jj, kk] = analytic_sol_Hy(x, y, z, (Nt - 0.5) * analytic.dt)

            x = ii * dx + xmin
            y = (jj + 0.5) * dy + ymin
            z = (kk + 0.5) * dz + zmin
            analytic.Hx[ii, jj, kk] = analytic_sol_Hx(x, y, z, (Nt - 0.5) * analytic.dt)

# Plot fields
plane = "XY"
if plane == "XY":
    xx, yy, zz = slice(0, Nx), slice(0, Ny), int(Nz // 2)  # plane XY
    title = "(x,y,Nz/2)"
    xax, yax = "y", "x"

fig, axs = plt.subplots(3, 3, tight_layout=True, figsize=[8, 6])
dims = {0: "x", 1: "y", 2: "z"}
lims = {
    0: np.max(np.abs(analytic.Hx[xx, yy, zz])),
    1: np.max(np.abs(analytic.Hy[xx, yy, zz])),
    2: np.max(np.abs(analytic.Hz[xx, yy, zz])),
}
# FIT
for i, ax in enumerate(axs[0, :]):
    # vmin, vmax = -np.max(np.abs(solverFIT.H[xx, yy, zz, dims[i]])), np.max(np.abs(solverFIT.H[xx, yy, zz, dims[i]]))
    vmin, vmax = -lims[i], lims[i]
    im = ax.imshow(
        solverFIT.H[xx, yy, zz, dims[i]], cmap="rainbow", vmin=-lims[i], vmax=lims[i]
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FIT H{dims[i]}{title}")


# FDTD
ax = axs[1, 0]
vmin, vmax = -lims[0], lims[0]
im = ax.imshow(solverFDTD.Hx[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"FDTD Hx{title}")

ax = axs[1, 1]
vmin, vmax = -lims[1], lims[1]
im = ax.imshow(solverFDTD.Hy[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"FDTD Hy{title}")

ax = axs[1, 2]
vmin, vmax = -lims[2], lims[2]
im = ax.imshow(solverFDTD.Hz[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"FDTD Hz{title}")
ax.set_xlabel(xax)
ax.set_ylabel(yax)

# Analytic
ax = axs[2, 0]
vmin, vmax = (
    -np.max(np.abs(analytic.Hx[xx, yy, zz])),
    np.max(np.abs(analytic.Hx[xx, yy, zz])),
)
im = ax.imshow(analytic.Hx[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"Analytic Hx{title}")
ax.set_xlabel(xax)
ax.set_ylabel(yax)

ax = axs[2, 1]
vmin, vmax = (
    -np.max(np.abs(analytic.Hy[xx, yy, zz])),
    np.max(np.abs(analytic.Hy[xx, yy, zz])),
)
im = ax.imshow(analytic.Hy[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"Analytic Hy{title}")
ax.set_xlabel(xax)
ax.set_ylabel(yax)

ax = axs[2, 2]
vmin, vmax = (
    -np.max(np.abs(analytic.Hz[xx, yy, zz])),
    np.max(np.abs(analytic.Hz[xx, yy, zz])),
)
im = ax.imshow(analytic.Hz[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"Analytic Hz{title}")
ax.set_xlabel(xax)
ax.set_ylabel(yax)

fig.suptitle(f"H field, timestep={Nt}")
fig.savefig(f"imgResH/analytic{plane}_TE{m}{n}{p}.png")
plt.clf()
plt.close(fig)

plane = "YZ"
if plane == "YZ":
    xx, yy, zz = int(Nx // 2), slice(0, Ny), slice(0, Nz)  # plane YZ
    title = "(Nx/2,y,z)"
    xax, yax = "z", "y"

fig, axs = plt.subplots(3, 3, tight_layout=True, figsize=[8, 6])
dims = {0: "x", 1: "y", 2: "z"}
lims = {
    0: np.max(np.abs(analytic.Hx[xx, yy, zz])),
    1: np.max(np.abs(analytic.Hy[xx, yy, zz])),
    2: np.max(np.abs(analytic.Hz[xx, yy, zz])),
}
# FIT
for i, ax in enumerate(axs[0, :]):
    # vmin, vmax = -np.max(np.abs(solverFIT.H[xx, yy, zz, dims[i]])), np.max(np.abs(solverFIT.H[xx, yy, zz, dims[i]]))
    im = ax.imshow(
        solverFIT.H[xx, yy, zz, dims[i]], cmap="rainbow", vmin=-lims[i], vmax=lims[i]
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FIT H{dims[i]}{title}")

# FDTD
ax = axs[1, 0]
vmin, vmax = -lims[0], lims[0]
im = ax.imshow(solverFDTD.Hx[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"FDTD Hx{title}")

ax = axs[1, 1]
vmin, vmax = -lims[1], lims[1]
im = ax.imshow(solverFDTD.Hy[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"FDTD Hy{title}")

ax = axs[1, 2]
vmin, vmax = -lims[2], lims[2]
im = ax.imshow(solverFDTD.Hz[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"FDTD Hz{title}")
ax.set_xlabel(xax)
ax.set_ylabel(yax)

# Analytic
ax = axs[2, 0]
vmin, vmax = -lims[0], lims[0]
im = ax.imshow(analytic.Hx[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"Analytic Hx{title}")
ax.set_xlabel(xax)
ax.set_ylabel(yax)

ax = axs[2, 1]
vmin, vmax = -lims[1], lims[1]
im = ax.imshow(analytic.Hy[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"Analytic Hy{title}")
ax.set_xlabel(xax)
ax.set_ylabel(yax)

ax = axs[2, 2]
vmin, vmax = -lims[2], lims[2]
im = ax.imshow(analytic.Hz[xx, yy, zz], cmap="rainbow", vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
ax.set_title(f"Analytic Hz{title}")
ax.set_xlabel(xax)
ax.set_ylabel(yax)

fig.suptitle(f"H field, timestep={Nt}")
fig.savefig(f"imgResH/analytic{plane}_TE{m}{n}{p}.png")
plt.clf()
plt.close(fig)
