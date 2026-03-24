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

from conductors3d import ConductorsAssembly, InCube
from grid3D import Grid3D
from solver3D import EMSolver3D
from solverFIT3D import SolverFIT3D

Z0 = np.sqrt(mu_0 / eps_0)

L = 1.0
# Number of mesh cells
N = 50
Nx = N
Ny = N
Nz = N
Lx = L
Ly = L
Lz = L
dx = L / Nx
dy = L / Ny
dz = L / Nz

xmin = -Lx / 2
xmax = Lx / 2
ymin = -Ly / 2
ymax = Ly / 2
zmin = -Lz / 2
zmax = Lz / 2

# Embedded cube
lx = Lx * 0.7
ly = Ly * 0.7
lz = Lz * 0.7
x_cent = 0
y_cent = 0
z_cent = 0
cube = InCube(lx, ly, lz, x_cent, y_cent, z_cent)  # noConductor()
conductors = ConductorsAssembly([cube])

NCFL = 1

# set FIT solver
gridFIT = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, "FIT")
solverFIT = SolverFIT3D(gridFIT)

# set FDTD solver
gridFDTD = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, "FDTD")
solverFDTD = EMSolver3D(gridFDTD, "FDTD", NCFL)

# Initial conditions
solverFIT.E[int(Nx / 2), int(Ny / 2), int(Nz / 2), "z"] = 1.0 * c_light
solverFDTD.Ez[int(Nx / 2), int(Ny / 2), int(Nz / 2)] = 1.0 * c_light

Nt = 50
plane = "XY"

if plane == "XY":
    x, y, z = slice(0, Nx), slice(0, Ny), int(Nz // 2)  # plane XY
    title = "(x,y,Nz/2)"
    xax, yax = "y", "x"

if plane == "YZ":
    x, y, z = int(Nx // 2), slice(0, Ny), slice(0, Nz)  # plane YZ
    title = "(Nx/2,y,z)"
    xax, yax = "z", "y"


def plot_E_field(solverFIT, solverFDTD, n):
    fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=[8, 6])
    dims = {0: "x", 1: "y", 2: "z"}
    vmin, vmax = -1.0e6, 1.0e6
    # FIT
    extent = (0, N, 0, N)
    for i, ax in enumerate(axs[0, :]):
        # vmin, vmax = -np.max(np.abs(solverFIT.E[x, y, z, dims[i]])), np.max(np.abs(solverFIT.E[x, y, z, dims[i]]))
        im = ax.imshow(
            solverFIT.E[x, y, z, dims[i]],
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
    # vmin, vmax = -np.max(np.abs(solverFDTD.Ex[x, y, z])), np.max(np.abs(solverFDTD.Ex[x, y, z]))
    im = ax.imshow(
        solverFDTD.Ex[x, y, z], cmap="rainbow", vmin=vmin, vmax=vmax, extent=extent
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Ex{title}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    ax = axs[1, 1]
    extent = (0, N, 0, N)
    # vmin, vmax = -np.max(np.abs(solverFDTD.Ey[x, y, z])), np.max(np.abs(solverFDTD.Ey[x, y, z]))
    im = ax.imshow(
        solverFDTD.Ey[x, y, z], cmap="rainbow", vmin=vmin, vmax=vmax, extent=extent
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Ey{title}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    ax = axs[1, 2]
    extent = (0, N, 0, N)
    # vmin, vmax = -np.max(np.abs(solverFDTD.Ez[x, y, z])), np.max(np.abs(solverFDTD.Ez[x, y, z]))
    im = ax.imshow(
        solverFDTD.Ez[x, y, z], cmap="rainbow", vmin=vmin, vmax=vmax, extent=extent
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Ez{title}")
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    fig.suptitle(f"E field, timestep={n}")
    fig.savefig("imgE/" + str(n).zfill(4) + ".png")
    plt.clf()
    plt.close(fig)


def plot_H_field(solverFIT, solverFDTD, n):
    fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=[8, 6])
    dims = {0: "x", 1: "y", 2: "z"}
    extent = (0, N, 0, N)
    vmin, vmax = -5e3, 5e3
    # FIT
    for i, ax in enumerate(axs[0, :]):
        # vmin, vmax = -np.max(np.abs(solverFIT.H[x, y, z, dims[i]])), np.max(np.abs(solverFIT.H[x, y, z, dims[i]]))
        im = ax.imshow(
            solverFIT.H[x, y, z, dims[i]],
            cmap="rainbow",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        fig.colorbar(
            im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        )
        ax.set_title(f"FIT H{dims[i]}{title}")

    # FDTD
    ax = axs[1, 0]
    # vmin, vmax = -np.max(np.abs(solverFDTD.Hx[x, y, z])), np.max(np.abs(solverFDTD.Hx[x, y, z]))
    im = ax.imshow(
        solverFDTD.Hx[x, y, z], cmap="rainbow", vmin=vmin, vmax=vmax, extent=extent
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Hx{title}")

    ax = axs[1, 1]
    # vmin, vmax = -np.max(np.abs(solverFDTD.Hy[x, y, z])), np.max(np.abs(solverFDTD.Hy[x, y, z]))
    im = ax.imshow(
        solverFDTD.Hy[x, y, z], cmap="rainbow", vmin=vmin, vmax=vmax, extent=extent
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Hy{title}")

    ax = axs[1, 2]
    # vmin, vmax = -np.max(np.abs(solverFDTD.Hz[x, y, z])), np.max(np.abs(solverFDTD.Hz[x, y, z]))
    im = ax.imshow(
        solverFDTD.Hz[x, y, z], cmap="rainbow", vmin=vmin, vmax=vmax, extent=extent
    )
    fig.colorbar(
        im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    )
    ax.set_title(f"FDTD Hz{title}")

    fig.suptitle(f"H field, timestep={n}")
    fig.savefig("imgH/" + str(n).zfill(4) + ".png")
    plt.clf()
    plt.close(fig)


for n in tqdm(range(Nt)):
    solverFIT.one_step()
    solverFDTD.one_step()

    plot_E_field(solverFIT, solverFDTD, n)
    plot_H_field(solverFIT, solverFDTD, n)

"""
for n in tqdm(range(Nt), 'FIT:'):

    solverFIT.one_step()

for n in tqdm(range(Nt), 'FDTD:'):

    solverFDTD.one_step()
"""
