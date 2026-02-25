import os

import numpy as np
from conductors3d import ConductorsAssembly, OutSphere
from grid3D import Grid3D

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0
from scipy.special import spherical_jn
from solver3D import EMSolver3D
from tqdm import tqdm

Z0 = np.sqrt(mu_0 / eps_0)

L = 1.0
# Number of mesh cells
N = 50
Nx = N
Ny = N
Nz = N
Lx = 0.5 * L
Ly = 0.5 * L
Lz = 0.5 * L
dx = L / Nx
dy = L / Ny
dz = L / Nz

r_sphere = 0.3
xmin = -L / 2  # + dx / 2
xmax = L / 2  # + dx / 2
ymin = -L / 2  # + dy / 2
ymax = L / 2  # + dy / 2
zmin = -L / 2  # + dz / 2
zmax = L / 2  # + dz / 2

x_cent = 0
y_cent = 0
z_cent = 0


sphere = OutSphere(r_sphere, x_cent, y_cent, z_cent)
conductors = ConductorsAssembly([sphere])

# sphere = noConductor()
# conductors = ConductorsAssembly([sphere])

# conductors = cube
sol_type = "FDTD"

grid = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, sol_type)
i_s = int(Nx / 2)
j_s = int(Ny / 2)
k_s = int(Nz / 2) + 10
NCFL = 1

flag_in_conductor = np.zeros((Nx, Ny, Nz), dtype=bool)

bc_low = ["dirichlet", "dirichlet", "dirichlet"]
bc_high = ["dirichlet", "dirichlet", "dirichlet"]
solver = EMSolver3D(grid, sol_type, NCFL, i_s, j_s, k_s, bc_low, bc_high)

# Constants
mu_r = 1
eps_r = 1

minpatch = np.ones((Nx, Ny))
small_patch = np.ones((Nx, Ny), dtype=bool)

Nborrow = np.zeros((Nx, Ny, Nz))
Nlend = np.zeros((Nx, Ny, Nz))

"""
for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):
            Nborrow[ii, jj, kk] = len(grid.borrowing_xy[ii, jj, kk])
            Nlend[ii, jj, kk] = len(grid.lending_xy[ii, jj, kk])
"""

m = 0
n = 1
p = 1

k = 2.7437 / r_sphere


def analytic_H(x, y, z, t):
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    theta = np.arctan2(np.sqrt(np.square(x) + np.square(y)), z)
    phi = np.arctan2(y, x)
    H_r = 0
    H_theta = 0
    H_phi = (
        k
        / (r_sphere * mu_r)
        * spherical_jn(1, k * r)
        * np.sin(theta)
        * np.cos(k * c_light * t)
    )

    H_x = (
        H_r * np.sin(theta) * np.cos(phi)
        + H_theta * np.cos(theta) * np.cos(phi)
        - H_phi * np.sin(phi)
    )
    H_y = (
        H_r * np.cos(theta) * np.cos(phi)
        + H_theta * np.cos(theta) * np.sin(phi)
        + H_phi * np.cos(phi)
    )
    H_z = H_r * np.cos(theta) - H_theta * np.sin(theta)

    return H_x, H_y, H_z


for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):
            if grid.flag_int_cell_xy[ii, jj, kk]:
                x = (ii + 0.5) * dx + xmin
                y = (jj + 0.5) * dy + ymin
                z = kk * dz + zmin
                solver.Hz[ii, jj, kk] = analytic_H(x, y, z, -0.5 * solver.dt)[2]

            if grid.flag_int_cell_zx[ii, jj, kk]:
                x = (ii + 0.5) * dx + xmin
                y = jj * dy + ymin
                z = (kk + 0.5) * dz + zmin
                solver.Hy[ii, jj, kk] = analytic_H(x, y, z, -0.5 * solver.dt)[1]

            if grid.flag_int_cell_yz[ii, jj, kk]:
                x = ii * dx + xmin
                y = (jj + 0.5) * dy + ymin
                z = (kk + 0.5) * dz + zmin
                solver.Hx[ii, jj, kk] = analytic_H(x, y, z, -0.5 * solver.dt)[0]

Tf = 6 * np.sqrt(2) * (r_sphere / c_light)
Nt = int(Tf / solver.dt)
# Nt = 1

res_Hy = np.zeros(Nt)
res_Ex = np.zeros(Nt)
res_Hz = np.zeros(Nt)

fields_norms = False
# Nt = 1
for t in tqdm(range(Nt)):
    if fields_norms:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
        norme = np.sqrt(
            np.square(solver.Ex[:, :, k_s])
            + np.square(solver.Ey[:, :, k_s])
            + np.square(solver.Ez[:, :, k_s])
        )
        normh = np.sqrt(
            np.square(solver.Hx[:, :, k_s])
            + np.square(solver.Hy[:, :, k_s])
            + np.square(solver.Hz[:, :, k_s])
        )

        im1 = axs[0].imshow(
            norme, cmap="jet", vmax=530, vmin=-530
        )  # ,extent=[0, L , 0, L ])
        axs[0].set_xlabel("x [m]")
        axs[0].set_ylabel("y [m]")
        axs[0].set_title("||E|| [V/m]")
        fig.colorbar(im1, ax=axs[0])
        im1 = axs[1].imshow(normh, cmap="jet", vmax=1, vmin=-1)
        axs[1].set_xlabel("x [m]")
        axs[1].set_ylabel("y [m]")
        axs[1].set_title("||H|| [V/m]")
        fig.colorbar(im1, ax=axs[1])
    else:
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)

        im1 = axs[0, 0].imshow(
            solver.Ex[:, :, k_s], cmap="jet", vmax=1500, vmin=-1500
        )  # ,extent=[0, L , 0, L ])
        axs[0, 0].set_xlabel("x [m]")
        axs[0, 0].set_ylabel("y [m]")
        axs[0, 0].set_title("Ex [V/m]")
        fig.colorbar(
            im1,
            ax=axs[0, 0],
        )
        im1 = axs[0, 1].imshow(solver.Ey[:, :, k_s], cmap="jet", vmax=1500, vmin=-1500)
        axs[0, 1].set_xlabel("x [m]")
        axs[0, 1].set_ylabel("y [m]")
        axs[0, 1].set_title("Ey [V/m]")
        fig.colorbar(im1, ax=axs[0, 1])
        im1 = axs[0, 2].imshow(solver.Ez[:, :, k_s], cmap="jet", vmax=5000, vmin=-5000)
        axs[0, 2].set_xlabel("x [m]")
        axs[0, 2].set_ylabel("y [m]")
        axs[0, 2].set_title("Ez [V/m]")
        fig.colorbar(im1, ax=axs[0, 2])
        im1 = axs[1, 0].imshow(
            solver.Hx[:, :, k_s], cmap="jet", vmax=8, vmin=-8
        )  # ,extent=[0, L , 0, L ])
        axs[1, 0].set_xlabel("x [m]")
        axs[1, 0].set_ylabel("y [m]")
        axs[1, 0].set_title("Hx [A/m]")
        fig.colorbar(
            im1,
            ax=axs[1, 0],
        )
        im1 = axs[1, 1].imshow(solver.Hy[:, :, k_s], cmap="jet", vmax=8, vmin=-8)
        axs[1, 1].set_xlabel("x [m]")
        axs[1, 1].set_ylabel("y [m]")
        axs[1, 1].set_title("Hy [A/m]")
        fig.colorbar(im1, ax=axs[1, 1])
        im1 = axs[1, 2].imshow(solver.Hz[:, :, k_s], cmap="jet", vmax=8, vmin=-8)
        axs[1, 2].set_xlabel("x [m]")
        axs[1, 2].set_ylabel("y [m]")
        axs[1, 2].set_title("Hz [A/m]")
        fig.colorbar(im1, ax=axs[1, 2])
    plt.suptitle(str(solver.time))

    folder = sol_type + "_sphere"
    if not os.path.exists(folder):
        os.mkdir(folder)

    filename = folder + "/%d.png" % t
    plt.savefig(filename)
    plt.close(fig)

    i_probe = int(Nx / 2)
    j_probe = int(Ny / 2)
    k_probe = int(Nz / 2)
    res_Hy[t] = solver.Hy[i_probe, j_probe, k_probe]
    res_Hz[t] = solver.Hz[i_probe, k_probe, k_probe]
    res_Ex[t] = solver.Ex[i_probe, j_probe, k_probe]

    solver.one_step()
    # solver.Jx[i_s, j_s, k_s] = solver.gauss(solver.time)
    # solver.Hz[i_s, j_s, k_s] += solver.gauss(solver.time)
    # solver.Jz[i_s, j_s, k_s] = solver.gauss(solver.time)
"""
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
axs[0, 0].imshow(grid.flag_int_cell_xy[:, :, k_s])
axs[0, 1].imshow(grid.Syz[:, :, k_s])
axs[0, 2].imshow(grid.Szx[:, :, k_s])
axs[1, 0].imshow(solver.Hz[:, :, k_s], cmap='jet')
axs[1, 1].imshow(solver.Hx[:, :, k_s], cmap='jet')
axs[1, 2].imshow(solver.Hy[:, :, k_s], cmap='jet')
"""
