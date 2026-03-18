import os

import numpy as np
from conductors3d import ConductorsAssembly, InSphere
from grid3D import Grid3D

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0
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

N_radius = 5
r_sphere = 0.15  # N_radius*dx
xmin = -L / 2  # + dx / 2
xmax = L / 2  # + dx / 2
ymin = -L / 2  # + dy / 2
ymax = L / 2  # + dy / 2
zmin = -L / 2  # + dz / 2
zmax = L / 2  # + dz / 2

x_cent = 0
y_cent = 0
z_cent = 0


sphere = InSphere(r_sphere, x_cent, y_cent, z_cent)
conductors = ConductorsAssembly([sphere])

# cond = noConductor()
# conductors = ConductorsAssembly([cond])

# conductors = cube
sol_type = "ECT"

grid = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, sol_type)
i_s = int(Nx / 4)
j_s = int(Ny / 4)
k_s = int(Nz / 2)
NCFL = 1

flag_in_conductor = np.zeros((Nx, Ny, Nz), dtype=bool)

bc_low = ["pml", "pml", "pml"]
bc_high = ["pml", "pml", "pml"]
solver = EMSolver3D(grid, sol_type, NCFL, i_s, j_s, k_s, bc_low, bc_high)

# Constants
mu_r = 1
eps_r = 1

minpatch = np.ones((Nx, Ny))
small_patch = np.ones((Nx, Ny), dtype=bool)

Nborrow = np.zeros((Nx, Ny, Nz))
Nlend = np.zeros((Nx, Ny, Nz))

Nt = 200

res_Hy = np.zeros(Nt)
res_Ex = np.zeros(Nt)
res_Hz = np.zeros(Nt)

Px = np.sqrt(2) / 2
Py = np.sqrt(2) / 2

fields_norms = False


def get_field_x_plane(field_str, solver, i_cut):
    bc_low = solver.bc_low
    bc_high = solver.bc_high

    if bc_low[1] == "pml" and bc_low[2] == "pml":
        f00 = solver.blocks_mat[1, 0, 0].__getattribute__(field_str)[i_cut, :, :]
    if bc_low[2] == "pml":
        f10 = solver.blocks_mat[1, 1, 0].__getattribute__(field_str)[i_cut, :, :]
    if bc_high[1] == "pml" and bc_low[2] == "pml":
        f20 = solver.blocks_mat[1, 2, 0].__getattribute__(field_str)[i_cut, :, :]
    if bc_low[1] == "pml":
        f01 = solver.blocks_mat[1, 0, 1].__getattribute__(field_str)[i_cut, :, :]

    f11 = solver.__getattribute__(field_str)[i_cut, :, :]

    if bc_high[1] == "pml":
        f21 = solver.blocks_mat[1, 2, 1].__getattribute__(field_str)[i_cut, :, :]
    if bc_low[1] == "pml" and bc_high[2] == "pml":
        f02 = solver.blocks_mat[1, 0, 2].__getattribute__(field_str)[i_cut, :, :]
    if bc_high[2] == "pml":
        f12 = solver.blocks_mat[1, 1, 2].__getattribute__(field_str)[i_cut, :, :]
    if bc_high[1] == "pml" and bc_high[2] == "pml":
        f22 = solver.blocks_mat[1, 2, 2].__getattribute__(field_str)[i_cut, :, :]

    f = f11

    if bc_low[1] == "pml":
        f = np.concatenate((f01, f), axis=0)
    if bc_high[1] == "pml":
        f = np.concatenate((f, f21), axis=0)

    if bc_low[2] == "pml":
        f3 = f10
        if bc_low[1] == "pml":
            f3 = np.concatenate((f00, f3), axis=0)
        if bc_high[1] == "pml":
            f3 = np.concatenate((f3, f20), axis=0)

        f = np.concatenate((f3, f), axis=1)

    if bc_high[2] == "pml":
        f3 = f12
        if bc_low[1] == "pml":
            f3 = np.concatenate((f02, f3), axis=0)
        if bc_high[1] == "pml":
            f3 = np.concatenate((f3, f22), axis=0)

        f = np.concatenate((f, f3), axis=1)

    return f


def get_field_y_plane(field_str, solver, j_cut):
    bc_low = solver.bc_low
    bc_high = solver.bc_high

    if bc_low[0] == "pml" and bc_low[2] == "pml":
        f00 = solver.blocks_mat[0, 1, 0].__getattribute__(field_str)[:, j_cut, :]
    if bc_low[2] == "pml":
        f10 = solver.blocks_mat[1, 1, 0].__getattribute__(field_str)[:, j_cut, :]
    if bc_high[0] == "pml" and bc_low[2] == "pml":
        f20 = solver.blocks_mat[2, 1, 0].__getattribute__(field_str)[:, j_cut, :]
    if bc_low[0] == "pml":
        f01 = solver.blocks_mat[0, 1, 1].__getattribute__(field_str)[:, j_cut, :]

    f11 = solver.__getattribute__(field_str)[:, j_cut, :]

    if bc_high[0] == "pml":
        f21 = solver.blocks_mat[2, 1, 1].__getattribute__(field_str)[:, j_cut, :]
    if bc_low[0] == "pml" and bc_high[2] == "pml":
        f02 = solver.blocks_mat[0, 1, 2].__getattribute__(field_str)[:, j_cut, :]
    if bc_high[2] == "pml":
        f12 = solver.blocks_mat[1, 1, 2].__getattribute__(field_str)[:, j_cut, :]
    if bc_high[0] == "pml" and bc_high[2] == "pml":
        f22 = solver.blocks_mat[2, 1, 2].__getattribute__(field_str)[:, j_cut, :]

    f = f11

    if bc_low[0] == "pml":
        f = np.concatenate((f01, f), axis=0)
    if bc_high[0] == "pml":
        f = np.concatenate((f, f21), axis=0)

    if bc_low[2] == "pml":
        f3 = f10
        if bc_low[0] == "pml":
            f3 = np.concatenate((f00, f3), axis=0)
        if bc_high[0] == "pml":
            f3 = np.concatenate((f3, f20), axis=0)

        f = np.concatenate((f3, f), axis=1)

    if bc_high[2] == "pml":
        f3 = f12
        if bc_low[0] == "pml":
            f3 = np.concatenate((f02, f3), axis=0)
        if bc_high[0] == "pml":
            f3 = np.concatenate((f3, f22), axis=0)

        f = np.concatenate((f, f3), axis=1)

    return f


def get_field_z_plane(field_str, solver, k_cut, pml=False):
    if pml:
        bc_low = solver.bc_low
        bc_high = solver.bc_high

        if bc_low[0] == "pml" and bc_low[1] == "pml":
            f00 = solver.blocks_mat[0, 0, 1].__getattribute__(field_str)[:, :, k_cut]
        if bc_low[1] == "pml":
            f10 = solver.blocks_mat[1, 0, 1].__getattribute__(field_str)[:, :, k_cut]
        if bc_high[0] == "pml" and bc_low[1] == "pml":
            f20 = solver.blocks_mat[2, 0, 1].__getattribute__(field_str)[:, :, k_cut]
        if bc_low[0] == "pml":
            f01 = solver.blocks_mat[0, 1, 1].__getattribute__(field_str)[:, :, k_cut]

        f11 = solver.__getattribute__(field_str)[:, :, k_cut]

        if bc_high[0] == "pml":
            f21 = solver.blocks_mat[2, 1, 1].__getattribute__(field_str)[:, :, k_cut]
        if bc_low[0] == "pml" and bc_high[1] == "pml":
            f02 = solver.blocks_mat[0, 2, 1].__getattribute__(field_str)[:, :, k_cut]
        if bc_high[1] == "pml":
            f12 = solver.blocks_mat[1, 2, 1].__getattribute__(field_str)[:, :, k_cut]
        if bc_high[0] == "pml" and bc_high[1] == "pml":
            f22 = solver.blocks_mat[2, 2, 1].__getattribute__(field_str)[:, :, k_cut]

        f = f11

        if bc_low[0] == "pml":
            f = np.concatenate((f01, f), axis=0)
        if bc_high[0] == "pml":
            f = np.concatenate((f, f21), axis=0)

        if bc_low[1] == "pml":
            f3 = f10
            if bc_low[0] == "pml":
                f3 = np.concatenate((f00, f3), axis=0)
            if bc_high[0] == "pml":
                f3 = np.concatenate((f3, f20), axis=0)

            f = np.concatenate((f3, f), axis=1)

        if bc_high[1] == "pml":
            f3 = f12
            if bc_low[0] == "pml":
                f3 = np.concatenate((f02, f3), axis=0)
            if bc_high[0] == "pml":
                f3 = np.concatenate((f3, f22), axis=0)

            f = np.concatenate((f, f3), axis=1)

        return f

    else:
        return solver.__getattribute__(field_str)[:, :, k_cut]


get_field_plane = get_field_z_plane

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
        # E1 = np.concatenate((solver.blocks_mat[1, 0, 0].Ex, solver.blocks_mat[1, 1, 0].Ex, solver.blocks_mat[1, 2, 0].Ex), axis=1)
        # E2 = np.concatenate((solver.blocks_mat[1, 0, 1].Ex, solver.Ex, solver.blocks_mat[1, 2, 1].Ex), axis=1)
        # E3 = np.concatenate((solver.blocks_mat[1, 0, 2].Ex, solver.blocks_mat[1, 1, 2].Ex, solver.blocks_mat[1, 2, 2].Ex), axis=1)
        # Ex = np.concatenate((E1, E2, E3), axis=2)
        Ex = get_field_plane("Ex", solver, k_s)
        im1 = axs[0, 0].imshow(
            Ex, cmap="jet", vmax=4, vmin=-4
        )  # , vmax=120, vmin=-120)  # ,extent=[0, L , 0, L ])
        axs[0, 0].set_xlabel("x [m]")
        axs[0, 0].set_ylabel("y [m]")
        axs[0, 0].set_title("Ex [V/m]")
        fig.colorbar(im1, ax=axs[0, 0])
        # E1 = np.concatenate((solver.blocks_mat[1, 0, 0].Ey, solver.blocks_mat[1, 1, 0].Ey, solver.blocks_mat[1, 2, 0].Ey), axis=1)
        # E2 = np.concatenate((solver.blocks_mat[1, 0, 1].Ey, solver.Ey, solver.blocks_mat[1, 2, 1].Ey), axis=1)
        # E3 = np.concatenate((solver.blocks_mat[1, 0, 2].Ey, solver.blocks_mat[1, 1, 2].Ey, solver.blocks_mat[1, 2, 2].Ey), axis=1)
        # Ey = np.concatenate((E1, E2, E3), axis=2)
        Ey = get_field_plane("Ey", solver, k_s)
        im1 = axs[0, 1].imshow(
            Ey, cmap="jet", vmax=4, vmin=-4
        )  # , vmax=1e-5, vmin=-1e-5)
        axs[0, 1].set_xlabel("x [m]")
        axs[0, 1].set_ylabel("y [m]")
        axs[0, 1].set_title("Ey [V/m]")
        fig.colorbar(im1, ax=axs[0, 1])
        # E1 = np.concatenate((solver.blocks_mat[1, 0, 0].Ez, solver.blocks_mat[1, 1, 0].Ez, solver.blocks_mat[1, 2, 0].Ez), axis=1)
        # E2 = np.concatenate((solver.blocks_mat[1, 0, 1].Ez, solver.Ez, solver.blocks_mat[1, 2, 1].Ez), axis=1)
        # E3 = np.concatenate((solver.blocks_mat[1, 0, 2].Ez, solver.blocks_mat[1, 1, 2].Ez, solver.blocks_mat[1, 2, 2].Ez), axis=1)
        # Ez = np.concatenate((E1, E2, E3), axis=2)
        Ez = get_field_plane("Ez", solver, k_s)
        im1 = axs[0, 2].imshow(
            Ez, cmap="jet", vmax=4, vmin=-4
        )  # , vmax=5000, vmin=-5000)
        axs[0, 2].set_xlabel("x [m]")
        axs[0, 2].set_ylabel("y [m]")
        axs[0, 2].set_title("Ez [V/m]")
        fig.colorbar(im1, ax=axs[0, 2])
        # E1 = np.concatenate((solver.blocks_mat[1, 0, 0].Hx, solver.blocks_mat[1, 1, 0].Hx, solver.blocks_mat[1, 2, 0].Hx), axis=1)
        # E2 = np.concatenate((solver.blocks_mat[1, 0, 1].Hx, solver.Hx, solver.blocks_mat[1, 2, 1].Hx), axis=1)
        # E3 = np.concatenate((solver.blocks_mat[1, 0, 2].Hx, solver.blocks_mat[1, 1, 2].Hx, solver.blocks_mat[1, 2, 2].Hx), axis=1)
        # Hx = np.concatenate((E1, E2, E3), axis=2)
        Hx = get_field_plane("Hx", solver, k_s)
        im1 = axs[1, 0].imshow(
            Hx, cmap="jet", vmax=0.01, vmin=-0.01
        )  # ,extent=[0, L , 0, L ])
        axs[1, 0].set_xlabel("x [m]")
        axs[1, 0].set_ylabel("y [m]")
        axs[1, 0].set_title("Hx [A/m]")
        # E1 = np.concatenate((solver.blocks_mat[1, 0, 0].Hy, solver.blocks_mat[1, 1, 0].Hy, solver.blocks_mat[1, 2, 0].Hy), axis=1)
        # E2 = np.concatenate((solver.blocks_mat[1, 0, 1].Hy, solver.Hy, solver.blocks_mat[1, 2, 1].Hy), axis=1)
        # E3 = np.concatenate((solver.blocks_mat[1, 0, 2].Hy, solver.blocks_mat[1, 1, 2].Hy, solver.blocks_mat[1, 2, 2].Hy), axis=1)
        # Hy = np.concatenate((E1, E2, E3), axis=2)
        Hy = get_field_plane("Hy", solver, k_s)
        fig.colorbar(
            im1,
            ax=axs[1, 0],
        )
        im1 = axs[1, 1].imshow(
            Hy, cmap="jet", vmax=0.01, vmin=-0.01
        )  # , vmax=0.35, vmin=-0.35)
        axs[1, 1].set_xlabel("x [m]")
        axs[1, 1].set_ylabel("y [m]")
        axs[1, 1].set_title("Hy [A/m]")
        fig.colorbar(im1, ax=axs[1, 1])
        # E1 = np.concatenate((solver.blocks_mat[1, 0, 0].Hz, solver.blocks_mat[1, 1, 0].Hz, solver.blocks_mat[1, 2, 0].Hz), axis=1)
        # E2 = np.concatenate((solver.blocks_mat[1, 0, 1].Hz, solver.Hz, solver.blocks_mat[1, 2, 1].Hz), axis=1)
        # E3 = np.concatenate((solver.blocks_mat[1, 0, 2].Hz, solver.blocks_mat[1, 1, 2].Hz, solver.blocks_mat[1, 2, 2].Hz), axis=1)
        # Hz = np.concatenate((E1, E2, E3), axis=2)
        Hz = get_field_plane("Hz", solver, k_s)
        im1 = axs[1, 2].imshow(Hz, cmap="jet", vmax=0.01, vmin=-0.01)
        axs[1, 2].set_xlabel("x [m]")
        axs[1, 2].set_ylabel("y [m]")
        axs[1, 2].set_title("Hz [A/m]")
        fig.colorbar(im1, ax=axs[1, 2])
    plt.suptitle(str(solver.time))

    folder = sol_type + "_open_space"
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
    # print(solver.blocks_mat[1,1,2].Ex[0, 0, 0])
    solver.one_step()
    # solver.Jx[i_s, j_s, k_s] = solver.gauss(solver.time)
    # solver.Ex[:, :, k_s] += Px*solver.gauss(solver.time)
    # solver.Ey[:, :, k_s] += Py*solver.gauss(solver.time)
    # solver.Hx[:, :, k_s] += -Py*np.sqrt(eps_0/mu_0)*solver.gauss(solver.time + solver.dt)
    # solver.Hy[:, :, k_s] += Px*np.sqrt(eps_0/mu_0)*solver.gauss(solver.time + solver.dt)
    solver.Hz[i_s, j_s, :] += np.sqrt(eps_0 / mu_0) * solver.gauss(
        solver.time + solver.dt
    )
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
