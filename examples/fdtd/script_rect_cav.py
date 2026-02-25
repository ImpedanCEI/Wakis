import os

import matplotlib.pyplot as plt
import numpy as np
from conductors import ConductorsAssembly, OutRect, Plane
from grid2D import Grid2D
from scipy.constants import c as c_light
from scipy.constants import mu_0 as mu_0
from solver2D import EMSolver2D
from tqdm import tqdm

L = 1.0
# Number of mesh cells
N = 50
Nx = N
Ny = N
dx = L / Nx
dy = L / Ny

r_circ = 0.4
xmin = -L / 2
xmax = L / 2
ymin = -L / 2
ymax = L / 2
Lx = L * 0.75
Ly = L * 0.75
x_cent = 0
y_cent = 0

rect = OutRect(Lx, Ly, x_cent, y_cent)

theta = np.pi / 2 * 0.25
m_plane1 = np.tan(theta)
q_plane1 = Ly / 2
m_plane2 = np.tan(theta)
q_plane2 = -Ly / 2
m_plane3 = -1 / np.tan(theta)
q_plane3 = (Ly / 2) / np.tan(theta)
m_plane4 = -1 / np.tan(theta)
q_plane4 = -(Ly / 2) / np.tan(theta)
# -dx/3+2*dx
plane1 = Plane(m_plane1, q_plane1, tol=0, sign=1)
plane2 = Plane(m_plane2, q_plane2, tol=0, sign=-1)
plane3 = Plane(m_plane3, q_plane3, tol=0, sign=1)
plane4 = Plane(m_plane4, q_plane4, tol=0, sign=-1)

conductors = ConductorsAssembly([plane1, plane2, plane3, plane4])

# theta = 0
# conductors = ConductorsAssembly([rect])
sol_type = "ECT"

grid = Grid2D(xmin, xmax, ymin, ymax, Nx, Ny, conductors, sol_type)
i_s = int(1 * Nx / 2.0)
j_s = int(1 * Ny / 2.0)
NCFL = 1
bc_low = ["dirichlet", "dirichlet", "dirichlet"]
bc_high = ["dirichlet", "dirichlet", "dirichlet"]

solver = EMSolver2D(grid, sol_type, NCFL, i_s, j_s, bc_low, bc_high)

# Constants
mu_r = 1
eps_r = 1

Nborrow = np.zeros((Nx, Ny))
Nlend = np.zeros((Nx, Ny))
minpatch = np.ones((Nx, Ny))
small_patch = np.ones((Nx, Ny), dtype=bool)
"""
for ii in range(Nx):
    for jj in range(Ny):
        Nborrow[ii, jj] = len(grid.borrowing[ii, jj])
        Nlend[ii, jj] = len(grid.lending[ii, jj])

        for (_, _, patch, _) in grid.borrowing[ii, jj]:
            if patch < minpatch[ii, jj]:
                minpatch[ii, jj] = patch
"""
for ii in range(Nx):
    for jj in range(Ny):
        small_patch[ii, jj] = minpatch[ii, jj] < grid.S_stab[ii, jj]


def analytic_sol(x, y, t):
    Rm = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    [x_0, y_0] = np.dot(Rm, np.array([x, y]))

    return (
        np.cos(np.pi / Lx * (x_0 - Lx / 2))
        * np.cos(np.pi / Ly * (y_0 - Lx / 2))
        * np.cos(np.sqrt(2) * np.pi / Lx * c_light * t)
    )


for ii in range(Nx):
    for jj in range(Ny):
        x = (ii) * dx + xmin
        y = (jj) * dy + ymin

        if grid.flag_int_cell[ii, jj]:
            solver.Hz[ii, jj] = analytic_sol(x, y, 0)  # -k*c_light*solver.dt/2)


max_sol = solver.Hz

t_f = 1.1 * np.sqrt(2) * Lx / c_light
Nt = int(t_f / solver.dt)
# Nt = 100

sol = np.zeros_like(solver.Hz)

for t in tqdm(range(Nt)):
    """
    for ii in range(-6, 7):
        for jj in range(-6, 7):
            iii = solver.i_s + ii
            jjj = solver.j_s + jj
            y = jjj*dy + ymin
            x = iii*dx + xmin
            amp = solver.gauss(
                solver.time) * ramped_sine(solver.time, 5 * per) * gauss_space(x) * gauss_space(y)
            if abs(x) > 1e-14:
                theta = np.arctan(np.divide(y, x))
            else:
                theta = np.pi / 2

            solver.Jx[iii, jjj] = amp
            #solver.Jy[iii, jjj] = amp
    """
    # for ii in range(Nx + 1):
    #    for jj in range(Ny + 1):
    #       x = (ii + 0.5) * dx + xmin
    #        y = (jj + 0.5) * dy + ymin
    #        solver.Hz[ii, jj] = analytic_sol(x, y, t*solver.dt)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
    im1 = axs[0].imshow(
        solver.Ex, cmap="jet", vmax=200, vmin=-200
    )  # ,extent=[0, L , 0, L ])
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[0].set_title("Ex [V/m]")
    fig.colorbar(
        im1,
        ax=axs[0],
    )
    im1 = axs[1].imshow(solver.Ey, cmap="jet", vmax=200, vmin=-200)
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].set_title("Ey [V/m]")
    fig.colorbar(im1, ax=axs[1])
    im1 = axs[2].imshow(solver.Hz, cmap="jet", vmax=1, vmin=-1)
    axs[2].set_xlabel("x [m]")
    axs[2].set_ylabel("y [m]")
    axs[2].set_title("Hz [A/m]")
    fig.colorbar(im1, ax=axs[2])
    plt.suptitle(str(solver.time))
    # im1 = axs[1,0].imshow(Hx.T,cmap='jet',origin='lower')#,extent=[0, L , 0, L ])
    # axs[1,0].set_xlabel('x [m]')
    # axs[1,0].set_ylabel('y [m]')
    # axs[1,0].set_title('Hx [A/m]')
    # fig.colorbar(im1, ax=axs[1,0],)
    # im1 = axs[1,1].imshow(Hy.T,cmap='jet',origin='lower')
    # axs[1,1].set_xlabel('x [m]')
    # axs[1,1].set_ylabel('y [m]')
    # axs[1,1].set_title('Hy [A/m]')
    # fig.colorbar(im1, ax=axs[1,1])
    # im1 = axs[1,2].imshow(Ez.T/Z_0,cmap='jet',origin='lower')
    # axs[1,2].set_xlabel('x [m]')
    # axs[1,2].set_ylabel('y [m]')
    # axs[1,2].set_title('Ez [V/m]')
    # fig.colorbar(im1, ax=axs[1,2])
    folder = sol_type + "_images_rect"
    if not os.path.exists(folder):
        os.mkdir(folder)

    filename = folder + "/%d.png" % t
    plt.savefig(filename)
    plt.close(fig)
    solver.time += solver.dt
    solver.one_step()
    # solver.Hz[i_s, j_s] += solver.gauss(solver.time)
# Compute the analytic solution
sol = np.zeros_like(solver.Hz)

for ii in range(Nx):
    for jj in range(Ny):
        x = (ii + 0.5) * dx + xmin
        y = (jj + 0.5) * dy + ymin
        sol[ii, jj] = analytic_sol(x, y, solver.time)

err = np.linalg.norm(solver.Hz - sol) * dx * dy / np.linalg.norm(max_sol)
print(err)
