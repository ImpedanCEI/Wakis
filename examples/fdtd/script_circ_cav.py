import os

import matplotlib.pyplot as plt
import numpy as np
from conductors import ConductorsAssembly, OutCircle, OutRect, Plane
from grid2D import Grid2D
from scipy.constants import c as c_light
from scipy.constants import mu_0 as mu_0
from scipy.special import jv
from solver2D import EMSolver2D
from tqdm import tqdm

L = 1.0
# Number of mesh cells
N = 50
Nx = N
Ny = N
dx = L / Nx
dy = L / Ny

dt = 0.01

flag_i_inside = np.zeros(Nx + 2, "bool")
flag_j_inside = np.zeros(Ny + 2, "bool")

flag_in_conductors = np.zeros((Nx, Ny), dtype="bool")

x_circ = 0.0
y_circ = 0.0
r_circ = 0.4
xmin = -L / 2
xmax = L / 2
ymin = -L / 2
ymax = L / 2
Lx = L - 4 * dx
Ly = L - 4 * dy
x_cent = 0.0
y_cent = 0.0

rect = OutRect(Lx, Ly, x_cent, y_cent)
circ = OutCircle(r_circ, x_circ, y_circ)
m_plane = -0.1
q_plane = dx / 5  # -dx/3+2*dx
plane = Plane(m_plane, q_plane, tol=0)
conductors = ConductorsAssembly([circ])
sol_type = "ECT"

grid = Grid2D(xmin, xmax, ymin, ymax, Nx, Ny, conductors, sol_type)
i_s = int(1 * Nx / 2.0)
j_s = int(1 * Ny / 2.0)
NCFL = 0.99

bc_low = ["dirichlet", "dirichlet", "dirichlet"]
bc_high = ["dirichlet", "dirichlet", "dirichlet"]
solver = EMSolver2D(grid, sol_type, NCFL, i_s, j_s, bc_low, bc_low)

Nt = 5000
Tf = Nt * dt

# Constants
mu_r = 1
eps_r = 1

Nborrow = np.zeros((Nx, Ny))
Nlend = np.zeros((Nx, Ny))

# for ii in range(Nx):
#    for jj in range(Ny):
#        Nborrow[ii, jj] = len(grid.borrowing[ii, jj])
#        Nlend[ii, jj] = len(grid.lending[ii, jj])

k = 8.5363 / r_circ
for ii in range(Nx):
    for jj in range(Ny):
        x = (ii + 0.5) * dx + xmin
        y = (jj + 0.5) * dy + ymin
        if abs(x) > 1e-14:
            thetacane = np.arctan(np.divide(y, x))
            if x > 0 and y > 0:
                theta = thetacane
            elif x < 0 and y > 0:
                theta = np.pi - thetacane
            elif x < 0 and y < 0:
                theta = np.pi + thetacane
            else:
                theta = -thetacane
        else:
            theta = np.pi / 2
        r = np.sqrt(np.square(x) + np.square(y))
        if grid.flag_int_cell[ii, jj]:
            solver.Hz[ii, jj] = (
                jv(1, r * k) * np.cos(theta) * np.cos(0)
            )  # -k*c_light*solver.dt/2)

max_sol = solver.Hz

t_f = 1.1 * 2 * np.pi / (k * c_light)
Nt = int(t_f / solver.dt)
# Nt = 0


def gauss_space(x):
    sigma = dx / 10
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-np.square(x) / (2 * sigma))


kappa = 8.5363 / r_circ
w_t = c_light * kappa
freq_t = w_t / (2 * np.pi)
per = 1 / freq_t

ramp = lambda tt: 0.5 * (1 - np.cos(np.pi * tt))


def modulation(tt, r_time):
    if tt < 2 * r_time:
        return ramp(tt / r_time)
    else:
        return 0


def ramped_sine(tt, r_time):
    return modulation(tt, r_time) * np.sin(w_t * tt)


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

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
    im1 = axs[0].imshow(
        solver.Ex, cmap="jet", vmax=80, vmin=-80
    )  # ,extent=[0, L , 0, L ])
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[0].set_title("Ex [V/m]")
    fig.colorbar(
        im1,
        ax=axs[0],
    )
    im1 = axs[1].imshow(solver.Ey, cmap="jet", vmax=150, vmin=-150)
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].set_title("Ey [V/m]")
    fig.colorbar(im1, ax=axs[1])
    im1 = axs[2].imshow(solver.Hz, cmap="jet", vmax=0.5, vmin=-0.5)
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
    folder = "CFDTD_images"
    if not os.path.exists(folder):
        os.mkdir(folder)

    filename = folder + "/%d.png" % t
    plt.savefig(filename)
    plt.close(fig)

    solver.one_step()

# Compute the analytic solution
sol = np.zeros_like(solver.Hz)

for ii in range(Nx + 1):
    for jj in range(Ny + 1):
        x = (ii + 0.5) * dx + xmin
        y = (jj + 0.5) * dy + ymin
        if abs(x) > 1e-14:
            thetacane = np.arctan(np.divide(y, x))
            if x > 0 and y > 0:
                theta = thetacane
            elif x < 0 and y > 0:
                theta = np.pi - thetacane
            elif x < 0 and y < 0:
                theta = np.pi + thetacane
            else:
                theta = -thetacane
        else:
            theta = np.pi / 2
        r = np.sqrt(np.square(x) + np.square(y))
        sol[ii, jj] = jv(1, r * k) * np.cos(theta) * np.cos(-k * c_light * solver.time)

err = np.linalg.norm(solver.Hz - sol) * dx * dy / np.linalg.norm(max_sol)
print(err)
