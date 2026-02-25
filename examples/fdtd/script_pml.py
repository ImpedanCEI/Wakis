import os

import matplotlib.pyplot as plt
import numpy as np
from conductors import (
    ConductorsAssembly,
    OutRect,
    Plane,
    noConductor,
)
from grid2D import Grid2D
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


conductors = ConductorsAssembly([noConductor()])

# theta = 0
# conductors = ConductorsAssembly([rect])
sol_type = "DM"

grid = Grid2D(xmin, xmax, ymin, ymax, Nx, Ny, conductors, sol_type)
i_s = int(1 * Nx / 2.0)
j_s = int(1 * Ny / 2.0)
NCFL = 1
bc_low = ["pml", "pml", "pml"]
bc_high = ["pml", "pml", "pml"]

solver = EMSolver2D(grid, sol_type, NCFL, i_s, j_s, bc_low, bc_high)

Nt = 300
"""
solver.pml_lxly.Hz = 1*np.ones_like(solver.pml_lxly.Hz)
solver.pml_ly.Hz = 2*np.ones_like(solver.pml_ly.Hz)
solver.pml_rxly.Hz = 3*np.ones_like(solver.pml_rxly.Hz)
solver.pml_lx.Hz = 4*np.ones_like(solver.pml_lx.Hz)
solver.Hz = 5*np.ones_like(solver.Hz)
solver.pml_rx.Hz = 6*np.ones_like(solver.pml_rx.Hz)
solver.pml_lxry.Hz = 7*np.ones_like(solver.pml_lxry.Hz)
solver.pml_ry.Hz = 8*np.ones_like(solver.pml_ry.Hz)
solver.pml_rxry.Hz = 9*np.ones_like(solver.pml_rxry.Hz)
E1 = np.concatenate((solver.pml_lxly.Hz, solver.pml_ly.Hz, solver.pml_rxly.Hz), axis=0)
E2 = np.concatenate((solver.pml_lx.Hz, solver.Hz, solver.pml_rx.Hz), axis=0)
E3 = np.concatenate((solver.pml_lxry.Hz, solver.pml_ry.Hz, solver.pml_rxry.Hz), axis=0)
Hz = np.concatenate((E1, E2, E3), axis=1)

solver.sigma_star_x = np.zeros_like(solver.Ex)
solver.sigma_star_y = np.zeros_like(solver.Ey)

E1 = np.concatenate((solver.pml_lxly.sigma_star_x, solver.pml_ly.sigma_star_x, solver.pml_rxly.sigma_star_x), axis=0)
E2 = np.concatenate((solver.pml_lx.sigma_star_x, solver.sigma_star_x, solver.pml_rx.sigma_star_x), axis=0)
E3 = np.concatenate((solver.pml_lxry.sigma_star_x, solver.pml_ry.sigma_star_x, solver.pml_rxry.sigma_star_x), axis=0)
sigma_star_x = np.concatenate((E1, E2, E3), axis=1)
plt.subplot(121)
plt.imshow(sigma_star_x, cmap='jet')
plt.colorbar()
plt.show()

E1 = np.concatenate((solver.pml_lxly.sigma_star_y, solver.pml_ly.sigma_star_y, solver.pml_rxly.sigma_star_y), axis=0)
E2 = np.concatenate((solver.pml_lx.sigma_star_y, solver.sigma_star_y, solver.pml_rx.sigma_star_y), axis=0)
E3 = np.concatenate((solver.pml_lxry.sigma_star_y, solver.pml_ry.sigma_star_y, solver.pml_rxry.sigma_star_y), axis=0)
sigma_star_y = np.concatenate((E1, E2, E3), axis=1)
plt.subplot(122)
plt.imshow(sigma_star_y, cmap='jet')
plt.colorbar()
plt.show()
"""
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

    E1 = np.concatenate(
        (
            solver.pml_lxly.Ex[:, :-1],
            solver.pml_ly.Ex[:, :-1],
            solver.pml_rxly.Ex[:, :-1],
        ),
        axis=0,
    )
    E2 = np.concatenate(
        (solver.pml_lx.Ex[:, :-1], solver.Ex[:, :-1], solver.pml_rx.Ex[:, :-1]), axis=0
    )
    E3 = np.concatenate(
        (
            solver.pml_lxry.Ex[:, :-1],
            solver.pml_ry.Ex[:, :-1],
            solver.pml_rxry.Ex[:, :-1],
        ),
        axis=0,
    )
    Ex = np.concatenate((E1, E2, E3), axis=1)
    # Ex = np.concatenate((solver.pml_lx.Ex[:, :-1], solver.Ex[:, :-1]))

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
    im1 = axs[0].imshow(
        Ex.T, cmap="jet", vmax=1000, vmin=-1000, origin="lower"
    )  # ,extent=[0, L , 0, L ])
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[0].set_title("Ex [V/m]")
    fig.colorbar(
        im1,
        ax=axs[0],
    )
    E1 = np.concatenate(
        (
            solver.pml_lxly.Ey[:-1, :],
            solver.pml_ly.Ey[:-1, :],
            solver.pml_rxly.Ey[:-1, :],
        ),
        axis=0,
    )
    E2 = np.concatenate(
        (solver.pml_lx.Ey[:-1, :], solver.Ey[:-1, :], solver.pml_rx.Ey[:-1, :]), axis=0
    )
    E3 = np.concatenate(
        (
            solver.pml_lxry.Ey[:-1, :],
            solver.pml_ry.Ey[:-1, :],
            solver.pml_rxry.Ey[:-1, :],
        ),
        axis=0,
    )
    Ey = np.concatenate((E1, E2, E3), axis=1)
    # Ey = np.concatenate((solver.pml_lx.Ey[:-1,:],solver.Ey[:-1,:]))
    im1 = axs[1].imshow(Ey.T, cmap="jet", vmax=1000, vmin=-1000, origin="lower")
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].set_title("Ey [V/m]")
    fig.colorbar(im1, ax=axs[1])
    E1 = np.concatenate(
        (solver.pml_lxly.Hz, solver.pml_ly.Hz, solver.pml_rxly.Hz), axis=0
    )
    E2 = np.concatenate((solver.pml_lx.Hz, solver.Hz, solver.pml_rx.Hz), axis=0)
    E3 = np.concatenate(
        (solver.pml_lxry.Hz, solver.pml_ry.Hz, solver.pml_rxry.Hz), axis=0
    )
    Hz = np.concatenate((E1, E2, E3), axis=1)
    # Hz = np.concatenate((solver.pml_lx.Hz,solver.Hz))
    im1 = axs[2].imshow(Hz.T, cmap="jet", vmax=2, vmin=-2, origin="lower")
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
    folder = sol_type + "_images_pml"
    if not os.path.exists(folder):
        os.mkdir(folder)

    filename = folder + "/%d.png" % t
    plt.savefig(filename)
    plt.close(fig)
    solver.one_step()
    solver.Hz[i_s, j_s] += solver.gauss(solver.time)
