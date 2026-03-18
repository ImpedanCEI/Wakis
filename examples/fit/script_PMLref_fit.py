import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import c as c_light
from scipy.constants import mu_0 as mu_0
from tqdm import tqdm

sys.path.append("../")

from gridFIT3D import GridFIT3D
from solverFIT3D import SolverFIT3D
from sources import Pulse

# ---------- Domain setup ---------

# Number of mesh cells
Nx = 200
Ny = 200
Nz = 20

# Domain bounds: box 10cmx10cmx30cm
xmin, xmax, ymin, ymax, zmin, zmax = -Nx / 2, Nx / 2, -Ny / 2, Ny / 2, -Nz / 2, Nz / 2

# boundary conditions
bc_low = ["pec", "pec", "pec"]
bc_high = ["pec", "pec", "pec"]

# set grid
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

# set solver
solver = SolverFIT3D(grid, dt=0.5 * grid.dx / c_light, bc_low=bc_low, bc_high=bc_high)

# set source
source = Pulse(
    field="Ez",
    xs=int(Nx / 2),
    ys=int(Ny / 2),
    zs=slice(0, Nz),  # zs=int(Nz/2),
    shape="harris",
    L=50 * solver.dx,
)

# ------------ Time loop ----------------

Nt = 200
plot = False
for n in tqdm(range(Nt)):
    # Advance
    solver.one_step()

    # Update source
    source.update(solver, n * solver.dt)

    # Clean z-dir
    # for k in range(1,Nz-1):
    #    solver.E[:,:,k,'z'] = solver.E[:,:,int(Nz/2),'z']

    # Plot
    if plot and n % 2 == 0:
        solver.plot2D(
            field="E",
            component="Abs",
            plane="XY",
            pos=0.5,
            norm=None,
            vmin=0,
            vmax=None,
            figsize=[8, 4],
            cmap="jet",
            title="imgEpml/Ez",
            off_screen=True,
            n=n,
            interpolation="spline36",
        )


# Save
hf = h5py.File("Ez.h5", "w")
hf["x"], hf["y"], hf["z"] = solver.x, solver.y, solver.z
hf["t"] = np.arange(0, Nt * solver.dt, solver.dt)
hf["Ez"] = solver.E[:, :, int(Nz / 2), "z"]
hf.close()

# Plot 2D
plot = True
if plot:
    Y_r, X_r = np.meshgrid(solver.y, solver.x)
    field = np.abs(solver.E[:, :, int(Nz / 2), "z"])

    fig = plt.figure(1, tight_layout=True, dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    # ax.plot_surface(X_R, Y_R, abs(R), cmap='jet', rstride=1,  cstride=1, linewidth=0, alpha=1, antialiased=False)
    ax.plot_surface(
        X_r,
        Y_r,
        field,
        cmap="jet",
        rstride=1,
        cstride=1,
        linewidth=0,
        alpha=1,
        antialiased=False,
    )

    ax.set_xlabel("x [a.u.]", labelpad=10)
    ax.set_ylabel("y [a.u.]", labelpad=20)
    ax.set_zlabel("Ez [V/m]", labelpad=10)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zticks(
        np.linspace(0, np.max(field), 2), labels=["0", "{:.2e}".format(np.max(field))]
    )
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 1.0, 0.4, 1]))

    plt.show()
