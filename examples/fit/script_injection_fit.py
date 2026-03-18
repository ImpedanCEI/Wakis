import sys

import h5py
import numpy as np
from scipy.constants import c as c_light
from tqdm import tqdm

sys.path.append("../")

from gridFIT3D import GridFIT3D
from solverFIT3D import SolverFIT3D

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 50
Ny = 50
Nz = 100

# Domain bounds
xmin, xmax = -2.5e-2, +2.5e-2
ymin, ymax = -2.5e-2, +2.5e-2
zmin, zmax = -5e-2, +5e-2
Lx, Ly, Lz = (xmax - xmin), (ymax - ymin), (zmax - zmin)

# boundary conditions
bc_low = ["pec", "pec", "abc"]
bc_high = ["pec", "pec", "abc"]

# set FIT solver
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

solver = SolverFIT3D(grid, bc_low=bc_low, bc_high=bc_high, bg="vacuum")

# ------------ Beam source ----------------
# Beam parameters
sigmaz = 5e-2  # [m]
q = 1e-9  # [C]
beta = 1.0  # beam beta
xs = 0.0  # x source position [m]
ys = 0.0  # y source position [m]
xt = 0.0  # x test position [m]
yt = 0.0  # y test position [m]
tinj = 8.53 * sigmaz / c_light  # injection time offset [s]

x, y, z = solver.x, solver.y, solver.z
ixs, iys = np.abs(x - xs).argmin(), np.abs(y - ys).argmin()
ixt, iyt = np.abs(x - xt).argmin(), np.abs(y - yt).argmin()


def beam(solver, t, gap=0):
    # Define gaussian
    if gap == 0:
        izs = slice(0, Nz)
    else:
        izs = slice(gap, Nz - gap)

    z = solver.z[izs]
    s0 = z.min() - c_light * tinj
    s = z - c_light * t
    profile = (
        1 / np.sqrt(2 * np.pi * sigmaz**2) * np.exp(-((s - s0) ** 2) / (2 * sigmaz**2))
    )

    # Update current
    current = q * c_light * profile / solver.dx / solver.dy
    solver.J[ixs, iys, izs, "z"] = current


# ------------ Time loop ----------------
# Define wake length
wakelength = 10 * 1e-2  # [m]

# Obtain simulation time
tmax = (wakelength + tinj * c_light + (zmax - zmin)) / c_light  # [s]
Nt = int(tmax / solver.dt)
t = np.arange(0, Nt * solver.dt, solver.dt)
ninj = int(tinj / solver.dt)

# Prepare save files
save = False
plot = True

if save:
    hf = h5py.File("imgInj/Ez_gap0_pec.h5", "w")
    hf2 = h5py.File("imgInj/Jz.h5", "w")
    hf["x"], hf["y"], hf["z"], hf["t"] = x, y, z, t

for n in tqdm(range(Nt)):
    # Initial condition
    beam(solver, n * solver.dt)

    # Advance
    solver.one_step()

    # Plot 2D
    if n > (ninj - 1000) and n % 20 == 0 and plot:
        solver.plot2D(
            field="E",
            component="y",
            plane="ZY",
            pos=0.5,
            norm=None,
            vmin=-5e5,
            vmax=5e5,
            cmap="bwr",
            title="imgInj/Ez",
            off_screen=True,
            n=n,
            interpolation="spline36",
        )

    # Save
    if save:
        hf["#" + str(n).zfill(5)] = solver.E[ixt, iyt, :, "z"]
        hf2["#" + str(n).zfill(5)] = solver.J[ixt, iyt, :, "z"]

# Close save file
if save:
    hf.close()
    hf2.close()


# Compare results TODO
if plot:
    pass
