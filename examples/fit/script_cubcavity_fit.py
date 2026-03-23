import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.constants import c as c_light
from tqdm import tqdm

sys.path.append("../")

from gridFIT3D import GridFIT3D
from solverFIT3D import SolverFIT3D

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 50
Ny = 50
Nz = 150

# Embedded boundaries
stl_file = "stl/cubcavity.stl"
surf = pv.read(stl_file)

stl_solids = {"Cavity": stl_file}
stl_materials = {"Cavity": "vacuum"}

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax - xmin), (ymax - ymin), (zmax - zmin)

# boundary conditions
bc_low = ["pec", "pec", "abc"]
bc_high = ["pec", "pec", "abc"]

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
    stl_materials=stl_materials,
)

solver = SolverFIT3D(grid, bc_low=bc_low, bc_high=bc_high, use_stl=True, bg="pec")

# ------------ Beam source ----------------
# Beam parameters
sigmaz = 18.5e-3  # [m]
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


def beam(solver, t):
    # Define gaussian
    s0 = z.min() - c_light * tinj
    s = z - c_light * t
    profile = (
        1 / np.sqrt(2 * np.pi * sigmaz**2) * np.exp(-((s - s0) ** 2) / (2 * sigmaz**2))
    )

    # Update current
    current = q * c_light * profile / solver.dx / solver.dy
    solver.J[ixs, iys, :, "z"] = current


# ------------ Time loop ----------------
# Define wake length
wakelength = 1000 * 1e-3  # [m]

# Obtain simulation time
tmax = (wakelength + tinj * c_light + (zmax - zmin)) / c_light  # [s]
Nt = int(tmax / solver.dt)
t = np.arange(0, Nt * solver.dt, solver.dt)
ninj = int(tinj / solver.dt)

# Prepare save files
save = True
plot = False

if save:
    hf = h5py.File("results/Ez.h5", "w")
    hf2 = h5py.File("results/Jz.h5", "w")
    hf["x"], hf["y"], hf["z"], hf["t"] = x, y, z, t

for n in tqdm(range(Nt)):
    # Initial condition
    beam(solver, n * solver.dt)

    # Advance
    solver.one_step()

    # Plot 2D on-the-fly
    if n > ninj and n % 10 == 0 and plot:
        solver.plot2D(
            field="E",
            component="y",
            plane="ZY",
            pos=0.5,
            norm=None,
            vmin=-5e5,
            vmax=5e5,
            cmap="rainbow",
            patch_alpha=0.5,
            add_patch="Cavity",
            patch_reverse=True,
            title="imgCav/Ey",
            off_screen=True,
            n=n,
            interpolation="spline36",
        )

    # Save in hdf5 format (needed for Impedance)
    if save:
        hf["#" + str(n).zfill(5)] = solver.E[ixt, iyt, :, "z"]
        # hf2['#'+str(n).zfill(5)]=solver.J[ixt, iyt, :, 'z']

# Close save file
if save:
    hf.close()
    hf2.close()

# Plot 3D built-in
plot3d = False
if plot3d:
    solver.plot3D(
        field="E",
        component="Abs",
        clim=None,
        add_stl="Cavity",
        stl_opacity=0.0,
        stl_colors="white",
        field_on_stl=True,
        nan_opacity=1.0,
        field_opacity=1.0,
        cmap="rainbow",
        off_screen=False,
        zoom=1.0,
        n=n,
    )

# ------------ Beam-coupling Impedance ----------------
from wakeSolver import WakeSolver

# Initialize wake solver with beam information
wake = WakeSolver(
    q=q,
    sigmaz=sigmaz,
    beta=beta,
    ti=tinj,
    xsource=xs,
    ysource=ys,
    xtest=xt,
    ytest=yt,
    save=True,
    logfile=True,
    Ez_file="results/Ez.h5",
)

# Compute beam-coupling impedance and wake potential
wake.calc_long_WP()
wake.calc_long_Z(samples=1001)

plot = True
if plot:
    fig, ax = plt.subplots(1, 2, figsize=[12, 4], dpi=150)
    ax[0].plot(wake.s * 1e3, wake.WP, c="r", lw=1.5)
    ax[0].set_xlabel("s [mm]")
    ax[0].set_ylabel("Longitudinal wake potential [V/pC]", color="r")

    ax[1].plot(wake.f * 1e-9, np.abs(wake.Z), c="b", lw=1.5)
    ax[1].set_xlabel("f [GHz]")
    ax[1].set_ylabel("Longitudinal impedance [Abs][$\Omega$]", color="b")

    fig.tight_layout()
    fig.savefig("results/longitudinal.png")

    plt.show()
