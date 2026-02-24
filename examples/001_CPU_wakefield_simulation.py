# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# (!) If using MKL intel backend only
import os

os.environ["MKL_NUM_THREADS"] = (
    "1"  # change the number of threads to be used in the simualtion
)

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from wakis import GridFIT3D, SolverFIT3D, WakeSolver

# ---------- Domain setup ---------

# Geometry & Materials
solid_1 = "data/001_vacuum_cavity.stl"
solid_2 = "data/001_lossymetal_shell.stl"

stl_solids = {"cavity": solid_1, "shell": solid_2}

stl_materials = {
    "cavity": "vacuum",
    "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
}

# BONUS: Visualize geometry - Uncomment for plotting!
# pl = pv.Plotter()
# pl.add_mesh(pv.read(solid_1),color='tab:orange', specular=0.5, smooth_shading=True)
# pl.add_mesh(pv.read(solid_2),color='tab:blue', opacity=0.5, specular=0.5, smooth_shading=True)
# pl.set_background('mistyrose', top='white')
# pl.show()

# Extract domain bounds from geometry
solids = pv.read(solid_1) + pv.read(solid_2)
xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

# Number of mesh cells
Nx = 80
Ny = 80
Nz = 141

# set grid and geometry
grid = GridFIT3D(
    *solids.bounds,
    Nx,
    Ny,
    Nz,
    stl_solids=stl_solids,
    stl_materials=stl_materials,
    stl_scale=1.0,
    stl_rotate=[0, 0, 0],
    stl_translate=[0, 0, 0],
    verbose=1,
)

# BONUS: Visualize grid - Uncomment for plotting!
grid.inspect(add_stl=["cavity", "shell"], stl_opacity=1.0)

# BONUS: Visualize imported solids - Uncomment for plotting!
grid.plot_solids(
    bounding_box=True, opacity=1.0, specular=0.5, anti_aliasing=None
)

# ------------ Beam source & Wake ----------------
# Beam parameters
sigmaz = 10e-2  # [m] -> 2 GHz
q = 1e-9  # [C]
beta = 1.0  # beam beta
xs = 0.0  # x source position [m]
ys = 0.0  # y source position [m]
xt = 0.0  # x test position [m]
yt = 0.0  # y test position [m]
# [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s]

# Simualtion
wakelength = 10.0  # [m]
add_space = 10  # no. cells to skip from boundaries - removes BC artifacts

results_folder = "001_results_float64/"
wake = WakeSolver(
    q=q,
    sigmaz=sigmaz,
    beta=beta,
    xsource=xs,
    ysource=ys,
    xtest=xt,
    ytest=yt,
    add_space=add_space,
    results_folder=results_folder,
    Ez_file=results_folder + "001_Ez.h5",
)

# ----------- Solver & Simulation ----------
# boundary conditions``
bc_low = ["pec", "pec", "pec"]
bc_high = ["pec", "pec", "pec"]

# on-the-fly plotting parameters
if not os.path.exists("001_img/"):
    os.mkdir("001_img/")

plotkw2D = {
    "title": "001_img/Ez",
    "add_patch": "cavity",
    "patch_alpha": 1.0,
    "patch_reverse": True,  # patch logical_not('cavity')
    "vmin": -1e3,
    "vmax": 1e3,  # colormap limits
    "cmap": "rainbow",
    "plane": [
        int(Nx / 2),  # x
        slice(0, Ny),  # y
        slice(add_space, -add_space),
    ],
}  # z

# Solver setup
solver = SolverFIT3D(
    grid,
    wake,
    bc_low=bc_low,
    bc_high=bc_high,
    use_stl=True,
    bg="pec",  # Background material
    dtype=np.float64,  # support for single-precision np.float32
)

# BONUS: Inspect material tensors - Uncomment to show!
# solver.ieps.inspect(plane='YZ', cmap='bwr', dpi=100)  # [1/eps] permittivity tensor ^-1
# solver.sigma.inspect(plane='XY', cmap='bwr', dpi=100) # [sigma] conductivity tersor S/m
# fig, ax = solver.imu.inspect(x=int(Nx-10), y=slice(0,Ny), z=slice(0,Nz),  # [1/mu] permeability tensor ^-1
#                              show=False, handles=True) # return handles to further adjust

# BONUS: Access and modify tensors - Uncomment to alter simulation!
# solver.sigma[:,:,10, 'x'] = 0. # turns XY plane at z=10*dz to 0.0 in the x-direction
# solver.E[10,10,10, 'z'] = 1e3  # introduces a point perturbation in Ez component of 1000 V/m
# solver.J[:,10,10, 'x'] = 10    # introduces a line perturbation in Jx component of 10 A/m2

# Solver run
solver.wakesolve(
    wakelength=wakelength,
    add_space=add_space,
    plot=True,  # turn False for speedup
    plot_every=30,
    plot_until=3000,
    **plotkw2D,
)

# ----------- 1d plot results --------------------
# Plot longitudinal wake potential and impedance
fig1, ax = plt.subplots(1, 2, figsize=[12, 4], dpi=150)
ax[0].plot(wake.s * 1e2, wake.WP, c="r", lw=1.5, label="Wakis")
ax[0].set_xlabel("s [cm]")
ax[0].set_ylabel("Longitudinal wake potential [V/pC]", color="r")
ax[0].legend()
ax[0].set_xlim(xmax=wakelength * 1e2)

ax[1].plot(wake.f * 1e-9, np.abs(wake.Z), c="b", lw=1.5, label="Wakis")
ax[1].set_xlabel("f [GHz]")
ax[1].set_ylabel("Longitudinal impedance [Abs][$\Omega$]", color="b")
ax[1].legend()

fig1.tight_layout()
fig1.savefig(results_folder + "001_longitudinal.png")
plt.show()

# Plot transverse x wake potential and impedance
fig2, ax = plt.subplots(1, 2, figsize=[12, 4], dpi=150)
ax[0].plot(wake.s * 1e2, wake.WPx, c="r", lw=1.5, label="Wakis")
ax[0].set_xlabel("s [cm]")
ax[0].set_ylabel("Transverse wake potential X [V/pC]", color="r")
ax[0].legend()
ax[0].set_xlim(xmax=wakelength * 1e2)

ax[1].plot(wake.f * 1e-9, np.abs(wake.Zx), c="b", lw=1.5, label="Wakis")
ax[1].set_xlabel("f [GHz]")
ax[1].set_ylabel("Transverse impedance X [Abs][$\Omega$]", color="b")
ax[1].legend()

fig2.tight_layout()
fig2.savefig(results_folder + "001_transverse_x.png")
plt.show()

# Plot transverse y wake potential and impedance
fig3, ax = plt.subplots(1, 2, figsize=[12, 4], dpi=150)
ax[0].plot(wake.s * 1e2, wake.WPy, c="r", lw=1.5, label="Wakis")
ax[0].set_xlabel("s [cm]")
ax[0].set_ylabel("Transverse wake potential Y [V/pC]", color="r")
ax[0].legend()
ax[0].set_xlim(xmax=wakelength * 1e2)

ax[1].plot(wake.f * 1e-9, np.abs(wake.Zy), c="b", lw=1.5, label="Wakis")
ax[1].set_xlabel("f [GHz]")
ax[1].set_ylabel("Transverse impedance Y [Abs][$\Omega$]", color="b")
ax[1].legend()

fig3.tight_layout()
fig3.savefig(results_folder + "001_transverse_y.png")
plt.show()

# Plot Electric field component in 2D using imshow
solver.plot1D(
    field="E",
    component="z",
    line="z",
    pos=0.5,
    xscale="linear",
    yscale="linear",
    off_screen=True,
    title="001_img/Ez1d",
)
plt.show()

# ----------- 2d plots results --------------------
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list(
    "name", plt.cm.jet(np.linspace(0.1, 0.9))
)  # CST's colormap

# Plot Electric field component in 2D using imshow
solver.plot2D(
    field="E",
    component="z",
    plane="XY",
    pos=0.5,
    cmap=cmap,
    vmin=-500,
    vmax=500.0,
    interpolation="hanning",
    add_patch="cavity",
    patch_reverse=True,
    patch_alpha=0.8,
    off_screen=True,
    title="001_img/Ez2d",
)
plt.show()


# BONUS: Generate an animation from the plots generated during the simulation
#        Needs imagemagick package -> `apt install imagemagick`
# os.system('convert -loop 0 -delay 5 001_img/Ez_*.png 001_img/Ez_sim.gif')

# ----------- 3d plots results --------------------
# Plot Electric field component in 3D using pyvista.plotter
solver.plot3D(
    "E",
    component="Abs",
    cmap="rainbow",
    clim=[0, 500],
    add_stl=["cavity", "shell"],
    stl_opacity=0.1,
    clip_interactive=True,
    clip_normal="-y",
)

# Plot Abs Electric field on STL solid `cavity`
solver.plot3DonSTL(
    "E",
    component="Abs",
    cmap="rainbow",
    clim=[0, 500],
    stl_with_field="cavity",
    field_opacity=1.0,
    stl_transparent="shell",
    stl_opacity=0.1,
    stl_colors="white",
    # clip_plane=True, clip_normal='-y', clip_origin=[0,0,0], #coming in v0.5.0
    off_screen=False,
    zoom=1.2,
    title="001_img/Ez3d",
)
