import sys

import pyvista as pv
from scipy.constants import c

sys.path.append("../wakis")

from wakis import GridFIT3D, SolverFIT3D
from wakis.sources import WavePacket

print("\n---------- Initializing simulation ------------------")
# Number of mesh cells
Nx = 40
Ny = 40
Nz = 200

# Embedded boundaries
stl_file = "tests/stl/003_sphere.stl"
surf = pv.read(stl_file)

stl_solids = {"Sphere": stl_file}
stl_materials = {"Sphere": [10.0, 1.0]}  # dielectric [eps_r, mu_r]
stl_rotate = [0, 0, 0]
stl_scale = 1e-3

surf = surf.rotate_x(stl_rotate[0])
surf = surf.rotate_y(stl_rotate[1])
surf = surf.rotate_z(stl_rotate[2])
surf = surf.scale(stl_scale)

# Domain bounds and grid
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
padx, pady, padz = (xmax - xmin) * 0.2, (ymax - ymin) * 0.2, (zmax - zmin) * 1.0

xmin, ymin, zmin = (xmin - padx), (ymin - pady), (zmin - padz)
xmax, ymax, zmax = (xmax + padx), (ymax + pady), (zmax + padz)

global grid
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
    stl_rotate=stl_rotate,
    stl_scale=stl_scale,
    stl_materials=stl_materials,
)

# Boundary conditions
bc_low = ["periodic", "periodic", "pec"]
bc_high = ["periodic", "periodic", "abc"]

# solver setup
global solver
solver = SolverFIT3D(
    grid,
    use_stl=False,
    bc_low=bc_low,
    bc_high=bc_high,
    bg="vacuum",
    dt=0.5 * grid.dz / c,
)

# source
f = 15 / ((solver.z.max() - solver.z.min()) / c)
# source = PlaneWave(xs=slice(1, Nx), ys=slice(1,Ny), zs=1,
#                    f=f, beta=1.0)
source = WavePacket(
    xs=slice(0, Nx),
    ys=slice(0, Ny),
    zs=0,
    sigmaz=(grid.zmax - grid.zmin) / 6,
    f=f,
    tinj=(grid.zmax - grid.zmin),
)

# plotting
kwargs = {
    "field": "H",
    "component": "y",
    "plane": "ZY",
    "pos": 0.5,
    "title": "tests/008_img/Hy",
    "vmin": -2,
    "vmax": 2,
    "cmap": "rainbow",
    "patch_reverse": False,
    "off_screen": True,
    "interpolation": "spline36",
}


def callback(solver, t):
    n = int(t / solver.dt)
    if n % 20 == 0:
        solver.plot1D(
            field="H",
            component="y",
            line="z",
            pos=[0.6, 0.5, 0.4],
            xscale="linear",
            yscale="linear",
            xlim=None,  # ylim=(-2, 2),
            figsize=[8, 4],
            title="tests/008_img/Hy1d",
            off_screen=True,
            n=n,
        )

        # fig, ax = solver.H.inspect(show=False, handles=True)
        # fig.savefig('tests/008_img/Hy_inspect'+str(n).zfill(5))
        # plt.close(fig)


Nt = int(
    2.0 * (solver.z.max() - solver.z.min()) / c / solver.dt
    + (grid.zmax - grid.zmin) / c / solver.dt
)
solver.emsolve(Nt, source, callback, plot=True, plot_every=20, **kwargs)
