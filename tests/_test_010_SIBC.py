import numpy as np
import pyvista as pv

from wakis import GridFIT3D

# Read solid
surf = pv.read("stl/007_lossymetal_shell.stl")

# Generate grid
Nx = 50
Ny = 50
Nz = 50
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds

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
    stl_solids={"shell": "stl/007_lossymetal_shell.stl"},
    stl_materials={"shell": [1.0, 1.0, 10]},
    verbose=1,
)


# using numpy gradient
# mask = np.reshape(grid.grid['shell'], (Nx, Ny, Nz)).astype(int)
# dsc_dx, dsc_dy, dsc_dz = np.gradient(mask, grid.dx, grid.dy, grid.dz)
# grad = np.sqrt(dsc_dx**2 + dsc_dy**2 + dsc_dz**2)
# grid.grid["grad_mag"] = grad.ravel(order="F")

# using pyvista gradient: du/dx, du/dy, du/dz
grad = np.array(
    grid.grid.compute_derivative(scalars="shell", gradient="gradient")["gradient"]
)
grad = np.sqrt(grad[:, 0] ** 2 + grad[:, 1] ** 2 + grad[:, 2] ** 2)
grid.grid["grad"] = grad.astype(bool)

# --- Interactive plotting routine ---
pl = pv.Plotter()
pl.add_mesh(grid.grid, scalars=None, style="wireframe", color="grey", opacity=0.2)
_ = pl.add_mesh_clip_box(
    grid.grid, scalars="grad", cmap="viridis", rotation_enabled=False
)
pl.show()
