import numpy as np
import pyvista as pv

from wakis import GridFIT3D

# Read solid
surf = pv.read("stl/007_lossymetal_shell.stl")

# Generate grid
Nx = 50
Ny = 60
Nz = 70
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

# import using voxelize_rectilinear
surf = grid.read_stl("shell")
vox = surf.voxelize_rectilinear(spacing=[grid.dx.min(), grid.dy.min(), grid.dz.min()])
mask = np.reshape(grid.grid["shell"], (Nx, Ny, Nz)).astype(bool)
# mask = vox['mask']
grid.grid["shell"] = mask.ravel(order="C")
grid.plot_stl_mask("shell")

# boundaries using numpy gradient
# mask = np.reshape(grid.grid['shell'], (Nx, Ny, Nz)).astype(int)
# dsc_dx, dsc_dy, dsc_dz = np.gradient(mask, grid.dx, grid.dy, grid.dz)
# grad = np.sqrt(dsc_dx**2 + dsc_dy**2 + dsc_dz**2)
# grid.grid["grad_mag"] = grad.ravel(order="F")

# boundaries using pyvista gradient: du/dx, du/dy, du/dz
# grad = np.array(
#     grid.grid.compute_derivative(scalars="shell", gradient="gradient")["gradient"]
# )
# grad = np.sqrt(grad[:, 0] ** 2 + grad[:, 1] ** 2 + grad[:, 2] ** 2)
# grid.grid["grad"] = grad.astype(bool)

# --- Interactive plotting routine ---
# pl = pv.Plotter()
# pl.add_mesh(surf, scalars=None, silhouette=True, color="red", opacity=0.2)
# _ = pl.add_mesh_clip_box(
#     vox, scalars="mask", cmap="viridis", rotation_enabled=False
# )
# pl.add_axes()
# pl.show()

# -----------------------------
import numpy as np
import pyvista as pv

from wakis import GridFIT3D

# Read solid
surf = pv.read("stl/VSMIO_body_decimated.stl")
# surf = surf.decimate_pro(0.75)
# surf.save("stl/VSMIO_body_decimated.stl")
# Generate grid
Nx = 100
Ny = 100
Nz = 200
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
    stl_solids={"shell": "stl/VSMIO_body_decimated.stl"},
    stl_materials={"shell": "stainless steel"},
    verbose=1,
)

grid.plot_stl_mask("shell")

# import using voxelize_rectilinear
surf = pv.read("stl/VSMIO_body_decimated.stl")  # grid.read_stl("shell")
vox = surf.voxelize_rectilinear(dimensions=(Nx + 1, Ny + 1, Nz + 1))
# mask = np.reshape(grid.grid["shell"], (Nx, Ny, Nz)).astype(bool)
vox = vox.point_data_to_cell_data()
mask = np.reshape(vox["mask"], (Nx, Ny, Nz), order="F").astype(bool)
grid.grid["shell"] = np.reshape(mask, (Nx * Ny * Nz), order="C")
grad = np.array(
    grid.grid.compute_derivative(scalars="shell", gradient="gradient")["gradient"]
)
grad = np.sqrt(grad[:, 0] ** 2 + grad[:, 1] ** 2 + grad[:, 2] ** 2)
grid.grid["shell"] = grad  # grad.astype(bool

grid.plot_stl_mask("shell")

pl = pv.Plotter()
pl.add_mesh(surf, scalars=None, silhouette=True, color="red", opacity=0.2)
_ = pl.add_mesh_clip_box(vox, scalars="mask", cmap="viridis", rotation_enabled=False)
pl.add_axes()
pl.show()

grid.grid["shell"] = mask.ravel(order="C")
grid.plot_stl_mask("shell")


# --------------------------------
# Reference volume:

import numpy as np
import pyvista as pv

from wakis import GridFIT3D

# Read solid
surf = pv.read("stl/VSMIO_body_decimated.stl")
# surf = surf.decimate_pro(0.75)
# surf.save("stl/VSMIO_body_decimated.stl")
# Generate grid
Nx = 200
Ny = 200
Nz = 300
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
    stl_solids={"shell": "stl/VSMIO_body_decimated.stl"},
    stl_materials={"shell": "stainless steel"},
    verbose=1,
)

dx = (xmax - xmin) / (Nx)
dy = (ymax - ymin) / (Ny)
dz = (zmax - zmin) / (Nz)

reference_vol = pv.ImageData(
    dimensions=(Nx, Ny, Nz),
    origin=(xmin + dx / 2, ymin + dy / 2, zmin + dz / 2),
    spacing=(dx, dy, dz),
)


surf = pv.read("stl/VSMIO_body_decimated.stl")  # grid.read_stl("shell")
vox = surf.voxelize_rectilinear(reference_volume=reference_vol)


mask = np.reshape(vox["mask"], (Nx, Ny, Nz), order="F").astype(bool)
grid.grid["shell"] = np.reshape(mask, (Nx * Ny * Nz))


grid.plot_stl_mask_slice(
    "shell",
)

grid.plot_stl_mask("shell", stl_opacity=0.0)


# --------------------------------
# Reference volume on fingers

import numpy as np
import pyvista as pv

from wakis import GridFIT3D

# Read solid
surf = pv.read("stl/VSMIO_fingers_decimated.stl")
# surf = surf.decimate_pro(0.75).scale(1e-3)
# surf.save("stl/VSMIO_fingers_decimated.stl")

# Generate grid
Nx = 100
Ny = 100
Nz = 200
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
    stl_solids={"shell": "stl/VSMIO_fingers_decimated.stl"},
    stl_materials={"shell": "stainless steel"},
    verbose=1,
)

dx = (xmax - xmin) / (Nx)
dy = (ymax - ymin) / (Ny)
dz = (zmax - zmin) / (Nz)

reference_vol = pv.ImageData(
    dimensions=(Nx, Ny, Nz),
    origin=(xmin + dx / 2, ymin + dy / 2, zmin + dz / 2),
    spacing=(dx, dy, dz),
)


surf = pv.read("stl/VSMIO_fingers_decimated.stl")  # grid.read_stl("shell")
vox = surf.voxelize_rectilinear(reference_volume=reference_vol)


mask = np.reshape(vox["mask"], (Nx, Ny, Nz), order="F").astype(bool)
grid.grid["shell"] = np.reshape(mask, (Nx * Ny * Nz))
grid.plot_stl_mask("shell")
