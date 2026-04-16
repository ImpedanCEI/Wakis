import sys

import numpy as np
from scipy.constants import c
from tqdm import tqdm

sys.path.append("../")
import wakis

n_pml_arr = [5, 10, 15, 20, 25, 30]
lo_pml_arr = [
    1e-5,
    5e-5,
]  # 1.e-4,5.e-4,1.e-3,5.e-3,1.e-2,5.e-2,1.e-1]

# for n_pml in n_pml_arr:
for lo_pml in lo_pml_arr:
    print("\n---------- Initializing simulation ------------------")
    # Domain bounds and grid
    xmin, xmax = -1.0, 1.0
    ymin, ymax = -1.0, 1.0
    zmin, zmax = 0.0, 10.0

    Nx, Ny = 20, 20
    Nz = 200

    grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

    # Source
    amplitude = 100.0
    nodes = 15
    planeWave = wakis.sources.PlaneWave(
        xs=slice(0, Nx),
        ys=slice(0, Ny),
        zs=0,
        nodes=nodes,
        beta=1.0,
        amplitude=amplitude,
    )

    # Boundary conditions and solver
    bc_low = ["pec", "pec", "pec"]
    bc_high = ["pec", "pec", "pml"]

    solver = wakis.SolverFIT3D(
        grid, use_stl=False, bc_low=bc_low, bc_high=bc_high, n_pml=10
    )

    solver.pml_lo = lo_pml  # np.sqrt(1.e-4)
    solver.pml_hi = np.sqrt(1.0e-1)
    solver.pml_func = np.geomspace
    solver.fill_pml_sigmas()
    solver.update_tensors("sigma")

    # Simulation
    Nt = int(2.0 * (zmax - zmin) / c / solver.dt)

    for n in tqdm(range(Nt)):
        if n < Nt / 4:  # Fill half the domain
            planeWave.update(solver, n * solver.dt)

        solver.one_step()

        if False:  # n == 1000:
            Hi = solver.H.copy()
            Ei = solver.E.copy()

        if n % 3500 == 0 and n != 0:
            solver.plot2D(
                "Hy",
                plane="ZY",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude,
                vmax=amplitude,
                off_screen=True,
                title=f"005_img/Hy_lin_n{solver.n_pml}_{solver.pml_lo:.1e}",
            )

        if n == 3500:
            Hf = solver.H.copy()
            Ef = solver.E.copy()

    reflection = Hf.get_abs()[:, :, 20:].max()
    results = np.array(
        [
            [
                solver.n_pml,
                f"{solver.pml_lo:.1e}",
                f"{solver.pml_hi:.1e}",
                solver.pml_func.__name__,
                reflection,
            ]
        ],
        dtype=object,
    )

    with open("005_results.txt", "a") as f:
        np.savetxt(f, results, fmt="%s", delimiter=" ")
