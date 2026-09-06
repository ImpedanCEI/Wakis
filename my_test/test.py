# CPU + MPI FAST RUN — no plots, large CFL, float32
# Writes: results_folder/{wake_s.npy, wake_WP.npy, Z_f_Hz.npy, Z_ohm.npy, summary.txt}
import sys
sys.path.insert(0, '/net/phase/store/users/sara.dastan/wakis_clean_test/PR_Wakis')
import os
# pin one thread per MPI rank (set BEFORE numpy import)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np
if not hasattr(np, "float_"): np.float_ = np.float64
#if not hasattr(np, "bool"):   np.bool   = np.bool_

import pyvista as pv
from wakis import *
from wakis.postprocessing import compute_wake_factors
import wakis

print('='*50)
print('wakis loaded from:')
print(wakis.__file__)
print('='*50)

os.environ["PYVISTA_OFF_SCREEN"] = "true"


# ---------- INPUTS ----------
scale = 1.0  # set 1e-3 if STL is in mm

solid_1 = 'component.stl'
solid_2 = 'shell1.stl'
solid_3 = 'shell2.stl'
solid_4 = 'shell3.stl'

stl_solids = {
    'cavity': solid_1,
    'shell1': solid_2,
    'shell2': solid_3,
    'shell3': solid_4,
}
stl_materials = {
    'cavity': 'vacuum',
    'shell1': 'pec',
    'shell2': 'pec',
    'shell3': 'pec',
}
stl_scale     = {k: 1.0 for k in stl_solids}
stl_rotate    = {k: (0.0, 0.0, 0.0) for k in stl_solids}
stl_translate = {k: (0.0, 0.0, 0.0) for k in stl_solids}

# Mesh (yours)
Nx, Ny, Nz = 133, 160, 940

# Beam / wake
sigmaz      = 4.5e-3     # [m]
q           = 1e-9     # [C]
beta        = 1.0
xs = 0.0; ys = 1e-3; xt = 0.0; yt = 0.0;
wakelength  = 10.0      # [m]
add_space   = 75       # cells skipped at boundaries

results_folder = '001_results_fast_cpu/'
if rank == 0 and not os.path.exists(results_folder):
    os.makedirs(results_folder, exist_ok=True)
comm.Barrier()

# ---------- DOMAIN BOUNDS ----------
solids = pv.read(solid_1) + pv.read(solid_2) + pv.read(solid_3) + pv.read(solid_4)
xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds
xmin *= scale; xmax *= scale
ymin *= scale; ymax *= scale
zmin *= scale; zmax *= scale

dx = (xmax-xmin)/Nx
dy = (ymax-ymin)/Ny
dz = (zmax-zmin)/Nz
#if rank == 0:
#    print("dx, dy, dz [mm] =", 1e3*dx, 1e3*dy, 1e3*dz)
if rank == 0:
    print("==== STL Bounds Debug ====", flush=True)
    for name, path in stl_solids.items():
        mesh = pv.read(path)
        print(name, mesh.bounds, flush=True)
    print("==========================", flush=True)


# ---------- GRID (MPI on) ----------
grid = GridFIT3D(
    xmin, xmax, ymin, ymax, zmin, zmax,
    Nx, Ny, Nz,
    use_mpi=True,
    stl_solids=stl_solids,
    stl_materials=stl_materials,
    stl_scale=stl_scale,
    stl_rotate=stl_rotate,
    stl_translate=stl_translate,
    verbose=1 if rank == 0 else 0
)

# ---------- WAKE SETUP ----------
wake = WakeSolver(
    q=q, sigmaz=sigmaz, beta=beta,
    xsource=xs, ysource=ys, xtest=xt, ytest=yt,
    add_space=add_space,
    results_folder=results_folder,
    Ez_file=os.path.join(results_folder, 'Ez.h5'),
)

# ---------- SOLVER (CPU, MPI, fast knobs) ----------
bc_low  = ['pec','pec','pml']
bc_high = ['pec','pec','pml']

solver = SolverFIT3D(
    grid, wake,
    bc_low=bc_low, bc_high=bc_high,
    use_stl=True,
    use_mpi=True,        # domain decomposition across ranks
    use_gpu=False,       # CPU path scales with MPI
    dtype=np.float64,    # more accurate, slower
    cfln=0.8,           # near-stability CFL (fewer timesteps)
    n_pml=8,             # slightly thinner PML (reduce if unstable)
    bg='vacuum',
    verbose=0
)

# ---------- RUN (no plotting) ----------
solver.wakesolve(
    wakelength=wakelength,
    add_space=add_space,
    plot=False
)

# ---------- Compute wake factors ----------
comm.Barrier()

if rank == 0:
    # load results the Wakis way
    wake.load_results(folder=results_folder)

    res = compute_wake_factors(
        s=wake.s,
        WP=wake.WP,
        WPx=wake.WPx,
        WPy=wake.WPy,
        lambdas=wake.lambdas,
        x_offset=xs,
        y_offset=ys,
        filename=wake.folder + "wake_factors.txt"
    )
    
# ---------- POST (rank 0) ----------
comm.Barrier()

if rank == 0:
    # Optional trimming/window for FFT (causal part)
    mask = (wake.s >= 0.0) & (wake.s <= wakelength)
    win = np.ones(mask.sum(), dtype=np.float32)

    # Compute longitudinal impedance (adjust fmax/samples as needed)
    wake.calc_long_Z(
        samples=4097,
        fmax=23e9,
        WP=wake.WP[mask]*win,
        s=wake.s[mask],
        lambdas=wake.lambdas[mask]
    )

    # Quick text summary
    with open(os.path.join(results_folder, 'summary.txt'), 'w') as f:
        f.write(f"Ranks: {size}\n")
        f.write(f"Mesh: Nx={Nx} Ny={Ny} Nz={Nz}\n")
        f.write(f"wakelength={wakelength} m, add_space={add_space} cells\n")
        f.write(f"dtype=float64, cfln=0.8, n_pml=8, CPU+MPI\n")
        f.write(f"sigmaz={sigmaz} m, xs={xs}, ys ={ys}, xt={xt}, yt={yt} m\n")
        f.write(f"Saved: wake_s.npy, wake_WP.npy, Z_f_Hz.npy, Z_ohm.npy\n")

# Make sure all ranks finish cleanly
comm.Barrier()
if rank == 0:
    with open(os.path.join(results_folder, "DONE.txt"), "w") as f:
        f.write("Simulation finished successfully.\n")
