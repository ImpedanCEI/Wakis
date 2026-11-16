import os
import sys

import numpy as np
import pytest
from scipy.constants import c, mu_0
from scipy.constants import epsilon_0 as eps_0
from tqdm import tqdm

sys.path.append("../")
import wakis

flag_interactive = False  # Set to true to run plot tests


class TestPML:
    def test_timestep(self):
        # TODO: check for dt < relaxation time
        pass

    def test_reflection_planewave(self, use_gpu):
        print("\n---------- Initializing simulation ------------------")
        # Domain bounds and grid
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0
        zmin, zmax = 0.0, 1.0

        Nx, Ny = 20, 20
        Nz = 200

        grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

        # Boundary conditions and solver
        bc_low = ["periodic", "periodic", "pec"]
        bc_high = ["periodic", "periodic", "pml"]

        # Test different eps_r and sigma case
        eps_r = 1.0
        sigma = 0.0

        # Solver
        solver = wakis.SolverFIT3D(
            grid,
            use_stl=False,
            use_gpu=use_gpu,
            bg=[eps_r, 1.0, sigma],
            bc_low=bc_low,
            bc_high=bc_high,
            n_pml=30,
            dtype=np.float32,
        )

        # Source
        amplitude = 100.0
        nodes = 7
        f = 15 / ((solver.z.max() - solver.z.min()) / c)
        planeWave = wakis.sources.PlaneWave(
            xs=slice(0, Nx),
            ys=slice(0, Ny),
            zs=0,
            beta=1.0,
            amplitude=amplitude,
            f=f,
            nodes=nodes,
            phase=np.pi / 2,
        )
        solver.dt = 1 / f / 200  # ensure right amplitude

        # Simulation time is extended by a factor eps_r
        # to ensure the wave is fully absorbed in the PML

        # dz contains physical cell widths, with no terminal ghost entry.
        z_pml = solver.dz[-solver.n_pml :].sum()
        Nt = int(eps_r * 2.0 * (zmax - zmin - z_pml) / c / solver.dt)

        for n in tqdm(range(Nt)):
            planeWave.update(solver, n * solver.dt)
            solver.one_step()

            if flag_interactive and n % int(Nt / 100) == 0:
                solver.plot1D(
                    "Hy",
                    ylim=(-amplitude, amplitude),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Hy",
                    n=n,
                )
                solver.plot1D(
                    "Ex",
                    ylim=(-amplitude * c * mu_0, amplitude * c * mu_0),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Ex",
                    n=n,
                )

        reflectionH = solver.H.get_abs()[Nx // 2, Ny // 2, :].max()
        reflectionE = solver.E.get_abs()[Nx // 2, Ny // 2, :].max() / (mu_0 * c)
        assert reflectionH <= 10, (
            f"PML H reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )
        assert reflectionE <= 10, (
            f"PML E reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )

        if flag_interactive:
            os.system("convert -delay 10 -loop 0 005_Hy*.png 005_Hy_planewave.gif")
            os.system("convert -delay 10 -loop 0 005_Ex*.png 005_Ex_planewave.gif")
            os.system("rm 005_Hy*.png")
            os.system("rm 005_Ex*.png")

            solver.plot2D(
                "Ex",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude * c * mu_0,
                vmax=amplitude * c * mu_0,
                off_screen=True,
                title="005_Ex2d",
            )

            solver.plot2D(
                "Hy",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude,
                vmax=amplitude,
                off_screen=True,
                title="005_Hy2d",
            )

    def test_reflection_planewave_resistive_material(self, use_gpu):
        print("\n---------- Initializing simulation ------------------")
        # Domain bounds and grid
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0
        zmin, zmax = 0.0, 1.0

        Nx, Ny = 20, 20
        Nz = 200

        grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

        # Boundary conditions and solver
        bc_low = ["periodic", "periodic", "pec"]
        bc_high = ["periodic", "periodic", "pml"]

        # Test different eps_r and sigma case
        eps_r = 3.0
        sigma = 1.0

        # Solver
        solver = wakis.SolverFIT3D(
            grid,
            use_stl=False,
            use_gpu=use_gpu,
            bg=[eps_r, 1.0, sigma],
            bc_low=bc_low,
            bc_high=bc_high,
            dtype=np.float32,
            n_pml=30,
        )

        # Source
        amplitude = 100.0
        nodes = 7
        f = 15 / ((solver.z.max() - solver.z.min()) / c)
        planeWave = wakis.sources.PlaneWave(
            xs=slice(0, Nx),
            ys=slice(0, Ny),
            zs=0,
            beta=1.0,
            amplitude=amplitude,
            f=f,
            nodes=nodes,
            phase=np.pi / 2,
        )
        solver.dt = 1 / f / 200  # ensure right amplitude

        # Simulation

        # Simulation time is extended by a factor eps_r
        # to ensure the wave is fully absorbed in the PML
        # dz contains physical cell widths, with no terminal ghost entry.
        z_pml = solver.dz[-solver.n_pml :].sum()
        Nt = int(eps_r * 2.0 * (zmax - zmin - z_pml) / c / solver.dt)

        for n in tqdm(range(Nt)):
            planeWave.update(solver, n * solver.dt)
            solver.one_step()

            if flag_interactive and n % int(Nt / 100) == 0:
                solver.plot1D(
                    "Hy",
                    ylim=(-amplitude, amplitude),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Hy",
                    n=n,
                )
                solver.plot1D(
                    "Ex",
                    ylim=(-amplitude * c * mu_0, amplitude * c * mu_0),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Ex",
                    n=n,
                )

        reflectionH = solver.H.get_abs()[Nx // 2, Ny // 2, :].max()
        reflectionE = solver.E.get_abs()[Nx // 2, Ny // 2, :].max() / (mu_0 * c)
        assert reflectionH <= 10, (
            f"PML H reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )
        assert reflectionE <= 10, (
            f"PML E reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )

        if flag_interactive:
            # os.system(f'convert -delay 10 -loop 0 005_Hy*.png 005_Hy_planewave.gif')
            # os.system(f'convert -delay 10 -loop 0 005_Ex*.png 005_Ex_planewave.gif')
            # os.system(f'rm 005_Hy*.png')
            # os.system(f'rm 005_Ex*.png')

            solver.plot2D(
                "Ex",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude * c * mu_0,
                vmax=amplitude * c * mu_0,
                off_screen=True,
                title="005_Ex2d",
            )

            solver.plot2D(
                "Hy",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude,
                vmax=amplitude,
                off_screen=True,
                title="005_Hy2d",
            )

    def test_reflection_planewave_high_resistivity_material(self, use_gpu):
        print("\n---------- Initializing simulation ------------------")
        # Domain bounds and grid
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0
        zmin, zmax = 0.0, 1.0

        Nx, Ny = 20, 20
        Nz = 200

        grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

        # Boundary conditions and solver
        bc_low = ["periodic", "periodic", "pec"]
        bc_high = ["periodic", "periodic", "pml"]

        # Test different eps_r and sigma case
        eps_r = 1.0
        sigma = 1

        # Solver
        solver = wakis.SolverFIT3D(
            grid,
            use_stl=False,
            use_gpu=use_gpu,
            bg=[eps_r, 1.0, sigma],
            bc_low=bc_low,
            bc_high=bc_high,
            dtype=np.float32,
            n_pml=30,
        )

        # Source
        amplitude = 100.0
        nodes = 7
        f = 15 / ((solver.z.max() - solver.z.min()) / c)
        planeWave = wakis.sources.PlaneWave(
            xs=slice(0, Nx),
            ys=slice(0, Ny),
            zs=0,
            beta=1.0,
            amplitude=amplitude,
            f=f,
            nodes=nodes,
            phase=np.pi / 2,
        )
        solver.dt = 1 / f / 200  # ensure right amplitude

        # Simulation

        # Simulation time is extended by a factor eps_r
        # to ensure the wave is fully absorbed in the PML
        # dz contains physical cell widths, with no terminal ghost entry.
        z_pml = solver.dz[-solver.n_pml :].sum()
        Nt = int(eps_r * 2.0 * (zmax - zmin - z_pml) / c / solver.dt)

        for n in tqdm(range(Nt)):
            planeWave.update(solver, n * solver.dt)
            solver.one_step()

            if flag_interactive and n % int(Nt / 100) == 0:
                solver.plot1D(
                    "Hy",
                    ylim=(-amplitude, amplitude),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Hy",
                    n=n,
                )
                solver.plot1D(
                    "Ex",
                    ylim=(-amplitude * c * mu_0, amplitude * c * mu_0),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Ex",
                    n=n,
                )

        reflectionH = solver.H.get_abs()[Nx // 2, Ny // 2, :].max()
        reflectionE = solver.E.get_abs()[Nx // 2, Ny // 2, :].max() / (mu_0 * c)
        assert reflectionH <= 10, (
            f"PML H reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )
        assert reflectionE <= 10, (
            f"PML E reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )

        if flag_interactive:
            # os.system(f'convert -delay 10 -loop 0 005_Hy*.png 005_Hy_planewave.gif')
            # os.system(f'convert -delay 10 -loop 0 005_Ex*.png 005_Ex_planewave.gif')
            # os.system(f'rm 005_Hy*.png')
            # os.system(f'rm 005_Ex*.png')

            solver.plot2D(
                "Ex",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude * c * mu_0,
                vmax=amplitude * c * mu_0,
                off_screen=True,
                title="005_Ex2d",
            )

            solver.plot2D(
                "Hy",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude,
                vmax=amplitude,
                off_screen=True,
                title="005_Hy2d",
            )

    def _pml_func():
        pml_lo = 0.005
        pml_hi = 0.1
        n_pml = 10

        lin = np.linspace(pml_lo, pml_hi, n_pml)
        geom = np.geomspace(pml_lo, pml_hi, n_pml)  # r=(pml_hi/pml_lo)**(1/(n_pml-1))

        x = np.linspace(0, 1, n_pml)
        quad = pml_lo + (pml_hi - pml_lo) * x**2
        cub = pml_lo + (pml_hi - pml_lo) * x**3
        quart = pml_lo + (pml_hi - pml_lo) * x**4
        quint = pml_lo + (pml_hi - pml_lo) * x**5

        # Plot func profiles
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(lin, label="linspace")
        ax.plot(geom, label="geomspace")
        ax.plot(quad, label="quadratic")
        ax.plot(cub, label="cubic")
        ax.plot(quart, label="quartic")
        ax.plot(quint, label="quintic")
        ax.set_xlabel("PML cells n")
        ax.set_ylabel(r"conductivity profile $\sigma(n)$")
        ax.legend()

        # plt.show()

        # Plot func derivative
        fig, ax = plt.subplots(dpi=200)

        ax.plot(lin[1:] - lin[:-1], label="linspace")
        ax.plot(geom[1:] - geom[:-1], label="geomspace")
        ax.plot(quad[1:] - quad[:-1], label="quadratic")
        ax.plot(cub[1:] - cub[:-1], label="cubic")
        ax.plot(quart[1:] - quart[:-1], label="quartic")
        ax.plot(quint[1:] - quint[:-1], label="quintic")
        ax.set_xlabel("PML cells n")
        ax.set_ylabel(r"gradient of conductivity profile $\sigma(n)$")
        ax.legend()


def _pml_probe_reflection(eps_r, mu_r, sigma, match_bg=True):
    """Measure reflected energy at a probe against a long-domain reference."""
    Nx, Ny = 2, 2
    dz = 0.005
    Nz_short = 201  # zmax = 1.0 m
    Nz_long = 401  # zmax = 2.0 m (same dz)
    z_probe = 120
    n_pml = 30
    f = 15.0 * c
    dt = 1.0 / f / 200.0
    z_pml = n_pml * dz
    n_idx = np.sqrt(eps_r * mu_r)
    Nt = int(n_idx * 2.2 * (1.0 - z_pml) / c / dt)

    traces = {}
    for label, Nz in (("short", Nz_short), ("ref", Nz_long)):
        zmax = (Nz - 1) * dz
        grid = wakis.GridFIT3D(0.0, 0.02, 0.0, 0.02, 0.0, zmax, Nx, Ny, Nz, verbose=0)
        bg = [eps_r, mu_r, sigma] if (label == "ref" or match_bg) else [1.0, 1.0, 0.0]
        solver = wakis.SolverFIT3D(
            grid,
            use_stl=False,
            bg=bg,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pml"],
            n_pml=n_pml,
            dt=dt,
            dtype=np.float64,
            verbose=0,
        )
        if label == "short" and not match_bg:
            for d in ("x", "y", "z"):
                solver.ieps[:, :, :-n_pml, d] = 1.0 / (eps_r * eps_0)
                solver.imu[:, :, :-n_pml, d] = 1.0 / (mu_r * mu_0)
                solver.sigma[:, :, :-n_pml, d] = sigma
            solver.update_tensors(tensor="all")

        source = wakis.sources.PlaneWave(
            xs=slice(0, Nx),
            ys=slice(0, Ny),
            zs=0,
            beta=1.0 / n_idx,
            amplitude=100.0,
            f=f / n_idx,
            nodes=7,
            phase=np.pi / 2,
        )
        trace = np.empty(Nt)
        for n in range(Nt):
            source.update(solver, n * dt)
            solver.one_step()
            trace[n] = solver.E[0, 0, z_probe, "x"]
        traces[label] = trace

    w_inc = np.sum(traces["ref"] ** 2)
    w_refl = np.sum((traces["short"] - traces["ref"]) ** 2)
    return w_refl / w_inc, w_refl, w_inc


@pytest.mark.parametrize(
    "eps_r,mu_r,sigma",
    [(4.0, 1.0, 0.0), (4.0, 1.0, 0.01), (1.0, 4.0, 0.0)],
    ids=["lossless_eps4", "lossy_eps4_sigma0p01", "magnetic_mu4"],
)
def test_pml_background_material_reflection(eps_r, mu_r, sigma):
    """PML matched to dielectric/magnetic/lossy background suppresses interface reflection."""
    r_vac, w_refl_vac, w_inc_vac = _pml_probe_reflection(1.0, 1.0, 0.0, match_bg=True)
    r_matched, w_refl_m, w_inc_m = _pml_probe_reflection(
        eps_r, mu_r, sigma, match_bg=True
    )
    r_unmatched, w_refl_u, _ = _pml_probe_reflection(eps_r, mu_r, sigma, match_bg=False)
    print(
        f"\nPML reflection (eps_r={eps_r}, mu_r={mu_r}, sigma={sigma}): "
        f"R_vac={r_vac:.4e} (W_refl={w_refl_vac:.4e}, W_inc={w_inc_vac:.4e}), "
        f"R_matched={r_matched:.4e} (W_refl={w_refl_m:.4e}, W_inc={w_inc_m:.4e}), "
        f"R_unmatched={r_unmatched:.4e} (W_refl={w_refl_u:.4e}), "
        f"R_matched/R_vac={r_matched / r_vac:.4f}x, "
        f"R_unmatched/R_matched={r_unmatched / r_matched:.1f}x"
    )
    assert r_unmatched > 10.0 * r_matched
    assert r_matched < 3.0 * r_vac


def test_pml_vacuum_default_bit_identical(use_gpu):
    """Default vacuum PML is bit-identical to the pre-change vacuum expressions transcribed below."""
    grid = wakis.GridFIT3D(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 6, 6, 80, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        use_stl=False,
        use_gpu=use_gpu,
        bg=[1.0, 1.0, 0.0],
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pml"],
        n_pml=20,
        dtype=np.float32,
        verbose=0,
    )
    ref = wakis.SolverFIT3D(
        wakis.GridFIT3D(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 6, 6, 80, verbose=0),
        use_stl=False,
        use_gpu=use_gpu,
        bg=[1.0, 1.0, 0.0],
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pml"],
        n_pml=20,
        dtype=np.float32,
        verbose=0,
    )
    # Explicitly populate ref's high-z PML with the pre-change vacuum expressions
    sz = np.zeros(ref.Nz)
    sz[-ref.n_pml :] = np.geomspace(5.0e-3, 1.0, ref.n_pml)
    for d in ["x", "y", "z"]:
        for k in range(ref.n_pml):
            k += 1
            ref.ieps[:, :, -k, d] = 1.0 / eps_0
            ref.imu[:, :, -k, d] = 1.0 / mu_0
            ref.sigma[:, :, -k, d] = sz[-k]
    ref.update_tensors(tensor="all")

    for attr in ("ieps", "imu", "sigma"):
        assert np.array_equal(getattr(solver, attr).array, getattr(ref, attr).array)

    f = 15.0 * c
    dt = 1.0 / f / 200.0
    for sim in (solver, ref):
        sim.dt = dt
        src = wakis.sources.PlaneWave(
            xs=slice(0, 6),
            ys=slice(0, 6),
            zs=0,
            beta=1.0,
            amplitude=100.0,
            f=f,
            nodes=7,
            phase=np.pi / 2,
        )
        for n in range(100):
            src.update(sim, n * dt)
            sim.one_step()

    for attr in ("E", "H", "J"):
        assert np.array_equal(
            getattr(solver, attr).toarray(), getattr(ref, attr).toarray()
        )
