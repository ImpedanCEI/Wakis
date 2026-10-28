import os
import sys

import numpy as np
import pyvista as pv
from scipy.constants import c, mu_0
from tqdm import tqdm

sys.path.append("../wakis")
import wakis

flag_interactive = False  # Set to true to run plot tests


class TestCPML:
    """Test CPML implementation in SolverFIT3D. First test is a reflection test with a Gaussian packet,
    second test is a TFSF simulation of a cubic cavity. It benchmarks the impedance of the cavity against
    the current simulation results with CPML and TF/SF as reference."""

    Zabs = np.array(
        [
            3.47315133e00,
            5.97329754e00,
            2.23361279e00,
            1.22917272e01,
            1.22076884e01,
            1.28088394e01,
            2.28882167e01,
            1.75346671e01,
            2.56085649e01,
            3.21286970e01,
            2.44043304e01,
            4.00341264e01,
            3.93705047e01,
            3.53608561e01,
            5.52641249e01,
            4.46828066e01,
            5.18836721e01,
            7.02169647e01,
            4.96412173e01,
            7.44927998e01,
            8.35124696e01,
            5.87993539e01,
            1.03398811e02,
            9.34446294e01,
            7.99431714e01,
            1.38897083e02,
            9.82048394e01,
            1.21771463e02,
            1.81743969e02,
            9.79354402e01,
            1.95823981e02,
            2.34161176e02,
            1.08591429e02,
            3.30394031e02,
            3.03926539e02,
            2.10753205e02,
            6.35704801e02,
            4.32451982e02,
            7.58190254e02,
            2.32941757e03,
            3.30912165e03,
            2.96933364e03,
            1.52564097e03,
            1.95724858e02,
            7.92389971e02,
            5.27546795e02,
            2.82779938e02,
            5.13469152e02,
            3.23116748e02,
            3.25493121e02,
        ],
    )

    def test_reflection_gaussianPacket(self, use_gpu):
        print("\n---------- Initializing simulation ------------------")
        # Domain bounds and grid
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0
        zmin, zmax = 0.0, 1.0

        Nx, Ny = 8, 8
        Nz = 200

        grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

        # Boundary conditions and solver
        bc_low = ["periodic", "periodic", "pec"]
        bc_high = ["periodic", "periodic", "cpml"]

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
            n_pml=8,
            kappa_max=5,
            alpha_max=0.05,
            sigma_factor=1,
            pml_exp=4,
            dtype=np.float32,
        )

        # Source
        amplitude = 1.0
        gaussianPacket = wakis.sources.GaussianPacket(
            xs=slice(0, Nx),
            ys=slice(0, Ny),
            sigmaz=15e-3,
            sigmaxy=100.0,
            amplitude=amplitude,
        )

        Nt = int(gaussianPacket.tinj + 2.0 * (zmax - zmin) / c / solver.dt)
        forward = int((gaussianPacket.tinj + 0.5 * (zmax - zmin)) / c / solver.dt)
        backward = int((gaussianPacket.tinj + 1.5 * (zmax - zmin)) / c / solver.dt)

        for n in tqdm(range(Nt)):
            gaussianPacket.update(solver, n * solver.dt)
            solver.one_step()
            if n == forward:
                Exfor = solver.E[Nx // 2, Ny // 2, : -solver.n_pml, "x"].copy()
            if n == backward:
                Exback = solver.E[Nx // 2, Ny // 2, : -solver.n_pml, "x"].copy()

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

        reflection_factor = (np.abs(Exback).max() / np.abs(Exfor).max()) ** 2
        assert reflection_factor <= 1.1e-6, (
            f"CPML Ex reflection factor in average > 1e-6 with eps_r={eps_r}, sigma={sigma}, reflection_factor={reflection_factor}"
        )

        t = solver.z[: -solver.n_pml] / c
        Sfor = np.abs(np.fft.fft(Exfor))
        Sback = np.abs(np.fft.fft(Exback))
        S = (Sback / Sfor) ** 2
        f = np.fft.fftfreq(len(t), d=t[1] - t[0])
        mask = (0 <= f) & (f <= 6.66e9)

        assert S[mask].max() <= 1e-4, (
            f"Maximal CPML Ex reflection factor over all frequencies >1e-4 with eps_r={eps_r}, sigma={sigma}, Smax={S[mask].max()}"
        )

        if flag_interactive:
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

    def test_tfsf_simulation(self, use_gpu):
        print("\n---------- Initializing simulation ------------------")
        # Number of mesh cells
        Nx = 50
        Ny = 50
        Nz = 150

        # Embedded boundaries
        stl_file = "tests/stl/001_cubic_cavity.stl"
        surf = pv.read(stl_file)

        stl_solids = {"cavity": stl_file}
        stl_materials = {"cavity": "vacuum"}

        # Domain bounds
        xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds

        # set grid and geometry
        global grid
        grid = wakis.GridFIT3D(
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
            verbose=2,
        )

        # Beam parameters
        beta = 1.0  # beam beta
        sigmaz = 18.5e-3 * beta  # [m]
        q = 1e-9  # [C]
        xs = 0.0  # x source position [m]
        ys = 0.0  # y source position [m]
        xt = 0.0  # x test position [m]
        yt = 0.0  # y test position [m]

        global wake
        skip_cells = 8  # no. cells to skip in WP integration
        wakelength = 1.0  # [m]
        wake = wakis.WakeSolver(
            wakelength=wakelength,
            q=q,
            sigmaz=sigmaz,
            beta=beta,
            xsource=xs,
            ysource=ys,
            xtest=xt,
            ytest=yt,
            save=False,
            Ez_file="tests/014_Ez.h5",
            skip_cells=skip_cells,
            verbose=2,
        )

        # boundary conditions
        bc_low = ["pec", "pec", "cpml"]
        bc_high = ["pec", "pec", "cpml"]

        # set Solver object
        solver = wakis.SolverFIT3D(
            grid,
            wake,
            bc_low=bc_low,
            bc_high=bc_high,
            use_stl=True,
            bg="pec",
            dtype=np.float32,
            use_gpu=use_gpu,
            verbose=2,
            n_pml=4,
        )

        solver.wakesolve(wakelength=wakelength, save_J=False)
        os.remove("tests/014_Ez.h5")

    def test_long_impedance(self, plot_comparison):
        global wake
        frequencies = wake.f[::20]
        impedance = np.abs(wake.Z)[::20]
        plot_comparison(impedance, self.Zabs, "CPML impedance magnitude")

        # The first three bins (< 0.3 GHz) are sensitive to small float32
        # differences in the low-frequency baseline. Bound their absolute error
        # without relaxing the relative comparison of the main spectrum.
        low_frequency = frequencies < 0.3e9
        low_frequency_error = np.abs(
            impedance[low_frequency] - self.Zabs[low_frequency]
        )
        assert np.max(low_frequency_error) < 6e-3
        np.testing.assert_allclose(
            impedance[~low_frequency], self.Zabs[~low_frequency], rtol=5e-4, atol=5e-4
        )

        # The dominant cavity resonance should remain at the same sampled
        # frequency and retain its reference amplitude.
        peak = np.argmax(self.Zabs)
        assert frequencies[np.argmax(impedance)] == frequencies[peak]
        assert np.isclose(impedance[peak], self.Zabs[peak], rtol=5e-4, atol=5e-4)
