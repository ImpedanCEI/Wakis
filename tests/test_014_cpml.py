import os
import sys

import numpy as np
import pytest
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

    # Float32 regression samples on the boundary-inclusive symmetric grid.
    Zabs = np.array(
        [
            18.49585683,
            16.19635935,
            25.20411577,
            16.02416801,
            19.57833625,
            35.232466,
            9.19646643,
            34.35405426,
            42.46661832,
            9.82798517,
            52.19158755,
            45.72192545,
            27.21097532,
            70.22140859,
            44.82656799,
            51.40633105,
            86.78512971,
            41.78677403,
            81.34314558,
            100.23222394,
            44.90325211,
            117.04192884,
            108.57805873,
            67.7476287,
            158.95025075,
            109.33661187,
            116.68320142,
            208.35239325,
            100.29542781,
            199.53259566,
            268.86562186,
            90.91492375,
            344.26424134,
            352.43056069,
            169.92774012,
            663.18985431,
            518.88816955,
            665.54735142,
            2324.55127956,
            3427.04599589,
            3169.29268253,
            1715.86072242,
            178.04754852,
            785.85894015,
            570.87671942,
            239.40239336,
            510.17186805,
            329.17328483,
            299.24634159,
            385.85367419,
        ]
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

        # Float32 off-resonance baselines vary across NumPy versions. Use the
        # same relative error budget for the absolute floor and resonance peak.
        np.testing.assert_allclose(
            impedance, self.Zabs, rtol=5e-4, atol=5e-4 * np.max(self.Zabs)
        )

        # The dominant cavity resonance should remain at the same sampled
        # frequency and retain its reference amplitude.
        peak = np.argmax(self.Zabs)
        assert frequencies[np.argmax(impedance)] == frequencies[peak]
        assert np.isclose(impedance[peak], self.Zabs[peak], rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("axis", range(3), ids=list("xyz"))
@pytest.mark.parametrize("n_pml", [1, 4])
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_cpml_symmetry(axis, n_pml, graded, plot_comparison):
    """A centred pulse sees identical low/high CPML on every staggered axis.

    Symmetric graded meshes exercise physical edge-midpoint sampling; a
    one-cell layer also exposes corrections assigned only to terminal storage.
    """
    nodes = np.linspace(0, 1, 65)
    if graded:
        nodes += 0.3 * np.sin(2 * np.pi * nodes) / (2 * np.pi)
    coordinates = {name: np.array([0.0, 1.0]) for name in "xyz"}
    coordinates["xyz"[axis]] = nodes
    grid = wakis.GridFIT3D(**coordinates, verbose=0)
    boundaries = ["periodic"] * 3
    boundaries[axis] = "cpml"
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=boundaries.copy(),
        bc_high=boundaries.copy(),
        n_pml=n_pml,
        verbose=0,
    )
    electric = "xyz"[(axis + 1) % 3]
    magnetic = "xyz"[(axis + 2) % 3]
    shape = [1, 1, 1]
    shape[axis] = len(nodes)
    solver.E.to_matrix(electric)[:] = np.exp(-(((nodes - 0.5) / 0.05) ** 2)).reshape(
        shape
    )
    line = [0, 0, 0]
    line[axis] = slice(None)
    traces = np.empty((240, 2))
    for step in range(len(traces)):
        solver.one_step()
        field = solver.E.to_matrix(electric)[tuple(line)]
        traces[step] = field[10], field[-11]
    plot_comparison(traces[:, 1], traces[:, 0], "High / low CPML pulse symmetry")
    np.testing.assert_allclose(traces[:, 1], traces[:, 0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(field, field[::-1], rtol=1e-10, atol=1e-12)
    field_h = solver.H.to_matrix(magnetic)[tuple(line)][:-1] * mu_0 * c
    np.testing.assert_allclose(field_h, -field_h[::-1], rtol=1e-10, atol=1e-12)
