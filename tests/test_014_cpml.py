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

    # Float32 regression samples with staggered primal/dual material masks.
    Zabs = np.array(
        [
            15.66068141,
            13.75843464,
            21.05945605,
            14.1730965,
            15.42627219,
            29.55483539,
            9.94307263,
            26.45701899,
            36.72666106,
            7.49252699,
            40.90149401,
            41.71954378,
            16.99605601,
            56.81065764,
            44.03772636,
            34.03720329,
            73.42533235,
            43.58420453,
            57.0682826,
            90.11253441,
            41.7609701,
            87.04235347,
            106.05698795,
            45.06377503,
            126.0754581,
            119.9864022,
            67.17657131,
            178.2767921,
            129.65169459,
            121.23329576,
            252.71752336,
            130.54801693,
            230.51944511,
            374.63690142,
            113.44174809,
            484.64026927,
            651.72986315,
            144.24335063,
            1616.02898442,
            3045.51192188,
            3415.08629615,
            2428.18052686,
            753.04282824,
            569.66287702,
            798.26879188,
            265.95133708,
            447.54601941,
            494.76402423,
            221.05188276,
            405.00970056,
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
        assert reflection_factor <= 1.2e-6, (
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

        # The dominant mode is close to the closed square-cavity TM110 mode;
        # allow one physical wake-record frequency resolution for port loading.
        width_x = grid.xmax - grid.xmin
        width_y = grid.ymax - grid.ymin
        expected_frequency = c / 2 * np.sqrt(1 / width_x**2 + 1 / width_y**2)
        peak_frequency = wake.f[np.argmax(np.abs(wake.Z))]
        resolution = wake.v / wake.wakelength
        assert abs(peak_frequency - expected_frequency) < resolution


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
