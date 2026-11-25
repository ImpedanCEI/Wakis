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
            15.75764965,
            13.85779274,
            21.16003422,
            14.25772982,
            15.50898055,
            29.65309051,
            9.96798193,
            26.53629941,
            36.81299176,
            7.3859594,
            40.99576167,
            41.7773939,
            16.92637497,
            56.91937479,
            44.04693687,
            34.03521084,
            73.53811852,
            43.52426329,
            57.12658748,
            90.21286986,
            41.62376802,
            87.15107178,
            106.12415506,
            44.90552051,
            126.21946796,
            119.99707039,
            67.11952177,
            178.43445819,
            129.58144254,
            121.30565008,
            252.859898,
            130.37273836,
            230.68400914,
            374.72528473,
            113.14491437,
            484.83134478,
            651.69845519,
            144.16316761,
            1615.99132114,
            3044.79910198,
            3413.83904278,
            2427.13952109,
            752.81189389,
            568.97599326,
            797.63507479,
            265.71440505,
            446.76309584,
            494.19057395,
            220.49691821,
            404.1397193,
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


def test_cpml_beam_float32_timing(tmp_path, monkeypatch):
    """Keep float32 CPML beam timing and source profiles in float64."""
    import h5py

    from wakis.sources import Beam, Pulse

    grid = wakis.GridFIT3D(-0.05, 0.05, -0.05, 0.05, -0.2, 0.2, 12, 12, 36, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        use_stl=False,
        dtype=np.float32,
        n_pml=4,
        verbose=0,
    )
    nt = 200
    expected_times = np.arange(nt) * float(solver.dt)
    callback_times = []
    beam = Beam(q=1e-9, sigmaz=18.5e-3, beta=1.0)
    solver.emsolve(
        nt,
        source=beam,
        save=False,
        callback=lambda s, t: callback_times.append(t),
    )
    times = np.asarray(callback_times)
    assert times.dtype == np.float64
    assert np.array_equal(times, expected_times)

    t = float(np.float32(100 * solver.dt))
    injected_j = []
    for scalar_t in (np.float32(t), float(t), np.float64(t)):
        solver.reset_fields()
        Beam(q=1e-9, sigmaz=18.5e-3, beta=1.0).update(solver, scalar_t)
        injected_j.append(solver.J.toarray().copy())
    assert np.array_equal(injected_j[0], injected_j[1])
    assert np.array_equal(injected_j[1], injected_j[2])

    pulse = Pulse(xs=6, ys=6, zs=18)
    pulse.update(solver, 0.0)
    assert float(pulse.L) == 50 * c * float(solver.dt)

    monkeypatch.chdir(tmp_path)
    solver.emsolve(nt, save=True, fields=["Ez"], save_every=20)
    with h5py.File(tmp_path / "Ez.h5", "r") as hf:
        saved_times = hf["t"][()]
    (tmp_path / "Ez.h5").unlink()
    assert saved_times.dtype == np.float64
    assert np.array_equal(saved_times, expected_times[::20])
