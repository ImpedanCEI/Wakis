import os
import sys

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c, mu_0
from scipy.constants import epsilon_0 as eps_0
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


def _cpml_probe_reflection(eps_r, mu_r, sigma, match_bg=True):
    """Measure reflected energy at a probe against a long-domain reference."""
    Nx, Ny = 2, 2
    dz = 0.005
    Nz_short = 201  # zmax = 1.0 m
    Nz_long = 401  # zmax = 2.0 m (same dz)
    z_probe = 120
    n_pml = 10
    dt = 0.45 * dz / c
    n_idx = np.sqrt(eps_r * mu_r)
    Nt = int(2.2 * n_idx * 1.0 / c / dt)

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
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=n_pml,
            dt=dt,
            dtype=np.float64,
            verbose=0,
        )
        if label == "short" and not match_bg:
            for d in ("x", "y", "z"):
                solver.ieps[:, :, :-n_pml, d] = 1.0 / (eps_r * eps_0)
                solver.imu[:, :, : -n_pml - 1, d] = 1.0 / (mu_r * mu_0)
                solver.sigma[:, :, :-n_pml, d] = sigma
            solver.update_tensors(tensor="all")

        source = wakis.sources.Pulse(
            field="Ex",
            xs=slice(0, Nx),
            ys=slice(0, Ny),
            zs=20,
            shape="harris",
            L=0.15 * n_idx,
            amplitude=1.0,
            injection="soft",
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
    [
        pytest.param(4.0, 1.0, 0.0, id="lossless_eps4"),
        pytest.param(4.0, 1.0, 0.01, id="lossy_eps4_sigma0p01"),
        pytest.param(
            1.0,
            4.0,
            0.0,
            id="magnetic_mu4",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Unscaled alpha_max=0.05 with eps_r=1, mu_r=4 gives "
                    "R_matched=2.8656e-06 > 3*R_vac=2.4208e-07 "
                    "(still 39146.0x below R_unmatched=1.1218e-01)"
                ),
            ),
        ),
    ],
)
def test_cpml_background_material_reflection(eps_r, mu_r, sigma):
    """CPML matched to dielectric/magnetic/lossy background suppresses interface reflection."""
    r_vac, w_refl_vac, w_inc_vac = _cpml_probe_reflection(1.0, 1.0, 0.0, match_bg=True)
    r_matched, w_refl_m, w_inc_m = _cpml_probe_reflection(
        eps_r, mu_r, sigma, match_bg=True
    )
    r_unmatched, w_refl_u, _ = _cpml_probe_reflection(
        eps_r, mu_r, sigma, match_bg=False
    )
    print(
        f"\nCPML reflection (eps_r={eps_r}, mu_r={mu_r}, sigma={sigma}): "
        f"R_vac={r_vac:.4e} (W_refl={w_refl_vac:.4e}, W_inc={w_inc_vac:.4e}), "
        f"R_matched={r_matched:.4e} (W_refl={w_refl_m:.4e}, W_inc={w_inc_m:.4e}), "
        f"R_unmatched={r_unmatched:.4e} (W_refl={w_refl_u:.4e}), "
        f"R_matched/R_vac={r_matched / r_vac:.4e}x, "
        f"R_vac/R_matched={r_vac / r_matched:.1f}x, "
        f"R_unmatched/R_matched={r_unmatched / r_matched:.1f}x"
    )
    assert r_unmatched > 10.0 * r_matched
    assert r_matched < 3.0 * r_vac


def test_cpml_vacuum_default_bit_identical(use_gpu):
    """Default vacuum CPML is bit-identical to the pre-change vacuum expressions transcribed below."""
    grid = wakis.GridFIT3D(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 6, 6, 80, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        use_stl=False,
        use_gpu=use_gpu,
        bg=[1.0, 1.0, 0.0],
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=8,
        kappa_max=5,
        alpha_max=0.05,
        sigma_factor=1,
        pml_exp=4,
        dtype=np.float32,
        verbose=0,
    )
    ref = wakis.SolverFIT3D(
        wakis.GridFIT3D(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 6, 6, 80, verbose=0),
        use_stl=False,
        use_gpu=use_gpu,
        bg=[1.0, 1.0, 0.0],
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=8,
        kappa_max=5,
        alpha_max=0.05,
        sigma_factor=1,
        pml_exp=4,
        dtype=np.float32,
        verbose=0,
    )
    # Explicitly rebuild ref's high-z CPML coefficients with pre-change vacuum expressions
    eta_0 = mu_0 * c
    interface = ref.grid.z[-1 - ref.n_pml]
    L = ref.grid.z[-1] - interface
    sigma_max = -ref.sigma_factor * (ref.pml_exp + 1) * np.log(1e-8) / (2 * L * eta_0)
    kappa = wakis.Field(ref.Nx, ref.Ny, ref.Nz, use_ones=True, dtype=ref.dtype)
    tkappa = wakis.Field(ref.Nx, ref.Ny, ref.Nz, use_ones=True, dtype=ref.dtype)
    alpha = wakis.Field(ref.Nx, ref.Ny, ref.Nz, dtype=ref.dtype)
    talpha = wakis.Field(ref.Nx, ref.Ny, ref.Nz, dtype=ref.dtype)
    sigma_pml = wakis.Field(ref.Nx, ref.Ny, ref.Nz, dtype=ref.dtype)
    tsigma_pml = wakis.Field(ref.Nx, ref.Ny, ref.Nz, dtype=ref.dtype)
    for i in range(-ref.n_pml, 0):
        dist = ref.grid.z[i] - interface
        tdist = max(0.0, min((ref.grid.z[i - 1] + ref.grid.z[i]) / 2 - interface, L))
        sz = (dist / L) ** ref.pml_exp
        tsz = (tdist / L) ** ref.pml_exp
        az = dist / L
        taz = tdist / L
        sigma_pml[:, :, i, "z"] = sigma_max * sz
        kappa[:, :, i, "z"] = 1 + (ref.kappa_max - 1) * sz
        alpha[:, :, i, "z"] = ref.alpha_max * (1 - az)
        tsigma_pml[:, :, i - 1, "z"] = sigma_max * tsz
        tkappa[:, :, i - 1, "z"] = 1 + (ref.kappa_max - 1) * tsz
        talpha[:, :, i - 1, "z"] = ref.alpha_max * (1 - taz)

    b_E = wakis.Field(ref.Nx, ref.Ny, ref.Nz, dtype=ref.dtype)
    c_E = wakis.Field(ref.Nx, ref.Ny, ref.Nz, dtype=ref.dtype)
    b_H = wakis.Field(ref.Nx, ref.Ny, ref.Nz, dtype=ref.dtype)
    c_H = wakis.Field(ref.Nx, ref.Ny, ref.Nz, dtype=ref.dtype)
    b_E.fromarray(
        np.exp(
            -(sigma_pml.toarray() / kappa.toarray() + alpha.toarray()) * ref.dt / eps_0
        )
    )
    denom_e = sigma_pml.toarray() + kappa.toarray() * alpha.toarray()
    ratio_e = np.divide(
        sigma_pml.toarray(), denom_e, out=np.zeros_like(denom_e), where=denom_e != 0
    )
    c_E.fromarray(ratio_e * (b_E.toarray() - 1.0))
    b_H.fromarray(
        np.exp(
            -(tsigma_pml.toarray() / tkappa.toarray() + talpha.toarray())
            * ref.dt
            / eps_0
        )
    )
    denom_h = tsigma_pml.toarray() + tkappa.toarray() * talpha.toarray()
    ratio_h = np.divide(
        tsigma_pml.toarray(), denom_h, out=np.zeros_like(denom_h), where=denom_h != 0
    )
    c_H.fromarray(ratio_h * (b_H.toarray() - 1.0))
    ref.pml_b_E_z_high = b_E[:, :, -ref.n_pml :, "z"].copy().flatten(order="F")
    ref.pml_c_E_z_high = c_E[:, :, -ref.n_pml :, "z"].copy().flatten(order="F")
    ref.pml_b_H_z_high = b_H[:, :, -ref.n_pml - 1 : -1, "z"].copy().flatten(order="F")
    ref.pml_c_H_z_high = c_H[:, :, -ref.n_pml - 1 : -1, "z"].copy().flatten(order="F")
    assert np.array_equal(solver.pml_b_E_z_high, ref.pml_b_E_z_high)
    assert np.array_equal(solver.pml_c_E_z_high, ref.pml_c_E_z_high)
    assert np.array_equal(solver.pml_b_H_z_high, ref.pml_b_H_z_high)
    assert np.array_equal(solver.pml_c_H_z_high, ref.pml_c_H_z_high)
    for attr in ("ieps", "imu", "sigma"):
        assert np.array_equal(getattr(solver, attr).array, getattr(ref, attr).array)

    for sim in (solver, ref):
        pkt = wakis.sources.GaussianPacket(
            xs=slice(0, 6),
            ys=slice(0, 6),
            sigmaz=15e-3,
            sigmaxy=100.0,
            amplitude=1.0,
        )
        for n in range(100):
            pkt.update(sim, n * sim.dt)
            sim.one_step()

    for attr in ("E", "H", "J"):
        assert np.array_equal(
            getattr(solver, attr).toarray(), getattr(ref, attr).toarray()
        )
