import os
import sys

import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


@pytest.mark.slow
class TestPecCubicCavity:
    # Float32 regression samples on the boundary-inclusive symmetric grid.
    # fmt: off
    WP = np.array([-9.84762617e-17, -2.21797690e-14, -3.58136978e-12, -3.64847913e-10,
     -2.35042707e-08, -9.58830151e-07, -2.47806631e-05, -4.05656485e-04,
     -4.20169170e-03, -2.74697747e-02, -1.12678756e-01, -2.84932342e-01,
     -4.17249416e-01, -2.52058336e-01,  2.35688500e-01,  6.48885216e-01,
      5.78352481e-01,  8.50462152e-02, -3.86531407e-01, -4.54228905e-01,
     -6.41301582e-02,  4.35260732e-01,  4.91953068e-01, -2.96645987e-02,
     -5.41425783e-01, -4.16498978e-01,  1.73835115e-01,  5.24408759e-01,
      3.01899997e-01, -1.96814425e-01, -4.87052035e-01, -2.99862338e-01,
      2.28657881e-01,  5.57097911e-01,  2.56193527e-01, -3.66540653e-01,
     -5.58486927e-01, -1.06097013e-01,  4.21106993e-01,  4.61789879e-01,
      4.62818892e-02, -3.98141089e-01, -4.59319565e-01, -3.37377551e-02,
      4.79939894e-01,  4.74881586e-01, -9.69699752e-02, -5.56920175e-01,
     -3.61438667e-01,  2.10384436e-01,  5.06425481e-01,  2.77607240e-01,
     -2.09653494e-01, -5.01841972e-01, -2.86450379e-01,  2.80646732e-01,
      5.72404285e-01,  1.96919905e-01, -4.12904806e-01, -5.24630159e-01,
     -5.51322735e-02,  4.22372351e-01,  4.40260552e-01,  3.17071048e-02,
     -4.19107090e-01, -4.64766894e-01,  1.16154105e-02,  5.22435508e-01,
      4.40131594e-01, -1.61706148e-01, -5.54573053e-01, -3.08880197e-01,
      2.32003206e-01,  4.92555396e-01,  2.64167454e-01, -2.30810043e-01,
     -5.24441016e-01, -2.59568155e-01,  3.39510763e-01,  5.70823939e-01,
      1.30567052e-01, -4.42213231e-01, -4.84595831e-01, -1.88999447e-02,
      4.20928094e-01,  4.31061002e-01,  1.44292157e-02, -4.52006742e-01,
     -4.62326244e-01,  7.03782654e-02,  5.54270771e-01,  3.90664787e-01,
     -2.15774227e-01, -5.38518486e-01, -2.66142543e-01])

    Z = np.array([ 1.81934847e+01-0.00000000e+00j, -1.21053587e+01-1.01553777e+01j,
     -2.68658349e+00+2.57680541e+01j,  1.59797783e+01+1.03276374e+00j,
     -1.81197979e+01+8.97351457e+00j,  7.56429752e+00+3.62221968e+01j,
      8.80701076e+00+4.65924720e+00j, -1.94038830e+01+3.09456946e+01j,
      1.68046153e+01+4.19068238e+01j, -1.79413161e+00+1.29674562e+01j,
     -1.51059333e+01+5.36433485e+01j,  2.25975149e+01+4.37630253e+01j,
     -1.38347099e+01+2.78609294e+01j, -5.16719514e+00+7.48157279e+01j,
      2.27361625e+01+4.37696583e+01j, -2.46146055e+01+5.06507740e+01j,
      9.45414990e+00+9.21564391e+01j,  1.54481043e+01+4.50792578e+01j,
     -3.08073187e+01+8.18849064e+01j,  2.65777118e+01+1.03637457e+02j,
     -2.86550050e-01+5.22319904e+01j, -2.85554554e+01+1.21184147e+02j,
      4.25227104e+01+1.08003674e+02j, -2.43232301e+01+7.15889932e+01j,
     -1.35242013e+01+1.67131507e+02j,  5.16170049e+01+1.05582116e+02j,
     -5.47544603e+01+1.12413354e+02j,  1.92713466e+01+2.17299432e+02j,
      4.50133888e+01+9.98838183e+01j, -8.67928270e+01+1.90074886e+02j,
      7.65649162e+01+2.68665025e+02j,  6.86515894e+00+1.01853781e+02j,
     -1.09656533e+02+3.37795270e+02j,  1.72282851e+02+3.19494294e+02j,
     -1.03087892e+02+1.47073993e+02j, -9.15326613e+01+6.69385507e+02j,
      3.65721223e+02+3.81316050e+02j, -5.06668888e+02+4.44459550e+02j,
      2.42895428e+02+2.32556219e+03j,  3.20922117e+03+1.21774668e+03j,
      2.23516553e+03-2.23333726e+03j, -7.07153314e+02-1.54883320e+03j,
     -6.62578220e+01+1.79962309e+02j,  4.26703987e+02-6.45034749e+02j,
     -4.39065696e+02-3.49019851e+02j,  1.80435077e+02+1.73531152e+02j,
      1.35123172e+02-4.75941547e+02j, -3.28847299e+02+1.40124674e+01j,
      2.88372432e+02+9.78423379e+01j, -6.36610831e+01-3.63446121e+02j])
    # fmt: on

    dtype = np.float32

    def test_simulation(self, use_gpu):
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
        skip_cells = 12  # no. cells to skip in WP integration
        wakelength = 1.0  # [m]
        wake = WakeSolver(
            wakelength=wakelength,
            q=q,
            sigmaz=sigmaz,
            beta=beta,
            xsource=xs,
            ysource=ys,
            xtest=xt,
            ytest=yt,
            save=False,
            Ez_file="tests/001_Ez.h5",
            skip_cells=skip_cells,
            verbose=2,
        )

        # boundary conditions
        bc_low = ["pec", "pec", "pec"]
        bc_high = ["pec", "pec", "pec"]

        # set Solver object
        solver = SolverFIT3D(
            grid,
            wake,
            bc_low=bc_low,
            bc_high=bc_high,
            use_stl=True,
            bg="pec",
            dtype=self.dtype,
            use_gpu=use_gpu,
            verbose=2,
        )

        solver.wakesolve(wakelength=wakelength, save_J=False)
        os.remove("tests/001_Ez.h5")

    def test_long_wake_potential(self, plot_comparison):
        global wake
        tol = dict(rtol=50 * 1e-5, atol=50 * 1e-5)
        plot_comparison(wake.WP[::50], self.WP, "Wake potential")
        assert np.allclose(wake.WP[::50], self.WP, **tol), (
            "Wake potential samples failed"
        )
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
            1325.6968037037557, 0.1
        ), "Wake potential cumsum failed"

    def test_long_impedance(self, plot_comparison):
        global wake
        # Off-resonance bins need an absolute error budget scaled to the
        # resonance, rather than their small baseline, across NumPy versions.
        tol = dict(rtol=5e-4, atol=5e-4 * np.max(np.abs(self.Z)))
        plot_comparison(np.abs(wake.Z)[::20], np.abs(self.Z), "Impedance magnitude")
        assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z), **tol), (
            "Abs Impedance samples failed"
        )
        plot_comparison(np.real(wake.Z)[::20], np.real(self.Z), "Impedance real part")
        assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z), **tol), (
            "Real Impedance samples failed"
        )
        plot_comparison(
            np.imag(wake.Z)[::20], np.imag(self.Z), "Impedance imaginary part"
        )
        assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z), **tol), (
            "Imag Impedance samples failed"
        )
        assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(
            372019.59123029554, 0.1
        ), "Abs Impedance cumsum failed"

    def test_grid_save_to_h5(self):
        global grid
        grid.save_to_h5("tests/001_grid.h5")
        assert os.path.exists("tests/001_grid.h5"), "Grid save_to_h5 failed"

    def test_grid_load_from_h5(self):
        global grid
        grid2 = GridFIT3D(load_from_h5="tests/001_grid.h5", verbose=2)

        assert (grid2.Nx, grid2.Ny, grid2.Nz) == (grid.Nx, grid.Ny, grid.Nz)
        for metric in ("L", "iA", "tL", "itA"):
            np.testing.assert_allclose(
                getattr(grid2, metric).toarray(), getattr(grid, metric).toarray()
            )

        assert np.array_equal(grid.x, grid2.x), "Grid load_from_h5 x-coords failed"
        assert np.array_equal(grid.y, grid2.y), "Grid load_from_h5 y-coords failed"
        assert np.array_equal(grid.z, grid2.z), "Grid load_from_h5 z-coords failed"
        assert np.array_equal(grid.grid["cavity"], grid2.grid["cavity"]), (
            "Grid load_from_h5 solid mask failed"
        )
        os.remove("tests/001_grid.h5")
