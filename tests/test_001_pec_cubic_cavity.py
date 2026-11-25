import os
import sys

import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest
from scipy.constants import c

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


@pytest.mark.slow
class TestPecCubicCavity:
    # Float32 regression samples with staggered primal/dual material masks.
    # fmt: off
    WP = np.array([-6.77452087e-17, -1.45282947e-14, -2.39179071e-12, -2.46912154e-10,
     -1.60600910e-08, -6.60751741e-07, -1.72455231e-05, -2.86201054e-04,
     -3.02538482e-03, -2.03792500e-02, -8.71581720e-02, -2.32793626e-01,
     -3.65108477e-01, -2.46939908e-01,  1.90997329e-01,  5.92355886e-01,
      5.35194351e-01,  6.12428092e-02, -3.90332653e-01, -4.33410390e-01,
     -2.16911803e-02,  4.56452132e-01,  4.44993671e-01, -1.04548644e-01,
     -5.42538833e-01, -3.32893228e-01,  2.36731799e-01,  4.99587852e-01,
      2.29401365e-01, -2.54465199e-01, -4.86817807e-01, -2.13049533e-01,
      3.35447294e-01,  5.34878326e-01,  9.83545430e-02, -4.57880030e-01,
     -4.60073640e-01,  4.52553092e-02,  4.53907676e-01,  3.73978516e-01,
     -7.66994939e-02, -4.66636695e-01, -3.77289962e-01,  1.63656439e-01,
      5.51607955e-01,  2.88499404e-01, -3.11607604e-01, -5.21139314e-01,
     -1.47830961e-01,  3.41613241e-01,  4.59642127e-01,  1.14813647e-01,
     -3.74869946e-01, -4.90991083e-01, -3.46889557e-02,  4.93516810e-01,
      4.38357320e-01, -1.25388200e-01, -5.06902386e-01, -3.14689792e-01,
      1.79005891e-01,  4.74357902e-01,  2.91537203e-01, -2.24736007e-01,
     -5.36285061e-01, -2.31813520e-01,  3.67655834e-01,  5.27994188e-01,
      7.44083360e-02, -4.21252522e-01, -4.30839229e-01, -9.18729356e-03,
      4.15220940e-01,  4.27624060e-01, -3.74234584e-02, -5.06916290e-01,
     -3.99022293e-01,  1.92363786e-01,  5.43873865e-01,  2.59331224e-01,
     -2.76272497e-01, -4.80308799e-01, -1.95226358e-01,  2.91355402e-01,
      5.02063878e-01,  1.60354757e-01, -4.05651481e-01, -5.13473902e-01,
     -7.91223609e-03,  4.85492997e-01,  4.02653781e-01, -9.43625562e-02,
     -4.55387830e-01, -3.51596147e-01,  1.20336828e-01])

    Z = np.array([ 1.54574615e+01-0.00000000e+00j, -1.00283514e+01-8.92673934e+00j,
     -3.02521829e+00+2.17036777e+01j,  1.42058524e+01+1.68490514e+00j,
     -1.49150546e+01+5.91631463e+00j,  4.57918246e+00+3.12534615e+01j,
      9.62594427e+00+4.80718025e+00j, -1.70644042e+01+2.31014803e+01j,
      1.21574060e+01+3.78798312e+01j,  2.40222415e+00+1.05198860e+01j,
     -1.58035164e+01+4.17694856e+01j,  1.84182854e+01+4.18082474e+01j,
     -6.79698184e+00+2.02341326e+01j, -1.05843132e+01+6.10367393e+01j,
      2.19762846e+01+4.36630598e+01j, -1.69218380e+01+3.54320184e+01j,
     -1.09486062e+00+7.98093246e+01j,  2.12623157e+01+4.46484753e+01j,
     -2.64081831e+01+5.77190852e+01j,  1.26168218e+01+9.67753125e+01j,
      1.44919966e+01+4.68671239e+01j, -3.29618224e+01+8.88985702e+01j,
      2.99266819e+01+1.10436314e+02j, -3.80513880e-01+5.39119431e+01j,
     -3.31685963e+01+1.31161769e+02j,  4.92441382e+01+1.19217193e+02j,
     -2.57918270e+01+7.21116659e+01j, -2.16509435e+01+1.87651009e+02j,
      6.71986405e+01+1.21757402e+02j, -6.50251872e+01+1.13675122e+02j,
      1.12321955e+01+2.64302841e+02j,  7.64671581e+01+1.17723686e+02j,
     -1.24041306e+02+2.06893506e+02j,  8.78107275e+01+3.77146723e+02j,
      5.73345546e+01+1.10813508e+02j, -2.21159552e+02+4.44987391e+02j,
      2.92766307e+02+5.96274666e+02j, -8.56692641e+01+1.30187454e+02j,
     -5.02427829e+02+1.55063749e+03j,  2.09820568e+03+2.22188477e+03j,
      3.25160428e+03-1.02571670e+03j,  2.50570666e+02-2.39858181e+03j,
     -7.00363421e+02-2.59647729e+02j,  5.33120289e+02-1.82922948e+02j,
     -1.36745246e+02-7.68995846e+02j, -2.45929745e+02+1.17425702e+02j,
      3.84477141e+02-2.10427616e+02j, -2.52137373e+02-4.07025738e+02j,
     -4.65648469e+01+2.33801097e+02j,  2.88743387e+02-2.64471966e+02j])
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
            1382.5556285373132, 0.1
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
        assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(378471.6286411207, 0.1), (
            "Abs Impedance cumsum failed"
        )

        # The narrow beam pipes perturb the closed-cavity TM110 mode. Its peak
        # should stay within one physical wake-record frequency resolution.
        # Rectangular mode formula: Wolski, CERN Accelerator School (2010).
        # https://cas.web.cern.ch/sites/default/files/lectures/ebeltoft-2010/wolski-2.pdf
        width_x = grid.xmax - grid.xmin
        width_y = grid.ymax - grid.ymin
        expected_frequency = c / 2 * np.sqrt(1 / width_x**2 + 1 / width_y**2)
        peak_frequency = wake.f[np.argmax(np.abs(wake.Z))]
        resolution = wake.v / wake.wakelength
        assert abs(peak_frequency - expected_frequency) < resolution

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
        for masks in ("primal_masks", "dual_masks"):
            np.testing.assert_array_equal(
                getattr(grid, masks)["cavity"].array,
                getattr(grid2, masks)["cavity"].array,
            )
        os.remove("tests/001_grid.h5")
