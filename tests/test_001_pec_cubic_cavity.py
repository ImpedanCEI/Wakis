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
    WP = np.array([-7.08304758e-17, -1.53286144e-14, -2.51157713e-12, -2.58057892e-10,
     -1.67069565e-08, -6.84194587e-07, -1.77756099e-05, -2.93657067e-04,
     -3.09017657e-03, -2.07218728e-02, -8.82194486e-02, -2.34491781e-01,
     -3.65597969e-01, -2.43989651e-01,  1.95971006e-01,  5.94520960e-01,
      5.32158405e-01,  5.59035861e-02, -3.93277829e-01, -4.31421211e-01,
     -1.63289300e-02,  4.59573171e-01,  4.41674886e-01, -1.10776868e-01,
     -5.43998899e-01, -3.28029500e-01,  2.41762774e-01,  4.99485106e-01,
      2.24978139e-01, -2.58749189e-01, -4.86824939e-01, -2.08167504e-01,
      3.40253660e-01,  5.33577761e-01,  9.21344434e-02, -4.61362282e-01,
     -4.56908528e-01,  5.07685585e-02,  4.55829551e-01,  3.70923844e-01,
     -8.16886268e-02, -4.68703429e-01, -3.73635032e-01,  1.69518576e-01,
      5.52543102e-01,  2.83113153e-01, -3.16622982e-01, -5.20050290e-01,
     -1.42637519e-01,  3.45198032e-01,  4.58430761e-01,  1.09917169e-01,
     -3.78756702e-01, -4.89150318e-01, -2.85398251e-02,  4.96586052e-01,
      4.34496367e-01, -1.31218235e-01, -5.07964340e-01, -3.10578757e-01,
      1.83659114e-01,  4.75190237e-01,  2.87511421e-01, -2.29918698e-01,
     -5.36588868e-01, -2.26214909e-01,  3.72473423e-01,  5.26147279e-01,
      6.85649304e-02, -4.24227262e-01, -4.28388387e-01, -4.22962310e-03,
      4.17990661e-01,  4.25136357e-01, -4.31951772e-02, -5.09388498e-01,
     -3.94724837e-01,  1.98281974e-01,  5.44240108e-01,  2.54288588e-01,
     -2.80659125e-01, -4.79855734e-01, -1.90754908e-01,  2.95657017e-01,
      5.01555200e-01,  1.54814641e-01, -4.10018607e-01, -5.11062080e-01,
     -1.67075880e-03,  4.87958148e-01,  3.99079518e-01, -9.94459700e-02,
     -4.56963961e-01, -3.48344120e-01,  1.25541748e-01])

    Z = np.array([ 1.53568977e+01+0.00000000e+00j, -9.96110881e+00-8.85229549e+00j,
     -3.02011642e+00+2.16031832e+01j,  1.41171022e+01+1.74865910e+00j,
     -1.48255643e+01+5.93489169e+00j,  4.49987147e+00+3.11650309e+01j,
      9.56737083e+00+4.91513843e+00j, -1.70061388e+01+2.30505725e+01j,
      1.19912444e+01+3.78420690e+01j,  2.37827062e+00+1.06359873e+01j,
     -1.58365400e+01+4.16563940e+01j,  1.81811625e+01+4.18506262e+01j,
     -6.80448504e+00+2.03118992e+01j, -1.07667972e+01+6.08922982e+01j,
      2.16973301e+01+4.37963115e+01j, -1.69609342e+01+3.54237529e+01j,
     -1.47340444e+00+7.96872110e+01j,  2.09742702e+01+4.48553604e+01j,
     -2.65623655e+01+5.75882941e+01j,  1.20172211e+01+9.67472850e+01j,
      1.42157241e+01+4.70942838e+01j, -3.33538101e+01+8.86359877e+01j,
      2.91118422e+01+1.10582272e+02j, -6.60049027e-01+5.40634273e+01j,
     -3.39631161e+01+1.30806196e+02j,  4.82572963e+01+1.19608184e+02j,
     -2.61647379e+01+7.20441273e+01j, -2.30601535e+01+1.87319047e+02j,
      6.61213320e+01+1.22427254e+02j, -6.57309286e+01+1.13192245e+02j,
      8.92708932e+00+2.64242446e+02j,  7.54072501e+01+1.18615185e+02j,
     -1.25654005e+02+2.05726697e+02j,  8.41631338e+01+3.77880943e+02j,
      5.63751269e+01+1.11632375e+02j, -2.25305516e+02+4.42685115e+02j,
      2.86600761e+02+5.99290651e+02j, -8.67729910e+01+1.29570956e+02j,
     -5.18675096e+02+1.54531344e+03j,  2.07449938e+03+2.24499354e+03j,
      3.26391770e+03-9.90125612e+02j,  2.77419301e+02-2.39667987e+03j,
     -6.97725592e+02-2.67357491e+02j,  5.35889130e+02-1.76833749e+02j,
     -1.27962664e+02-7.71159543e+02j, -2.47380857e+02+1.14992999e+02j,
      3.87671582e+02-2.06130442e+02j, -2.47672742e+02-4.10455256e+02j,
     -4.92012589e+01+2.33832605e+02j,  2.92555742e+02-2.61526720e+02j])
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
            1383.0255747420938, 0.1
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
        assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(378608.1621916529, 0.1), (
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
