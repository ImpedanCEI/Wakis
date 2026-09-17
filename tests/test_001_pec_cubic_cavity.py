import os
import sys

import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


@pytest.mark.slow
class TestPecCubicCavity:
    # Regression data
    # fmt: off
    WP = np.array([-9.44332595e-17, -1.90593430e-14, -2.92432824e-12, -2.85966621e-10,
        -1.78775256e-08, -7.15669085e-07, -1.83583718e-05, -3.01738609e-04,
        -3.17451120e-03, -2.13264221e-02, -9.09471961e-02, -2.42170990e-01,
        -3.80942164e-01, -2.70215476e-01,  1.63723476e-01,  5.86828703e-01,
        5.78425842e-01,  1.40326739e-01, -3.14918593e-01, -4.14008552e-01,
        -1.00341426e-01,  3.47488664e-01,  4.55707614e-01,  3.12301510e-02,
        -4.67954590e-01, -4.14751349e-01,  1.31785366e-01,  4.91686569e-01,
        2.79958815e-01, -1.96277565e-01, -4.33751390e-01, -2.40362919e-01,
        2.04148362e-01,  4.73990162e-01,  2.26365693e-01, -3.22863060e-01,
        -5.14151085e-01, -8.56697123e-02,  4.28214666e-01,  4.18693782e-01,
        -3.63103145e-02, -4.03970520e-01, -3.49289428e-01,  4.75824918e-02,
        4.22758055e-01,  3.69793808e-01, -1.28731294e-01, -5.18144741e-01,
        -2.93339490e-01,  2.82700986e-01,  5.00387114e-01,  1.45041055e-01,
        -3.21364175e-01, -4.20348811e-01, -1.05629844e-01,  3.20952689e-01,
        4.44675483e-01,  7.38541659e-02, -4.30011048e-01, -4.47382113e-01,
        7.89662940e-02,  4.99504584e-01,  3.21308113e-01, -1.86062433e-01,
        -4.45060460e-01, -2.44846000e-01,  1.88536046e-01,  4.51379995e-01,
        2.47529690e-01, -2.72913212e-01, -5.17073038e-01, -1.42964155e-01,
        4.07066400e-01,  4.58068049e-01, -8.11280162e-03, -4.13854650e-01,
        -3.59550001e-01,  4.03289095e-02,  4.00914687e-01,  3.69118465e-01,
        -8.31046304e-02, -4.92825246e-01, -3.37255129e-01,  2.36689164e-01,
        5.21574819e-01,  1.88715987e-01, -3.19144678e-01, -4.37645870e-01,
        -1.12207153e-01,  3.07143625e-01,  4.30919175e-01,  1.02079124e-01,
        -3.86863965e-01, -4.64999222e-01,  2.10214892e-02,  4.92317381e-01],)

    Z = np.array([-5.12228082e+00   +0.j,   8.74802421e-01   +5.64167799j,
        -2.29034622e+00   +3.14009729j, -4.28204864e+00  +13.49615494j,
        2.32502792e+00  +13.321102j,   -5.36792535e+00  +14.68933047j,
        -1.17714699e+00  +25.72710812j,  1.43437646e+00  +20.50530279j,
        -7.38252819e+00  +28.80590918j,  2.89251921e+00  +36.04913828j,
        -1.95806099e+00  +29.12426822j, -6.92833532e+00  +44.94792056j,
        6.39622577e+00  +44.20261146j, -7.26872064e+00  +41.10070955j,
        -3.07547564e+00  +61.92576056j,  7.45466785e+00  +50.83926611j,
        -1.30126832e+01  +58.25532521j,  4.35193653e+00  +78.05791427j,
        4.12745568e+00  +57.82811987j, -1.67543448e+01  +81.99904217j,
        1.43894531e+01  +91.47524504j, -5.17740910e+00  +68.63401325j,
        -1.51213462e+01 +112.98799433j,  2.44580766e+01 +100.65983537j,
        -2.11205720e+01  +88.87548838j, -3.86583900e+00 +150.7750233j,
        2.96814665e+01 +105.41704079j, -4.25414436e+01 +127.44216196j,
        2.21346007e+01 +193.52299164j,  2.15440874e+01 +108.86061213j,
        -6.48125798e+01 +199.53903442j,  6.94170396e+01 +237.94436543j,
        -1.60152482e+01 +122.63894239j, -7.54404758e+01 +337.75127188j,
        1.49497564e+02 +280.07236224j, -1.24084853e+02 +187.88759905j,
        -3.32632999e+01 +652.35640735j,  3.06764067e+02 +321.36800633j,
        -5.31737679e+02 +560.55823748j,  4.56077028e+02+2304.80902956j,
        3.19267822e+03 +897.87015732j,  1.88671765e+03-2277.05835854j,
        -7.86859618e+02-1290.30287787j,  5.66636366e+01 +207.07935809j,
        3.59985611e+02 -685.87627934j, -4.57196960e+02 -245.5963575j,
        2.43368571e+02 +163.79602585j,  7.57041898e+01 -486.50523561j,
        -3.17335611e+02  +90.50517683j,  3.21226858e+02  +69.69050002j],)
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

    def test_long_wake_potential(self):
        global wake
        tol = dict(rtol=50 * 1e-4, atol=50 * 1e-4)
        assert np.allclose(wake.WP[::50], self.WP, **tol), (
            "Wake potential samples failed"
        )
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
            1325.6968037037557, 0.1
        ), "Wake potential cumsum failed"

    def test_long_impedance(self):
        global wake
        tol = dict(rtol=50 * 1e-4, atol=50 * 1e-4)
        assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z), **tol), (
            "Abs Impedance samples failed"
        )
        assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z), **tol), (
            "Real Impedance samples failed"
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

        assert np.array_equal(grid.x, grid2.x), "Grid load_from_h5 x-coords failed"
        assert np.array_equal(grid.y, grid2.y), "Grid load_from_h5 y-coords failed"
        assert np.array_equal(grid.z, grid2.z), "Grid load_from_h5 z-coords failed"
        assert np.array_equal(grid.grid["cavity"], grid2.grid["cavity"]), (
            "Grid load_from_h5 solid mask failed"
        )
        os.remove("tests/001_grid.h5")
