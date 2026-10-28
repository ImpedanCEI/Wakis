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
    WP = np.array([-9.43204400e-17, -1.90365527e-14, -2.92098926e-12, -2.85649944e-10, -1.78582417e-08,
        -7.14906330e-07, -1.83392085e-05, -3.01426854e-04, -3.17124910e-03, -2.13046180e-02,
        -9.08548431e-02, -2.41925939e-01, -3.80559683e-01, -2.69947594e-01,  1.63552484e-01,
        5.86233036e-01,  5.77837028e-01,  1.40175729e-01, -3.14615802e-01, -4.13608670e-01,
        -1.00259426e-01,  3.47120168e-01,  4.55227499e-01,  3.11737173e-02, -4.67514531e-01,
        -4.14366479e-01,  1.31619287e-01,  4.91157460e-01,  2.79639144e-01, -1.96121822e-01,
        -4.33361687e-01, -2.40169019e-01,  2.03893734e-01,  4.73462744e-01,  2.26083771e-01,
        -3.22596263e-01, -5.13696302e-01, -8.56471769e-02,  4.27719205e-01,  4.18206200e-01,
        -3.63445946e-02, -4.03638371e-01, -3.49016181e-01,  4.74557712e-02,  4.22252654e-01,
        3.69339914e-01, -1.28688767e-01, -5.17714812e-01, -2.93137982e-01,  2.82322410e-01,
        4.99788647e-01,  1.44796399e-01, -3.21144378e-01, -4.20032385e-01, -1.05631660e-01,
        3.20520482e-01,  4.44116730e-01,  7.36648767e-02, -4.29698460e-01, -4.47054923e-01,
        7.87633249e-02,  4.98877691e-01,  3.20857021e-01, -1.86007075e-01, -4.44748516e-01,
        -2.44738341e-01,  1.88207167e-01,  4.50785379e-01,  2.47137010e-01, -2.72787020e-01,
        -5.16705090e-01, -1.42974285e-01,  4.06502758e-01,  4.57450939e-01, -8.26470444e-03,
        -4.13603514e-01, -3.59355328e-01,  4.01192879e-02,  4.00341972e-01,  3.68574556e-01,
        -8.31971068e-02, -4.92510661e-01, -3.37098722e-01,  2.36267291e-01,  5.20865204e-01,
        1.88336874e-01, -3.19016365e-01, -4.37402240e-01, -1.12292092e-01,  3.06634588e-01,
        4.30284994e-01,  1.01771034e-01, -3.86683617e-01, -4.64744983e-01,  2.07866357e-02,
        4.91607525e-01],)

    Z = np.array([-5.11677666e+00+0.00000000e+00j,  8.73892985e-01+5.63563600e+00j,
        -2.28832842e+00+3.13690507e+00j, -4.27775052e+00+1.34826455e+01j,
        2.32265208e+00+1.33075604e+01j, -5.36267199e+00+1.46745185e+01j,
        -1.17596870e+00+2.57011993e+01j,  1.43286436e+00+2.04845281e+01j,
        -7.37521234e+00+2.87768839e+01j,  2.88960621e+00+3.60127830e+01j,
        -1.95618542e+00+2.90948093e+01j, -6.92144886e+00+4.49026427e+01j,
        6.38978545e+00+4.41579976e+01j, -7.26152509e+00+4.10591980e+01j,
        -3.07243376e+00+6.18633871e+01j,  7.44714943e+00+5.07879247e+01j,
        -1.29997260e+01+5.81965686e+01j,  4.34754340e+00+7.79792904e+01j,
        4.12325513e+00+5.77697082e+01j, -1.67376261e+01+8.19164336e+01j,
        1.43750017e+01+9.13830912e+01j, -5.17229843e+00+6.85647141e+01j,
        -1.51062502e+01+1.12874268e+02j,  2.44335268e+01+1.00558398e+02j,
        -2.10994911e+01+8.87858550e+01j, -3.86200697e+00+1.50623360e+02j,
        2.96516670e+01+1.05310763e+02j, -4.24988955e+01+1.27313834e+02j,
        2.21123872e+01+1.93328414e+02j,  2.15224221e+01+1.08750824e+02j,
        -6.47476993e+01+1.99338412e+02j,  6.93474442e+01+2.37705189e+02j,
        -1.59992989e+01+1.22515304e+02j, -7.53649158e+01+3.37412095e+02j,
        1.49347702e+02+2.79790879e+02j, -1.23960621e+02+1.87698556e+02j,
        -3.32299509e+01+6.51701880e+02j,  3.06456535e+02+3.21045033e+02j,
        -5.31204923e+02+5.59995809e+02j,  4.55620257e+02+2.30249838e+03j,
        3.18947850e+03+8.96968914e+02j,  1.88482611e+03-2.27477722e+03j,
        -7.86071167e+02-1.28901014e+03j,  5.66069741e+01+2.06870910e+02j,
        3.59624570e+02-6.85189780e+02j, -4.56738769e+02-2.45350894e+02j,
        2.43124665e+02+1.63630876e+02j,  7.56280842e+01-4.86018475e+02j,
        -3.17017533e+02+9.04137214e+01j,  3.20904857e+02+6.96196841e+01j],)
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
        tol = dict(rtol=50 * 1e-5, atol=50 * 1e-5)
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

        assert np.array_equal(grid.x, grid2.x), "Grid load_from_h5 x-coords failed"
        assert np.array_equal(grid.y, grid2.y), "Grid load_from_h5 y-coords failed"
        assert np.array_equal(grid.z, grid2.z), "Grid load_from_h5 z-coords failed"
        assert np.array_equal(grid.grid["cavity"], grid2.grid["cavity"]), (
            "Grid load_from_h5 solid mask failed"
        )
        os.remove("tests/001_grid.h5")
