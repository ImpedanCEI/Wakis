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
    WP = np.array([-9.95458549e-17, -2.05046155e-14, -3.13946919e-12, -3.06897233e-10,
       -1.92003072e-08, -7.69621207e-07, -1.97695468e-05, -3.25246046e-04,
       -3.42179648e-03, -2.29496511e-02, -9.74681785e-02, -2.57605933e-01,
       -4.00433017e-01, -2.77971803e-01,  1.74965578e-01,  6.06108048e-01,
        5.93615999e-01,  1.48213195e-01, -3.13849000e-01, -4.18423138e-01,
       -1.06055653e-01,  3.46718391e-01,  4.61725985e-01,  3.75969195e-02,
       -4.69307323e-01, -4.22517889e-01,  1.27354182e-01,  4.95498135e-01,
        2.86543650e-01, -1.93994231e-01, -4.36868927e-01, -2.46039348e-01,
        2.00696576e-01,  4.77730108e-01,  2.34450998e-01, -3.20973718e-01,
       -5.21458944e-01, -9.21702645e-02,  4.30032192e-01,  4.24952714e-01,
       -3.23271769e-02, -4.05244228e-01, -3.54719092e-01,  4.29930706e-02,
        4.24183971e-01,  3.76540160e-01, -1.24009110e-01, -5.21691624e-01,
       -3.01353114e-01,  2.80371055e-01,  5.06105865e-01,  1.51118487e-01,
       -3.21008872e-01, -4.24863392e-01, -1.10982371e-01,  3.19529863e-01,
        4.50130203e-01,  8.10542918e-02, -4.30801702e-01, -4.55687907e-01,
        7.42336927e-02,  5.03432059e-01,  3.27840927e-01, -1.83622835e-01,
       -4.48129076e-01, -2.50575523e-01,  1.85580579e-01,  4.54734769e-01,
        2.54338635e-01, -2.70248891e-01, -5.22500181e-01, -1.50415469e-01,
        4.07080605e-01,  4.65165698e-01, -3.17568505e-03, -4.15589815e-01,
       -3.64899855e-01,  3.59083120e-02,  4.01596031e-01,  3.75713287e-01,
       -7.75370758e-02, -4.96148372e-01, -3.45729644e-01,  2.34144828e-01,
        5.27290473e-01,  1.94868331e-01, -3.18596495e-01, -4.42170990e-01,
       -1.17578617e-01,  3.06112222e-01,  4.35816760e-01,  1.08211242e-01,
       -3.86427725e-01, -4.71772210e-01,  1.47990082e-02,  4.94782071e-01],)

    Z = np.array([-5.24909426e+00   +0.j,  8.16923909e-01   +6.02403378j,
       -2.06427345e+00   +3.71312638j, -4.10217932e+00  +13.95118407j,
        2.23378636e+00  +14.14187912j, -5.17574116e+00  +15.86678186j,
       -1.00358385e+00  +26.794005j  ,  1.37326004e+00  +21.99028187j,
       -7.10389218e+00  +30.57858605j,  3.05622057e+00  +37.71516299j,
       -1.95199677e+00  +31.30903732j, -6.53120427e+00  +47.29988326j,
        6.54902893e+00  +46.49366121j, -7.14109397e+00  +44.00962949j,
       -2.54551548e+00  +64.83955663j,  7.60232249e+00  +53.80472334j,
       -1.26928769e+01  +61.90508678j,  5.01436094e+00  +81.51964238j,
        4.29051965e+00  +61.54320948j, -1.61542604e+01  +86.39397846j,
        1.51663301e+01  +95.48299727j, -4.94965599e+00  +73.20543895j,
       -1.41345777e+01 +118.11203214j,  2.53102053e+01 +105.23943652j,
       -2.07262234e+01  +94.44891956j, -2.36795665e+00 +156.58071027j,
        3.05472038e+01 +110.65179411j, -4.17823007e+01 +134.21338547j,
        2.42858032e+01 +199.91670612j,  2.23451222e+01 +114.95217815j,
       -6.33013844e+01 +207.77796539j,  7.23884153e+01 +244.76270925j,
       -1.53485068e+01 +130.05803814j, -7.23435365e+01 +347.87989904j,
        1.53530144e+02 +287.02245535j, -1.23514190e+02 +197.90812683j,
       -2.61806726e+01 +665.33665972j,  3.12540412e+02 +327.7072119j ,
       -5.30405716e+02 +579.26803187j,  4.86001847e+02+2327.59151824j,
        3.23055241e+03 +878.05386917j,  1.87824258e+03-2309.86790488j,
       -8.08239099e+02-1285.08243448j,  5.88954657e+01 +217.25217304j,
        3.56057712e+02 -686.69210039j, -4.64168091e+02 -233.95310497j,
        2.46994361e+02 +172.91480762j,  7.19317366e+01 -481.61037549j,
       -3.19808635e+02 +104.88892043j,  3.25327940e+02  +79.96383153j],)
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
        wake = WakeSolver(
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
        )

        wakelength = 1.0  # [m]
        solver.wakesolve(wakelength=wakelength, save_J=False)
        os.remove("tests/001_Ez.h5")

    def test_long_wake_potential(self):
        global wake
        tol = dict(rtol=50 * 1e-6, atol=50 * 1e-6)
        assert np.allclose(wake.WP[::50], self.WP, **tol), (
            "Wake potential samples failed"
        )
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
            1325.6968037037557, 0.1
        ), "Wake potential cumsum failed"

    def test_long_impedance(self):
        global wake
        tol = dict(rtol=50 * 1e-5, atol=50 * 1e-5)
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

        assert np.array_equal(grid.x, grid2.x), (
            "Grid load_from_h5 x-coords failed"
        )
        assert np.array_equal(grid.y, grid2.y), (
            "Grid load_from_h5 y-coords failed"
        )
        assert np.array_equal(grid.z, grid2.z), (
            "Grid load_from_h5 z-coords failed"
        )
        assert np.array_equal(grid.grid["cavity"], grid2.grid["cavity"]), (
            "Grid load_from_h5 solid mask failed"
        )
        os.remove("tests/001_grid.h5")
