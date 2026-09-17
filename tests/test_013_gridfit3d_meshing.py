import sys

import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


@pytest.mark.slow
class TestGridFIT3DMeshing:
    # Reference data
    tol = dict(rtol=50e-5, atol=50e-4)
    dtype = np.float32

    # fmt: off
    WP = np.array([ 3.03999377e-18, -1.57436678e-14, -5.81744067e-11, -4.16962878e-08,
                    -8.37576052e-06, -5.10188367e-04, -9.65024317e-03, -5.62474460e-02,
                    -8.92658495e-02,  1.94104416e-02,  1.20731894e-01,  6.94716656e-02,
                    -2.25460627e-02, -4.91365346e-02, -2.81756714e-02, -1.07040993e-02,
                    6.24067287e-02,  7.76899724e-02, -6.41804617e-02, -1.09100607e-01,
                    3.17122367e-02,  9.95300771e-02,  1.59930539e-02, -6.11959223e-02,
                    -3.38419464e-02, -3.31974531e-03,  1.97809615e-02,  7.32837232e-02,
                    2.14655914e-03, -9.54641911e-02, -3.93029761e-02,  7.85793202e-02,
                    6.92603412e-02, -4.44469131e-02, -5.61724850e-02, -4.31207455e-03,
                    1.37319512e-02,  4.47538136e-02,  3.30728875e-02, -4.91526078e-02,
                    -7.17331871e-02,  2.30389026e-02,  9.07601155e-02,  4.60074950e-03,
                    -6.82039506e-02, -2.42327580e-02,  1.47484248e-02,  3.33755469e-02,
                    3.07827068e-02, -1.10044189e-02, -6.00957767e-02, -2.83398905e-02,
                    7.01179625e-02,  5.07391705e-02, -4.55863327e-02, -5.23725044e-02,
                    3.64085044e-03,  3.55489172e-02,  2.52530595e-02,  3.40678159e-03,
                    -3.39325723e-02, -4.70488102e-02,  2.66807614e-02,  6.77546427e-02,
                    -3.49550941e-03, -5.86094480e-02, -2.46501822e-02,  3.33310938e-02,
                    3.19647879e-02,  3.80802801e-03, -1.52754979e-02, -4.11351141e-02,
                    -7.28641157e-03,  5.23813004e-02,  3.22026092e-02, -3.76132883e-02,
                    -4.92387915e-02,  1.42774059e-02,  4.05807388e-02,  1.10590376e-02,
                    -1.09048635e-02, -2.85400383e-02, -2.08799885e-02,  2.56681667e-02,
                    4.33091263e-02, -4.11216871e-03, -5.22413075e-02, -1.57761296e-02,
                    3.85064574e-02,  2.44861946e-02, -5.94644029e-03, -2.37867277e-02,
                    -2.08974540e-02,  6.12977218e-03,  3.41808685e-02,  2.11225797e-02,
                    -3.48074853e-02, -3.68432258e-02,  1.84538236e-02,  3.62614155e-02,
                    5.71392888e-03, -2.22282929e-02, -2.07595900e-02, -2.14059628e-03])


    Z = np.array([ 4.87770399e+00   -0.j,         -9.37231333e-01   +9.76902j,
                    -4.24056580e-02   +5.75895605j,  7.64718800e+00  +15.05633851j,
                    2.25805798e-02  +22.94919652j,  1.81227395e+00  +18.44213153j,
                    9.25440294e+00  +29.07627434j,  2.80600688e-01  +36.56571882j,
                    3.21773007e+00  +31.43991496j,  1.09670695e+01  +44.23182763j,
                    4.43604660e-02  +51.73186851j,  4.57961315e+00  +45.61130327j,
                    1.31989216e+01  +61.78568579j, -7.53366266e-01  +69.74316051j,
                    6.17983511e+00  +62.00941222j,  1.65219994e+01  +83.7460061j,
                    -2.43993772e+00  +92.90805422j,  8.45866414e+00  +82.50488264j,
                    2.21727722e+01 +114.27862094j, -5.88329858e+00 +126.41846549j,
                    1.25234418e+01 +111.41469792j,  3.38300774e+01 +164.58570432j,
                    -1.36120025e+01 +186.02864687j,  2.28328211e+01 +163.1007426j,
                    6.98071354e+01 +281.32459768j, -3.41036324e+01 +359.14919895j,
                    9.25735014e+01 +348.74599877j,  8.21857579e+02+1199.51083954j,
                    1.12890901e+03-1003.57407371j, -1.01266995e+01 -186.88815086j,
                    -4.70789813e+01 -262.98926442j,  1.12077022e+02  -91.51376311j,
                    -1.54554302e+01  +54.34806626j, -2.88613554e+01  -36.43009739j,
                    8.69109385e+01  +55.12056128j, -1.08808342e+01 +164.39050487j,
                    -1.17645744e+01  +86.56414324j,  1.18563803e+02 +213.48816771j,
                    1.20556558e+01 +395.18631788j,  9.57219766e+01 +399.30569715j,
                    1.16332782e+03 +970.86145883j,  1.04839496e+03-1020.32813672j,
                    6.54871934e+01 -429.14499028j,  2.24857075e+01 -363.85175326j,
                    6.78880242e+01 -188.44076494j, -1.98911757e+01 -125.63772553j,
                    2.05538171e+01 -141.29209289j,  3.79217091e+01  -41.08975005j,
                    -2.73138490e+01  -38.01589641j,  3.61895934e+01  -59.80608412j])

    #Ez = np.array([])

    # fmt: on
    def test_voxelize_rectilinear(self, use_gpu):
        """
        Tests 'voxelize_rectilinear' and subpixel smoothing using the
        exact cavity and shell gridLogs configuration.
        """

        # Geometry & Materials
        solid_1 = "tests/stl/007_vacuum_cavity.stl"  # logs["stl_solids"]["cavity"]
        solid_2 = "tests/stl/007_lossymetal_shell.stl"  # logs["stl_solids"]["shell"]

        stl_solids = {"cavity": solid_1, "shell": solid_2}

        stl_materials = {
            "cavity": "vacuum",
            "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
        }

        # Extract domain bounds from geometry
        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

        # Number of mesh cells
        Nx = 60  # logs["Nx"]
        Ny = 60  # logs["Ny"]
        Nz = 140  # logs["Nz"]

        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,  # Global domain zmin
            zmax,  # Global domain zmax
            Nx,
            Ny,
            Nz,  # Global domain Nz
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_method="voxelize_rectilinear",
            subpixel_smoothing=False,
            stl_scale=1.0,
            stl_rotate=[0, 0, 0],
            stl_translate=[0, 0, 0],
            verbose=1,
        )

        # number of cells in the mask
        n_inside = grid.grid.threshold(scalars="shell", value=0.5).n_cells
        n_inside_expected = 61258
        assert n_inside == n_inside_expected, (
            f"Number of cells masked inside the shell is {n_inside}, expected {n_inside_expected}"
        )

        # volume
        vol = n_inside * np.min(grid.dx) * np.min(grid.dy) * np.min(grid.dz)
        vol_expected = 0.02629232100267493
        assert np.allclose(vol, vol_expected, rtol=1e-5), (
            f"Volume of the shell mask is {vol}, expected {vol_expected}"
        )

    def test_subpixel_smoothing(self):
        """
        Tests 'voxelize_rectilinear' and subpixel smoothing using the
        exact cavity and shell gridLogs configuration.
        """

        # Geometry & Materials
        solid_1 = "tests/stl/007_vacuum_cavity.stl"  # logs["stl_solids"]["cavity"]
        solid_2 = "tests/stl/007_lossymetal_shell.stl"  # logs["stl_solids"]["shell"]

        stl_solids = {"cavity": solid_1, "shell": solid_2}

        stl_materials = {
            "cavity": "vacuum",
            "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
        }

        # Extract domain bounds from geometry
        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

        # Number of mesh cells
        Nx = 60  # logs["Nx"]
        Ny = 60  # logs["Ny"]
        Nz = 140  # logs["Nz"]

        global grid
        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,  # Global domain zmin
            zmax,  # Global domain zmax
            Nx,
            Ny,
            Nz,  # Global domain Nz
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_method="voxelize_rectilinear",
            subpixel_smoothing=True,
            subpixel_smoothing_factor=4,
            subpixel_smoothing_bool=True,
            subpixel_smoothing_threshold=0.3,
            stl_scale=1.0,
            stl_rotate=[0, 0, 0],
            stl_translate=[0, 0, 0],
            verbose=1,
        )

        # number of cells in the mask
        n_inside = grid.grid.threshold(scalars="shell", value=0.5).n_cells
        n_inside_expected = 89256
        assert n_inside == n_inside_expected, (
            f"Number of cells masked inside the shell is {n_inside}, expected {n_inside_expected}"
        )

        # volume
        vol = n_inside * np.min(grid.dx) * np.min(grid.dy) * np.min(grid.dz)
        vol_expected = 0.038309239665264186
        assert np.allclose(vol, vol_expected, rtol=1e-5), (
            f"Volume of the shell mask is {vol}, expected {vol_expected}"
        )

    def test_long_wake_potential_and_impedance(self, use_gpu):
        global grid
        # ------------ Beam source ----------------
        # Beam parameters
        sigmaz = 10e-2  # [m] -> 2 GHz
        q = 1e-9  # [C]
        beta = 1.0  # beam beta
        xs = 0.0  # x source position [m]
        ys = 0.0  # y source position [m]
        xt = 0.0  # x test position [m]
        yt = 0.0  # y test position [m]
        # [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s]

        # ----------- Wake Solver  setup  ----------
        # Wakefield post-processor
        wakelength = 10.0  # [m] -> Partially decayed
        skip_cells = 20  # no. cells to skip at zlo/zhi for wake integration
        results_folder = "tests/013_results/"

        wake = WakeSolver(
            q=q,
            sigmaz=sigmaz,
            beta=beta,
            xsource=xs,
            ysource=ys,
            xtest=xt,
            ytest=yt,
            skip_cells=skip_cells,
            results_folder=results_folder,
            Ez_file=results_folder + "Ez.h5",
        )

        # ----------- Solver & Simulation ----------
        # boundary conditions
        bc_low = ["pec", "pec", "pec"]
        bc_high = ["pec", "pec", "pec"]

        # Solver setup
        solver = SolverFIT3D(
            grid,
            wake,
            bc_low=bc_low,
            bc_high=bc_high,
            use_stl=True,
            bg="pec",  # Background material
            dtype=self.dtype,
            use_gpu=use_gpu,
        )

        # Run simulation
        solver.wakesolve(wakelength=wakelength)

        # print(wake.WP[::50])
        np.cumsum(np.abs(wake.WP))[-1]
        assert len(wake.WP) == 5195, "Wake potential mesh samples length mismatch"
        assert np.allclose(wake.WP[::50], self.WP, **self.tol), (
            "Wake potential mesh samples failed"
        )
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
            179.95393780891274, 0.1
        ), "Wake potential cumsum mesh failed"

        assert len(wake.Z) == 998, "Impedance samples length mismatch"
        assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z), **self.tol), (
            "Abs Impedance samples mesh failed"
        )
        assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z), **self.tol), (
            "Real Impedance samples mesh failed"
        )
        assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z), **self.tol), (
            "Imag Impedance samples mesh failed"
        )
        assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(
            249395.46953432143, 0.1
        ), "Abs Impedance cumsum mesh failed"
