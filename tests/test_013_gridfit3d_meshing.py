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


    Z = np.array([ 5.16965227e+00   -0.j,        -1.06561926e+00  +10.10730216j,
                -4.11369040e-02   +6.13938248j,  7.82656129e+00  +15.81793808j,
                3.79179404e-02  +24.04034048j,  1.85647348e+00  +19.56490683j,
                9.52003168e+00  +30.53857243j,  4.02697448e-01  +38.3995175j,
                3.34651599e+00  +33.29286014j,  1.13638745e+01  +46.40886279j,
                3.10753446e-01  +54.32536692j,  4.83569168e+00  +48.20467763j,
                1.37868343e+01  +64.69501453j, -2.85943193e-01  +73.13066016j,
                6.61757376e+00  +65.36517895j,  1.73879531e+01  +87.41310469j,
                -1.68553103e+00  +97.14708083j,  9.15680782e+00  +86.65742844j,
                2.34612327e+01 +118.73922447j, -4.69278184e+00 +131.61010937j,
                1.36187052e+01 +116.41525215j,  3.58404623e+01 +169.8880007j,
                -1.16491084e+01 +192.38658625j,  2.46562215e+01 +169.02454354j,
                7.34870752e+01 +287.50304403j, -2.99946405e+01 +367.37878608j,
                9.70190908e+01 +355.51417916j,  8.39860646e+02+1201.22387426j,
                1.12139992e+03-1013.71454889j, -1.24422459e+01 -179.71074921j,
                -4.87155328e+01 -255.07196115j,  1.12234421e+02  -85.97008488j,
                -1.43947190e+01  +63.05360618j, -2.87693059e+01  -27.57445929j,
                8.89254807e+01  +63.76507921j, -8.09437817e+00 +175.31317057j,
                -9.39551694e+00  +97.40639893j,  1.24046801e+02 +224.68127368j,
                1.92620767e+01 +410.09800524j,  1.05341678e+02 +413.38947667j,
                1.19853299e+03 +977.32791986j,  1.04746227e+03-1045.98818149j,
                5.97976208e+01 -426.03330595j,  1.60736714e+01 -357.72176423j,
                6.56091290e+01 -181.12829036j, -2.08608242e+01 -114.24362128j,
                1.90235643e+01 -128.7294329j,   3.83310532e+01  -27.23991147j,
                -2.38991796e+01  -20.46349505j,  5.00051233e+01  -36.70012155j])

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

        print(wake.Z[::20])
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
