
import sys
import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest
from wakis import GridFIT3D, SolverFIT3D, WakeSolver


# Run with:
# mpiexec -n 2 python -m pytest --color=yes -v -s tests/test_013_gridfit3d_meshing.py


@pytest.mark.slow
class TestGridFIT3DMeshing:
    # Regression data
    # fmt: off
    WP = np.array([ 9.67345087e-19, -2.17164213e-13, -2.46909922e-09, -3.35736557e-06, -6.68783545e-04,
        -2.03471539e-02, -8.93366466e-02,  2.21782798e-03,  1.21944850e-01,  2.65765433e-02,
        -4.95708206e-02, -2.88028306e-02,  9.05319186e-04,  9.19114857e-02, -2.93135252e-02,
        -1.05165644e-01,  7.22711895e-02,  6.04058770e-02, -5.66535414e-02, -2.71704419e-02,
        -5.21572263e-04,  6.24378705e-02,  9.82923044e-03, -9.71731801e-02,  2.22139919e-02,
        8.85327605e-02, -4.62353298e-02, -3.97817186e-02,  6.97264916e-03,  3.95994154e-02,
        2.65785066e-02, -7.19997432e-02, -1.89248689e-02,  9.12103520e-02, -1.71569383e-02,
        -5.61388341e-02,  7.71421643e-03,  3.33498980e-02,  2.32096595e-02, -4.10197933e-02,
        -4.41299903e-02,  7.45925725e-02,  1.52849901e-02, -6.26857268e-02, -2.71268610e-03,
        3.61423830e-02,  1.65933166e-02, -2.10142816e-02, -4.81579782e-02,  4.63937593e-02,
        3.98669695e-02, -5.42217321e-02, -2.16236505e-02,  4.06084475e-02,  1.35533369e-02,
        -1.16436235e-02, -4.15325567e-02,  2.11196806e-02,  4.88665705e-02, -3.40779255e-02,
        -3.92676324e-02,  3.71275977e-02,  1.92633580e-02, -1.12574493e-02, -3.11576144e-02,
        3.82191895e-03,  4.52120404e-02, -1.08903261e-02, -4.90844003e-02,  2.58177495e-02,
        2.74693885e-02, -1.00551312e-02, -2.61124913e-02, -2.73028041e-03,  3.41625232e-02,
        7.47955334e-03, -4.73350653e-02,  8.06941570e-03,  3.51520091e-02, -6.47884339e-03,
        -2.46368796e-02, -4.59794705e-03])


    Z = np.array([ 5.19585047e+00   -0.j        , -1.14584967e+00  +10.07240661j,  8.57752415e-02   +6.01789639j,
        7.82619161e+00  +16.00298465j, -2.09913641e-01  +23.79009263j,  2.25936860e+00  +19.33604521j,
        9.44234793e+00  +30.93420192j,  1.15160305e-02  +37.83475196j,  4.13830902e+00  +32.99371877j,
        1.11279023e+01  +47.06880521j, -2.17806106e-01  +53.27943471j,  6.20341923e+00  +47.87777531j,
        1.32490192e+01  +65.72089242j, -9.64938404e-01  +71.3043722j ,  8.89282637e+00  +65.07658368j,
        1.62659217e+01  +88.99341956j, -2.55604121e+00  +93.95087188j,  1.29871525e+01  +86.53428044j,
        2.11294381e+01 +121.2423969j , -5.85354188e+00 +125.70421173j,  2.04773483e+01 +116.73572463j,
        3.06238179e+01 +174.1142225j , -1.33926893e+01 +179.84849633j,  3.89537992e+01 +170.39341032j,
        5.86719481e+01 +295.3922762j , -3.46162682e+01 +327.27250813j,  1.43593011e+02 +353.34433544j,
        6.01920709e+02+1202.0840238j ,  1.37069557e+03 -858.17451862j, -8.45765393e+01 -261.48097504j,
        1.73388593e+00 -298.6092324j ,  1.10792221e+02  -48.66599352j, -4.81755071e+01  +34.36500335j,
        5.33563341e+00  -48.23594643j,  8.34238121e+01  +94.72344872j, -3.62433387e+01 +144.60058139j,
        2.90362116e+01  +79.15226046j,  1.09096072e+02 +256.59678218j, -1.84928388e+01 +346.47133802j,
        1.66004913e+02 +362.92496325j,  8.80510046e+02+1078.09377996j,  1.35373690e+03 -881.94842291j,
        2.80079423e+01 -530.396313j  ,  5.88141964e+01 -388.69913707j,  4.77113269e+01 -169.50329412j,
        -2.74628280e+01 -149.86592623j,  4.08748760e+01 -123.17588073j,  1.25959867e+01  -23.45937766j,
        -1.79762955e+01  -53.60614807j,  6.52952860e+01  -16.41154217j,  2.03685498e+01  +40.09814247j])

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
        n_inside = grid.grid.threshold(scalars='shell', value=0.5).n_cells
        n_inside_expected = 61258
        assert n_inside == n_inside_expected, f"Number of cells masked inside the shell is {n_inside}, expected {n_inside_expected}"

        # volume
        vol=n_inside*np.min(grid.dx)*np.min(grid.dy)*np.min(grid.dz)
        vol_expected = 0.02629232100267493
        assert np.allclose(vol, vol_expected, rtol=1e-5), f"Volume of the shell mask is {vol}, expected {vol_expected}"

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
        n_inside = grid.grid.threshold(scalars='shell', value=0.5).n_cells
        n_inside_expected = 89256
        assert n_inside == n_inside_expected, f"Number of cells masked inside the shell is {n_inside}, expected {n_inside_expected}"

        # volume
        vol=n_inside*np.min(grid.dx)*np.min(grid.dy)*np.min(grid.dz)
        vol_expected = 0.038309239665264186
        assert np.allclose(vol, vol_expected, rtol=1e-5), f"Volume of the shell mask is {vol}, expected {vol_expected}"

    def test_long_wake_potential_and_impedance(self):

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

        assert len(wake.WP) == 4090, "Wake potential mesh samples length mismatch"
        assert np.allclose(wake.WP[::50], self.WP, **self.tol), (
                "Wake potential mesh samples failed"
        )
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
                141.8402107359091, 0.1
        ), "Wake potential cumsum mesh failed"

        # print(wake.Z[::20])
        assert len(wake.Z) == 1001, "Impedance samples length mismatch"
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