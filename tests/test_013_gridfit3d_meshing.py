import sys

import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


@pytest.mark.slow
class TestGridFIT3DMeshing:
    # Float32 regression samples on the boundary-inclusive symmetric grid.
    tol = dict(rtol=50e-5, atol=50e-4)
    dtype = np.float32

    # fmt: off
    WP = np.array([ 3.69497033e-18, -1.76312597e-14, -7.15643513e-11, -5.31938036e-08,
     -1.07457252e-05, -6.40165139e-04, -1.15214912e-02, -6.20450494e-02,
     -8.61496815e-02,  3.32663077e-02,  1.22169573e-01,  6.06792870e-02,
     -2.77738841e-02, -4.82990017e-02, -2.93178485e-02, -8.89325479e-03,
      7.24423878e-02,  7.33755874e-02, -7.73087974e-02, -1.05067621e-01,
      4.20113079e-02,  9.75587361e-02,  1.02308788e-02, -5.94117615e-02,
     -3.20849012e-02, -8.79245166e-03,  2.31801153e-02,  8.01548511e-02,
     -3.61939470e-03, -1.00205160e-01, -3.57803410e-02,  8.17645397e-02,
      6.59394480e-02, -4.39457310e-02, -5.14388194e-02, -9.50912100e-03,
      8.73759607e-03,  5.08178851e-02,  3.83665363e-02, -5.31764934e-02,
     -7.59498516e-02,  2.42307031e-02,  9.11925132e-02,  5.39993713e-03,
     -6.30954545e-02, -2.53319652e-02,  5.47113321e-03,  3.28113171e-02,
      4.00533792e-02, -6.96024499e-03, -6.63175474e-02, -3.36937521e-02,
      7.07995712e-02,  5.35101313e-02, -4.00765997e-02, -5.03564243e-02,
     -4.48021569e-03,  2.79555573e-02,  3.14969382e-02,  1.42706289e-02,
     -3.49031457e-02, -5.66810886e-02,  2.21895849e-02,  7.14825169e-02,
      3.49330403e-03, -5.39872237e-02, -2.99564252e-02,  2.26026253e-02,
      3.14060675e-02,  1.59347219e-02, -8.00680595e-03, -4.91986352e-02,
     -1.78949917e-02,  5.25540354e-02,  4.08000453e-02, -3.00624983e-02,
     -5.09744336e-02,  3.17461107e-03,  3.40268423e-02,  1.93036899e-02,
      1.84891188e-03, -2.94608019e-02, -3.42813709e-02,  1.92206429e-02,
      5.03493953e-02,  6.81489903e-03, -4.96466361e-02, -2.53482650e-02,
      2.81322459e-02,  2.64629132e-02,  7.77319131e-03, -1.67554672e-02,
     -3.12805505e-02, -6.32519473e-03,  3.54976392e-02,  3.35698391e-02,
     -2.68630952e-02, -4.30736535e-02,  6.14047593e-03,  3.22922861e-02,
      1.62074181e-02, -9.62441796e-03, -2.41878946e-02])


    Z = np.array([ 5.15202899e+00  -0.j        , -2.33824590e-01 +10.3505354j ,
     -7.82134452e-01  +6.88917031j,  7.83802688e+00 +14.2572602j ,
      2.25640130e+00 +25.19886143j, -3.98357683e-01 +21.53789521j,
      9.21155739e+00 +27.33638382j,  4.40354421e+00 +40.53936685j,
     -6.86554320e-01 +36.94514669j,  1.04840116e+01 +41.16987277j,
      6.72329714e+00 +57.64584631j, -1.50838906e+00 +54.32304289j,
      1.19431334e+01 +56.67145523j,  9.71893178e+00 +78.06612842j,
     -3.05176395e+00 +75.42050646j,  1.38914587e+01 +75.19095941j,
      1.42592909e+01+104.6305348j , -5.87098171e+00+103.6984012j ,
      1.69272196e+01 +99.44840294j,  2.24996720e+01+143.92882467j,
     -1.14514404e+01+148.03090294j,  2.28924871e+01+136.80464665j,
      4.23012301e+01+217.14592203j, -2.45723643e+01+242.56818539j,
      4.26523620e+01+220.67861184j,  1.40281896e+02+459.58172901j,
     -2.41192513e+01+796.91394068j,  1.79682962e+03+212.8042844j ,
     -1.93609882e+01-791.9012444j ,  1.81652348e+02-271.92411769j,
      2.68607263e+00  -4.57992744j, -5.20085205e+01 -96.05850052j,
      8.68132046e+01 -20.10396324j,  6.27635508e+00+120.85228382j,
     -3.67120292e+01 +46.40523796j,  8.92495047e+01+106.97974096j,
      1.88934269e+01+263.7055151j , -1.63938733e+01+212.25161781j,
      2.11409479e+02+385.57583155j,  3.35004375e+02+990.45690034j,
      1.70930966e+03-298.8572478j ,  9.59726746e+01-768.69487401j,
      1.39793883e+02-386.93885329j,  1.09373443e+01-168.17963674j,
     -1.16649778e+01-196.3939974j ,  6.29608486e+01-104.46485619j,
     -1.40190719e+01 -28.82456426j, -2.93977755e+00 -79.48748635j,
      7.18758251e+01  +1.43390517j,  7.41722473e-01 +36.72008401j])

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

        # Terminal computational cells have zero physical volume.
        vol = abs(grid.grid.threshold(scalars="shell", value=0.5).volume)
        vol_expected = 0.026895873465686084
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

        # Terminal computational cells have zero physical volume.
        vol = abs(grid.grid.threshold(scalars="shell", value=0.5).volume)
        vol_expected = 0.03914833869508424
        assert np.allclose(vol, vol_expected, rtol=1e-5), (
            f"Volume of the shell mask is {vol}, expected {vol_expected}"
        )

    def test_long_wake_potential_and_impedance(self, use_gpu, plot_comparison):
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
        assert len(wake.WP) == 5135, "Wake potential mesh samples length mismatch"
        plot_comparison(wake.WP[::50], self.WP, "Wake potential")
        assert np.allclose(wake.WP[::50], self.WP, **self.tol), (
            "Wake potential mesh samples failed"
        )
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
            np.float64(179.77371899827358), 0.1
        ), "Wake potential cumsum mesh failed"

        assert len(wake.Z) == 995, "Impedance samples length mismatch"
        plot_comparison(np.abs(wake.Z)[::20], np.abs(self.Z), "Impedance magnitude")
        assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z), **self.tol), (
            "Abs Impedance samples mesh failed"
        )
        plot_comparison(np.real(wake.Z)[::20], np.real(self.Z), "Impedance real part")
        assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z), **self.tol), (
            "Real Impedance samples mesh failed"
        )
        plot_comparison(
            np.imag(wake.Z)[::20], np.imag(self.Z), "Impedance imaginary part"
        )
        assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z), **self.tol), (
            "Imag Impedance samples mesh failed"
        )
        assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(
            np.float64(249145.4203489436), 0.1
        ), "Abs Impedance cumsum mesh failed"
