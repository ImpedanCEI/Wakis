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
    WP = np.array([ 3.04630498e-18, -1.57008581e-14, -5.80925142e-11, -4.16489867e-08, -8.36718278e-06,
        -5.09692562e-04, -9.64107933e-03, -5.61940775e-02, -8.91793160e-02,  1.93935027e-02,
        1.20613671e-01,  6.94030548e-02, -2.25229088e-02, -4.90874776e-02, -2.81476860e-02,
        -1.06940498e-02,  6.23438033e-02,  7.76134829e-02, -6.41150535e-02, -1.08992801e-01,
        3.16790898e-02,  9.94316382e-02,  1.59779094e-02, -6.11351005e-02, -3.38082136e-02,
        -3.31629015e-03,  1.97599390e-02,  7.32110603e-02,  2.14580395e-03, -9.53691052e-02,
        -3.92647802e-02,  7.85002701e-02,  6.91921761e-02, -4.44018857e-02, -5.61169510e-02,
        -4.30780071e-03,  1.37179657e-02,  4.47084785e-02,  3.30408758e-02, -4.91026240e-02,
        -7.16624192e-02,  2.30146382e-02,  9.06701415e-02,  4.59715946e-03, -6.81361410e-02,
        -2.42086139e-02,  1.47330810e-02,  3.33421270e-02,  3.07522528e-02, -1.09926539e-02,
        -6.00356189e-02, -2.83128629e-02,  7.00475018e-02,  5.06898169e-02, -4.55404417e-02,
        -5.23208454e-02,  3.63697295e-03,  3.55132257e-02,  2.52280010e-02,  3.40381185e-03,
        -3.38981266e-02, -4.70026904e-02,  2.66534441e-02,  6.76875230e-02, -3.49122472e-03,
        -5.85509627e-02, -2.46264082e-02,  3.32978590e-02,  3.19330192e-02,  3.80438155e-03,
        -1.52599193e-02, -4.10939013e-02, -7.28016879e-03,  5.23290530e-02,  3.21712155e-02,
        -3.75754719e-02, -4.91900187e-02,  1.42625933e-02,  4.05405702e-02,  1.10479911e-02,
        -1.08936364e-02, -2.85114349e-02, -2.08594664e-02,  2.56420070e-02,  4.32661161e-02,
        -4.10720122e-03, -5.21893669e-02, -1.57609914e-02,  3.84680009e-02,  2.44619636e-02,
        -5.94037275e-03, -2.37626472e-02, -2.08768692e-02,  6.12328696e-03,  3.41465390e-02,
        2.11021297e-02, -3.47723067e-02, -3.68070278e-02,  1.84349961e-02,  3.62253989e-02,
        5.70862670e-03, -2.22060960e-02, -2.07388017e-02, -2.13881202e-03])


    Z = np.array([ 4.87260032e+00+0.00000000e+00j, -9.36159483e-01+9.75925492e+00j,
        -4.23812469e-02+5.75318587e+00j,  7.63955058e+00+1.50411662e+01j,
        2.26106579e-02+2.29261382e+01j,  1.81047525e+00+1.84235628e+01j,
        9.24516940e+00+2.90470001e+01j,  2.80373359e-01+3.65289411e+01j,
        3.21454038e+00+3.14082467e+01j,  1.09561316e+01+4.41873035e+01j,
        4.43757543e-02+5.16798242e+01j,  4.57507206e+00+4.55653651e+01j,
        1.31857595e+01+6.17235105e+01j, -7.52543696e-01+6.96730056e+01j,
        6.17370355e+00+6.19469795e+01j,  1.65055227e+01+8.36617709e+01j,
        -2.43741455e+00+9.28146352e+01j,  8.45026457e+00+8.24218590e+01j,
        2.21506555e+01+1.14163748e+02j, -5.87730933e+00+1.26291433e+02j,
        1.25109915e+01+1.11302667e+02j,  3.37963196e+01+1.64420403e+02j,
        -1.35982405e+01+1.85841886e+02j,  2.28100831e+01+1.62936912e+02j,
        6.97374304e+01+2.81042365e+02j, -3.40692544e+01+3.58789105e+02j,
        9.24810577e+01+3.48396266e+02j,  8.21035524e+02+1.19830910e+03j,
        1.12777868e+03-1.00257057e+03j, -1.01164346e+01-1.86701558e+02j,
        -4.70309413e+01-2.62727113e+02j,  1.11964395e+02-9.14237793e+01j,
        -1.54399970e+01+5.42926190e+01j, -2.88327046e+01-3.63945122e+01j,
        8.68237423e+01+5.50642404e+01j, -1.08699194e+01+1.64224905e+02j,
        -1.17530368e+01+8.64765120e+01j,  1.18444830e+02+2.13273048e+02j,
        1.20435243e+01+3.94789328e+02j,  9.56251136e+01+3.98904716e+02j,
        1.16216172e+03+9.69886929e+02j,  1.04734217e+03-1.01930503e+03j,
        6.54216937e+01-4.28714357e+02j,  2.24633870e+01-3.63486721e+02j,
        6.78202690e+01-1.88251762e+02j, -1.98707701e+01-1.25511578e+02j,
        2.05337605e+01-1.41150121e+02j,  3.78848226e+01-4.10479730e+01j,
        -2.72833483e+01-3.79761545e+01j,  3.61577248e+01-5.97509649e+01j])

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
        assert len(wake.WP) == 5195, "Wake potential mesh samples length mismatch"
        plot_comparison(wake.WP[::50], self.WP, "Wake potential")
        assert np.allclose(wake.WP[::50], self.WP, **self.tol), (
            "Wake potential mesh samples failed"
        )
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
            np.float64(179.77371899827358), 0.1
        ), "Wake potential cumsum mesh failed"

        assert len(wake.Z) == 998, "Impedance samples length mismatch"
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
