import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.append("../wakis")

import pytest
from scipy.constants import c
from tqdm import tqdm

from wakis import GridFIT3D, SolverFIT3D, WakeSolver
from wakis.sources import Beam

# Run with:
# mpiexec -n 2 python -m pytest --color=yes -v -s tests/test_007_mpi_lossy_cavity.py


def test_mpi_single_rank():
    """Reject MPI decomposition without multiple ranks; serial remains valid."""
    MPI = pytest.importorskip("mpi4py.MPI")
    if MPI.COMM_WORLD.Get_size() != 1:
        pytest.skip("run with one MPI rank")

    bounds = (0, 1, 0, 1, 0, 1, 4, 4, 6)
    with pytest.raises(ValueError):
        GridFIT3D(*bounds, use_mpi=True, verbose=0)
    grid = GridFIT3D(*bounds, use_mpi=False, verbose=0)
    solver = SolverFIT3D(grid, verbose=0)
    solver.one_step()
    assert np.isfinite(solver.E.toarray()).all()


def test_mpi_rounds_uneven_node_count():
    """Round up the global node count while preserving the domain bounds."""
    MPI = pytest.importorskip("mpi4py.MPI")
    size = MPI.COMM_WORLD.Get_size()
    if size < 2:
        pytest.skip("run with multiple MPI ranks")

    grid = GridFIT3D(0, 1, 0, 1, 0, 1, 4, 4, 2 * size + 1, use_mpi=True, verbose=0)
    assert grid.NZ == 3 * size
    np.testing.assert_allclose(grid.Z, np.linspace(0, 1, 3 * size))
    assert grid.logger.grid["Nz"] == grid.NZ
    np.testing.assert_allclose(grid.logger.grid["dz"], np.diff(grid.Z))


def test_mpi_custom_axes():
    """MPI partitions supplied nonuniform nodes without changing their positions."""
    MPI = pytest.importorskip("mpi4py.MPI")
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if size < 2:
        pytest.skip("run with multiple MPI ranks")

    x = np.array([0.0, 0.2, 1.0])
    y = np.array([0.0, 0.3, 1.0])
    z = np.linspace(0.0, 1.0, 3 * size) ** 2
    grid = GridFIT3D(x=x, y=y, z=z, use_mpi=True, verbose=0)
    start = 3 * rank - (rank > 0)
    stop = 3 * (rank + 1) + (rank < size - 1)
    np.testing.assert_array_equal(grid.Z, z)
    np.testing.assert_array_equal(grid.z, z[start:stop])
    np.testing.assert_allclose(grid.dz, np.diff(grid.z))
    solver = SolverFIT3D(
        grid, use_mpi=True, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
    )
    solver.one_step()
    assert np.isfinite(solver.E.toarray()).all()
    gathered = grid.mpi_gather_asGrid()
    if rank == 0:
        np.testing.assert_array_equal(gathered.x, x)
        np.testing.assert_array_equal(gathered.y, y)
        np.testing.assert_array_equal(gathered.z, z)

    with pytest.raises(ValueError, match="Custom z"):
        GridFIT3D(x=x, y=y, z=z[:-1], use_mpi=True, verbose=0)


@pytest.mark.slow
class TestMPILossyCavity:
    # Float32 regression samples with staggered primal/dual material masks.
    # fmt: off
    tol = dict(rtol=50e-3, atol=50e-3)
    dtype = np.float32

    WP = np.array([-0.000403  ,  0.00027331, -0.00018145, -0.0008381 ,
                   -0.00059366, -0.00149418, -0.01341081, -0.06546433,
                   -0.08187606,  0.03575794,  0.11591192,  0.05952239,
                   -0.02400019, -0.04472129, -0.02970571, -0.01264781,
                    0.06488926,  0.07575511, -0.06227177, -0.10617149,
                    0.02628806,  0.09249132,  0.02099129, -0.04944804,
                   -0.03627527, -0.01338947,  0.01196185,  0.07632031,
                    0.01910652, -0.09054648, -0.05430678,  0.06076495,
                    0.07596456, -0.02177092, -0.05187292, -0.01764839,
                   -0.00295882,  0.04000526,  0.05371512, -0.02780972,
                   -0.08196381, -0.00901327,  0.08144971,  0.03676088,
                   -0.04857485, -0.03628262, -0.00670256,  0.01644544,
                    0.04602184,  0.01835088, -0.04994927, -0.0601927 ,
                    0.03461758,  0.07417205, -0.00730479, -0.04961481,
                   -0.02336022,  0.00955677,  0.02951386,  0.031034  ,
                   -0.00703649, -0.05973925, -0.02348343,  0.06078824,
                    0.04242064, -0.03216786, -0.04321695, -0.00578835,
                    0.02466572,  0.02572693,  0.01531771, -0.02855488,
                   -0.0511813 ,  0.01649569,  0.05775541,  0.01176292,
                   -0.04431081, -0.0302384 ,  0.01583838,  0.02447339,
                    0.0191856 , -0.00255221, -0.04062323, -0.02389878,
                    0.0375593 ,  0.04238003, -0.01435875, -0.04607814,
                   -0.00737676,  0.02520504,  0.01946724,  0.01016393,
                   -0.02070681, -0.03378118,  0.0018897 ,  0.04257063,
                    0.02228231, -0.03546749, -0.03160103,  0.01123792,
                    0.02479243,  0.01458273, -0.00545324])

    Z = np.array([-2.95878260e-03   +0.j        ,
                   5.73267545e+00   +1.51458174j,
                   5.30341065e+00  +14.25704418j,
                  -1.63713344e-01  +15.15675621j,
                   7.15039652e+00  +17.58553883j,
                   5.93486249e+00  +30.20411655j,
                   3.30882278e-01  +29.71120371j,
                   8.77673348e+00  +32.71916382j,
                   6.25619554e+00  +46.58098636j,
                   1.07851832e-01  +44.66968548j,
                   1.05700200e+01  +48.60241432j,
                   6.48085013e+00  +64.97085974j,
                  -7.11332473e-01  +61.21426881j,
                   1.31663313e+01  +66.5183092j ,
                   6.92036393e+00  +87.43067639j,
                  -2.23719695e+00  +81.01463257j,
                   1.75926558e+01  +88.59403254j,
                   7.96693920e+00 +118.1083489j ,
                  -5.10050566e+00 +107.53873739j,
                   2.62220041e+01 +119.61124229j,
                   1.05678326e+01 +167.88801456j,
                  -1.14638921e+01 +150.59551719j,
                   4.76298394e+01 +174.82395835j,
                   2.01893974e+01 +280.46311917j,
                  -3.03158588e+01 +259.92271521j,
                   1.59609420e+02 +358.25669088j,
                   2.94202532e+02+1085.07028966j,
                   1.51119195e+03 -542.88891629j,
                  -1.20807214e+02 -342.58744594j,
                   5.75902286e+01 -292.58870595j,
                   9.32462188e+01   -7.95034471j,
                  -6.25294706e+01  +14.85332975j,
                   2.74959872e+01  -43.68046508j,
                   6.89919309e+01 +113.07936728j,
                  -4.91085499e+01 +127.2531281j ,
                   3.68514077e+01  +75.91770038j,
                   8.85382223e+01 +258.27641554j,
                  -4.33749142e+01 +316.31361968j,
                   1.40939374e+02 +329.37796842j,
                   7.59550775e+02+1075.30385269j,
                   1.26176316e+03 -804.9021747j ,
                   3.20548264e+01 -434.96070127j,
                   5.06924944e+01 -334.97864511j,
                   5.66876132e+01 -142.47162272j,
                  -1.82270110e+01 -108.70554109j,
                   2.94540257e+01  -99.08379951j,
                   2.29788472e+01  -13.38851612j,
                  -1.30439212e+01  -19.75261626j,
                   4.09286599e+01  -17.8430034j ,
                   1.83699746e+01  +55.59384025j])

    Ez = np.array([  18.061344 ,   -9.153261 ,  -11.58048  ,   -8.437785 ,   -6.873616 ,
      -10.673922 ,  -31.06836  ,  -67.36018  , -111.72592  , -153.16042  ,
     -184.0861   , -204.05917  , -214.5473   , -216.9742   , -213.10362  ,
     -202.92325  , -187.07582  , -166.50366  , -140.19456  , -107.53213  ,
      -70.57386  ,  -36.18224  ,  -13.983852 ,   -4.422415 ,   -1.6875826,
       -1.0345211,   -1.159411 ,   -1.5268996])

    # fmt: on

    gridLogs = {
        "use_mesh_refinement": False,
        "Nx": 60,
        "Ny": 60,
        "Nz": 140,
        "dx": 0.008813558998754467,
        "dy": 0.008813558998754467,
        "dz": 0.005755395769215315,
        "xmin": -0.25999999046325684,
        "xmax": 0.25999999046325684,
        "ymin": -0.25999999046325684,
        "ymax": 0.25999999046325684,
        "zmin": -0.25,
        "zmax": 0.550000011920929,
        "stl_solids": {
            "cavity": "tests/stl/007_vacuum_cavity.stl",
            "shell": "tests/stl/007_lossymetal_shell.stl",
        },
        "stl_materials": {"cavity": [1.0, 1.0, 0.0], "shell": [30, 1.0, 30]},
        "gridInitializationTime": 0,
    }

    solverLogs = {
        "use_gpu": False,  # updated in test_log_file
        "use_mpi": False,  # updated in test_log_file
        "source_type": "direct",
        "background": "pec",
        "bc_low": ["pec", "pec", "pec"],
        "bc_high": ["pec", "pec", "pec"],
        "dt": dtype(7.051852157768934e-12),
        "solverInitializationTime": 0,
    }

    wakeSolverLogs = {
        "ti": 2.8516132094735135e-09,
        "q": 1e-09,
        "sigmaz": 0.1,
        "beta": 1.0,
        "xsource": 0.0,
        "ysource": 0.0,
        "xtest": 0.0,
        "ytest": 0.0,
        "chargedist": None,
        "skip_cells": 10,
        "results_folder": "tests/007_results/",
        "wakelength": 10.0,
        "simulationTime": 0,
    }

    img_folder = "tests/007_img/"

    def test_mpi_import(self):
        # ---------- MPI setup ------------
        global use_mpi
        try:
            # can be skipped since it is handled inside GridFIT3D
            from mpi4py import MPI

            comm = MPI.COMM_WORLD  # Get MPI communicator
            size = comm.Get_size()  # Total number of MPI processes
            if size > 1:
                use_mpi = True
            else:
                use_mpi = False
        except Exception as e:
            print(f"[!] MPI not available: {e}")
            use_mpi = False

        print(f"Using mpi: {use_mpi}")

    def test_mpi_simulation(self, use_gpu, plot_comparison):
        # ---------- Domain setup ---------

        # Geometry & Materials
        solid_1 = "tests/stl/007_vacuum_cavity.stl"
        solid_2 = "tests/stl/007_lossymetal_shell.stl"

        stl_solids = {"cavity": solid_1, "shell": solid_2}

        stl_materials = {
            "cavity": "vacuum",
            "shell": [30, 1.0, 30],  # [eps_r, mu_r, sigma[S/m]]
        }

        # Extract domain bounds from geometry
        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, ZMIN, ZMAX = solids.bounds

        # Number of mesh cells
        Nx = 60
        Ny = 60
        NZ = 140
        global use_mpi
        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            ZMIN,  # Global domain zmin
            ZMAX,  # Global domain zmax
            Nx,
            Ny,
            NZ,  # Global domain Nz
            use_mpi=use_mpi,  # Enables MPI subdivision of the domain
            stl_solids=stl_solids,
            stl_materials=stl_materials,
            stl_scale=1.0,
            stl_rotate=[0, 0, 0],
            stl_translate=[0, 0, 0],
            verbose=1,
        )
        if use_mpi:
            print(
                f"Process {grid.rank}: Handling Z range {grid.zmin} to {grid.zmax} with {grid.Nz} cells"
            )

        # ------------ Beam source & Wake ----------------
        # Beam parameters
        sigmaz = 10e-2  # [m] -> 2 GHz
        q = 1e-9  # [C]
        beta = 1.0  # beam beta
        xs = 0.0  # x source position [m]
        ys = 0.0  # y source position [m]
        ti = 3 * sigmaz / c  # injection time [s]

        beam = Beam(q=q, sigmaz=sigmaz, beta=beta, xsource=xs, ysource=ys, ti=ti)

        # ----------- Solver & Simulation ----------
        # boundary conditions
        bc_low = ["pec", "pec", "pec"]
        bc_high = ["pec", "pec", "pec"]

        # Solver setup
        global solver
        solver = SolverFIT3D(
            grid,
            bc_low=bc_low,
            bc_high=bc_high,
            use_stl=True,
            use_mpi=use_mpi,  # Activate MPI
            bg="pec",  # Background material
            dtype=self.dtype,
            use_gpu=use_gpu,
        )

        # -------------- Output folder ---------------------
        if use_mpi and solver.rank == 0:
            if not os.path.exists(self.img_folder):
                os.mkdir(self.img_folder)
        elif not use_mpi:
            if not os.path.exists(self.img_folder):
                os.mkdir(self.img_folder)

        # Late-time float32 ringing varies between numerical backends. Scale
        # the absolute floor to the pulse peak, while checking its main shape
        # more tightly than the general wake/impedance regression tolerance.
        field_tol = dict(rtol=5e-3, atol=5e-3 * np.max(np.abs(self.Ez)))

        # -------------- Custom time loop  -----------------
        if use_mpi:
            Nt = 3000
            for n in tqdm(range(Nt)):
                beam.update(solver, n * solver.dt)
                solver.one_step()  # MPI handled internally

            Ez = solver.mpi_gather("Ez", x=int(Nx / 2), y=int(Ny / 2))
            if solver.rank == 0:
                # print(Ez)
                # print(len(Ez))
                assert len(Ez) == NZ, "Electric field Ez samples length mismatch"
                plot_comparison(Ez[::5], self.Ez, "Electric field Ez (MPI)")
                assert np.allclose(Ez[np.s_[::5]], self.Ez, **field_tol), (
                    "Electric field Ez samples MPI failed"
                )
        else:
            Nt = 3000
            for n in tqdm(range(Nt)):
                beam.update(solver, n * solver.dt)
                solver.one_step()

            Ez = solver.E[int(Nx / 2), int(Ny / 2), np.s_[::5], "z"]
            # print(Ez)
            assert len(solver.E[int(Nx / 2), int(Ny / 2), :, "z"]) == NZ, (
                "Electric field Ez samples length mismatch"
            )
            plot_comparison(Ez, self.Ez, "Electric field Ez")
            assert np.allclose(Ez, self.Ez, **field_tol), (
                "Electric field Ez samples failed"
            )

    def test_mpi_save_state(self, tmp_path):
        """Save current solver state to disk on CPU and MPI."""
        global solver
        filename = tmp_path / "solver_state_007.h5"

        solver.save_state(str(filename))

        if not use_mpi or solver.rank == 0:
            assert os.path.exists(filename)

    def test_mpi_load_state(self, tmp_path):
        """Reload a previously saved solver state and check fields are restored."""
        global solver
        filename = tmp_path / "solver_state_007_roundtrip.h5"

        # Save current (non-zero) state
        solver.save_state(str(filename))

        # Overwrite fields and load back
        solver.reset_fields()
        solver.load_state(str(filename))

        if not use_mpi or solver.rank == 0:
            Ez_restored = np.asarray(solver.E.toarray())
            assert np.any(Ez_restored != 0.0)

    def test_mpi_gather_asField(self, flag_offscreen):
        # Plot inspect after mpi gather
        global solver
        if use_mpi:
            E = solver.mpi_gather_asField("E")
            if solver.rank == 0:  # custom plots go in rank 0
                fig, ax = E.inspect(
                    figsize=[20, 6],
                    plane="YZ",
                    off_screen=flag_offscreen,
                    handles=True,
                )
                fig.savefig(self.img_folder + "Einspect_" + str(3000).zfill(4) + ".png")
                plt.close(fig)
        else:
            fig, ax = solver.E.inspect(
                figsize=[20, 6],
                plane="YZ",
                off_screen=flag_offscreen,
                handles=True,
            )
            fig.savefig(self.img_folder + "Einspect_" + str(3000).zfill(4) + ".png")
            plt.close(fig)

    def test_mpi_plot2D(self, flag_offscreen):
        # Plot E abs in 2D every 20 timesteps
        global solver
        solver.plot2D(
            field="E",
            component="Abs",
            plane="YZ",
            pos=0.5,
            cmap="rainbow",
            vmin=0,
            vmax=500.0,
            interpolation="hanning",
            off_screen=flag_offscreen,
            title=self.img_folder + "Ez2d",
            n=3000,
        )

    def test_mpi_plot1D(self, flag_offscreen):
        # Plot E z in 1D at diferent transverse positions `pos` every 20 timesteps
        global solver
        solver.plot1D(
            field="E",
            component="z",
            line="z",
            pos=[0.45, 0.5, 0.55],
            xscale="linear",
            yscale="linear",
            off_screen=flag_offscreen,
            title=self.img_folder + "Ez1d",
            n=3000,
        )

    def test_mpi_wakefield(self, use_gpu):
        # Reset fields
        global solver
        solver.reset_fields()

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
        skip_cells = 10  # no. cells to skip at zlo/zhi for wake integration
        results_folder = "tests/007_results/"

        global wake
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

        # Run simulation
        solver.wakesolve(wakelength=wakelength, wake=wake)

    def test_long_wake_potential(self, plot_comparison):
        global wake
        global solver
        if use_mpi:
            if solver.rank == 0:
                # MPI wake samples near zero carry small interface noise.
                tol = dict(rtol=0.1, atol=5e-3)
                assert len(wake.WP) == 5135, (
                    "Wake potential MPI samples length mismatch"
                )
                plot_comparison(wake.WP[::50], self.WP, "Wake potential (MPI)")
                assert np.allclose(wake.WP[::50], self.WP, **tol), (
                    "Wake potential MPI samples failed"
                )
                assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
                    175.6712188427506, 0.1
                ), "Wake potential cumsum MPI failed"
        else:
            assert len(wake.WP) == 5135, "Wake potential samples length mismatch"
            plot_comparison(wake.WP[::50], self.WP, "Wake potential")
            assert np.allclose(wake.WP[::50], self.WP, **self.tol), (
                "Wake potential samples failed"
            )
            assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
                175.6712188427506, 0.1
            ), "Wake potential cumsum MPI failed"

    def test_long_impedance(self, plot_comparison):
        global wake
        global solver
        if use_mpi:
            if solver.rank == 0:
                tol = dict(rtol=0.1)
                assert len(wake.Z) == 995, "Impedance samples length mismatch"
                plot_comparison(
                    np.abs(wake.Z)[::20], np.abs(self.Z), "Impedance magnitude (MPI)"
                )
                assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z), **tol), (
                    "Abs Impedance samples MPI failed"
                )
                plot_comparison(
                    np.real(wake.Z)[::20], np.real(self.Z), "Impedance real part (MPI)"
                )
                assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z), **tol), (
                    "Real Impedance samples MPI failed"
                )
                plot_comparison(
                    np.imag(wake.Z)[::20],
                    np.imag(self.Z),
                    "Impedance imaginary part (MPI)",
                )
                assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z), **tol), (
                    "Imag Impedance samples MPI failed"
                )
                assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(
                    230800.9373039502, 0.1
                ), "Abs Impedance cumsum MPI failed"
        else:
            # print(wake.Z[::20])
            assert len(wake.Z) == 995, "Impedance samples length mismatch"
            plot_comparison(np.abs(wake.Z)[::20], np.abs(self.Z), "Impedance magnitude")
            assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z), **self.tol), (
                "Abs Impedance samples failed"
            )
            plot_comparison(
                np.real(wake.Z)[::20], np.real(self.Z), "Impedance real part"
            )
            assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z), **self.tol), (
                "Real Impedance samples failed"
            )
            plot_comparison(
                np.imag(wake.Z)[::20], np.imag(self.Z), "Impedance imaginary part"
            )
            assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z), **self.tol), (
                "Imag Impedance samples failed"
            )
            assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(
                230800.9373039502, 0.1
            ), "Abs Impedance cumsum failed"

    def test_log_file(self, use_gpu):
        # Helper function to compare nested dicts with float tolerance
        def assert_dict_allclose(d1, d2, rtol=1e-6, atol=1e-6, path=""):
            assert set(d1.keys()) == set(d2.keys()), (
                f"Key mismatch at {path}: {set(d1.keys())} != {set(d2.keys())}"
            )

            for k in d1:
                v1, v2 = d1[k], d2[k]
                p = f"{path}.{k}" if path else k

                # nested dict
                if isinstance(v1, dict) and isinstance(v2, dict):
                    assert_dict_allclose(v1, v2, rtol, atol, p)

                # floats
                elif isinstance(v1, float) and isinstance(v2, float):
                    if k == "dt":
                        assert v1 <= v2, "Timestep bigger than for uniform grid"
                    else:
                        assert np.isclose(v1, v2, rtol=rtol, atol=atol), (
                            f"Float mismatch at {p}: {v1} != {v2}"
                        )

                # np.floats
                elif isinstance(v1, np.floating) and isinstance(v2, np.floating):
                    if k == "dt":
                        assert v1 <= v2, "Timestep bigger than for uniform grid"
                    else:
                        assert np.isclose(v1, v2, rtol=rtol, atol=atol), (
                            f"Float mismatch at {p}: {v1} != {v2}"
                        )

                # lists/tuples/arrays
                elif isinstance(v1, (list, tuple, np.ndarray)) and isinstance(
                    v2, (list, tuple, np.ndarray)
                ):
                    assert len(v1) == len(v2), f"Length mismatch at {p}"
                    for i, (a, b) in enumerate(zip(v1, v2)):
                        if isinstance(a, float) and isinstance(b, float):
                            assert np.isclose(a, b, rtol=rtol, atol=atol), (
                                f"Float mismatch at {p}[{i}]: {a} != {b}"
                            )
                        else:
                            assert a == b, f"Value mismatch at {p}[{i}]: {a} != {b}"

                # list vs single float
                elif isinstance(v1, (list, tuple, np.ndarray)) and isinstance(
                    v2, float
                ):
                    for i, a in enumerate(v1):
                        assert np.isclose(a, v2, rtol=rtol, atol=atol), (
                            f"Float mismatch at {p}[{i}]: {a} != {v2}"
                        )

                elif isinstance(v1, float) and isinstance(
                    v2, (list, tuple, np.ndarray)
                ):
                    for i, b in enumerate(v2):
                        assert np.isclose(v1, b, rtol=rtol, atol=atol), (
                            f"Float mismatch at {p}[{i}]: {v1} != {b}"
                        )

                # everything else → exact match
                else:
                    assert v1 == v2, f"Mismatch at {p}: {v1} != {v2}"

        global solver
        # Exclude timing info from comparison as they can vary between runs
        solver.logger.grid["gridInitializationTime"] = 0
        solver.logger.solver["solverInitializationTime"] = 0
        solver.logger.wakeSolver["simulationTime"] = 0
        self.solverLogs["use_mpi"] = use_mpi
        self.solverLogs["use_gpu"] = use_gpu

        # Check log file exists
        logfile = os.path.join(solver.logger.wakeSolver["results_folder"], "wakis.log")
        assert os.path.exists(logfile), "Log file not created"

        # Compare log dict contents
        assert_dict_allclose(solver.logger.grid, self.gridLogs)
        assert_dict_allclose(solver.logger.solver, self.solverLogs)
        assert_dict_allclose(solver.logger.wakeSolver, self.wakeSolverLogs)
