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


@pytest.mark.slow
class TestMPILossyCavity:
    # Float32 regression samples on the boundary-inclusive symmetric grid.
    # fmt: off
    tol = dict(rtol=50e-3, atol=50e-3)
    dtype = np.float32

    WP = np.array([-6.71674913e-04,  3.70437594e-05, -2.09607843e-05, -8.98969097e-04,
     -9.43441174e-04, -1.57932410e-03, -1.36473162e-02, -6.95587915e-02,
     -8.75929572e-02,  3.99073112e-02,  1.23350702e-01,  5.87594902e-02,
     -2.91380108e-02, -4.70907276e-02, -2.91980295e-02, -4.96408223e-03,
      7.62279860e-02,  6.38072905e-02, -8.74054135e-02, -9.41303820e-02,
      5.50457125e-02,  9.05124648e-02, -1.94161533e-03, -5.96854674e-02,
     -2.71024128e-02, -6.68621296e-03,  3.42843618e-02,  7.52560568e-02,
     -2.70132081e-02, -9.96208946e-02, -1.03050248e-02,  9.13886260e-02,
      4.22874630e-02, -5.63545815e-02, -4.13372405e-02, -3.79811920e-03,
      1.73229719e-02,  5.60931252e-02,  1.65100438e-02, -7.23442471e-02,
     -5.44386124e-02,  5.54356053e-02,  7.70507710e-02, -2.73170818e-02,
     -5.86510488e-02, -1.07589483e-02,  1.40807778e-02,  3.94702427e-02,
      2.82323963e-02, -3.28499613e-02, -6.71796210e-02,  7.93275391e-03,
      8.22445022e-02,  1.39160593e-02, -5.76421887e-02, -3.17600884e-02,
      1.32505073e-02,  3.21207718e-02,  2.63361076e-02, -6.90919671e-03,
     -5.33080951e-02, -2.82377451e-02,  5.94753692e-02,  4.75041764e-02,
     -3.61512507e-02, -4.90487338e-02, -8.81286153e-04,  3.39892722e-02,
      2.29106973e-02,  3.90008314e-03, -3.21750668e-02, -4.24626961e-02,
      2.63970697e-02,  5.75723545e-02, -2.03724268e-03, -5.28778636e-02,
     -2.15254735e-02,  2.96754349e-02,  2.75155132e-02,  6.09396933e-03,
     -1.69470772e-02, -3.88458441e-02, -1.75356148e-03,  4.81637463e-02,
      2.49068342e-02, -3.65212085e-02, -4.14376424e-02,  1.65900103e-02,
      3.27582689e-02,  1.06948170e-02, -1.00106892e-02, -3.00821943e-02,
     -1.52079425e-02,  2.76072255e-02,  3.82026481e-02, -1.25541835e-02,
     -4.64979231e-02, -5.36892381e-03,  3.30134665e-02,  1.90749061e-02,
     -5.81061107e-03, -2.33756014e-02, -1.88052473e-02])

    Z = np.array([ 1.55208453e+01   -0.j        , -3.14979966e-01  +11.90183337j,
     -1.19660417e-01   +3.58435564j,  1.35301819e+01  +16.26722179j,
      1.17649991e+00  +30.00554074j,  2.64322468e-01  +19.84365855j,
      1.50131988e+01  +31.70061472j,  2.14439406e+00  +46.88671322j,
     -1.20020089e-02  +35.58817162j,  1.66465351e+01  +47.45560445j,
      2.83480502e+00  +65.16066123j, -7.40668233e-01  +52.25031888j,
      1.90899766e+01  +64.78829561j,  3.64409865e+00  +86.64096686j,
     -1.89618754e+00  +71.30680463j,  2.31543213e+01  +85.46427126j,
      4.87108871e+00 +114.43025893j, -3.84886405e+00  +95.28470339j,
      3.03832416e+01 +112.85712744j,  6.98238561e+00 +155.40294211j,
     -7.73699067e+00 +129.98623541j,  4.53718242e+01 +155.7118103j ,
      1.19600766e+01 +230.86769883j, -1.71656173e+01 +195.94144974j,
      9.16485834e+01 +251.29599025j,  4.24818473e+01 +468.24911087j,
     -3.69230631e+01 +495.45466571j,  1.47585086e+03 +885.54956374j,
      4.33035526e+02-1075.05506221j,  1.55652780e+02 -155.24082635j,
     -7.80591018e+01  -94.91933544j,  3.99549331e+01 -123.22995462j,
      6.62792294e+01  +59.79987597j, -4.38347026e+01  +66.41146537j,
      3.91439259e+01  +34.51646665j,  6.13713872e+01 +179.47382472j,
     -3.11823573e+01 +189.94450983j,  7.80993710e+01 +191.87381297j,
      1.21588946e+02 +451.90820089j,  7.23799294e+01 +628.09363035j,
      1.57620844e+03 +592.14032817j,  5.45673205e+02-1054.21395899j,
      1.44565767e+02 -325.36838268j, -3.45279079e+01 -243.14931681j,
      5.42953598e+01 -204.82827496j,  3.55150868e+01  -55.22680716j,
     -3.30723734e+01  -71.8019739j ,  4.40213775e+01  -65.61836905j,
      2.57543507e+01  +39.5004643j , -2.65157620e+01   +7.50939865j])

    Ez = np.array([ 29.8853    ,   3.8917696 ,   3.5878048 ,   5.388985  ,   8.009888  ,
      18.127728  ,  45.042206  ,  99.547295  , 166.13406   , 219.75793   ,
     245.6054    , 243.57756   , 219.86266   , 181.38158   , 136.38814   ,
      92.54131   ,  54.776005  ,  25.220638  ,   5.623968  ,  -4.053369  ,
      -7.1510763 ,  -5.981537  ,  -2.5126505 ,  -0.34398004,  -0.27764353,
       0.95293194,   0.6373872 ,   1.0908693 ])

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
                    184.43818552913254, 0.1
                ), "Wake potential cumsum MPI failed"
        else:
            assert len(wake.WP) == 5135, "Wake potential samples length mismatch"
            plot_comparison(wake.WP[::50], self.WP, "Wake potential")
            assert np.allclose(wake.WP[::50], self.WP, **self.tol), (
                "Wake potential samples failed"
            )
            assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
                184.43818552913254, 0.1
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
                    250910.51090497518, 0.1
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
                250910.51090497518, 0.1
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
