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
    # Regression data
    # fmt: off
    tol = dict(rtol=50e-3, atol=50e-3)
    dtype = np.float32

    WP = np.array([-1.20513623e-18 ,-3.43161145e-14 ,-9.12255548e-11 ,-5.96997512e-08,
                    -1.14173019e-05 ,-6.67499094e-04 ,-1.20239747e-02 ,-6.53062215e-02,
                    -9.26520639e-02 , 2.78321623e-02  ,1.25518516e-01 , 7.30670041e-02,
                    -1.90239810e-02 ,-4.85596408e-02 ,-3.20357619e-02 ,-1.41604640e-02,
                    6.08356174e-02 , 8.46570691e-02 ,-5.47181776e-02 ,-1.16180823e-01,
                    1.96111216e-02 , 9.92679050e-02 , 2.71878152e-02 ,-5.20035156e-02,
                    -4.06137472e-02 ,-1.13185667e-02 , 1.23530805e-02 , 7.68524041e-02,
                    2.25859758e-02 ,-9.37436643e-02 ,-5.75287185e-02 , 6.39890431e-02,
                    8.01280003e-02 ,-2.25671230e-02 ,-5.80154107e-02 ,-1.67525750e-02,
                    1.87720028e-03 , 4.10247646e-02  ,5.36731566e-02 ,-3.30682943e-02,
                    -8.35234400e-02 ,-3.49965114e-03 , 8.57976513e-02 , 3.52580927e-02,
                    -5.59291793e-02 ,-3.74168173e-02 ,-1.57790071e-03 , 2.16477574e-02,
                    4.65687493e-02 , 1.13356268e-02 ,-5.61946649e-02 ,-5.45040228e-02,
                    4.41993189e-02  ,7.38920452e-02 ,-1.60854441e-02 ,-5.54896162e-02,
                    -1.77145556e-02 , 1.61884877e-02 , 3.33170207e-02 , 2.63165394e-02,
                    -1.83733826e-02 ,-5.94726314e-02 ,-1.15493543e-02 , 6.90379083e-02,
                    3.30361171e-02 ,-4.32087737e-02 ,-4.14862834e-02 , 2.95742481e-03,
                    3.11653273e-02 , 2.43513549e-02 , 5.17266702e-03 ,-3.70591561e-02,
                    -4.24818163e-02 , 3.09681557e-02 , 5.76902933e-02 ,-4.78519848e-03,
                    -5.02625318e-02 ,-2.09821868e-02 , 2.46893459e-02 , 2.77009485e-02,
                    1.12395692e-02 ,-1.34117335e-02 ,-4.22375110e-02 ,-8.34037808e-03,
                    4.81709861e-02 , 3.05819110e-02 ,-2.98222467e-02 ,-4.38079436e-02,
                    6.51694708e-03 , 3.11637911e-02 , 1.61475003e-02 ,-1.03231050e-03,
                    -2.84581274e-02 ,-2.64695160e-02  ,1.92431962e-02 , 4.34036616e-02,
                    3.00897180e-03 ,-4.36364833e-02 ,-2.09601909e-02 , 2.48741294e-02,
                    2.47321674e-02  ,5.34334063e-03 ,-1.56819510e-02 ,-2.75503078e-02])

    Z = np.array([ 1.39494420e+01   -0.j,   -1.92596139e+00  +12.14277743j,
                    -2.04348603e+00   +3.02851583j,  1.22695563e+01  +15.8399473j,
                    -6.50605645e-01  +29.9225005j,  -1.65424766e+00  +18.94989745j,
                    1.37576308e+01  +31.01645375j,  2.65950658e-01  +46.55181503j,
                    -1.97534475e+00  +34.40198795j,  1.53779924e+01  +46.52226957j,
                    8.60779742e-01  +64.60337834j, -2.81193463e+00  +50.74670707j,
                    1.77811468e+01  +63.57088258j,  1.52087428e+00  +85.82446456j,
                    -4.10843816e+00  +69.38387661j,  2.18343119e+01  +83.88119354j,
                    2.59651772e+00 +113.28087363j, -6.14559798e+00  +92.80429367j,
                    2.91887955e+01 +110.85988556j,  4.60993585e+00 +153.89071454j,
                    -9.94451904e+00 +126.83885701j,  4.45478983e+01 +153.43414465j,
                    9.47258576e+00 +229.06989569j, -1.89013483e+01 +192.00422647j,
                    9.15962914e+01 +249.432772j,    3.91409984e+01 +465.98109678j,
                    -3.54400946e+01 +488.20015838j,  1.46569804e+03 +892.95179609j,
                    4.50786904e+02-1081.60659854j,  1.48737786e+02 -160.81052458j,
                    -7.68871185e+01 -106.30116174j,  4.12323110e+01 -125.48455177j,
                    5.87559863e+01  +53.38395571j, -4.24625985e+01  +54.36630224j,
                    3.99792315e+01  +33.96133138j,  4.96479510e+01 +169.90128475j,
                    -2.68588737e+01 +170.13079256j,  8.04887980e+01 +193.09475383j,
                    9.01859274e+01 +426.74907513j,  7.31057816e+01 +541.50361688j,
                    1.40645468e+03 +756.68000225j,  7.28899882e+02-1078.53088928j,
                    1.17812854e+02 -322.42673477j, -4.35713455e+01 -287.17594956j,
                    7.52009772e+01 -210.85748915j,  2.00727569e+01  -50.34061675j,
                    -4.02209245e+01  -97.78101888j,  5.93485923e+01  -70.71854011j,
                    1.83947873e+01  +43.94667805j, -3.23582668e+01  -20.89422029j])

    Ez = np.array([ 173.0133,     -16.503159,    -7.379827,    -7.106013,   -10.790575,
                    -2.9021797,   -9.144537,   -23.47881,    -46.128967,   -71.88119,
                    -88.91851,   -106.84356,   -117.83744,   -136.86569,   -144.82545,
                    -146.83551,   -142.0079,    -137.68463,   -117.18314,    -92.1836,
                    -58.804626,   -31.48718,     -6.6670427,   -5.2338524,   -3.8040538,
                    -4.508667,    -1.3254653,    3.6782658])

    # fmt: on

    gridLogs = {
        "use_mesh_refinement": False,
        "Nx": 60,
        "Ny": 60,
        "Nz": 140,
        "dx": 0.00866666634877522,
        "dy": 0.00866666634877522,
        "dz": 0.005714285799435207,
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
        "dt": dtype(6.970326659611059e-12),
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

    def test_mpi_simulation(self, use_gpu):
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
                assert np.allclose(Ez[np.s_[::5]], self.Ez, **self.tol), (
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
            assert np.allclose(Ez, self.Ez, **self.tol), (
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

    def test_long_wake_potential(self):
        global wake
        global solver
        if use_mpi:
            if solver.rank == 0:
                tol = dict(rtol=0.1)
                assert len(wake.WP) == 5195, (
                    "Wake potential MPI samples length mismatch"
                )
                assert np.allclose(wake.WP[::50], self.WP, **tol), (
                    "Wake potential MPI samples failed"
                )
                assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
                    184.43818552913254, 0.1
                ), "Wake potential cumsum MPI failed"
        else:
            assert len(wake.WP) == 5195, "Wake potential samples length mismatch"
            assert np.allclose(wake.WP[::50], self.WP, **self.tol), (
                "Wake potential samples failed"
            )
            assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(
                184.43818552913254, 0.1
            ), "Wake potential cumsum MPI failed"

    def test_long_impedance(self):
        global wake
        global solver
        if use_mpi:
            if solver.rank == 0:
                tol = dict(rtol=0.1)
                assert len(wake.Z) == 998, "Impedance samples length mismatch"
                assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z), **tol), (
                    "Abs Impedance samples MPI failed"
                )
                assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z), **tol), (
                    "Real Impedance samples MPI failed"
                )
                assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z), **tol), (
                    "Imag Impedance samples MPI failed"
                )
                assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(
                    250910.51090497518, 0.1
                ), "Abs Impedance cumsum MPI failed"
        else:
            # print(wake.Z[::20])
            assert len(wake.Z) == 998, "Impedance samples length mismatch"
            assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z), **self.tol), (
                "Abs Impedance samples failed"
            )
            assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z), **self.tol), (
                "Real Impedance samples failed"
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
