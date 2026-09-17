"""Run with: mpiexec -n 2 python -m pytest -q tests/test_015_tfsf_mpi.py"""

import numpy as np
import pytest

from wakis import GridFIT3D, SolverFIT3D
from wakis.sources import Beam


def test_tfsf_matches_serial_across_mpi_boundary():
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        pytest.skip("requires exactly two MPI ranks")

    bounds = (-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 6, 6, 40)
    solver_kw = dict(
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        n_pml=3,
        bg=[1.0, 1.0, 0.0],
        verbose=0,
    )
    grid = GridFIT3D(*bounds, use_mpi=True, verbose=0)
    solver = SolverFIT3D(grid, use_mpi=True, **solver_kw)
    beam = Beam(sigmaz=0.1, ti=1e-9)
    for n in range(180):
        beam.update(solver, n * solver.dt)
        solver.one_step()

    mpi_ez = solver.mpi_gather("Ez", x=3, y=3)
    if comm.Get_rank() == 0:
        serial_grid = GridFIT3D(*bounds, verbose=0)
        serial_solver = SolverFIT3D(serial_grid, **solver_kw)
        serial_beam = Beam(sigmaz=0.1, ti=1e-9)
        for n in range(180):
            serial_beam.update(serial_solver, n * serial_solver.dt)
            serial_solver.one_step()
        np.testing.assert_allclose(
            mpi_ez, serial_solver.E[3, 3, :, "z"], rtol=1e-8, atol=1e-9
        )
