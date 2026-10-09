"""TF/SF boundary and beam regressions, including optional MPI comparisons.

Run this file normally for serial tests, or with mpiexec -n 2 or -n 3 for MPI.
Add --debug-plots to show the gathered Ez field and rank bounds.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c

from wakis import GridFIT3D, SolverFIT3D
from wakis.sources import Beam
from wakis.sources.beam import Z0


@pytest.mark.parametrize(
    "low_z, high_z, source_type, expected_start, expected_stop",
    [
        ("pec", "pec", "tfsf", 1, 22),
        ("pec", "pml", "direct", 1, 19),
        ("pml", "pec", "direct", 4, 22),
    ],
)
def test_serial_tfsf_injection_and_lifecycle(
    low_z, high_z, source_type, expected_start, expected_stop
):
    """Exercise non-CPML signs, face offsets, and a complete PEC beam passage."""
    grid = GridFIT3D(-0.3, 0.3, -0.3, 0.3, 0, 1, 6, 6, 24, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["pec", "pec", low_z],
        bc_high=["pec", "pec", high_z],
        source_type=source_type,
        n_pml=3,
        verbose=0,
    )
    assert solver.source_type == "tfsf"
    assert solver.logger.solver["source_type"] == "tfsf"
    beam = Beam(sigmaz=0.04, ti=0.0)
    beam.update(solver, 0.0)
    assert (beam.j_start, beam.j_stop) == (expected_start, expected_stop)
    assert beam.high_plane == solver.z[expected_stop]
    assert set(beam._tfsf_field_templates) == {"low", "high"}
    for (
        electric_x,
        electric_y,
        magnetic_x,
        magnetic_y,
    ) in beam._tfsf_field_templates.values():
        assert electric_x.shape == electric_y.shape == (solver.Nx, solver.Ny)
        np.testing.assert_allclose(magnetic_x, -electric_y / Z0)
        np.testing.assert_allclose(magnetic_y, electric_x / Z0)

    # Initially E and H are zero, so the first H update isolates the TF/SF correction.
    expected_hx = (
        -solver.dt * solver.imu.field_x * (solver.tf_dxz * solver.E_trans.field_y)
    )
    expected_hy = (
        solver.dt * solver.imu.field_y * (solver.tf_dyz * solver.E_trans.field_x)
    )
    assert np.any(expected_hx) or np.any(expected_hy)
    solver.one_step()
    np.testing.assert_allclose(solver.H.field_x, expected_hx)
    np.testing.assert_allclose(solver.H.field_y, expected_hy)

    base_e = solver.dt * (
        solver.itDaiDepsDstC * solver.H.toarray()
        - solver.ieps.toarray() * solver.J.toarray()
    )
    expected_ex = base_e[: solver.N] - solver.dt * solver.ieps.field_x * (
        solver.tf_dtxz * solver.H_trans.field_y
    )
    expected_ey = base_e[solver.N : 2 * solver.N] + solver.dt * solver.ieps.field_y * (
        solver.tf_dtyz * solver.H_trans.field_x
    )
    np.testing.assert_allclose(solver.E.field_x, expected_ex)
    np.testing.assert_allclose(solver.E.field_y, expected_ey)

    # Continue until the beam tail leaves the high interface and injection is removed.
    assert not solver.injection_done
    for n in range(1, 600):
        beam.update(solver, n * solver.dt)
        solver.one_step()
        if solver.injection_done:
            break
    assert solver.injection_done
    assert not hasattr(solver, "E_trans")
    assert not hasattr(solver, "tf_dxz")
    assert not hasattr(beam, "_tfsf_field_templates")
    for step in range(n + 1, n + 6):
        beam.update(solver, step * solver.dt)
        solver.one_step()
    assert np.isfinite(solver.E.toarray()).all()
    assert np.any(solver.E.toarray())


def test_serial_cpml_tfsf_injection():
    """Confirm CPML selects TF/SF and uses the two physical z-face offsets."""
    grid = GridFIT3D(-0.3, 0.3, -0.3, 0.3, 0, 1, 6, 6, 24, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        source_type="direct",
        n_pml=3,
        verbose=0,
    )
    beam = Beam(sigmaz=0.04, ti=0.0)
    beam.update(solver, 0.0)
    assert solver.source_type == solver.logger.solver["source_type"] == "tfsf"
    assert (beam.j_start, beam.j_stop) == (4, 19)
    assert beam.high_plane == solver.z[19]
    solver.one_step()
    assert np.isfinite(solver.E.toarray()).all()
    assert np.any(solver.J.toarray())


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
        if n == 0 and comm.Get_rank() == 0:
            # The final physical cell on a non-last rank must receive beam current.
            assert beam.j_stop == solver.Nz - 1
            assert solver.J[beam.ixs, beam.iys, solver.Nz - 2, "z"] != 0
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


def test_007_geometry_with_tfsf_across_mpi_boundaries(request):
    """Use test 007's lossy cavity and PEC boundaries with TF/SF on 2 or 3 ranks."""
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD
    if comm.Get_size() not in (2, 3):
        pytest.skip("run with two or three MPI ranks")

    solids = {
        "cavity": "tests/stl/007_vacuum_cavity.stl",
        "shell": "tests/stl/007_lossymetal_shell.stl",
    }
    materials = {"cavity": "vacuum", "shell": [30, 1.0, 30]}
    bounds = (pv.read(solids["cavity"]) + pv.read(solids["shell"])).bounds
    # Keep test 007's geometry and materials, with fewer cells and steps for CI.
    grid_kw = dict(
        stl_solids=solids,
        stl_materials=materials,
        stl_method="implicit_distance",
        verbose=0,
    )
    solver_kw = dict(
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        use_stl=True,
        bg="pec",
        source_type="tfsf",
        dtype=np.float32,
        verbose=0,
    )
    beam_kw = dict(q=1e-9, sigmaz=0.1, beta=1.0, ti=0.3 / c)
    nx, ny, nz = 60, 60, 180  # nz must divide evenly across two or three ranks.
    x_probe, y_probe = nx // 2, ny // 2
    mpi_grid = GridFIT3D(*bounds, nx, ny, nz, use_mpi=True, **grid_kw)
    mpi_solver = SolverFIT3D(mpi_grid, use_mpi=True, **solver_kw)
    mpi_beam = Beam(**beam_kw)

    for n in range(400):
        mpi_beam.update(mpi_solver, n * mpi_solver.dt)
        if n == 0 and comm.Get_rank() < comm.Get_size() - 1:
            assert mpi_beam.j_stop == mpi_solver.Nz - 1
            assert mpi_solver.J[mpi_beam.ixs, mpi_beam.iys, mpi_solver.Nz - 2, "z"] != 0
        mpi_solver.one_step()

    mpi_ez = mpi_solver.mpi_gather("Ez", x=x_probe, y=y_probe)
    if request.config.getoption("--debug-plots"):
        # plot2D gathers on every rank; only rank 0 receives figure handles.
        local_bounds = comm.gather((mpi_grid.zmin, mpi_grid.zmax), root=0)
        handles = mpi_solver.plot2D(
            field="Ez",
            plane="ZY",
            pos=0.5,
            cmap="RdBu_r",
            interpolation="nearest",
            return_handles=True,
            figsize=(10, 6),
        )
        if comm.Get_rank() == 0:
            import matplotlib.pyplot as plt

            fig, ax = handles
            for rank, (z_start, z_end) in enumerate(local_bounds):
                color = f"C{rank}"
                ax.axvline(
                    z_start, color=color, linestyle="--", label=f"rank {rank} start"
                )
                ax.axvline(z_end, color=color, linestyle=":", label=f"rank {rank} end")
            ax.legend(loc="upper right")
            plt.show()
            plt.close(fig)

    if comm.Get_rank() == 0:
        serial_grid = GridFIT3D(*bounds, nx, ny, nz, **grid_kw)
        serial_solver = SolverFIT3D(serial_grid, **solver_kw)
        serial_beam = Beam(**beam_kw)
        for n in range(400):
            serial_beam.update(serial_solver, n * serial_solver.dt)
            serial_solver.one_step()
        np.testing.assert_allclose(
            mpi_ez, serial_solver.E[x_probe, y_probe, :, "z"], rtol=5e-3, atol=1e-3
        )
