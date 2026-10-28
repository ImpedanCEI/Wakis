"""Test beam-driven TF/SF boundary setup, field updates, and MPI propagation.

Serial tests cover boundary offsets, field physics, and precision. The MPI test
checks propagation through the lossy STL geometry used by test 007. Run with
``--debug-plots`` to display the gathered field and rank bounds.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.constants import c

from wakis import GridFIT3D, SolverFIT3D
from wakis.sources import Beam
from wakis.sources.beam import Z0


def _make_serial_solver(
    low_z="pec", high_z="pec", source_type="tfsf", dtype=np.float64
):
    """Create the small serial solver used by focused TF/SF tests."""
    grid = GridFIT3D(
        xmin=-0.3,
        xmax=0.3,
        ymin=-0.3,
        ymax=0.3,
        zmin=0,
        zmax=1,
        Nx=6,
        Ny=6,
        Nz=24,
        verbose=0,
    )
    return SolverFIT3D(
        grid=grid,
        bc_low=["pec", "pec", low_z],
        bc_high=["pec", "pec", high_z],
        source_type=source_type,
        n_pml=3,
        dtype=dtype,
        verbose=0,
    )


@pytest.mark.parametrize(
    "low_z, high_z, source_type, expected_start, expected_stop",
    [
        ("pec", "pec", "tfsf", 1, 22),
        ("pec", "pml", "direct", 1, 19),
        ("pml", "pec", "direct", 4, 22),
        ("cpml", "cpml", "direct", 4, 19),
    ],
)
def test_serial_tfsf_boundary_configuration(
    low_z, high_z, source_type, expected_start, expected_stop
):
    """Check TF/SF selection and injection-plane offsets at each boundary."""
    solver = _make_serial_solver(
        low_z=low_z,
        high_z=high_z,
        source_type=source_type,
    )

    # Solver setup selects TF/SF automatically when the boundary requires it.
    assert solver.source_type == "tfsf"

    # Updating the source locates its injection planes for this boundary setup.
    beam = Beam(sigmaz=0.04, ti=0.0)
    beam.update(solver, 0.0)
    assert (beam.j_start, beam.j_stop) == (expected_start, expected_stop)
    assert beam.high_plane == solver.z[expected_stop]

    # Confirm that injection produces a valid first solver update.
    solver.one_step()
    assert np.isfinite(solver.E.toarray()).all()
    assert np.any(solver.J.toarray())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_serial_tfsf_field_update_preserves_precision(dtype):
    """Check TF/SF field physics in both supported solver precisions."""
    solver = _make_serial_solver(dtype=dtype)
    beam = Beam(sigmaz=0.04, ti=0.0)
    beam.update(solver, 0.0)

    # The injected fields and corrections must use the solver's selected precision.
    assert solver.E_trans.toarray().dtype == dtype
    assert solver.H_trans.toarray().dtype == dtype
    for operator_name in ("tf_dxz", "tf_dyz", "tf_dtxz", "tf_dtyz"):
        assert getattr(solver, operator_name).dtype == dtype

    # The public slice method returns E and H in the solver precision.
    injected_fields = beam.get_injected_2D_slice(
        solver, solver.z[beam.j_start], 0.0, side="low"
    )
    assert all(field.dtype == dtype for field in injected_fields)
    electric_x, electric_y, magnetic_x, magnetic_y = injected_fields
    # Templates divide by impedance before scaling by the beam current, while
    # these checks scale first; allow a few float32 rounding steps.
    field_tol = dict(rtol=5e-7) if dtype == np.float32 else {}
    np.testing.assert_allclose(magnetic_x, -electric_y / Z0, **field_tol)
    np.testing.assert_allclose(magnetic_y, electric_x / Z0, **field_tol)

    # With zero initial fields, the first H update isolates the TF/SF correction.
    expected_hx = (
        -solver.dt * solver.imu.field_x * (solver.tf_dxz * solver.E_trans.field_y)
    )
    expected_hy = (
        solver.dt * solver.imu.field_y * (solver.tf_dyz * solver.E_trans.field_x)
    )
    assert np.any(expected_hx) or np.any(expected_hy)
    solver.one_step()
    assert solver.E.toarray().dtype == dtype
    assert solver.H.toarray().dtype == dtype
    np.testing.assert_allclose(solver.H.field_x, expected_hx, **field_tol)
    np.testing.assert_allclose(solver.H.field_y, expected_hy, **field_tol)

    # The corresponding E update combines the regular curl/current terms and TF/SF.
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
    np.testing.assert_allclose(solver.E.field_x, expected_ex, **field_tol)
    np.testing.assert_allclose(solver.E.field_y, expected_ey, **field_tol)


def test_serial_tfsf_injection_lifecycle():
    """Remove temporary TF/SF state after the beam crosses the domain."""
    solver = _make_serial_solver()
    beam = Beam(sigmaz=0.04, ti=0.0)
    beam.update(solver, 0.0)

    # Continue until the beam tail passes the high plane and injection completes.
    assert not solver.injection_done
    for n in range(1, 600):
        beam.update(solver, n * solver.dt)
        solver.one_step()
        if solver.injection_done:
            break
    assert solver.injection_done
    # The public solver fields remain finite during subsequent regular updates.
    for step in range(n + 1, n + 6):
        beam.update(solver, step * solver.dt)
        solver.one_step()
    assert np.isfinite(solver.E.toarray()).all()
    assert np.any(solver.E.toarray())


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


def test_tfsf_matches_serial_across_mpi_boundary():
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD
    if comm.Get_size() not in (2, 3):
        pytest.skip("run with two or three MPI ranks")

    # With three ranks the middle rank has no CPML face and steps with
    # `_mpi_one_step`; its ghost exchanges must match those of the CPML ranks.
    bounds = (-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 6, 6, 42)
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
    mpi_e = solver.mpi_gather_asField("E")
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
        np.testing.assert_allclose(
            mpi_e.toarray(), serial_solver.E.toarray(), rtol=1e-8, atol=1e-9
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

    # Propagate the beam and verify the source reaches each non-final rank boundary.
    for n in range(400):
        mpi_beam.update(mpi_solver, n * mpi_solver.dt)
        if n == 0 and comm.Get_rank() < comm.Get_size() - 1:
            assert mpi_beam.j_stop == mpi_solver.Nz - 1
            assert mpi_solver.J[mpi_beam.ixs, mpi_beam.iys, mpi_solver.Nz - 2, "z"] != 0
        mpi_solver.one_step()

    # Gather a probe line for comparison with a serial run on rank zero.
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
        # Check that decomposition leaves the sampled physical field unchanged.
        np.testing.assert_allclose(
            mpi_ez, serial_solver.E[x_probe, y_probe, :, "z"], rtol=5e-3, atol=1e-3
        )


def test_mpi_wake_output(tmp_path):
    """Gather physical Ez edges and reproduce serial wake integration."""
    import h5py

    from wakis import WakeSolver

    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD
    if comm.Get_size() not in (2, 3):
        pytest.skip("run with two or three MPI ranks")
    folder = comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None, root=0)
    bounds = (-0.04, 0.04, -0.04, 0.04, -0.04, 0.08, 6, 6, 24)
    options = dict(
        bc_low=["pec", "pec", "cpml"],
        bc_high=["pec", "pec", "cpml"],
        n_pml=2,
        verbose=0,
    )
    wake_options = dict(
        q=1e-12,
        sigmaz=0.01,
        ti=0.01 / c,
        skip_cells=2,
        results_folder=folder,
        save=False,
        verbose=0,
    )
    wake = WakeSolver(Ez_file=folder + "/mpi.h5", **wake_options)
    grid = GridFIT3D(*bounds, use_mpi=True, verbose=0)
    solver = SolverFIT3D(grid, wake, use_mpi=True, **options)
    solver.wakesolve(wakelength=0.03)

    if comm.Get_rank() == 0:
        serial_wake = WakeSolver(Ez_file=folder + "/serial.h5", **wake_options)
        serial_grid = GridFIT3D(*bounds, verbose=0)
        serial_solver = SolverFIT3D(serial_grid, serial_wake, **options)
        serial_solver.wakesolve(wakelength=0.03)
        with (
            h5py.File(wake.Ez_file) as mpi_hf,
            h5py.File(serial_wake.Ez_file) as serial_hf,
        ):
            for key in ("x", "y", "z", "dz", "t"):
                np.testing.assert_allclose(mpi_hf[key][()], serial_hf[key][()])
            for key in mpi_hf:
                if key.startswith("#"):
                    np.testing.assert_allclose(
                        mpi_hf[key][()], serial_hf[key][()], rtol=1e-8, atol=1e-9
                    )
        np.testing.assert_allclose(wake.s, serial_wake.s)
        np.testing.assert_allclose(wake.WP, serial_wake.WP, rtol=1e-8, atol=1e-12)
