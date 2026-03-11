import numpy as np

from wakis import GridFIT3D, SolverFIT3D, WakeSolver
from wakis.field import Field
from wakis.field_monitors import FieldMonitor


def _make_constant_field(Nx=2, Ny=2, Nz=2, values=(1.0, 2.0, 3.0)):
    """Create a Field with spatially constant components."""

    field = Field(Nx, Ny, Nz, dtype=float, use_ones=False, use_gpu=False)

    Ex = np.full((Nx, Ny, Nz), values[0], dtype=float)
    Ey = np.full((Nx, Ny, Nz), values[1], dtype=float)
    Ez = np.full((Nx, Ny, Nz), values[2], dtype=float)

    field.from_matrix(Ex, "x")
    field.from_matrix(Ey, "y")
    field.from_matrix(Ez, "z")

    return field


def test_initial_state():
    freqs = [1.0e9, 2.0e9]
    monitor = FieldMonitor(freqs)

    assert isinstance(monitor.frequencies, np.ndarray)
    assert np.allclose(monitor.frequencies, freqs)
    assert monitor.time_index == 0
    assert monitor.dt is None
    assert monitor.Ex_acc is None
    assert monitor.Ey_acc is None
    assert monitor.Ez_acc is None
    assert monitor.shape is None
    assert monitor.xp is None


def test_single_update_initializes_and_accumulates():
    dt = 1.0e-9
    field = _make_constant_field(Nx=2, Ny=2, Nz=2, values=(1.0, 2.0, 3.0))
    monitor = FieldMonitor([0.0])  # zero frequency -> unit phase

    monitor.update(field, dt)
    comps = monitor.get_components()

    assert monitor.dt == dt
    assert monitor.shape == (2, 2, 2)
    assert monitor.time_index == 1
    assert monitor.xp is field.xp

    for key in ("Ex", "Ey", "Ez"):
        arr = comps[key]
        assert arr is not None
        assert arr.shape == (1, 2, 2, 2)
        assert np.iscomplexobj(arr)

    Ex_expected = field.to_matrix("x")
    Ey_expected = field.to_matrix("y")
    Ez_expected = field.to_matrix("z")

    assert np.allclose(comps["Ex"][0], Ex_expected)
    assert np.allclose(comps["Ey"][0], Ey_expected)
    assert np.allclose(comps["Ez"][0], Ez_expected)


def test_multiple_updates_accumulate_linearly_for_zero_frequency():
    dt = 0.1
    n_steps = 5
    field = _make_constant_field(Nx=2, Ny=2, Nz=2, values=(1.0, 0.0, 0.0))
    monitor = FieldMonitor([0.0])  # exp(0) == 1 for all timesteps

    for _ in range(n_steps):
        monitor.update(field, dt)

    comps = monitor.get_components()

    assert monitor.time_index == n_steps
    assert monitor.dt == dt  # dt is fixed at first call

    Ex_expected = n_steps * field.to_matrix("x")
    assert np.allclose(comps["Ex"][0], Ex_expected)
    assert np.allclose(comps["Ey"], 0.0)
    assert np.allclose(comps["Ez"], 0.0)


def test_frequency_phase_accumulation_matches_geometric_series():
    # Choose f and dt such that the complex exponential alternates sign:
    # r = exp(-2j*pi*f*dt) with f*dt = 0.5 -> r = exp(-j*pi) = -1.
    f = 0.5
    dt = 1.0
    n_steps = 4

    field = _make_constant_field(Nx=1, Ny=1, Nz=1, values=(1.0, 0.0, 0.0))
    monitor = FieldMonitor([f])

    for _ in range(n_steps):
        monitor.update(field, dt)

    comps = monitor.get_components()
    Ex = comps["Ex"][0]

    # Geometric series: sum_{n=0}^{3} (-1)^n = 0
    assert np.allclose(Ex, 0.0, atol=1e-12)


def test_get_components_before_update_returns_none_accumulators():
    monitor = FieldMonitor([1.0])
    comps = monitor.get_components()

    assert comps["Ex"] is None
    assert comps["Ey"] is None
    assert comps["Ez"] is None


def test_field_monitor_integration_with_wakesolve():
    """Run a tiny wake simulation with a FieldMonitor attached.

    This exercises the integration between SolverFIT3D.wakesolve and
    FieldMonitor.update, ensuring that the monitor is updated during the
    time-stepping loop and produces non-empty accumulators.
    """

    # Small, uniform grid
    Nx = Ny = Nz = 4
    grid = GridFIT3D(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        zmin=0.0,
        zmax=1.0,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        verbose=0,
    )

    # Minimal wake configuration (no STL). Use a dedicated Ez file so that
    # the solver writes its Ez snapshots without interfering with other
    # tests or examples.
    ez_file = "tests/012_Ez_monitor.h5"
    wake = WakeSolver(
        q=1e-9,
        sigmaz=1e-2,
        beta=1.0,
        xsource=0.0,
        ysource=0.0,
        xtest=0.0,
        ytest=0.0,
        Ez_file=ez_file,
        save=False,
        results_folder="results/",
    )

    # For this integration test we only care that ``wakesolve`` runs its
    # time-stepping loop and calls the field monitor. The heavy wake
    # post-processing (HDF5 reads, impedance calculations, etc.) is
    # exercised elsewhere in the test suite, so we replace ``solve`` with
    # a lightweight no-op to keep this test fast and robust.
    wake.solve = lambda *args, **kwargs: None

    solver = SolverFIT3D(
        grid,
        wake,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        use_stl=False,
        bg="pec",
        dtype=np.float32,
        use_gpu=False,
    )

    # Attach a monitor at a single frequency
    monitor = FieldMonitor([1.0e9])

    # Run a very short wake simulation to keep the test fast
    wakelength = 0.1  # [m]
    solver.wakesolve(
        wakelength=wakelength,
        add_space=0,
        plot=False,
        use_field_monitor=True,
        field_monitor=monitor,
    )

    comps = monitor.get_components()

    # After the run, the monitor should have been initialized and updated
    assert monitor.dt is not None
    assert monitor.time_index > 0
    assert monitor.shape == (Nx, Ny, Nz)

    for key in ("Ex", "Ey", "Ez"):
        arr = comps[key]
        assert arr is not None
        assert arr.shape[0] == 1  # one frequency
        assert arr.shape[1:] == (Nx, Ny, Nz)

    # Clean up Ez file created for this test if it exists
    import os

    if os.path.exists(ez_file):
        os.remove(ez_file)
