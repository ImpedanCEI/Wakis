"""Shared solver fixtures and propagation runner for source tests."""

import numpy as np
import pytest

from wakis import GridFIT3D, SolverFIT3D


@pytest.fixture
def unit_solver():
    """Return a small solver for focused source-profile and injection tests."""
    grid = GridFIT3D(
        xmin=-0.05,
        xmax=0.05,
        ymin=-0.05,
        ymax=0.05,
        zmin=0.0,
        zmax=0.1,
        Nx=5,
        Ny=5,
        Nz=8,
        verbose=0,
    )
    return SolverFIT3D(grid, source_type="direct", verbose=0)


@pytest.fixture
def propagation_solver(request):
    """Return the larger solver used by end-to-end propagation tests.

    The transverse boundary defaults to periodic and can be selected through
    indirect parametrization when a source requires a different boundary.
    """
    transverse_boundary = getattr(request, "param", "periodic")
    grid = GridFIT3D(
        xmin=-0.03,
        xmax=0.03,
        ymin=-0.03,
        ymax=0.03,
        zmin=0.0,
        zmax=0.12,
        Nx=7,
        Ny=7,
        Nz=40,
        verbose=0,
    )
    return SolverFIT3D(
        grid,
        bc_low=[transverse_boundary, transverse_boundary, "abc"],
        bc_high=[transverse_boundary, transverse_boundary, "abc"],
        source_type="direct",
        verbose=0,
    )


@pytest.fixture
def simulate_source(request):
    """Run a source and return its injection, propagation, and profile traces."""

    def simulate(
        solver,
        source,
        injected_component,
        observed_component,
        companion_component,
        *,
        probe_z=20,
        steps=60,
    ):
        center_x = solver.Nx // 2
        center_y = solver.Ny // 2
        source_z = source.zs
        injected_field, injected_axis = injected_component
        observed_field, observed_axis = observed_component
        companion_field, companion_axis = companion_component

        injected = np.empty(steps)
        observed = np.empty(steps)
        companion = np.empty(steps)
        profile_x = np.empty((steps, solver.Nx))
        profile_y = np.empty((steps, solver.Ny))

        for step in range(steps):
            time = step * solver.dt
            source.update(solver, time)
            injected[step] = getattr(solver, injected_field)[
                center_x, center_y, source_z, injected_axis
            ]

            solver.one_step()
            field = getattr(solver, observed_field)
            observed[step] = field[center_x, center_y, probe_z, observed_axis]
            companion[step] = getattr(solver, companion_field)[
                center_x, center_y, probe_z, companion_axis
            ]
            profile_x[step] = field[:, center_y, probe_z, observed_axis]
            profile_y[step] = field[center_x, :, probe_z, observed_axis]

        result = {
            "time": np.arange(steps) * solver.dt,
            "injected": injected,
            "observed": observed,
            "companion": companion,
            "profile_x": profile_x,
            "profile_y": profile_y,
        }

        # Every end-to-end source test shares this basic propagation contract.
        assert np.max(np.abs(injected)) > 0.0
        assert np.max(np.abs(observed)) > 1e-8
        injected_start = np.flatnonzero(
            np.abs(injected) > 1e-4 * np.max(np.abs(injected))
        )[0]
        observed_start = np.flatnonzero(
            np.abs(observed) > 1e-4 * np.max(np.abs(observed))
        )[0]
        assert observed_start > injected_start

        if request.config.getoption("--debug-plots"):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            for label, values in {
                "Injection plane": injected,
                "Downstream field": observed,
                "Companion field": companion,
            }.items():
                scale = np.max(np.abs(values))
                normalized = values if scale == 0 else values / scale
                ax.plot(result["time"] * 1e9, normalized, label=label)
            ax.set(
                title=request.node.name,
                xlabel="Time [ns]",
                ylabel="Normalized field",
            )
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            plt.show()
            plt.close(fig)

        return result

    return simulate
