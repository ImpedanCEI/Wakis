"""Regression coverage for FIT boundary topology and longitudinal Mur ABC.

The topology checks verify that the periodic derivative matrices wrap in all
three Cartesian directions, preserve constants and translation invariance, and
retain correct metrics and PEC/PMC masks after the curl operator is rebuilt.
The short Harris-pulse case complements those algebraic checks by confirming
the expected longitudinal response of PEC, PMC, periodic, and Mur ABC faces.
"""

import numpy as np
import pytest

from wakis import GridFIT3D, SolverFIT3D
from wakis.sources import Pulse


def make_solver(bcs=None, **kwargs):
    grid = GridFIT3D(0, 1, 0, 1, 0, 1, 4, 5, 6, verbose=0)
    bcs = ["periodic"] * 3 if bcs is None else bcs
    return SolverFIT3D(grid, bc_low=bcs.copy(), bc_high=bcs.copy(), verbose=0, **kwargs)


@pytest.mark.parametrize("axis,name", enumerate("xyz"))
def test_periodic_derivative_matches_coordinate_roll(axis, name):
    solver = make_solver()
    values = np.random.default_rng(1).normal(size=(4, 5, 6))
    derivative = getattr(solver, "P" + name)
    expected = np.roll(values, -1, axis=axis) - values
    np.testing.assert_allclose(
        derivative @ values.ravel(order="F"), expected.ravel(order="F")
    )
    np.testing.assert_array_equal(derivative @ np.ones(solver.N), 0)
    np.testing.assert_array_equal(derivative.T @ np.ones(solver.N), 0)


def test_constant_fields_survive_periodic_first_step():
    solver = make_solver()
    solver.E.array[:] = 2.0
    solver.H.array[:] = 3.0
    solver.one_step()
    np.testing.assert_allclose(solver.E.array, 2.0, atol=1e-12)
    np.testing.assert_allclose(solver.H.array, 3.0, atol=1e-12)


def _longitudinal_harris_probe(boundary):
    """Return a probe trace for a transverse pulse in a 1-D-like domain."""
    grid = GridFIT3D(0, 0.02, 0, 0.02, 0, 1, 2, 2, 80, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", boundary],
        bc_high=["periodic", "periodic", boundary],
        verbose=0,
    )
    source = Pulse(
        field="Ex",
        xs=slice(0, solver.Nx),
        ys=slice(0, solver.Ny),
        zs=30,
        shape="harris",
        L=0.12,
    )
    trace = np.empty(280)
    for step in range(len(trace)):
        source.update(solver, step * solver.dt)
        solver.one_step()
        trace[step] = solver.E[1, 1, 8, "x"]
    return trace


def test_longitudinal_harris_pulse_boundary_signatures():
    """PEC, PMC, and periodic faces give distinct expected pulse returns."""
    traces = {
        boundary: _longitudinal_harris_probe(boundary)
        for boundary in ("pec", "pmc", "periodic")
    }

    # The initial left-going pulse has not reached a longitudinal face yet.
    for trace in traces.values():
        assert trace[100:130].max() > 0.8

    # A tangential electric field reflects with opposite sign at PEC and the
    # same sign at PMC.  A periodic face has no conductor-style return here.
    assert traces["pec"][155:195].min() < -0.7
    assert traces["pmc"][155:195].max() > 0.7
    assert np.abs(traces["periodic"][155:195]).max() < 0.1

    # The periodic wave subsequently reaches the probe after wrapping around.
    assert traces["periodic"][220:275].max() > 0.6


def test_abc_pairs_apply_mur_to_tangential_electric_field():
    """Both z faces use their current interior and prior-time E planes."""
    grid = GridFIT3D(0, 1, 0, 1, 0, 1, 4, 5, 6, verbose=0)
    bc = ["periodic", "periodic", "abc"]
    solver = SolverFIT3D(grid, bc_low=bc.copy(), bc_high=bc.copy(), verbose=0)

    rng = np.random.default_rng(42)
    solver.E.array[:] = rng.normal(size=solver.E.array.size)
    solver.H.array[:] = rng.normal(size=solver.H.array.size)
    old_E = {d: solver.E.to_matrix(d).copy() for d in "xyz"}
    solver.one_step()

    for d in "xy":
        values = solver.E.to_matrix(d)
        expected_low = old_E[d][:, :, 1] + solver.abc_low_coeff * (
            values[:, :, 1] - old_E[d][:, :, 0]
        )
        expected_high = old_E[d][:, :, -2] + solver.abc_high_coeff * (
            values[:, :, -2] - old_E[d][:, :, -1]
        )
        np.testing.assert_allclose(values[:, :, 0], expected_low)
        np.testing.assert_allclose(values[:, :, -1], expected_high)


def test_abc_does_not_modify_metric_operators():
    """Mur is a field update, so it leaves FIT metrics unchanged."""
    abc_grid = GridFIT3D(0, 1, 0, 1, 0, 1, 4, 4, 4, verbose=0)
    reference_grid = GridFIT3D(0, 1, 0, 1, 0, 1, 4, 4, 4, verbose=0)
    abc = SolverFIT3D(
        abc_grid,
        bc_low=["periodic", "periodic", "abc"],
        bc_high=["periodic", "periodic", "abc"],
        verbose=0,
    )
    reference = SolverFIT3D(
        reference_grid,
        bc_low=["periodic", "periodic", "pmc"],
        bc_high=["periodic", "periodic", "pmc"],
        verbose=0,
    )
    np.testing.assert_array_equal(abc.tDs.diagonal(), reference.tDs.diagonal())
    np.testing.assert_array_equal(abc.itDa.diagonal(), reference.itDa.diagonal())


@pytest.mark.parametrize("axis", [0, 1])
def test_transverse_abc_is_rejected(axis):
    """The first Mur implementation is deliberately longitudinal only."""
    grid = GridFIT3D(0, 1, 0, 1, 0, 1, 4, 4, 4, verbose=0)
    bc = ["periodic", "periodic", "periodic"]
    bc[axis] = "abc"
    with pytest.raises(NotImplementedError, match="longitudinal z faces"):
        SolverFIT3D(grid, bc_low=bc.copy(), bc_high=bc.copy(), verbose=0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_periodic_evolution_is_translation_invariant(axis):
    reference = make_solver()
    shifted = make_solver()
    rng = np.random.default_rng(12)
    for field in ("E", "H"):
        for component in "xyz":
            values = rng.normal(size=(4, 5, 6))
            getattr(reference, field).from_matrix(values, component)
            getattr(shifted, field).from_matrix(
                np.roll(values, 1, axis=axis), component
            )
    for _ in range(5):
        reference.one_step()
        shifted.one_step()
    for field in ("E", "H"):
        for component in "xyz":
            np.testing.assert_allclose(
                getattr(shifted, field).to_matrix(component),
                np.roll(getattr(reference, field).to_matrix(component), 1, axis=axis),
                rtol=1e-11,
                atol=1e-11,
            )


def test_nonuniform_periodic_metrics():
    grid = GridFIT3D(
        x=np.array([0.0, 0.1, 0.4, 1.0]),
        y=np.array([0.0, 0.2, 0.5, 1.0]),
        z=np.array([0.0, 0.3, 0.7, 1.0]),
        verbose=0,
    )
    solver = SolverFIT3D(grid, verbose=0)
    for axis, name in enumerate("xyz"):
        widths = getattr(grid, "d" + name)
        expected = np.concatenate(
            ((widths[:-1] + widths[1:]) / 2, [(widths[-1] + widths[0]) / 2])
        )
        shape = [1, 1, 1]
        shape[axis] = len(widths)
        np.testing.assert_allclose(
            solver.tL.to_matrix(name),
            np.broadcast_to(expected.reshape(shape), (3, 3, 3)),
        )
    np.testing.assert_allclose(
        solver.itA.field_x, 1 / (solver.tL.field_y * solver.tL.field_z)
    )


@pytest.mark.parametrize("boundary", ["pec", "pmc"])
def test_rebuilt_curl_preserves_conductor_masks(boundary):
    solver = make_solver(["periodic", "periodic", boundary])
    masked = np.flatnonzero(solver.BC.toarray() == 0)
    curl = solver.C.tocsr()
    selected = curl[:, masked] if boundary == "pec" else curl[masked, :]
    assert np.count_nonzero(selected.toarray()) == 0


def test_unpaired_periodic_rejected():
    grid = GridFIT3D(0, 1, 0, 1, 0, 1, 4, 4, 4, verbose=0)
    with pytest.raises(ValueError, match="must be paired"):
        SolverFIT3D(
            grid, bc_low=["pec"] * 3, bc_high=["pec", "pec", "periodic"], verbose=0
        )
