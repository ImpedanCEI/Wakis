"""Test pulse profiles, component injection, plotting, and propagation."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c

from wakis.sources import Pulse


@pytest.mark.parametrize(
    "shape,time_factor,expected",
    [
        ("harris", 0.5, 1.0),
        ("gaussian", 0.5, 1.0),
        ("rectangular", 0.5, 1.0),
    ],
)
def test_source_updates_component(unit_solver, shape, time_factor, expected):
    length = 0.02
    source = Pulse(field="Jz", shape=shape, L=length, amplitude=2.0)
    time = time_factor * length / c

    source.update(unit_solver, time)

    center = (unit_solver.Nx // 2, unit_solver.Ny // 2, unit_solver.Nz // 2)
    np.testing.assert_allclose(
        unit_solver.J[center[0], center[1], center[2], "z"], 2.0 * expected
    )
    assert not np.any(unit_solver.E.toarray())
    assert not np.any(unit_solver.H.toarray())


def test_default_pulse_length_is_expressed_as_distance(unit_solver):
    source = Pulse()

    source.update(unit_solver, 0.0)

    assert source.L == pytest.approx(50 * c * unit_solver.dt)


def test_pulse_plot_shows_injected_component(monkeypatch):
    length = 0.02
    source = Pulse(field="Jz", shape="gaussian", L=length, amplitude=2.0)
    time = np.array([length / (2 * c)])
    monkeypatch.setattr(plt, "show", lambda: None)

    source.plot(time)

    line = plt.gcf().axes[0].lines[0]
    assert line.get_label() == "Jz"
    assert line.get_ydata()[0] == pytest.approx(2.0)
    plt.close()


def test_pulse_propagates_a_transverse_harris_wave(propagation_solver, simulate_source):
    source = Pulse(
        field="Ex",
        xs=slice(None),
        ys=slice(None),
        zs=8,
        shape="harris",
        L=8 * c * propagation_solver.dt,
    )

    result = simulate_source(
        propagation_solver, source, ("E", "x"), ("E", "x"), ("H", "y")
    )

    assert np.max(np.abs(result["companion"])) > 1e-5
    assert np.min(result["injected"]) >= 0.0
    assert np.max(result["injected"]) == pytest.approx(1.0)
