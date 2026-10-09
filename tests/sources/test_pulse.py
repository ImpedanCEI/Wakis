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
def test_source_updates_component(source_solver, shape, time_factor, expected):
    length = 0.02
    source = Pulse(field="Jz", shape=shape, L=length, amplitude=2.0)
    time = time_factor * length / c

    source.update(source_solver, time)

    center = (source_solver.Nx // 2, source_solver.Ny // 2, source_solver.Nz // 2)
    np.testing.assert_allclose(
        source_solver.J[center[0], center[1], center[2], "z"], 2.0 * expected
    )
    assert not np.any(source_solver.E.toarray())
    assert not np.any(source_solver.H.toarray())


def test_default_pulse_length_is_expressed_as_distance(source_solver):
    source = Pulse()

    source.update(source_solver, 0.0)

    assert source.L == pytest.approx(50 * c * source_solver.dt)


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
