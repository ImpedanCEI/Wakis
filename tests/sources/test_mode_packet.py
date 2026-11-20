"""Test TE01 mode shape, plotting, and waveguide propagation."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wakis.sources import ModePacket


def test_source_updates_component(unit_solver):
    source = ModePacket(zs=2, f=1e9, amplitude=2.0, sigma_t=1e-9, tinj=0.0)

    source.update(unit_solver, 0.0)

    electric = unit_solver.E[:, :, 2, "x"]
    assert np.allclose(electric[:, 0], 0.0)
    assert np.allclose(electric[:, -1], 0.0, atol=1e-15)
    assert electric[:, unit_solver.Ny // 2].max() == pytest.approx(2.0)
    assert not np.any(unit_solver.H.toarray())


def test_mode_packet_plot_shows_modal_drive(monkeypatch):
    source = ModePacket(f=1e9, amplitude=2.0, sigma_t=1e-9, tinj=0.0)
    monkeypatch.setattr(plt, "show", lambda: None)

    source.plot(np.array([0.0]))

    line = plt.gcf().axes[0].lines[0]
    assert line.get_label() == "Ex"
    assert line.get_ydata()[0] == pytest.approx(2.0)
    plt.close()


@pytest.mark.parametrize("propagation_solver", ["pec"], indirect=True)
def test_mode_packet_propagates_the_te01_profile(propagation_solver, simulate_source):
    source = ModePacket(
        zs=8,
        f=5e9,
        sigma_t=4 * propagation_solver.dt,
        tinj=12 * propagation_solver.dt,
    )

    result = simulate_source(
        propagation_solver, source, ("E", "x"), ("E", "x"), ("H", "y")
    )

    assert np.max(np.abs(result["companion"])) > 1e-5
    peak = np.argmax(np.abs(result["observed"]))
    profile = result["profile_y"][peak]
    assert profile[0] == pytest.approx(0.0, abs=1e-12)
    assert profile[-1] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(profile, profile[::-1], rtol=1e-12, atol=1e-12)
    assert abs(profile[propagation_solver.Ny // 2]) == np.max(np.abs(profile))
