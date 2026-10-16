"""Test beam-current deposition, profile stability, and plotting."""

import matplotlib.pyplot as plt
import numpy as np

from wakis.sources import Beam


def test_source_updates_component(unit_solver):
    beam = Beam(sigmaz=0.01, ti=0.0)

    beam.update(unit_solver, 0.0)

    assert np.any(unit_solver.J[:, :, :, "z"])
    assert not np.any(unit_solver.J[:, :, :, "x"])
    assert not np.any(unit_solver.J[:, :, :, "y"])
    assert beam.Jold.shape == unit_solver.z.shape


def test_beam_plot_uses_longitudinal_current_profile(monkeypatch):
    beam = Beam(q=2e-9, sigmaz=0.01, ti=0.0)
    times = np.array([0.0, beam.sigmaz / beam.v])
    monkeypatch.setattr(plt, "show", lambda: None)

    beam.plot(times)

    current = plt.gcf().axes[0].lines[0].get_ydata()
    np.testing.assert_allclose(current[1] / current[0], np.exp(-0.5))
    plt.close()
