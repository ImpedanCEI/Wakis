"""Test plane-wave phase, impedance, plotting, and propagation."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c, mu_0

from wakis.sources import PlaneWave


def test_source_updates_component(unit_solver):
    source = PlaneWave(zs=2, nodes=1, f=1e9, amplitude=2.0)

    source.update(unit_solver, 0.0)

    np.testing.assert_allclose(unit_solver.H[:, :, 2, "y"], 2.0)
    assert np.all(unit_solver.E[:, :, 2, "x"] > 0.0)
    assert not np.any(unit_solver.H[:, :, 1, "y"])

    source.update(unit_solver, source.tmax + unit_solver.dt)

    assert not np.any(unit_solver.H[:, :, 2, "y"])
    assert not np.any(unit_solver.E[:, :, 2, "x"])


def test_plane_wave_follows_harmonic_phase_and_positive_z_direction(unit_solver):
    source = PlaneWave(zs=2, f=1e9, amplitude=2.0)
    period = 1.0 / source.f

    source.update(unit_solver, 0.0)
    assert np.all(unit_solver.E[:, :, 2, "x"] * unit_solver.H[:, :, 2, "y"] > 0)
    source.update(unit_solver, period / 4)
    np.testing.assert_allclose(unit_solver.H[:, :, 2, "y"], 0.0, atol=1e-14)
    source.update(unit_solver, period / 2)
    np.testing.assert_allclose(unit_solver.H[:, :, 2, "y"], -2.0)


def test_plane_wave_uses_medium_wave_impedance(unit_solver):
    source = PlaneWave(zs=2, f=1e9, amplitude=2.0, beta=0.8)

    source.update(unit_solver, 0.0)

    electric = unit_solver.E[unit_solver.Nx // 2, unit_solver.Ny // 2, 2, "x"]
    magnetic = unit_solver.H[unit_solver.Nx // 2, unit_solver.Ny // 2, 2, "y"]
    np.testing.assert_allclose(electric / magnetic, mu_0 * source.vp)


def test_plane_wave_uses_vacuum_impedance_at_beta_one(unit_solver):
    source = PlaneWave(zs=2, f=1e9, amplitude=2.0, beta=1.0)

    source.update(unit_solver, 0.0)

    electric = unit_solver.E[unit_solver.Nx // 2, unit_solver.Ny // 2, 2, "x"]
    magnetic = unit_solver.H[unit_solver.Nx // 2, unit_solver.Ny // 2, 2, "y"]
    np.testing.assert_allclose(electric / magnetic, mu_0 * c)


def test_plane_wave_requires_positive_frequency():
    with pytest.raises(ValueError, match="positive frequency"):
        PlaneWave()


def test_plane_wave_plot_matches_update_sign(monkeypatch):
    source = PlaneWave(f=1e9)
    monkeypatch.setattr(plt, "show", lambda: None)

    source.plot(np.array([0.0]))

    magnetic_axis, electric_axis = plt.gcf().axes
    assert [line.get_label() for line in magnetic_axis.lines] == ["Hy"]
    assert [line.get_label() for line in electric_axis.lines] == ["Ex"]
    magnetic = magnetic_axis.lines[0].get_ydata()[0]
    electric = electric_axis.lines[0].get_ydata()[0]
    assert magnetic > 0.0
    assert electric > 0.0
    plt.close()


def test_plane_wave_propagates_uniform_impedance_matched_fields(
    propagation_solver, simulate_source
):
    source = PlaneWave(zs=8, f=5e9, nodes=1)

    result = simulate_source(
        propagation_solver, source, ("H", "y"), ("E", "x"), ("H", "y")
    )

    peak = np.argmax(np.abs(result["observed"]))
    profile = result["profile_x"][peak]
    np.testing.assert_allclose(profile, profile[0], rtol=1e-12, atol=1e-12)
    assert result["observed"][peak] * result["companion"][peak] > 0.0
    assert result["observed"][peak] / result["companion"][peak] == pytest.approx(
        mu_0 * c, rel=0.25
    )
