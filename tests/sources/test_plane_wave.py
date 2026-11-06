"""Test plane-wave phase, impedance, plotting, and propagation."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c, mu_0

from wakis.sources import PlaneWave


def test_source_updates_component(unit_solver):
    source = PlaneWave(zs=2, nodes=1, f=1e9, amplitude=2.0)
    delay = (unit_solver.tz[1:][2] - unit_solver.z[2]) / source.vp

    source.update(unit_solver, delay)

    np.testing.assert_allclose(unit_solver.H[:, :, 2, "y"], 2.0)
    assert np.all(unit_solver.E[:, :, 2, "x"] > 0.0)
    assert not np.any(unit_solver.H[:, :, 1, "y"])

    source.update(unit_solver, source.tmax + delay + unit_solver.dt)

    assert not np.any(unit_solver.H[:, :, 2, "y"])
    assert not np.any(unit_solver.E[:, :, 2, "x"])


def test_plane_wave_follows_harmonic_phase_and_positive_z_direction(unit_solver):
    source = PlaneWave(zs=2, f=1e9, amplitude=2.0)
    period = 1.0 / source.f
    delay = (unit_solver.tz[1:][2] - unit_solver.z[2]) / source.vp

    source.update(unit_solver, delay)
    assert np.all(unit_solver.E[:, :, 2, "x"] * unit_solver.H[:, :, 2, "y"] > 0)
    source.update(unit_solver, delay + period / 4)
    np.testing.assert_allclose(unit_solver.H[:, :, 2, "y"], 0.0, atol=1e-14)
    source.update(unit_solver, delay + period / 2)
    np.testing.assert_allclose(unit_solver.H[:, :, 2, "y"], -2.0)


def test_plane_wave_uses_medium_wave_impedance(unit_solver):
    source = PlaneWave(zs=2, f=1e9, amplitude=2.0, beta=0.8)
    delay = (unit_solver.tz[1:][2] - unit_solver.z[2]) / source.vp
    source.update(unit_solver, delay)

    electric = unit_solver.E[unit_solver.Nx // 2, unit_solver.Ny // 2, 2, "x"]
    magnetic = unit_solver.H[unit_solver.Nx // 2, unit_solver.Ny // 2, 2, "y"]
    np.testing.assert_allclose(
        electric / magnetic, mu_0 * source.vp * np.cos(source.omega * delay)
    )


def test_plane_wave_uses_vacuum_impedance_at_beta_one(unit_solver):
    source = PlaneWave(zs=2, f=1e9, amplitude=2.0, beta=1.0)
    delay = (unit_solver.tz[1:][2] - unit_solver.z[2]) / source.vp
    source.update(unit_solver, delay)

    electric = unit_solver.E[unit_solver.Nx // 2, unit_solver.Ny // 2, 2, "x"]
    magnetic = unit_solver.H[unit_solver.Nx // 2, unit_solver.Ny // 2, 2, "y"]
    np.testing.assert_allclose(
        electric / magnetic, mu_0 * c * np.cos(source.omega * delay)
    )


def test_plane_wave_oblique_phase_uses_staggered_positions(unit_solver):
    """The oblique carrier reaches each field at its own physical position."""
    theta = np.pi / 6
    source = PlaneWave(zs=2, f=1e9, theta=theta, amplitude=2.0)
    time = 1 / source.f
    source.update(unit_solver, time)

    x_edge = unit_solver.tx[1:][:, None]
    x_node = unit_solver.x[:, None]
    dz = unit_solver.tz[1:][2] - unit_solver.z[2]
    phase_h = source.omega * time - source.k * (
        x_edge * np.sin(theta) + dz * np.cos(theta)
    )
    phase_ex = source.omega * time - source.k * x_edge * np.sin(theta)
    phase_ez = source.omega * time - source.k * (
        x_node * np.sin(theta) + dz * np.cos(theta)
    )

    np.testing.assert_allclose(unit_solver.H[:, 0, 2, "y"], 2 * np.cos(phase_h[:, 0]))
    np.testing.assert_allclose(
        unit_solver.E[:, 0, 2, "x"],
        2 * mu_0 * source.vp * np.cos(theta) * np.cos(phase_ex[:, 0]),
    )
    np.testing.assert_allclose(
        unit_solver.E[:, 0, 2, "z"],
        -2 * mu_0 * source.vp * np.sin(theta) * np.cos(phase_ez[:, 0]),
    )


def test_plane_wave_injects_multiple_z_planes(unit_solver):
    """A z slice broadcasts the oblique phase across x, y, and z."""
    source = PlaneWave(zs=slice(2, 4), f=1e9, theta=np.pi / 6)
    source.update(unit_solver, 1 / source.f)

    x_edge = unit_solver.tx[1:][:, None]
    z_offset = unit_solver.tz[1:][2:4] - unit_solver.z[2:4]
    phase = source.omega / source.f - source.k * (
        x_edge * np.sin(source.theta) + z_offset * np.cos(source.theta)
    )
    np.testing.assert_allclose(
        unit_solver.H[:, 0, 2:4, "y"], source.amplitude * np.cos(phase)
    )
    assert np.any(unit_solver.E[:, :, 2:4, "z"])


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
    # The terminal Ex sample has zero primal length on the periodic grid.
    profile = result["profile_x"][peak, :-1]
    np.testing.assert_allclose(profile, profile[0], rtol=1e-12, atol=1e-12)
    assert result["observed"][peak] * result["companion"][peak] > 0.0
    assert result["observed"][peak] / result["companion"][peak] == pytest.approx(
        mu_0 * c, rel=0.25
    )
