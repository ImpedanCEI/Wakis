"""Test wave-packet envelopes, oblique physics, plotting, and propagation."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c, mu_0

from wakis.sources import WavePacket


def test_source_updates_component(unit_solver):
    source = WavePacket(
        zs=2,
        sigmaz=0.02,
        sigmaxy=0.03,
        tinj=0.0,
        wavelength=0.1,
        amplitude=2.0,
    )

    source.update(unit_solver, 0.0)

    magnetic = unit_solver.H[:, :, 2, "y"]
    electric = unit_solver.E[:, :, 2, "x"]
    assert np.any(magnetic)
    np.testing.assert_allclose(electric, mu_0 * c * magnetic)
    assert np.all(electric * magnetic > 0.0)
    assert not np.any(unit_solver.E[:, :, 2, "y"])
    assert not np.any(unit_solver.E[:, :, 2, "z"])


def test_wave_packet_envelope_peak_and_width(unit_solver):
    source = WavePacket(
        zs=2,
        sigmaz=0.01,
        sigmaxy=0.03,
        tinj=0.03,
        f=0.0,
        beta=0.5,
    )
    peak_time = source.tinj / (source.beta * c)
    sigma_time = source.sigmaz / (source.beta * c)
    center = unit_solver.Nx // 2

    source.update(unit_solver, peak_time)
    peak = unit_solver.H[center, center, 2, "y"]
    source.update(unit_solver, peak_time + sigma_time)
    one_sigma = unit_solver.H[center, center, 2, "y"]

    np.testing.assert_allclose(one_sigma / peak, np.exp(-0.5))


def test_wave_packet_frequency_matches_wavelength_and_velocity():
    source = WavePacket(sigmaz=0.01, wavelength=0.2, beta=0.5)

    np.testing.assert_allclose(source.f, source.vp / source.wavelength)
    np.testing.assert_allclose(source.k, 2 * np.pi / source.wavelength)


def test_wave_packet_oblique_fields_are_transverse(unit_solver):
    theta = np.pi / 6
    source = WavePacket(
        zs=2,
        sigmaz=0.02,
        sigmaxy=0.03,
        tinj=0.0,
        f=0.0,
        amplitude=2.0,
        beta=0.8,
        theta=theta,
    )

    source.update(unit_solver, 0.0)

    magnetic = unit_solver.H[:, :, 2, "y"]
    electric_x = unit_solver.E[:, :, 2, "x"]
    electric_z = unit_solver.E[:, :, 2, "z"]
    np.testing.assert_allclose(
        np.sin(theta) * electric_x + np.cos(theta) * electric_z, 0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        np.hypot(electric_x, electric_z), mu_0 * source.vp * np.abs(magnetic)
    )
    np.testing.assert_allclose(
        (-electric_z * magnetic) / (electric_x * magnetic), np.tan(theta)
    )


def test_wave_packet_oblique_envelope_arrival_delay(unit_solver):
    theta = np.pi / 6
    source = WavePacket(
        zs=2,
        sigmaz=0.01,
        sigmaxy=1.0,
        tinj=0.03,
        f=0.0,
        theta=theta,
    )
    left, right = 1, 3
    center_y = unit_solver.Ny // 2
    peak_times = (
        source.tinj + unit_solver.x[[left, right]] * np.sin(theta)
    ) / source.vp

    source.update(unit_solver, peak_times[0])
    left_peak = unit_solver.H[left, center_y, 2, "y"]
    source.update(unit_solver, peak_times[1])
    right_peak = unit_solver.H[right, center_y, 2, "y"]

    expected_profile = np.exp(
        -(
            (unit_solver.x[[left, right]] * np.cos(theta)) ** 2
            + unit_solver.y[center_y] ** 2
        )
        / (2 * source.sigmaxy**2)
    )
    np.testing.assert_allclose([left_peak, right_peak], expected_profile)
    np.testing.assert_allclose(
        peak_times[1] - peak_times[0],
        (unit_solver.x[right] - unit_solver.x[left]) * np.sin(theta) / source.vp,
    )


def test_wave_packet_resolves_mesh_dependent_defaults(unit_solver):
    source = WavePacket(xs=slice(0, 4), zs=2)

    source.update(unit_solver, 0.0)

    np.testing.assert_allclose(source.sigmaz, 10 * np.mean(unit_solver.dz))
    np.testing.assert_allclose(source.tinj, 6 * source.sigmaz)
    np.testing.assert_allclose(source.wavelength, 10 * np.mean(unit_solver.dz))
    np.testing.assert_allclose(source.f, c / source.wavelength)
    assert unit_solver.H[:4, :, 2, "y"].shape == (4, unit_solver.Ny)


def test_wave_packet_plot_shows_injected_components(monkeypatch):
    theta = np.pi / 6
    source = WavePacket(
        sigmaz=0.01,
        sigmaxy=0.02,
        tinj=0.0,
        f=0.0,
        theta=theta,
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    source.plot(np.array([0.0]))

    magnetic_axis, electric_axis = plt.gcf().axes
    assert [line.get_label() for line in magnetic_axis.lines] == ["Hy"]
    assert [line.get_label() for line in electric_axis.lines] == ["Ex", "Ez"]

    magnetic_y = magnetic_axis.lines[0].get_ydata()[0]
    electric_x = electric_axis.lines[0].get_ydata()[0]
    electric_z = electric_axis.lines[1].get_ydata()[0]
    np.testing.assert_allclose(
        electric_x, mu_0 * source.vp * np.cos(theta) * magnetic_y
    )
    np.testing.assert_allclose(
        electric_z, -mu_0 * source.vp * np.sin(theta) * magnetic_y
    )
    plt.close()


def test_wave_packet_propagates_a_finite_impedance_matched_gaussian(
    propagation_solver, simulate_source
):
    source = WavePacket(
        zs=8,
        sigmaz=4 * c * propagation_solver.dt,
        sigmaxy=0.02,
        tinj=12 * c * propagation_solver.dt,
        f=5e9,
    )

    result = simulate_source(
        propagation_solver, source, ("H", "y"), ("E", "x"), ("H", "y")
    )

    peak = np.argmax(np.abs(result["observed"]))
    assert result["observed"][peak] * result["companion"][peak] > 0.0
    assert abs(result["profile_x"][peak, propagation_solver.Nx // 2]) > 1.4 * abs(
        result["profile_x"][peak, 0]
    )
    assert result["observed"][peak] / result["companion"][peak] == pytest.approx(
        mu_0 * c, rel=0.4
    )
