import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, mu_0

from wakis.sources import WavePacket


def test_source_updates_component(source_solver):
    source = WavePacket(
        zs=2,
        sigmaz=0.02,
        sigmaxy=0.03,
        tinj=0.0,
        wavelength=0.1,
        amplitude=2.0,
    )

    source.update(source_solver, 0.0)

    magnetic = source_solver.H[:, :, 2, "y"]
    electric = source_solver.E[:, :, 2, "x"]
    assert np.any(magnetic)
    np.testing.assert_allclose(electric, mu_0 * c * magnetic)
    assert np.all(electric * magnetic > 0.0)
    assert not np.any(source_solver.E[:, :, 2, "y"])
    assert not np.any(source_solver.E[:, :, 2, "z"])


def test_wave_packet_envelope_peak_and_width(source_solver):
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
    center = source_solver.Nx // 2

    source.update(source_solver, peak_time)
    peak = source_solver.H[center, center, 2, "y"]
    source.update(source_solver, peak_time + sigma_time)
    one_sigma = source_solver.H[center, center, 2, "y"]

    np.testing.assert_allclose(one_sigma / peak, np.exp(-0.5))


def test_wave_packet_frequency_matches_wavelength_and_velocity():
    source = WavePacket(sigmaz=0.01, wavelength=0.2, beta=0.5)

    np.testing.assert_allclose(source.f, source.vp / source.wavelength)
    np.testing.assert_allclose(source.k, 2 * np.pi / source.wavelength)


def test_wave_packet_oblique_fields_are_transverse(source_solver):
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

    source.update(source_solver, 0.0)

    magnetic = source_solver.H[:, :, 2, "y"]
    electric_x = source_solver.E[:, :, 2, "x"]
    electric_z = source_solver.E[:, :, 2, "z"]
    np.testing.assert_allclose(
        np.sin(theta) * electric_x + np.cos(theta) * electric_z, 0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        np.hypot(electric_x, electric_z), mu_0 * source.vp * np.abs(magnetic)
    )
    np.testing.assert_allclose(
        (-electric_z * magnetic) / (electric_x * magnetic), np.tan(theta)
    )


def test_wave_packet_oblique_envelope_arrival_delay(source_solver):
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
    center_y = source_solver.Ny // 2
    peak_times = (
        source.tinj + source_solver.x[[left, right]] * np.sin(theta)
    ) / source.vp

    source.update(source_solver, peak_times[0])
    left_peak = source_solver.H[left, center_y, 2, "y"]
    source.update(source_solver, peak_times[1])
    right_peak = source_solver.H[right, center_y, 2, "y"]

    expected_profile = np.exp(
        -(
            (source_solver.x[[left, right]] * np.cos(theta)) ** 2
            + source_solver.y[center_y] ** 2
        )
        / (2 * source.sigmaxy**2)
    )
    np.testing.assert_allclose([left_peak, right_peak], expected_profile)
    np.testing.assert_allclose(
        peak_times[1] - peak_times[0],
        (source_solver.x[right] - source_solver.x[left]) * np.sin(theta) / source.vp,
    )


def test_wave_packet_resolves_mesh_dependent_defaults(source_solver):
    source = WavePacket(xs=slice(0, 4), zs=2)

    source.update(source_solver, 0.0)

    np.testing.assert_allclose(source.sigmaz, 10 * np.mean(source_solver.dz))
    np.testing.assert_allclose(source.tinj, 6 * source.sigmaz)
    np.testing.assert_allclose(source.wavelength, 10 * np.mean(source_solver.dz))
    np.testing.assert_allclose(source.f, c / source.wavelength)
    assert source_solver.H[:4, :, 2, "y"].shape == (4, source_solver.Ny)


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
