import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c

from wakis.sources import GaussianPacket


def test_source_updates_component(source_solver):
    source = GaussianPacket(
        zs=2,
        sigmaz=0.02,
        sigmaxy=0.03,
        tinj=0.0,
        amplitude=2.0,
    )

    source.update(source_solver, 0.0)

    assert np.all(source_solver.H[:, :, 2, "y"] < 0.0)
    assert not np.any(source_solver.E.toarray())
    assert not np.any(source_solver.H[:, :, 2, "x"])


def test_gaussian_packet_can_inject_electric_field(source_solver):
    source = GaussianPacket(
        field="E",
        zs=2,
        sigmaz=0.02,
        sigmaxy=0.03,
        tinj=0.0,
        amplitude=2.0,
    )

    source.update(source_solver, 0.0)

    assert np.all(source_solver.E[:, :, 2, "x"] > 0.0)
    assert not np.any(source_solver.H.toarray())
    assert not np.any(source_solver.E[:, :, 2, "y"])


def test_gaussian_packet_envelope_peak_and_width(source_solver):
    source = GaussianPacket(
        zs=2,
        sigmaz=0.01,
        sigmaxy=0.02,
        tinj=0.03,
        beta=0.5,
    )
    peak_time = source.tinj / (source.beta * c)
    sigma_time = source.sigmaz / (source.beta * c)
    center = source_solver.Nx // 2

    source.update(source_solver, peak_time)
    peak = -source_solver.H[center, center, 2, "y"]
    one_sigma_x = -source_solver.H[center + 1, center, 2, "y"]
    source.update(source_solver, peak_time + sigma_time)
    one_sigma_t = -source_solver.H[center, center, 2, "y"]

    np.testing.assert_allclose(one_sigma_x / peak, np.exp(-0.5))
    np.testing.assert_allclose(one_sigma_t / peak, np.exp(-0.5))


def test_gaussian_packet_spatial_and_spectral_widths_are_reciprocal():
    from_space = GaussianPacket(sigmaz=0.02)
    from_frequency = GaussianPacket(sigmaf=from_space.sigmaf)

    np.testing.assert_allclose(from_space.sigmaf, c / (2 * np.pi * 0.02))
    np.testing.assert_allclose(from_frequency.sigmaz, 0.02)


@pytest.mark.parametrize(
    "field, label, expected_sign", [("E", "Ex", 1.0), ("H", "Hy", -1.0)]
)
def test_gaussian_packet_plot_shows_injected_component(
    monkeypatch, field, label, expected_sign
):
    source = GaussianPacket(
        field=field, sigmaz=0.01, sigmaxy=0.02, tinj=0.0, amplitude=2.0
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    source.plot(np.array([0.0]))

    line = plt.gcf().axes[0].lines[0]
    assert line.get_label() == label
    assert line.get_ydata()[0] == pytest.approx(expected_sign * source.amplitude)
    plt.close()
