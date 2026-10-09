import numpy as np
import pytest

from wakis.sources import (
    Beam,
    Dipole,
    GaussianPacket,
    ModePacket,
    PlaneWave,
    Pulse,
    Source,
    WaveformSource,
    WavePacket,
)


def test_waveform_sources_share_waveform_source_interface():
    sources = [
        Dipole(f=1e9),
        GaussianPacket(sigmaz=0.01),
        ModePacket(),
        PlaneWave(f=1e9),
        Pulse(L=0.01),
        WavePacket(wavelength=0.1),
    ]

    assert Source is WaveformSource
    assert all(isinstance(source, WaveformSource) for source in sources)
    assert not isinstance(Beam(sigmaz=0.01), WaveformSource)


def test_all_concrete_sources_provide_a_plot_method():
    sources = [
        Beam(sigmaz=0.01),
        Dipole(f=1e9),
        GaussianPacket(sigmaz=0.01),
        ModePacket(),
        PlaneWave(f=1e9),
        Pulse(L=0.01),
        WavePacket(wavelength=0.1),
    ]

    assert all(callable(source.plot) for source in sources)


def test_common_profiles_are_available_to_waveform_sources():
    source = PlaneWave(f=1e9)

    np.testing.assert_allclose(
        source.gaussian_profile(np.array([0.0, 1.0]), 1.0),
        np.array([1.0, np.exp(-0.5)]),
    )
    np.testing.assert_allclose(
        source.harmonic_carrier(np.array([0.0, np.pi])),
        np.array([1.0, -1.0]),
    )


def test_source_frame_uses_u_v_w_coordinates():
    theta = np.pi / 6
    direction = np.array([np.sin(theta), 0.0, np.cos(theta)])

    u, v, w = WaveformSource._to_source_frame(2.0, 3.0, 0.0, direction)

    assert u == pytest.approx(2.0 * np.cos(theta))
    assert v == pytest.approx(3.0)
    assert w == pytest.approx(2.0 * np.sin(theta))


def test_oscillating_sources_name_angular_frequency_omega():
    sources = [
        Dipole(f=1e9),
        ModePacket(f=1e9),
        PlaneWave(f=1e9),
        WavePacket(f=1e9),
    ]

    for source in sources:
        assert source.omega == pytest.approx(2 * np.pi * source.f)
        assert not hasattr(source, "w")


@pytest.mark.parametrize("injection, expected", [("hard", 2.0), ("soft", 5.0)])
def test_injection_method_controls_field_application(
    source_solver, injection, expected
):
    source = Dipole(
        field="Ex",
        xs=2,
        ys=2,
        zs=2,
        f=1e9,
        amplitude=2.0,
        phase=np.pi / 2,
        injection=injection,
    )
    source_solver.E[2, 2, 2, "x"] = 3.0

    source.update(source_solver, 0.0)

    assert source_solver.E[2, 2, 2, "x"] == pytest.approx(expected)


def test_source_rejects_unknown_injection_method():
    with pytest.raises(ValueError, match="injection"):
        PlaneWave(f=1e9, injection="replace")
