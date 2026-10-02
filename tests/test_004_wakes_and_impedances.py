import sys

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.integrate import trapezoid

sys.path.append("../wakis")
from wakis import WakeSolver as wk

def analytic_impedance(f, fr, amplitude, duration, plane):
    """Continuous transform of a finite cosine or sine wake."""

    def rectangular_pulse_transform(offset):
        return duration * np.exp(-1j * np.pi * offset * duration) * np.sinc(
            offset * duration
        )

    lower_sideband = rectangular_pulse_transform(f - fr)
    upper_sideband = rectangular_pulse_transform(f + fr)
    if plane == "longitudinal":
        return 0.5 * amplitude * (lower_sideband + upper_sideband)
    return 0.5 * amplitude * (lower_sideband - upper_sideband)

class TestImpedancesAndWakes:
    @pytest.mark.parametrize("analytic", [True, False])
    def test_charge_profile_file_units(self, tmp_path, analytic):
        s = np.linspace(-0.04, 0.04, 257)
        wake = wk(results_folder=str(tmp_path), save=True, verbose=0)
        wake.s = s

        if analytic:
            wake.calc_lambdas_analytic()
        else:
            profile = np.exp(-0.5 * (s / wake.sigmaz) ** 2)
            profile /= trapezoid(profile, s)
            wake.z = s
            wake.chargedist = wake.q * profile
            wake.calc_lambdas()

        header = (tmp_path / "lambda.txt").read_text().splitlines()[0]
        assert "s [m]" in header
        assert "Normalized charge distribution [1/m]" in header
        assert trapezoid(wake.lambdas, s) == pytest.approx(1.0, rel=1e-10)

    def test_dimensionless_spectrum_preserves_impedances(self, tmp_path):
        wake = wk(results_folder=str(tmp_path), save=True, verbose=0)
        wake.s = np.linspace(-0.04, 0.04, 257)
        wake.calc_lambdas_analytic()
        wake.WP = np.exp(-0.5 * ((wake.s - 0.002) / 0.006) ** 2)
        wake.WPx = 0.4 * wake.WP
        wake.WPy = -0.2 * wake.WP

        wake.calc_long_Z(samples=101, fmax=5e9)
        ds = wake.s[1] - wake.s[0]
        n = int((wake.v / ds) // 5e9 * 101)
        frequencies = np.fft.fftfreq(n, ds / wake.v)
        mask = np.logical_and(frequencies >= 0, frequencies < 5e9)
        old_spectrum = np.fft.fft(wake.lambdas * wake.v, n=n)[mask] * ds
        old_z = -(np.fft.fft(wake.WP * 1e12, n=n)[mask] * ds) / old_spectrum

        assert wake.lambdaf[0] == pytest.approx(1.0, rel=1e-10)
        np.testing.assert_allclose(wake.lambdaf * wake.v, old_spectrum, rtol=1e-12)
        np.testing.assert_allclose(wake.Z, old_z, rtol=1e-12)
        header = (tmp_path / "spectrum.txt").read_text().splitlines()[0]
        assert "[dimensionless]" in header

        wake.calc_trans_Z(samples=101, fmax=5e9)
        old_zx = 1j * np.fft.fft(wake.WPx * 1e12, n=n)[mask] * ds / old_spectrum
        old_zy = 1j * np.fft.fft(wake.WPy * 1e12, n=n)[mask] * ds / old_spectrum
        np.testing.assert_allclose(wake.Zx, old_zx, rtol=1e-12)
        np.testing.assert_allclose(wake.Zy, old_zy, rtol=1e-12)

    @pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
    def test_analytic_transform_and_round_trip(self, plane, plot_comparison):
        fr = 0.5e9
        amplitude = 100.0
        time = np.linspace(0, 100e-9, 3000)
        if plane == "longitudinal":
            wake = amplitude * np.cos(2 * np.pi * fr * time)
            wake[0] *= 0.5
        else:
            wake = amplitude * np.sin(2 * np.pi * fr * time)

        frequency, impedance = wk.calc_impedance_from_wake(
            [time, wake], plane=plane, verbose=False
        )
        duration = len(time) * np.mean(np.diff(time))
        expected_impedance = analytic_impedance(
            frequency, fr, amplitude, duration, plane
        )
        np.testing.assert_allclose(
            impedance, expected_impedance, rtol=2e-3, atol=4e-9
        )

        reconstructed_time, reconstructed_wake = wk.calc_wake_from_impedance(
            [frequency, impedance], plane=plane, verbose=False
        )
        expected_wake = amplitude * (
            np.cos(2 * np.pi * fr * reconstructed_time)
            if plane == "longitudinal"
            else np.sin(2 * np.pi * fr * reconstructed_time)
        )
        expected_wake[0] *= 0.5
        if plane == "transverse":
            expected_wake[-1] = 0.0

        plot_comparison(
            reconstructed_wake, expected_wake, f"{plane.title()} wake"
        )
        plot_comparison(
            np.abs(impedance),
            np.abs(expected_impedance),
            f"{plane.title()} impedance magnitude",
        )
        plot_comparison(
            np.real(impedance),
            np.real(expected_impedance),
            f"{plane.title()} impedance real part",
        )
        plot_comparison(
            np.imag(impedance),
            np.imag(expected_impedance),
            f"{plane.title()} impedance imaginary part",
        )
        np.testing.assert_allclose(reconstructed_wake, expected_wake, atol=0.05)

    def test_non_ultrarelativistic_distance_round_trip(self):
        gamma = 2.0
        beta = np.sqrt(1.0 - 1.0 / gamma**2)
        time = np.linspace(0, 100e-9, 3000)
        distance = time * beta * c_light
        wake = 100.0 * np.cos(2 * np.pi * 0.5e9 * time)
        wake[0] *= 0.5

        frequency, impedance = wk.calc_impedance_from_wake(
            wake, s=distance, gamma=gamma, verbose=False
        )
        reconstructed_time, _ = wk.calc_wake_from_impedance(
            [frequency, impedance], gamma=gamma, verbose=False
        )
        reference_time, _ = wk.calc_wake_from_impedance(
            [frequency, impedance], verbose=False
        )

        np.testing.assert_array_equal(reconstructed_time, reference_time)
        assert reconstructed_time[-1] == pytest.approx(time[-1], rel=5e-4)

    def test_legacy_positional_arguments(self):
        time = np.linspace(0, 20e-9, 128)
        wake = np.cos(2 * np.pi * 0.5e9 * time)

        frequency, impedance = wk.calc_impedance_from_wake(
            [time, wake], None, None, None, 64, False
        )
        reconstructed_time, reconstructed_wake = wk.calc_wake_from_impedance(
            impedance, frequency, 20e-9, 32, 0, False
        )

        assert len(frequency) == 64
        assert len(reconstructed_time) == len(reconstructed_wake) == 32
        assert reconstructed_time[-1] == pytest.approx(20e-9)
