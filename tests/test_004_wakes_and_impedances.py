import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.append("../wakis")
from wakis import WakeSolver as wk


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
            profile /= np.trapz(profile, s)
            wake.z = s
            wake.chargedist = wake.q * profile
            wake.calc_lambdas()

        header = (tmp_path / "lambda.txt").read_text().splitlines()[0]
        assert "s [m]" in header
        assert "Normalized charge distribution [1/m]" in header
        assert np.trapz(wake.lambdas, s) == pytest.approx(1.0, rel=1e-10)

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

    def test_sin_wake(self):
        fr = 0.5e9
        A = 100
        t = np.linspace(0, 100 * 1e-9, 3000)
        wake = A * np.sin(2 * np.pi * fr * t)

        f, Z = wk.calc_impedance_from_wake([t, wake])
        tt, wwake = wk.calc_wake_from_impedance([f, Z], samples=3000)
        ff, Zz = wk.calc_impedance_from_wake(
            [tt, wwake],
        )
        ttt, wwwake = wk.calc_wake_from_impedance([ff, Zz], samples=3000)

        assert np.allclose(wake, wwake, atol=1), "1st Transformed wake failed"
        assert np.allclose(wake, wwwake, atol=1), "2nd Transformed wake failed"

    def test_sin_impedance(self):
        fr = 0.5e9
        A = 100
        t = np.linspace(0, 100 * 1e-9, 3000)
        wake = A * np.sin(2 * np.pi * fr * t)

        f, Z = wk.calc_impedance_from_wake([t, wake])
        tt, wwake = wk.calc_wake_from_impedance([f, Z], samples=3000)
        ff, Zz = wk.calc_impedance_from_wake(
            [tt, wwake],
        )

        assert np.allclose(np.max(np.abs(Z)), A, atol=1), (
            "1st Transformed impedance Max. failed"
        )
        assert np.allclose(np.max(np.abs(Zz)), A, atol=1), (
            "2nd Transformed impedance Max. failed"
        )

        assert np.allclose(f[np.argmax(Z)], fr, atol=1e6), (
            "1st Transformed impedance fr failed"
        )
        assert np.allclose(ff[np.argmax(Zz)], fr, atol=1e6), (
            "2nd Transformed impedance fr. failed"
        )

    def plot_sin(self):
        fr = 0.5e9
        A = 100
        t = np.linspace(0, 100 * 1e-9, 3000)
        wake = A * np.sin(2 * np.pi * fr * t)

        f, Z = wk.calc_impedance_from_wake([t, wake])
        tt, wwake = wk.calc_wake_from_impedance([f, Z], samples=3000)
        ff, Zz = wk.calc_impedance_from_wake(
            [tt, wwake],
        )
        ttt, wwwake = wk.calc_wake_from_impedance([ff, Zz], samples=3000)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(t, wake, "-g", alpha=0.8, label="analytic")
        ax1.plot(tt, wwake, "--r", alpha=0.5, label="calc")
        ax1.plot(ttt, wwwake, "--b", alpha=0.5, label="calc, iter2")
        ax1.set_xlabel("time [s]")

        ax2.plot([fr, fr], [0.0, A], "-g", alpha=0.8, label="analytic")
        ax2.plot(f, np.abs(Z), "--r", alpha=0.5, label="calc")
        ax2.plot(ff, np.abs(Zz), "--b", alpha=0.5, label="calc, iter2")
        ax2.set_xlabel("frequency [Hz]")
        ax2.legend()

        fig.tight_layout()
        fig.savefig("test_004_sin.png")
        plt.show()
