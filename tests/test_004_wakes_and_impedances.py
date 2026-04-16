import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../wakis")
from wakis import WakeSolver as wk


class TestImpedancesAndWakes:
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
