import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

sys.path.append("../wakis")
from wakis import WakeSolver as wk


def analytic_impedance_from_sine_cosine_wake(f, fr, A, T):
    return -0.5 * A * (
        (np.exp(-1j * 2*np.pi*(f + fr)*T) - 1) / (-1j * 2*np.pi*(f + fr)) -
        (np.exp(-1j * 2*np.pi*(f - fr)*T) - 1) / (-1j * 2*np.pi*(f - fr))
    )


# Parameters
fr = 0.5e9
A = 100
Nsamples = 30000

t = np.linspace(0, 100e-9, Nsamples)

# Longitudinal
wake_l = A * np.cos(2 * np.pi * fr * t)
wake_l[0] *= 0.5  # fundamental theorem

f_l, Z_l = wk.calc_impedance_from_wake([t, wake_l])
tt_l, wwake_l = wk.calc_wake_from_impedance([f_l, Z_l])
ff_l, Zz_l = wk.calc_impedance_from_wake([tt_l, wwake_l])
ttt_l, wwwake_l = wk.calc_wake_from_impedance([ff_l, Zz_l])

# Analytical
dt = np.mean(np.diff(t))
T = Nsamples * dt
Z_analytical = analytic_impedance_from_sine_cosine_wake(f_l, fr, A, T)

# Transverse
wake_t = A * np.sin(2 * np.pi * fr * t)

f_t, Z_t = wk.calc_impedance_from_wake([t, wake_t], plane="transverse")
tt_t, wwake_t = wk.calc_wake_from_impedance([f_t, Z_t], plane="transverse")
ff_t, Zz_t = wk.calc_impedance_from_wake([tt_t, wwake_t], plane="transverse")
ttt_t, wwwake_t = wk.calc_wake_from_impedance([ff_t, Zz_t], plane="transverse")

# Same analytical formula applies
Z_analytical_t = analytic_impedance_from_sine_cosine_wake(f_t, fr, A, T)


# Plot: Longitudinal
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

fig1, axs = plt.subplots(4, 1, figsize=(11, 14))

axs[0].set_title("Longitudinal wake and impedance", fontsize=18)
axs[0].plot(t, wake_l, "-g", label="analytic", linewidth=3)
axs[0].plot(tt_l, wwake_l, "--r", label="calc")
axs[0].plot(ttt_l, wwwake_l, "--b", label="calc, iter2")
axs[0].set_ylabel("Wake [V/C]")
axs[0].legend()

axs[1].plot(f_l, np.abs(Z_analytical), "-g", label="analytic", linewidth=3)
axs[1].plot(f_l, np.abs(Z_l), "--r", label="calc")
axs[1].plot(ff_l, np.abs(Zz_l), "--b", label="calc, iter2")
axs[1].set_ylabel("|Z| [Ohm]")
axs[1].set_xlim([0, 1e9])
axs[1].legend()

axs[2].plot(f_l, np.real(Z_analytical), "-g", label="analytic", linewidth=3)
axs[2].plot(f_l, np.real(Z_l), "--r", label="calc")
axs[2].plot(ff_l, np.real(Zz_l), "--b", label="calc, iter2")
axs[2].set_ylabel("Re(Z) [Ohm]")
axs[2].set_xlim([0, 1e9])
axs[2].legend()

axs[3].plot(f_l, np.imag(Z_analytical), "-g", label="analytic", linewidth=3)
axs[3].plot(f_l, np.imag(Z_l), "--r", label="calc")
axs[3].plot(ff_l, np.imag(Zz_l), "--b", label="calc, iter2")
axs[3].set_ylabel("Im(Z) [Ohm]")
axs[3].set_xlabel("Frequency [Hz]")
axs[3].set_xlim([0, 1e9])
axs[3].legend()

fig1.tight_layout()


# Plot: Transverse
fig2, axs = plt.subplots(4, 1, figsize=(11, 14))

axs[0].set_title("Transverse wake and impedance", fontsize=18)
axs[0].plot(t, wake_t, "-g", label="analytic", linewidth=3)
axs[0].plot(tt_t, wwake_t, "--r", label="calc")
axs[0].plot(ttt_t, wwwake_t, "--b", label="calc, iter2")
axs[0].set_ylabel("Wake [V/C/m]")
axs[0].legend()

axs[1].plot(f_t, np.abs(Z_analytical_t), "-g", label="analytic", linewidth=3)
axs[1].plot(f_t, np.abs(Z_t), "--r", label="calc")
axs[1].plot(ff_t, np.abs(Zz_t), "--b", label="calc, iter2")
axs[1].set_ylabel("|Z| [Ohm/m]")
axs[1].set_xlim([0, 1e9])
axs[1].legend()

axs[2].plot(f_t, np.real(Z_analytical_t), "-g", label="analytic", linewidth=3)
axs[2].plot(f_t, np.real(Z_t), "--r", label="calc")
axs[2].plot(ff_t, np.real(Zz_t), "--b", label="calc, iter2")
axs[2].set_ylabel("Re(Z) [Ohm/m]")
axs[2].set_xlim([0, 1e9])
axs[2].legend()

axs[3].plot(f_t, np.imag(Z_analytical_t), "-g", label="analytic", linewidth=3)
axs[3].plot(f_t, np.imag(Z_t), "--r", label="calc")
axs[3].plot(ff_t, np.imag(Zz_t), "--b", label="calc, iter2")
axs[3].set_ylabel("Im(Z) [Ohm/m]")
axs[3].set_xlabel("Frequency [Hz]")
axs[3].set_xlim([0, 1e9])
axs[3].legend()

fig2.tight_layout()

plt.show()
