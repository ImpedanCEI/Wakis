# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Mode-packet source."""

import matplotlib.pyplot as plt
import numpy as np

from .source import WaveformSource


class ModePacket(WaveformSource):
    def __init__(
        self,
        zs=0,
        mode="TE01",
        f=2e9,  # Frequency [Hz]
        amplitude=1.0,
        sigma_t=None,  # Time-based gaussian sigma [s]
        tinj=None,
        phase=0,
        injection="hard",
    ):
        """Inject a Gaussian-windowed TE01 electric-field profile.

        Parameters
        ----------
        zs : int, optional
            Longitudinal index of the injection plane. Default is 0.
        mode : str, optional
            Waveguide mode. Only ``'TE01'`` is currently implemented.
        f : float, optional
            Carrier frequency [Hz]. Default is 2 GHz.
        amplitude : float, optional
            Electric-field amplitude [V/m]. Default is 1.0.
        sigma_t : float, optional
            Gaussian temporal standard deviation [s]. Default is ``3/f``.
        tinj : float, optional
            Time at which the envelope peaks [s]. Default is ``6*sigma_t``.
        phase : float, optional
            Carrier phase [rad]. Default is 0.
        injection : {'hard', 'soft'}, optional
            Hard injection assigns Ex; soft injection adds to it. Default is
            ``'hard'``.
        """
        super().__init__(injection=injection)
        self.xs = slice(None)
        self.ys = slice(None)
        self.zs = zs
        self.mode = mode
        self.f = f
        self.omega = 2 * np.pi * f
        self.amplitude = amplitude
        self.sigma_t = sigma_t
        self.tinj = tinj
        self.phase = phase
        if self.sigma_t is None:
            self.sigma_t = 3.0 / self.f
        if self.tinj is None:
            self.tinj = 6 * self.sigma_t

    def _initialize(self, solver):
        """Resolve the transverse TE01 profile from the solver grid."""
        self.ly = solver.y.max() - solver.y.min()
        y_norm = (solver.y - solver.y.min()) / self.ly
        if self.mode == "TE01":
            self.ExProfile = np.sin(np.pi * y_norm)[None, :]
        else:
            raise NotImplementedError("Only TE01 is currently implemented.")

    def update(self, solver, t):
        """Inject the Gaussian-windowed TE01 electric field at time ``t``."""
        self._ensure_initialized(solver)

        temporal = self.gaussian_profile(t - self.tinj, self.sigma_t)
        carrier = self.harmonic_carrier(self.omega * t + self.phase)
        electric_x = self.amplitude * self.ExProfile * temporal * carrier

        index = (self.xs, self.ys, self.zs, "x")
        self._inject(solver.E, index, electric_x)

    def plot(self, t):
        """Plot the time evolution of the modal electric-field drive."""
        temporal = self.gaussian_profile(np.asarray(t) - self.tinj, self.sigma_t)
        carrier = self.harmonic_carrier(self.omega * np.asarray(t) + self.phase)
        electric_x = self.amplitude * temporal * carrier

        fig, ax = plt.subplots()
        ax.plot(t, electric_x, label="Ex")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Electric field [V/m]")
        ax.legend()
        fig.tight_layout()
        plt.show()
