# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Mode-packet source."""

import numpy as np


class ModePacket:
    def __init__(
        self,
        zs=0,
        mode="TE01",
        f=2e9,  # Frequency [Hz]
        amplitude=1.0,
        sigma_t=None,  # Time-based gaussian sigma [s]
        tinj=None,
        phase=0,
    ):
        """
        Updates E and H fields to introduce a TE01 mode.
        Automatically handles both propagating (f > fc) and evanescent (f < fc) regimes.
        """
        self.zs = zs
        self.mode = mode
        self.f = f
        self.w = 2 * np.pi * f
        self.amplitude = amplitude
        self.sigma_t = sigma_t
        self.tinj = tinj
        self.phase = phase
        self.is_first_update = True
        if self.sigma_t is None:
            self.sigma_t = 3.0 / self.f
        if self.tinj is None:
            self.tinj = 6 * self.sigma_t

    def update(self, solver, t):
        if self.is_first_update:
            # Determine Waveguide geometry
            self.ly = solver.y.max() - solver.y.min()

            # Spatial profile for TE01 (E is strictly in x, varying along y)
            y_norm = (solver.y - solver.y.min()) / self.ly
            if self.mode == "TE01":
                self.ExProfile = np.sin(np.pi * y_norm)[None, :]
            else:
                raise NotImplementedError("Only TE01 is currently implemented.")

            self.is_first_update = False

        # Pure temporal Gaussian envelope
        gausst = np.exp(-((t - self.tinj) ** 2) / (2 * self.sigma_t**2))

        # Calculate pure E-field temporal drive
        Et = self.amplitude * np.cos(self.w * t + self.phase) * gausst

        # Inject ONLY into E_x. Let the solver natively compute H!
        solver.E[:, :, self.zs, "x"] = Et * self.ExProfile
