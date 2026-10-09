# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Dipole source."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light

from .source import WaveformSource


class Dipole(WaveformSource):
    def __init__(
        self,
        field="E",
        component="z",
        xs=None,
        ys=None,
        zs=None,
        nodes=10,
        f=None,
        amplitude=1.0,
        phase=0,
        injection="hard",
    ):
        """
        Updates the given field and component every timestep to introduce a dipole-like
        sinusoidal excitation.

        Parameters
        ----------
        field : str, optional
            Field to add source to. Supports component e.g. 'Ex'. Default is 'E'.
        component : str, optional
            If not specified in field, component of the field to add the source to.
            Default is 'z'.
        xs, ys, zs : int or slice, optional
            Positions of the source (indexes). Default is N/2.
        nodes : float, optional
            Number of nodes between z.min and z.max. Default is 10.
        f : float, optional
            Frequency of the excitation [Hz]. Overrides nodes param.
        amplitude : float, optional
            Amplitude of the dipole. Default is 1.0.
        phase : float, optional
            Phase offset [rad]. Default is 0.
        injection : {'hard', 'soft'}, optional
            Hard injection assigns the selected component; soft injection
            adds to it. Default is ``'hard'``.

        Attributes
        ----------
        field : str
            Field to add source to.
        component : str
            Field component.
        xs, ys, zs : int or slice
            Source positions.
        nodes : float
            Number of nodes.
        f : float
            Frequency [Hz].
        amplitude : float
            Amplitude of the dipole.
        phase : float
            Phase offset [rad].
        is_first_update : bool
            Flag for first update call.
        """
        super().__init__(injection=injection)

        # Check inputs and update self
        self.nodes = nodes
        self.xs, self.ys, self.zs = xs, ys, zs
        self.f = f
        self.field = field
        self.component = component
        self.amplitude = amplitude
        self.phase = phase
        self.omega = None if self.f is None else 2 * np.pi * self.f

        if len(field) == 2:  # support for e.g. field='Ex'
            self.component = field[1]
            self.field = field[0]

        self.field = self.field.upper()
        self.component = self.component.lower()
        if self.field not in ("E", "H", "J"):
            raise ValueError("field must be 'E', 'H', or 'J'")

    def _initialize(self, solver):
        """Resolve the default position and frequency from the solver."""
        if self.xs is None:
            self.xs = int(solver.Nx / 2)
        if self.ys is None:
            self.ys = int(solver.Ny / 2)
        if self.zs is None:
            self.zs = int(solver.Nz / 2)
        if self.f is None:
            period = (solver.z.max() - solver.z.min()) / c_light
            self.f = self.nodes / period
        self.omega = 2 * np.pi * self.f

    def update(self, solver, t):
        """Inject the selected sinusoidal field component at time ``t``."""
        self._ensure_initialized(solver)

        phase = self.omega * t + self.phase
        waveform = self.amplitude * self.harmonic_carrier(phase, kind="sin")
        index = (self.xs, self.ys, self.zs, self.component)

        if self.field == "E":
            self._inject(solver.E, index, waveform)
        elif self.field == "H":
            self._inject(solver.H, index, waveform)
        else:
            self._inject(solver.J, index, waveform)

    def plot(self, t):
        """Plot the time evolution of the injected dipole component."""
        if self.omega is None:
            raise ValueError("f must be provided or resolved by update() before plot()")

        phase = self.omega * np.asarray(t) + self.phase
        waveform = self.amplitude * self.harmonic_carrier(phase, kind="sin")
        units = {"E": "V/m", "H": "A/m", "J": "A/m²"}[self.field]
        quantity = {
            "E": "Electric field",
            "H": "Magnetic field",
            "J": "Current density",
        }[self.field]

        fig, ax = plt.subplots()
        ax.plot(t, waveform, label=f"{self.field}{self.component}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{quantity} [{units}]")
        ax.legend()
        fig.tight_layout()
        plt.show()
