# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Electromagnetic pulse source."""

import matplotlib.pyplot as plt
from scipy.constants import c as c_light

from .source import WaveformSource


class Pulse(WaveformSource):
    def __init__(
        self,
        field="E",
        component="z",
        xs=None,
        ys=None,
        zs=None,
        shape="Harris",
        L=None,
        amplitude=1.0,
        delay=0.0,
        injection="hard",
    ):
        """
        Injects an electromagnetic pulse at the given source point (xs, ys, zs), with
        the selected shape, length and amplitude.

        Parameters
        ----------
        field : str, optional
            Field to add source to. Supports component e.g. 'Ex'. Default is 'E'.
        component : str, optional
            If not specified in field, component of the field to add the source to.
            Default is 'z'.
        xs, ys, zs : int or slice, optional
            Positions of the source (indexes). Default is N/2.
        shape : str, optional
            Profile of the pulse in time: ['Harris', 'Gaussian', 'Rectangular'].
            Default is 'Harris'.
        L : float, optional
            Longitudinal pulse length [m]. Default is ``50*c*dt``.
        amplitude : float, optional
            Amplitude of the pulse. Default is 1.0.
        delay : float, optional
            Longitudinal pulse delay [m]. Default is 0.0.
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
        shape : str
            Pulse shape.
        L : float
            Pulse width.
        amplitude : float
            Amplitude of the pulse.
        delay : float
            Time delay for the pulse.
        is_first_update : bool
            Flag for first update call.

        Notes
        -----
        The Gaussian pulse peaks at ``L/2`` and has standard deviation ``L/10``.
        """
        super().__init__(injection=injection)

        # Check inputs and update self
        self.xs, self.ys, self.zs = xs, ys, zs
        self.field = field
        self.component = component
        self.amplitude = amplitude
        self.shape = shape
        self.L = L
        self.delay = delay

        if len(field) == 2:  # support for e.g. field='Ex'
            self.component = field[1]
            self.field = field[0]

        self.field = self.field.upper()
        self.component = self.component.lower()
        if self.field not in ("E", "H", "J"):
            raise ValueError("field must be 'E', 'H', or 'J'")

        if shape.lower() == "harris":
            self.tprofile = self.harris_pulse
        elif shape.lower() == "gaussian":
            self.tprofile = self.gaussian_pulse
        elif shape.lower() == "rectangular":
            self.tprofile = self.rectangular_pulse
        else:
            raise ValueError("shape must be 'Harris', 'Gaussian', or 'Rectangular'")

    def harris_pulse(self, t):
        """
        Harris pulse time profile.

        Parameters
        ----------
        t : float or array_like
            Time value(s) [s].

        Returns
        -------
        float or ndarray
            Harris pulse value(s).
        """
        coordinate = t * c_light - self.delay
        return self.harris_profile(coordinate, self.L)

    def gaussian_pulse(self, t):
        """
        Gaussian pulse time profile.

        Parameters
        ----------
        t : float or array_like
            Time value(s) [s].

        Returns
        -------
        float or ndarray
            Gaussian pulse value(s).
        """
        coordinate = t * c_light - self.delay
        return self.gaussian_profile(coordinate - self.L / 2, self.L / 10)

    def rectangular_pulse(self, t):
        """
        Rectangular pulse time profile.

        Parameters
        ----------
        t : float or array_like
            Time value(s) [s].

        Returns
        -------
        float or ndarray
            Rectangular pulse value(s).
        """
        coordinate = t * c_light - self.delay
        return self.rectangular_profile(coordinate, self.L)

    def _initialize(self, solver):
        """Resolve the default source position and pulse length."""
        if self.xs is None:
            self.xs = int(solver.Nx / 2)
        if self.ys is None:
            self.ys = int(solver.Ny / 2)
        if self.zs is None:
            self.zs = int(solver.Nz / 2)
        if self.L is None:
            self.L = 50 * c_light * solver.dt

    def update(self, solver, t):
        """Inject the selected pulse component at time ``t``."""
        self._ensure_initialized(solver)

        waveform = self.amplitude * self.tprofile(t)
        index = (self.xs, self.ys, self.zs, self.component)

        if self.field == "E":
            self._inject(solver.E, index, waveform)
        elif self.field == "H":
            self._inject(solver.H, index, waveform)
        else:
            self._inject(solver.J, index, waveform)

    def plot(self, t):
        """Plot the time evolution of the injected pulse component."""
        if self.L is None:
            raise ValueError("L must be provided or resolved by update() before plot()")

        waveform = self.amplitude * self.tprofile(t)
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
