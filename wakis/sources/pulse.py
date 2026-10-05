# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Electromagnetic pulse source."""

import numpy as np
from scipy.constants import c as c_light


class Pulse:
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
            Width of the pulse (~10*sigma). Default is 50*dt.
        amplitude : float, optional
            Amplitude of the pulse. Default is 1.0.
        delay : float, optional
            Time delay for the pulse [s]. Default is 0.0.

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
        Injection time for the gaussian pulse t0=5*L to ensure smooth derivative.
        """
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

        if shape.lower() == "harris":
            self.tprofile = self.harris_pulse
        elif shape.lower() == "gaussian":
            self.tprofile = self.gaussian_pulse
        elif shape.lower() == "rectangular":
            self.tprofile = self.rectangular_pulse
        else:
            print(
                '** shape does not, match available types: "Harris", "Gaussian", "Rectangular"'
            )

        self.is_first_update = True

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
        t = t * c_light - self.delay
        try:
            if t < self.L:
                return (
                    10
                    - 15 * np.cos(2 * np.pi / self.L * t)
                    + 6 * np.cos(4 * np.pi / self.L * t)
                    - np.cos(6 * np.pi / self.L * t)
                ) / 32  # L dividing (working)
            else:
                return 0.0
        except Exception:  # support for time arrays
            return (
                10
                - 15 * np.cos(2 * np.pi / self.L * t)
                + 6 * np.cos(4 * np.pi / self.L * t)
                - np.cos(6 * np.pi / self.L * t)
            ) / 32  # L dividing (working)

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
        t = t * c_light - self.delay
        return np.exp(-((t - 5 * (self.L / 10)) ** 2) / (2 * (self.L / 10) ** 2))

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
        t = t * c_light - self.delay
        if t < self.L and t > 0.0:
            return 1.0
        else:
            return 0.0

    def update(self, solver, t):
        """
        Update the specified field/component in the solver to represent the pulse at
        time t.

        Parameters
        ----------
        solver : object
            Solver object with E, H, and J field arrays.
        t : float
            Current simulation time [s].
        """
        if self.is_first_update:
            if self.xs is None:
                self.xs = int(solver.Nx / 2)
            if self.ys is None:
                self.ys = int(solver.Ny / 2)
            if self.zs is None:
                self.zs = int(solver.Nz / 2)
            if self.L is None:
                self.L = 50 * solver.dt

            self.is_first_update = False

        if self.field == "E":
            solver.E[self.xs, self.ys, self.zs, self.component] = (
                self.amplitude * self.tprofile(t)
            )
        elif self.field == "H":
            solver.H[self.xs, self.ys, self.zs, self.component] = (
                self.amplitude * self.tprofile(t)
            )
        elif self.field == "J":
            solver.J[self.xs, self.ys, self.zs, self.component] = (
                self.amplitude * self.tprofile(t)
            )
        else:
            print(f'Field "{self.field}" not valid, should be "E", "H" or "J"]')
