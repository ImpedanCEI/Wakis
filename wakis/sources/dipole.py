# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Dipole source."""

import numpy as np
from scipy.constants import c as c_light


class Dipole:
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
        # Check inputs and update self
        self.nodes = nodes
        self.xs, self.ys, self.zs = xs, ys, zs
        self.f = f
        self.field = field
        self.component = component
        self.amplitude = amplitude
        self.phase = phase

        if len(field) == 2:  # support for e.g. field='Ex'
            self.component = field[1]
            self.field = field[0]

        self.is_first_update = True

    def update(self, solver, t):
        """
        Update the specified field/component in the solver to represent the dipole at
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
            if self.f is None:
                T = (solver.z.max() - solver.z.min()) / c_light
                self.f = self.nodes / T

            self.w = 2 * np.pi * self.f
            self.is_first_update = False

        if self.field == "E":
            solver.E[self.xs, self.ys, self.zs, self.component] = (
                self.amplitude * np.sin(self.w * t + self.phase)
            )
        elif self.field == "H":
            solver.H[self.xs, self.ys, self.zs, self.component] = (
                self.amplitude * np.sin(self.w * t + self.phase)
            )
        elif self.field == "J":
            solver.J[self.xs, self.ys, self.zs, self.component] = (
                self.amplitude * np.sin(self.w * t + self.phase)
            )
        else:
            print(f'Field "{self.field}" not valid, should be "E", "H" or "J"]')
