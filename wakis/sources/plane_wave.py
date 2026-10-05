# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Plane-wave source."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import mu_0


class PlaneWave:
    """Harmonic Ex/Hy plane source intended to propagate in positive z.

    ``amplitude`` is the magnetic-field amplitude in A/m. The electric field is
    scaled to the wave impedance ``mu_0 * vp``.
    """

    def __init__(
        self,
        xs=None,
        ys=None,
        zs=0,
        nodes=None,
        f=None,
        amplitude=1.0,
        beta=1.0,
        phase=0,
    ):
        """
        Updates the fields E and H every timestep to introduce a planewave excitation at
        the given xs, ys slice, moving in z+ direction.

        Parameters
        ----------
        xs, ys : slice or None, optional
            Transverse positions of the source (indexes). Default is full range.
        zs : int or slice, optional
            Injection position in z. Default is 0.
        nodes : float, optional
            Number of periods to inject. By default injection continues
            indefinitely.
        f : float
            Frequency of the plane wave [Hz]. Must be positive.
        amplitude : float, optional
            Amplitude of the plane wave. Default is 1.0.
        beta : float, optional
            Relativistic beta. Default is 1.0.
        phase : float, optional
            Phase offset [rad]. Default is 0.

        Attributes
        ----------
        xs, ys, zs : slice or int
            Source positions.
        nodes : float
            Number of periods to inject.
        f : float
            Frequency [Hz].
        amplitude : float
            Amplitude of the plane wave.
        beta : float
            Relativistic beta.
        phase : float
            Phase offset [rad].
        vp : float
            Phase velocity.
        w : float
            Angular frequency.
        kz : float
            Wave number.
        tmax : float
            Maximum injection time.
        is_first_update : bool
            Flag for first update call.
        """
        if f is None or f <= 0:
            raise ValueError("f must be a positive frequency")
        if nodes is not None and nodes < 0:
            raise ValueError("nodes must be non-negative")

        # Check inputs and update self
        self.nodes = nodes
        self.beta = beta
        self.xs, self.ys = xs, ys
        self.zs = zs
        self.f = f
        self.amplitude = amplitude
        self.is_first_update = True
        self.phase = phase

        self.vp = self.beta * c_light  # wavefront velocity beta*c
        self.w = 2 * np.pi * self.f  # ang. frequency
        self.kz = self.w / c_light  # wave number
        self.tmax = np.inf

        if self.nodes is not None:
            self.tmax = self.nodes / self.f

    def update(self, solver, t):
        """
        Update the E and H fields in the solver to represent the plane wave at time t.

        Parameters
        ----------
        solver : object
            Solver object with E and H field arrays.
        t : float
            Current simulation time [s].
        """
        if self.is_first_update:
            if self.xs is None:
                self.xs = slice(0, solver.Nx)
            if self.ys is None:
                self.ys = slice(0, solver.Ny)

            self.is_first_update = False

        if t <= self.tmax:
            solver.H[self.xs, self.ys, self.zs, "y"] = self.amplitude * np.cos(
                self.w * t + self.phase
            )
            solver.E[self.xs, self.ys, self.zs, "x"] = (
                self.amplitude * mu_0 * self.vp * np.cos(self.w * t + self.phase)
            )
        else:
            solver.H[self.xs, self.ys, self.zs, "y"] = 0.0
            solver.E[self.xs, self.ys, self.zs, "x"] = 0.0

    def plot(self, t):
        """
        Plot the time evolution of the plane wave source fields.

        Parameters
        ----------
        t : array_like
            Array of time values [s].
        """
        fig, ax = plt.subplots()

        sourceH = self.amplitude * np.cos(self.w * t + self.phase)
        sourceE = self.amplitude * mu_0 * self.vp * np.cos(self.w * t + self.phase)

        sourceH[t > self.tmax] = 0.0
        sourceE[t > self.tmax] = 0.0

        ax.plot(t, sourceH, "b")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Magnetic field Hy [A/m]", color="b")
        ax.set_ylim(-np.abs(sourceH).max(), +np.abs(sourceH).max())

        axx = ax.twinx()
        axx.plot(t, sourceE, "r")
        axx.set_ylabel("Electric field Ex [V/m]", color="r")
        axx.set_ylim(-np.abs(sourceE).max(), +np.abs(sourceE).max())

        fig.tight_layout()
        plt.show()
