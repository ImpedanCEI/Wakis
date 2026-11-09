# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Plane-wave source."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import mu_0

from .source import WaveformSource


class PlaneWave(WaveformSource):
    """Harmonic TM plane source with configurable x-z propagation angle.

    ``amplitude`` is the magnetic-field amplitude in A/m. The electric field is
    scaled to the wave impedance ``mu_0 * vp``. ``theta`` is measured from
    positive z towards positive x.
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
        theta=0,
        injection="hard",
    ):
        """
        Update matched E and H fields to launch a plane wave in the x-z plane.

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
        theta : float, optional
            Propagation angle from positive z towards positive x [rad].
            Default is 0.
        injection : {'hard', 'soft'}, optional
            Hard injection assigns the fields; soft injection adds to them.
            Default is ``'hard'``.

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
        omega : float
            Angular frequency.
        k : float
            Wave number along propagation direction [rad/m].
        tmax : float
            Maximum injection time.
        is_first_update : bool
            Flag for first update call.
        """
        if f is None or f <= 0:
            raise ValueError("f must be a positive frequency")
        if nodes is not None and nodes < 0:
            raise ValueError("nodes must be non-negative")
        super().__init__(injection=injection)

        # Check inputs and update self
        self.nodes = nodes
        self.beta = beta
        self.xs, self.ys = xs, ys
        self.zs = zs
        self.f = f
        self.amplitude = amplitude
        self.phase = phase
        self.theta = theta

        self.vp = self.beta * c_light  # wavefront velocity beta*c
        self.omega = 2 * np.pi * self.f
        self.k = self.omega / self.vp
        self.tmax = np.inf

        if self.nodes is not None:
            self.tmax = self.nodes / self.f

    def _initialize(self, solver):
        """Resolve the default transverse extent from the solver grid."""
        if self.xs is None:
            self.xs = slice(0, solver.Nx)
        if self.ys is None:
            self.ys = slice(0, solver.Ny)

    def update(self, solver, t):
        """Inject matched TM fields for the plane wave at time ``t``."""
        self._ensure_initialized(solver)

        # Evaluate each component at its staggered field position.
        x_edge = solver.tx[1:][self.xs]
        z_node = solver.z[self.zs]
        z_edge = solver.tz[1:][self.zs]

        z_slice = isinstance(self.zs, slice)
        if z_slice:
            z_node = z_node[None, None, :]
            z_edge = z_edge[None, None, :]
            x_edge = x_edge[:, None, None]
        else:
            x_edge = x_edge[:, None]

        # Compute fields
        magnetic_y = self._waveform(
            t,
            x_edge * np.sin(self.theta) + (z_edge - z_node) * np.cos(self.theta),
        )
        impedance = mu_0 * self.vp
        electric_x = (
            impedance
            * np.cos(self.theta)
            * self._waveform(t, x_edge * np.sin(self.theta))
        )

        index = (self.xs, self.ys, self.zs)
        self._inject(solver.H, (*index, "y"), magnetic_y)
        self._inject(solver.E, (*index, "x"), electric_x)

        # Inject the z-component of the electric field if the wave is tilted.
        if self.theta != 0.0:
            x_node = solver.x[self.xs]
            x_node = x_node[:, None, None] if z_slice else x_node[:, None]
            electric_z = (
                -impedance
                * np.sin(self.theta)
                * self._waveform(
                    t,
                    x_node * np.sin(self.theta)
                    + (z_edge - z_node) * np.cos(self.theta),
                )
            )
            self._inject(solver.E, (*index, "z"), electric_z)

    def _waveform(self, t, w=0.0):
        """Return the windowed base waveform in A/m at coordinate ``w``."""
        retarded_time = np.asarray(t) - np.asarray(w) / self.vp
        phase = self.omega * np.asarray(t) - self.k * np.asarray(w) + self.phase
        waveform = self.amplitude * self.harmonic_carrier(phase)
        active = self.finite_window(retarded_time, stop=self.tmax)
        return np.where(active, waveform, 0.0)

    def plot(self, t):
        """
        Plot the time evolution of the plane wave source fields.

        Parameters
        ----------
        t : array_like
            Array of time values [s].
        """
        fig, ax = plt.subplots()

        magnetic_y = self._waveform(t)
        impedance = mu_0 * self.vp
        electric_x = impedance * np.cos(self.theta) * magnetic_y
        electric_z = -impedance * np.sin(self.theta) * magnetic_y

        ax.plot(t, magnetic_y, label="Hy")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Magnetic field [A/m]")
        ax.legend(loc="upper left")

        axx = ax.twinx()
        axx.plot(t, electric_x, label="Ex")
        if self.theta != 0.0:
            axx.plot(t, electric_z, label="Ez")
        axx.set_ylabel("Electric field [V/m]")
        axx.legend(loc="upper right")

        fig.tight_layout()
        plt.show()
