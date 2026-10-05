# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Gaussian packet source."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import mu_0


class GaussianPacket:
    """Magnetic soft source with a Gaussian spatial and temporal envelope.

    The source currently writes only ``Hy``; the solver evolves the companion
    electric field. ``sigmaf`` and ``sigmaz`` use the vacuum relation
    ``sigmaf = c / (2*pi*sigmaz)``.
    """

    def __init__(
        self,
        xs=None,
        ys=None,
        zs=0,
        sigmaz=None,
        sigmaxy=None,
        tinj=None,
        amplitude=1.0,
        beta=1.0,
        sigmaf=None,
        phase=0,
        theta=0,
    ):
        """
        Updates Hy every timestep to introduce a magnetic Gaussian soft source
        at the given xs, ys slice.

        Parameters
        ----------
        xs, ys : slice or None, optional
            Transverse positions of the source [index]. Default is full range.
        zs : int, optional
            Longitudinal position of the source [index]. Default is 0.
        sigmaz : float, optional
            Longitudinal gaussian sigma [m]. Default is 10*dz.
        sigmaxy : float, optional
            Transverse gaussian sigma [m]. Default is 5*dx.
        tinj : float, optional
            Injection time delay [m]. Default is 6*sigmaz.
        amplitude : float, optional
            Amplitude of the wave packet. Default is 1.0.
        beta : float, optional
            Relativistic beta. Default is 1.0.
        phase : float, optional
            Reserved for API compatibility. This nonoscillatory source does
            not currently apply a carrier phase.
        theta : float, optional
            Propagation angle with respect to z-axis [rad]. Default is 0.

        Attributes
        ----------
        xs, ys, zs : slice or int
            Source positions.
        sigmaz : float
            Longitudinal gaussian sigma [m].
        sigmaxy : float
            Transverse gaussian sigma [m].
        tinj : float
            Injection time delay [m].
        sigmaf : float
            Gaussian frequency width [Hz].
        amplitude : float
            Amplitude of the wave packet.
        beta : float
            Relativistic beta.
        phase : float
            Reserved phase value; currently unused.
        theta : float
            Propagation angle with respect to z-axis [rad].
        is_first_update : bool
            Flag for first update call.
        """
        # Check inputs and update self
        self.beta = beta
        self.xs, self.ys = xs, ys
        self.zs = zs
        self.sigmaxy = sigmaxy
        self.sigmaz = sigmaz
        self.sigmaf = sigmaf
        self.tinj = tinj
        self.amplitude = amplitude
        self.phase = phase
        self.theta = theta

        if self.sigmaf is not None and self.sigmaz is None:
            self.sigmaz = c_light / (2 * np.pi * self.sigmaf)
        if self.sigmaf is None and self.sigmaz is not None:
            self.sigmaf = c_light / (2 * np.pi * self.sigmaz)
        if self.tinj is None and self.sigmaz is not None:
            self.tinj = 6 * self.sigmaz

        self.is_first_update = True

    def update(self, solver, t):
        """
        Update the E and H fields in the solver to represent the wave packet at time t.

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
            if self.sigmaz is None:
                self.sigmaz = 10 * np.mean(
                    solver.dz
                )  # only feasible for not to ununiform grids
            if self.tinj is None:
                self.tinj = 6 * self.sigmaz
            if self.sigmaxy is None:
                self.sigmaxy = 5 * np.mean([np.mean(solver.dx), np.mean(solver.dy)])

            self.is_first_update = False

        # 2d gaussian
        X, Y = np.meshgrid(solver.x[self.xs], solver.y[self.ys], indexing="ij")
        zs_physical = solver.z[self.zs]
        s_spatial = X * np.sin(self.theta) + zs_physical * np.cos(self.theta)

        # reference shift
        s0 = zs_physical - self.tinj
        s = s_spatial - self.beta * c_light * t

        gaussxy = np.exp(-(X**2 + Y**2) / (2 * self.sigmaxy**2))
        gausst = np.exp(-((s - s0) ** 2) / (2 * self.sigmaz**2))

        # Update

        solver.H[self.xs, self.ys, self.zs, "y"] = -self.amplitude * gaussxy * gausst
        # solver.E[self.xs, self.ys, self.zs, "x"] = (
        #     self.amplitude
        #     * mu_0
        #     * c_light
        #     * gaussxy
        #     * gausst
        # )

    def plot(self, t, zmin=0):
        """
        Plot the time evolution of the wave packet source fields.

        Parameters
        ----------
        t : array_like
            Array of time values [s].
        zmin : float, optional
            Minimum z position for the reference shift. Default is 0.
        """
        fig, ax = plt.subplots()

        # compute source evolution
        s0 = zmin - self.tinj
        s = zmin - self.beta * c_light * t
        gausst = np.exp(-((s - s0) ** 2) / (2 * self.sigmaz**2))

        sourceH = -self.amplitude * gausst
        ax.plot(t, sourceH, "b")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Magnetic field Hy [A/m]", color="b")
        ax.set_ylim(-np.abs(sourceH).max(), +np.abs(sourceH).max())

        sourceE = self.amplitude * mu_0 * c_light * gausst
        axx = ax.twinx()
        axx.plot(t, sourceE, "r")
        axx.set_ylabel("Electric field Ex [V/m]", color="r")
        axx.set_ylim(-np.abs(sourceE).max(), +np.abs(sourceE).max())

        fig.tight_layout()
        plt.show()

    def spectrumPlot(self, t, zmin=0):
        """
        Plot the spectrum of the gaussian pulse.

        Parameters
        ----------
        t : array_like
            Array of time values [s].
        zmin : float, optional
            Minimum z position for the reference shift. Default is 0.

        Returns
        -------
        f : ndarray
            Frequency values [Hz].
        S : ndarray
            Spectrum values (arbitrary units).
        """
        s0 = zmin - self.tinj
        s = zmin - self.beta * c_light * t
        gausst = np.exp(-((s - s0) ** 2) / (2 * self.sigmaz**2))

        S = np.abs(np.fft.fft(gausst)) ** 2
        f = np.fft.fftfreq(len(t), d=t[1] - t[0])

        mask = f >= 0

        fig, ax = plt.subplots()
        ax.plot(f[mask] * 1e-9, S[mask] / np.max(S[mask]), "m")
        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel("Normalized Spectrum", color="m")
        ax.set_xlim(0, self.sigmaf * 3 * 1e-9)
        fig.tight_layout()
        plt.show()

        return
