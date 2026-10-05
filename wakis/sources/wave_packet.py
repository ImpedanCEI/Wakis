# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Gaussian wave-packet source."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import mu_0


class WavePacket:
    """Matched Ex/Hy Gaussian carrier injected on a transverse plane.

    The temporal envelope peaks at ``tinj / (beta*c)`` and has standard
    deviation ``sigmaz / (beta*c)``. In vacuum the injected fields satisfy
    ``Ex / Hy = mu_0*c``, giving propagation in the positive z direction.
    """

    def __init__(
        self,
        xs=None,
        ys=None,
        zs=0,
        sigmaz=None,
        sigmaxy=None,
        tinj=None,
        wavelength=None,
        f=None,
        amplitude=1.0,
        beta=1.0,
        phase=0,
    ):
        """
        Updates E and H fields every timestep to introduce a 2D gaussian wave packet at
        the given xs, ys slice travelling in z+ direction.

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
        wavelength : float, optional
            Wave packet wavelength [m]. Default is 10*mean(dz), resolved on
            the first update when neither wavelength nor frequency is given.
        f : float, optional
            Wave packet frequency [Hz], overrides wavelength.
        amplitude : float, optional
            Amplitude of the wave packet. Default is 1.0.
        beta : float, optional
            Relativistic beta. Default is 1.0.
        phase : float, optional
            Phase offset [rad]. Default is 0.

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
        wavelength : float
            Wave packet wavelength [m].
        f : float
            Frequency [Hz].
        amplitude : float
            Amplitude of the wave packet.
        beta : float
            Relativistic beta.
        phase : float
            Phase offset [rad].
        w : float
            Angular frequency.
        is_first_update : bool
            Flag for first update call.
        """
        # Check inputs and update self
        self.beta = beta
        self.xs, self.ys = xs, ys
        self.zs = zs
        self.f = f
        self.wavelength = wavelength
        self.sigmaxy = sigmaxy
        self.sigmaz = sigmaz
        self.tinj = tinj
        self.amplitude = amplitude
        self.phase = phase

        if self.f is None and self.wavelength is not None:
            self.f = c_light / self.wavelength
        self.w = None if self.f is None else 2 * np.pi * self.f

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
            if self.f is None:
                self.wavelength = 10 * np.mean(solver.dz)
                self.f = c_light / self.wavelength
                self.w = 2 * np.pi * self.f
            if self.sigmaxy is None:
                self.sigmaxy = 5 * np.mean([np.mean(solver.dx), np.mean(solver.dy)])
            if self.tinj is None:
                self.tinj = 6 * self.sigmaz

            self.is_first_update = False

        # reference shift
        s0 = solver.z.min() - self.tinj
        s = solver.z.min() - self.beta * c_light * t

        # 2d gaussian
        X, Y = np.meshgrid(solver.x[self.xs], solver.y[self.ys], indexing="ij")
        gaussxy = np.exp(-(X**2 + Y**2) / (2 * self.sigmaxy**2))
        gausst = np.exp(-((s - s0) ** 2) / (2 * self.sigmaz**2))

        # Update
        solver.H[self.xs, self.ys, self.zs, "y"] = (
            self.amplitude * np.cos(self.w * t + self.phase) * gaussxy * gausst
        )
        solver.E[self.xs, self.ys, self.zs, "x"] = (
            self.amplitude
            * mu_0
            * c_light
            * np.cos(self.w * t + self.phase)
            * gaussxy
            * gausst
        )

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

        sourceH = self.amplitude * np.cos(self.w * t + self.phase) * gausst
        ax.plot(t, sourceH, "b")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Magnetic field Hy [A/m]", color="b")
        ax.set_ylim(-np.abs(sourceH).max(), +np.abs(sourceH).max())

        sourceE = (
            self.amplitude * mu_0 * c_light * np.cos(self.w * t + self.phase) * gausst
        )
        axx = ax.twinx()
        axx.plot(t, sourceE, "r")
        axx.set_ylabel("Electric field Ex [V/m]", color="r")
        axx.set_ylim(-np.abs(sourceE).max(), +np.abs(sourceE).max())

        fig.tight_layout()
        plt.show()
