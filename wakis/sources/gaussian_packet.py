# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Gaussian packet source."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light

from .source import WaveformSource


class GaussianPacket(WaveformSource):
    """Carrier-free soft source with Gaussian temporal and transverse profiles.

    The source writes either ``Ex`` or ``Hy`` on a plane normal to the z-axis;
    the solver evolves the companion field. ``sigmaf`` and ``sigmaz`` use the
    vacuum relation ``sigmaf = c / (2*pi*sigmaz)``.
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
        field="H",
        injection="hard",
    ):
        """
        Inject a carrier-free Gaussian soft source on an xy plane.

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
            Amplitude of the injected field, in V/m for ``field='E'`` or A/m
            for ``field='H'``. Default is 1.0.
        beta : float, optional
            Relativistic beta. Default is 1.0.
        phase : float, optional
            Reserved for API compatibility. This nonoscillatory source does
            not currently apply a carrier phase.
        field : {'E', 'H'}, optional
            Field to inject. ``'E'`` writes Ex in V/m and ``'H'`` writes Hy
            in A/m. Default is ``'H'``.
        injection : {'hard', 'soft'}, optional
            Hard injection assigns the selected field; soft injection adds to
            it. Default is ``'hard'``.

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
            Amplitude of the injected field.
        beta : float
            Relativistic beta.
        phase : float
            Reserved phase value; currently unused.
        field : str
            Injected field, either ``'E'`` or ``'H'``.
        is_first_update : bool
            Flag for first update call.
        """
        super().__init__(injection=injection)

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
        self.field = str(field).upper()
        if self.field not in ("E", "H"):
            raise ValueError("field must be either 'E' or 'H'")

        if self.sigmaf is not None and self.sigmaz is None:
            self.sigmaz = c_light / (2 * np.pi * self.sigmaf)
        if self.sigmaf is None and self.sigmaz is not None:
            self.sigmaf = c_light / (2 * np.pi * self.sigmaz)
        if self.tinj is None and self.sigmaz is not None:
            self.tinj = 6 * self.sigmaz

    def _initialize(self, solver):
        """Resolve mesh-dependent packet defaults."""
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

    def update(self, solver, t):
        """Inject the selected carrier-free Gaussian field at time ``t``."""
        self._ensure_initialized(solver)

        X, Y = np.meshgrid(solver.x[self.xs], solver.y[self.ys], indexing="ij")
        zs_physical = solver.z[self.zs]
        spatial = self.gaussian_spatial_profile(X, Y, self.sigmaxy)
        temporal = self._compute_temporal_envelope(t, zs_physical)
        waveform = self.amplitude * spatial * temporal

        index = (self.xs, self.ys, self.zs)
        if self.field == "H":
            self._inject(solver.H, (*index, "y"), -waveform)
        else:
            self._inject(solver.E, (*index, "x"), waveform)

    def _compute_temporal_envelope(self, t, z_pos=0):
        """Return the Gaussian temporal envelope at a longitudinal position."""
        coordinate = self.tinj - self.beta * c_light * t
        return self.gaussian_profile(coordinate, self.sigmaz)

    def plot(self, t, zmin=0):
        """
        Plot the time evolution of the injected field.

        Parameters
        ----------
        t : array_like
            Array of time values [s].
        zmin : float, optional
            Minimum z position for the reference shift. Default is 0.
        """
        fig, ax = plt.subplots()

        waveform = self.amplitude * self._compute_temporal_envelope(t, zmin)
        waveform *= -1 if self.field == "H" else 1
        units = "A/m" if self.field == "H" else "V/m"
        component = "Hy" if self.field == "H" else "Ex"

        ax.plot(t, waveform, label=component)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{self.field} field [{units}]")
        ax.legend()
        fig.tight_layout()
        plt.show()

    def plot_spectrum(self, t, zmin=0):
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
        gausst = self._compute_temporal_envelope(t, zmin)

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

        return f[mask], S[mask]
