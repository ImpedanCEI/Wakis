# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Gaussian wave-packet source."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import mu_0

from .source import WaveformSource


class WavePacket(WaveformSource):
    """Matched TM Gaussian carrier injected on a transverse plane.

    The temporal envelope peaks at ``tinj / (beta*c)`` and has standard
    deviation ``sigmaz / (beta*c)`` at the centre of the plane. ``theta`` is
    measured from positive z towards positive x. The injected fields satisfy
    ``|E| / |H| = mu_0*beta*c`` and their Poynting vector follows the requested
    propagation direction.
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
        theta=0,
        injection="hard",
    ):
        """
        Updates E and H fields every timestep to introduce a 2D gaussian wave packet at
        the given xs, ys slice travelling in z+ direction.

        Parameters
        ----------
        xs, ys : slice or None, optional
            Transverse positions of the source [index]. Default is full range.
        zs : int, optional
            Longitudinal position of the injection plane [index]. Default is 0.
        sigmaz : float, optional
            Longitudinal gaussian sigma [m]. Default is 10*dz.
        sigmaxy : float, optional
            Transverse gaussian sigma [m]. Default is 5*dx.
        tinj : float, optional
            Injection time delay [m]. Default is 6*sigmaz.
        wavelength : float, optional
            Wave packet wavelength along its propagation direction [m].
            Default is 10*mean(dz), resolved on the first update when neither
            wavelength nor frequency is given.
        f : float, optional
            Wave packet frequency [Hz], overrides wavelength.
        amplitude : float, optional
            Magnetic-field amplitude $H_y$ [A/m]. Default is 1.0.
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
        theta : float
            Propagation angle from positive z towards positive x [rad].
        vp : float
            Phase velocity [m/s].
        omega : float
            Angular frequency.
        k : float
            Wave number along the propagation direction [rad/m].
        is_first_update : bool
            Flag for first update call.
        """
        super().__init__(injection=injection)

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
        self.theta = theta
        self.vp = self.beta * c_light

        if self.f is None and self.wavelength is not None:
            self.f = self.vp / self.wavelength
        self.omega = None if self.f is None else 2 * np.pi * self.f
        self.k = None if self.omega is None else self.omega / self.vp

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
        if self.f is None:
            self.wavelength = 10 * np.mean(solver.dz)
            self.f = self.vp / self.wavelength
            self.omega = 2 * np.pi * self.f
            self.k = self.omega / self.vp
        if self.sigmaxy is None:
            self.sigmaxy = 5 * np.mean([np.mean(solver.dx), np.mean(solver.dy)])
        if self.tinj is None:
            self.tinj = 6 * self.sigmaz

    def update(self, solver, t):
        """Inject the matched TM Gaussian wave packet at time ``t``."""
        self._ensure_initialized(solver)

        # tx[0] is the boundary, H starts at tx[1]
        x_edge, y_node = np.meshgrid(
            solver.tx[1:][self.xs], solver.y[self.ys], indexing="ij"
        )
        z_edge = solver.tz[1:][self.zs]
        z_node = solver.z[self.zs]
        impedance = mu_0 * self.vp
        index = (self.xs, self.ys, self.zs)

        # H_y: (x edge, y node, z edge); E_x: (x edge, y node, z node).
        components = [
            (solver.H, "y", x_edge, z_edge, 1.0),
            (solver.E, "x", x_edge, z_node, impedance * np.cos(self.theta)),
        ]

        # Apply rotation, add Ez if the packet is tilted
        if self.theta != 0.0:
            # The tilted packet also has E_z at (x node, y node, z edge).
            x_node, _ = np.meshgrid(solver.x[self.xs], solver.y[self.ys], indexing="ij")
            components.append(
                (solver.E, "z", x_node, z_edge, -impedance * np.sin(self.theta))
            )

        sin_theta = np.sin(self.theta)
        cos_theta = np.cos(self.theta)
        for field, axis, x, z, scale in components:
            # Measure each component's z position from the source's z node.
            offset_z = z - z_node
            u = x * cos_theta - offset_z * sin_theta
            w = x * sin_theta + offset_z * cos_theta
            waveform = (
                scale
                * self.amplitude
                * self.gaussian_spatial_profile(u, y_node, self.sigmaxy)
                * self.gaussian_profile(w - self.vp * t + self.tinj, self.sigmaz)
                * self.harmonic_carrier(self.omega * t - self.k * w + self.phase)
            )
            self._inject(field, (*index, axis), waveform)

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
        w0 = zmin - self.tinj
        w = zmin - self.vp * t
        temporal = self.gaussian_profile(w - w0, self.sigmaz)
        carrier = self.harmonic_carrier(self.omega * t + self.phase)
        magnetic_y = self.amplitude * temporal * carrier
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
