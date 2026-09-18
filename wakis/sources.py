# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
The `sources.py` script containts different classes
defining a time-dependent sources to be installed
in the electromagnetic simulation.

All sources need an update function that will be called
every simulation timestep, e.g.:
    def update(self, t, *args, **kwargs)`
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import mu_0


class Beam:
    def __init__(
        self, xsource=0.0, ysource=0.0, beta=1.0, q=1e-9, sigmaz=None, ti=None
    ):
        """
        Updates the current J every timestep to introduce a gaussian beam moving in +z
        direction.

        Parameters
        ----------
        xsource, ysource : float, optional
            Transverse position of the source [m]. Default is 0.
        beta : float, optional
            Relativistic beta of the beam [0-1.0]. Default is 1.0.
        q : float, optional
            Beam charge [C]. Default is 1e-9.
        sigmaz : float, optional
            Beam longitudinal sigma [m]. Required.
        ti : float, optional
            Injection time [s]. Default is 8.548921333333334 * sigmaz / v.

        Attributes
        ----------
        xsource, ysource : float
            Transverse position of the source [m].
        sigmaz : float
            Beam longitudinal sigma [m].
        q : float
            Beam charge [C].
        beta : float
            Relativistic beta of the beam.
        v : float
            Beam velocity [m/s].
        ti : float
            Injection time [s].
        is_first_update : bool
            Flag for first update call.
        ixs, iys : int
            Indices of the source in x and y (set on first update).
        zmin : float
            Minimum z position (set on first update).
        """

        self.xsource, self.ysource = xsource, ysource
        self.sigmaz = sigmaz
        self.q = q
        self.beta = beta
        self.v = c_light * beta
        if ti is not None:
            self.ti = ti
        else:
            self.ti = 8.548921333333334 * self.sigmaz / self.v
        self.is_first_update = True

    def update(self, solver, t):
        """
        Update the current density J in the solver to represent the beam at time t.

        Parameters
        ----------
        solver : object
            Solver object with attributes x, y, z, dx, dy, dz, and J.
        t : float
            Current simulation time [s].
        """
        if self.is_first_update:
            self.ixs, self.iys = (
                np.abs(solver.x - self.xsource).argmin(),
                np.abs(solver.y - self.ysource).argmin(),
            )
            if solver.source_type.lower() == "soft":
                self.Jold = np.zeros_like(solver.J[self.ixs, self.iys, :, "z"])
            self.is_first_update = False
            if hasattr(solver, "ZMIN"):  # support for MPI
                zminIdx = np.abs(solver.z - solver.ZMIN).argmin()
                self.zmin = solver.ZMIN + solver.dz[zminIdx] / 2
            else:
                self.zmin = solver.z.min()
        # reference shift
        s0 = self.zmin - self.v * self.ti
        s = solver.z - self.v * t
        # gaussian
        profile = (
            1
            / np.sqrt(2 * np.pi * self.sigmaz**2)
            * np.exp(-((s - s0) ** 2) / (2 * self.sigmaz**2))
        )
        # update
        if solver.source_type.lower() == "hard":
            solver.J[self.ixs, self.iys, :, "z"] = (
                self.q * self.v * profile / solver.dx[self.ixs] / solver.dy[self.iys]
            )

        if solver.source_type.lower() == "soft":
            Jprofile = self.q * self.v * profile / solver.dx[self.ixs] / solver.dy[self.iys]
            dJ = Jprofile - self.Jold
            solver.J[self.ixs, self.iys, :, "z"] += dJ
            self.Jold = Jprofile

    def plot(self, t):
        """
        Plot the time evolution of the beam current profile.

        Parameters
        ----------
        t : array_like
            Array of time values [s].
        """
        # reference shift
        s0 = -self.v * self.ti
        s = -self.v * t
        # gaussian
        profile = (
            1
            / np.sqrt(2 * np.pi * self.sigmaz**2)
            * np.exp(-((s - s0) ** 2) / (2 * self.sigmaz**2))
        )
        source = self.q * self.v * profile

        fig, ax = plt.subplots()
        ax.plot(t, source, "darkorange")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Current Jz [Cm/s]", color="darkorange")
        ax.set_ylim(0.0, +np.abs(source).max() * 1.3)

        fig.tight_layout()
        plt.show()


class PlaneWave:
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
            Number of nodes between z.min and z.max.
        f : float, optional
            Frequency of the plane wave [Hz]. Overrides nodes param.
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
            Number of nodes.
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
            solver.H[self.xs, self.ys, self.zs, "y"] = -self.amplitude * np.cos(
                self.w * t + self.phase
            )
            solver.E[self.xs, self.ys, self.zs, "x"] = (
                self.amplitude
                * np.cos(self.w * t + self.phase)
                / (self.kz / (mu_0 * self.vp))
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

        sourceH = -self.amplitude * np.cos(self.w * t + self.phase)
        sourceE = (
            self.amplitude
            * np.cos(self.w * t + self.phase)
            / (self.kz / (mu_0 * self.vp))
        )

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


class WavePacket:
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
            Wave packet wavelength [m]. Default is 10*dz.
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

        if self.f is None:
            self.f = c_light / self.wavelength
        self.w = 2 * np.pi * self.f

        if self.tinj is None:
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
            if self.sigmaxy is None:
                self.sigmaxy = 5 * np.mean([np.mean(solver.dx), np.mean(solver.dy)])
            if self.tinj is None:
                self.tinj = 6 * self.sigmaz

            self.is_first_update = False

        # reference shift
        s0 = solver.z.min() - self.tinj
        s = solver.z.min() - self.beta * c_light * t

        # 2d gaussian
        X, Y = np.meshgrid(solver.x[self.xs], solver.y[self.ys])
        gaussxy = np.exp(-(X**2 + Y**2) / (2 * self.sigmaxy**2))
        gausst = np.exp(-((s - s0) ** 2) / (2 * self.sigmaz**2))

        # Update
        solver.H[self.xs, self.ys, self.zs, "y"] = (
            -self.amplitude * np.cos(self.w * t + self.phase) * gaussxy * gausst
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

        sourceH = -self.amplitude * np.cos(self.w * t + self.phase) * gausst
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


class ModePacket:
    def __init__(
        self,
        zs=0,
        mode="TE01",
        f=2e9,          # Frequency [Hz]
        amplitude=1.0,
        sigma_t=None,   # Time-based gaussian sigma [s]
        tinj=None,
        phase=0,
    ):
        """
        Updates E and H fields to introduce a TE01 mode.
        Automatically handles both propagating (f > fc) and evanescent (f < fc) regimes.
        """
        self.zs = zs
        self.mode = mode
        self.f = f
        self.w = 2 * np.pi * f
        self.amplitude = amplitude
        self.sigma_t = sigma_t
        self.tinj = tinj
        self.phase = phase
        self.is_first_update = True
        if self.sigma_t is None:
            self.sigma_t = 3.0 / self.f
        if self.tinj is None:
            self.tinj = 6 * self.sigma_t

    def update(self, solver, t):
        if self.is_first_update:
            # Determine Waveguide geometry
            self.ly = solver.y.max() - solver.y.min()

            # Spatial profile for TE01 (E is strictly in x, varying along y)
            y_norm = (solver.y - solver.y.min()) / self.ly
            if self.mode == "TE01":
                self.ExProfile = np.sin(np.pi * y_norm)[None, :]
            else:
                raise NotImplementedError("Only TE01 is currently implemented.")
                
            self.is_first_update = False

        # Pure temporal Gaussian envelope
        gausst = np.exp(-((t - self.tinj) ** 2) / (2 * self.sigma_t**2))

        # Calculate pure E-field temporal drive
        Et = self.amplitude * np.cos(self.w * t + self.phase) * gausst

        # Inject ONLY into E_x. Let the solver natively compute H!
        solver.E[:, :, self.zs, "x"] = Et * self.ExProfile
        

class GaussianPacket:
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
            Wave packet wavelength [m]. Default is 10*dz.
        f : float, optional
            Wave packet frequency [Hz], overrides wavelength.
        amplitude : float, optional
            Amplitude of the wave packet. Default is 1.0.
        beta : float, optional
            Relativistic beta. Default is 1.0.
        phase : float, optional
            Phase offset [rad]. Default is 0.
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
            Propagation angle with respect to z-axis [rad].
        w : float
            Angular frequency.
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

        solver.H[self.xs, self.ys, self.zs, "y"] = (
            -self.amplitude * gaussxy * gausst
        )
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

        sourceE = (
            self.amplitude * mu_0 * c_light * gausst
        )
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
        ax.plot(f[mask] *1e-9, S[mask]/np.max(S[mask]), "m")
        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel("Normalized Spectrum", color="m")
        ax.set_xlim(0, self.sigmaf * 3 * 1e-9)
        fig.tight_layout()
        plt.show()

        return
    
class AngledWavePacket:
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
            Wave packet wavelength [m]. Default is 10*dz.
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
        theta : float
            Angle of the wave packet with respect to the x-axis [rad].
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
        self.theta = theta

        if self.f is None:
            if self.wavelength is None:
                raise ValueError("Either f or wavelength must be provided.")
            self.f = c_light / self.wavelength
        self.w = 2 * np.pi * self.f
        self.k = self.w / (self.beta * c_light)

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
            if self.sigmaxy is None:
                self.sigmaxy = 5 * np.mean([np.mean(solver.dx), np.mean(solver.dy)])
            if self.tinj is None:
                self.tinj = 6 * self.sigmaz

            self.is_first_update = False


        X, Y = np.meshgrid(solver.x[self.xs], solver.y[self.ys], indexing="ij")

        zs_physical = solver.z[self.zs]
        s_spatial = X * np.sin(self.theta) +zs_physical * np.cos(self.theta)
        # reference shift
        s0 = solver.z[self.zs] - self.tinj
        s = s_spatial - self.beta * c_light * t

        # 2d gaussian
        gaussxy = np.exp(-(X**2 + Y**2) / (2 * self.sigmaxy**2))
        gausst = np.exp(-((s - s0) ** 2) / (2 * self.sigmaz**2))

        carrier_phase = self.w * t - self.k * X * np.sin(self.theta) + self.phase

        # Update
        solver.H[self.xs, self.ys, self.zs, "y"] = (
            -self.amplitude * np.cos(carrier_phase) * gaussxy * gausst
        )
        solver.E[self.xs, self.ys, self.zs, "x"] = (
            self.amplitude
            * mu_0
            * c_light
            * np.cos(carrier_phase)
            * gaussxy
            * gausst
        )