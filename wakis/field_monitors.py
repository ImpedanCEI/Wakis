# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #


from typing import Sequence

import numpy as np

from wakis.field import Field


class FieldMonitor:
    """Accumulate time-domain fields into frequency-domain components.

    The monitor sums snapshots of the three electric-field components with a
    complex exponential kernel for each monitored frequency, producing
    frequency-domain accumulators compatible with either NumPy or CuPy as the
    array backend. Use :meth:`update` to feed time-domain ``Field`` snapshots
    and :meth:`get_components` to retrieve the accumulated spectra.

    Attributes
    ----------
    frequencies : numpy.ndarray
        Array of monitored frequencies in Hz.
    time_index : int
        Current time-step counter (increments on every :meth:`update` call).
    dt : float or None
        Time-step size (set on first :meth:`update` call).
    Ex_acc, Ey_acc, Ez_acc : array or None
        Complex accumulators with shape ``(n_freq, Nx, Ny, Nz)`` storing the
        frequency-domain sums for each field component.
    shape : tuple or None
        Spatial shape of the monitored fields as ``(Nx, Ny, Nz)``.
    xp : module or None
        Array library used for storage and computation (``numpy`` or ``cupy``).
    """

    def __init__(self, frequencies: Sequence[float]):
        """Create a new FieldMonitor.

        Parameters
        ----------
        frequencies : sequence of float
            Frequencies (in Hz) where the field will be monitored. Provide a
            sequence (list or array) of values; they are stored as a
            one-dimensional NumPy array.
        """
        self.frequencies = np.array(frequencies)
        self.time_index = 0
        self.dt = None
        self.Ex_acc = None
        self.Ey_acc = None
        self.Ez_acc = None
        self.shape = None
        self.xp = None

    def update(self, E: Field, dt: float):
        """Accumulate a time-domain ``Field`` snapshot into the monitors.

        On the first call this method initializes internal arrays and records
        the time-step ``dt``. For each monitored frequency f it multiplies the
        spatial field snapshot by exp(-2j*pi*f*t) and adds the result to the
        corresponding complex accumulator. Time ``t`` is computed as
        ``time_index * dt`` before incrementing ``time_index``.

        Parameters
        ----------
        E : Field
            Electric-field container providing ``Nx``, ``Ny``, ``Nz``, the
            backend ``xp`` (numpy or cupy) and a ``to_matrix(component)``
            method that returns a Cartesian component as a dense array.
        dt : float
            Time-step size (s). If ``dt`` was not previously set it is stored
            internally and used for subsequent updates.
        """
        if self.dt is None:
            self.dt = dt
            self.shape = (E.Nx, E.Ny, E.Nz)
            self.xp = E.xp

            n_freqs = len(self.frequencies)
            shape = (n_freqs, *self.shape)
            self.Ex_acc = self.xp.zeros(shape, dtype=self.xp.complex128)
            self.Ey_acc = self.xp.zeros(shape, dtype=self.xp.complex128)
            self.Ez_acc = self.xp.zeros(shape, dtype=self.xp.complex128)
            t = self.time_index * self.dt

        t = self.time_index * self.dt

        Ex = E.to_matrix("x")  # E.array[0:E.N]
        Ey = E.to_matrix("y")
        Ez = E.to_matrix("z")

        for i, f in enumerate(self.frequencies):
            phase = self.xp.exp(-2j * self.xp.pi * f * t)
            self.Ex_acc[i] += Ex * phase
            self.Ey_acc[i] += Ey * phase
            self.Ez_acc[i] += Ez * phase

        self.time_index += 1

    def get_components(self):
        """Return the accumulated frequency-domain components.

        Returns
        -------
        dict
            Dictionary with keys ``'Ex'``, ``'Ey'``, ``'Ez'`` mapping to the
            respective complex accumulator arrays. Each array has shape
            ``(n_freq, Nx, Ny, Nz)`` and uses the configured backend dtype
            (complex128).
        """

        return {"Ex": self.Ex_acc, "Ey": self.Ey_acc, "Ez": self.Ez_acc}
