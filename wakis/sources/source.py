# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Common building blocks for waveform sources."""

from abc import ABC, abstractmethod

import numpy as np


class WaveformSource(ABC):
    """Base class for non-beam sources driven by analytic waveforms.

    The class owns reusable profile functions, coordinate transforms, the
    one-time initialization lifecycle, and hard/soft injection mechanics.
    Concrete sources keep their complete field equations in ``update`` so the
    injected components, signs, and physical scaling remain visible together.
    """

    def __init__(self, injection="hard"):
        injection = str(injection).lower()
        if injection not in ("hard", "soft"):
            raise ValueError("injection must be either 'hard' or 'soft'")
        self.injection = injection
        self.is_first_update = True

    def _ensure_initialized(self, solver):
        """Resolve mesh-dependent state before the first field update."""
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False

    def _initialize(self, solver):
        """Resolve optional mesh-dependent source state."""

    @staticmethod
    def gaussian_profile(coordinate, sigma):
        """Return a unit Gaussian profile centred at zero."""
        coordinate = np.asarray(coordinate)
        return np.exp(-(coordinate**2) / (2 * sigma**2))

    @classmethod
    def gaussian_spatial_profile(cls, u, v, sigma):
        """Return a circular Gaussian profile in transverse coordinates."""
        return cls.gaussian_profile(np.hypot(u, v), sigma)

    @staticmethod
    def rectangular_profile(coordinate, length):
        """Return a unit rectangular profile over ``0 < coordinate < length``."""
        coordinate = np.asarray(coordinate)
        profile = np.where((coordinate > 0.0) & (coordinate < length), 1.0, 0.0)
        return profile.item() if profile.ndim == 0 else profile

    @staticmethod
    def harris_profile(coordinate, length):
        """Return a compact, smooth Harris pulse over one pulse length."""
        coordinate = np.asarray(coordinate)
        phase = 2 * np.pi * coordinate / length
        profile = (
            10 - 15 * np.cos(phase) + 6 * np.cos(2 * phase) - np.cos(3 * phase)
        ) / 32
        profile = np.where((coordinate >= 0.0) & (coordinate < length), profile, 0.0)
        return profile.item() if profile.ndim == 0 else profile

    @staticmethod
    def harmonic_carrier(phase, kind="cos"):
        """Return a sine or cosine carrier for the supplied phase."""
        if kind == "cos":
            return np.cos(phase)
        if kind == "sin":
            return np.sin(phase)
        raise ValueError("carrier kind must be either 'cos' or 'sin'")

    @staticmethod
    def finite_window(coordinate, start=0.0, stop=np.inf):
        """Return a mask selecting coordinates inside a closed interval."""
        coordinate = np.asarray(coordinate)
        return (coordinate >= start) & (coordinate <= stop)

    @staticmethod
    def _source_frame(direction):
        """Build an orthonormal ``(u, v, w)`` source frame.

        The ``w`` axis follows the propagation direction, while ``u`` and
        ``v`` span its transverse plane. The three axes form a right-handed
        coordinate system.
        """
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be a finite 3D vector")
        magnitude = np.linalg.norm(direction)
        if magnitude == 0:
            raise ValueError("direction must be non-zero")
        w_hat = direction / magnitude

        for reference in np.eye(3):
            u_hat = reference - np.dot(reference, w_hat) * w_hat
            u_norm = np.linalg.norm(u_hat)
            if u_norm > 1e-12:
                u_hat /= u_norm
                break
        v_hat = np.cross(w_hat, u_hat)
        return u_hat, v_hat, w_hat

    @classmethod
    def _to_source_frame(cls, x, y, z, direction, origin=(0.0, 0.0, 0.0)):
        """Transform Cartesian coordinates into source-frame ``(u, v, w)``."""
        x0, y0, z0 = origin
        dx = np.asarray(x) - x0
        dy = np.asarray(y) - y0
        dz = np.asarray(z) - z0
        u_hat, v_hat, w_hat = cls._source_frame(direction)
        u = dx * u_hat[0] + dy * u_hat[1] + dz * u_hat[2]
        v = dx * v_hat[0] + dy * v_hat[1] + dz * v_hat[2]
        w = dx * w_hat[0] + dy * w_hat[1] + dz * w_hat[2]
        return u, v, w

    def _inject(self, field, index, values):
        """Apply values to one explicitly selected field component."""
        if self.injection == "hard":
            field[index] = values
        else:
            field[index] = field[index] + values

    @abstractmethod
    def update(self, solver, t):
        """Evaluate the source and explicitly update solver fields."""


# Backward-compatible public name retained for existing user code.
Source = WaveformSource
