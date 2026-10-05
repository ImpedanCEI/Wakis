# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Particle-beam source."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.constants import c as c_light
from scipy.constants import epsilon_0, mu_0
from scipy.sparse.linalg import spsolve

Z0 = np.sqrt(mu_0 / epsilon_0)  # Wave impedance in free space [Ohms]


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
            self._initialize(solver, t)
        # reference shift
        s0 = self.zmin - self.v * self.ti
        s = solver.z - self.v * (t + solver.dt / 2)

        # gaussian
        profile = (
            1
            / np.sqrt(2 * np.pi * self.sigmaz**2)
            * np.exp(-((s - s0) ** 2) / (2 * self.sigmaz**2))
        )
        if solver.source_type == "tfsf":
            self._update_tfsf(solver, t, profile, s0)
        else:
            self._update_direct(solver, profile)

    def _initialize(self, solver, t):
        """Locate the beam and prepare the selected injection mode once."""
        self.is_first_update = False
        self.ixs, self.iys = (
            np.abs(solver.grid.x - self.xsource).argmin(),
            np.abs(solver.grid.y - self.ysource).argmin(),
        )
        if solver.source_type == "tfsf":
            self._initialize_tfsf(solver, t)
        else:
            self._initialize_direct(solver)

        if hasattr(solver, "ZMIN"):  # support for MPI
            self.zmin = solver.ZMIN + solver.dz[0] / 2
        else:
            self.zmin = solver.z.min()

    def _initialize_direct(self, solver):
        """Allocate the previous current profile for direct injection."""
        self.Jold = np.zeros_like(solver.J[self.ixs, self.iys, :, "z"])

    def _initialize_tfsf(self, solver, t):
        """Set TF/SF planes, current history, and incident field templates."""
        solver.injection_done = False
        self._set_tfsf_planes(solver)
        self.Jold = np.zeros_like(
            solver.J[self.ixs, self.iys, self.j_start : self.j_stop, "z"]
        )
        solver.J_max = (
            self.q
            * self.v
            / solver.tdx[self.ixs]
            / solver.tdy[self.iys]
            / np.sqrt(2 * np.pi * self.sigmaz**2)
        )
        if solver.verbose > 1:
            print(
                f"[!] Total-Field/Scattered-Field injection started at t={t:.3e}s, Jmax={solver.J_max:.3e} Cm/s"
            )
        if self.at_low_boundary:
            self._calculate_injected_fields(solver, z_pos=self.j_start, side="low")
        if self.at_high_boundary:
            self._calculate_injected_fields(solver, z_pos=self.j_stop, side="high")

    def _update_direct(self, solver, profile):
        """Update the longitudinal current over the complete local domain."""
        Jprofile = (
            self.q * self.v * profile / solver.tdx[self.ixs] / solver.tdy[self.iys]
        )
        dJ = Jprofile - self.Jold
        solver.J[self.ixs, self.iys, :, "z"] += dJ
        self.Jold = Jprofile

    def _update_tfsf(self, solver, t, profile, s0):
        """Update the current and transverse fields inside the TF/SF region."""
        Jprofile = (
            self.q
            * self.v
            * profile[self.j_start : self.j_stop]
            / solver.tdx[self.ixs]
            / solver.tdy[self.iys]
        )
        dJ = Jprofile - self.Jold
        solver.J[self.ixs, self.iys, self.j_start : self.j_stop, "z"] += dJ
        self.Jold = Jprofile

        if solver.injection_done:
            return

        # Update transverse E and H fields on the injection planes using the
        # pre-calculated 2D templates.
        if self.at_low_boundary:
            Einj_x, Einj_y, _, _ = self.get_injected_2D_slice(
                solver, solver.grid.z[self.j_start], t, side="low"
            )
            solver.E_trans[:, :, self.j_start, "x"] = Einj_x
            solver.E_trans[:, :, self.j_start, "y"] = Einj_y
            _, _, Hinj_x, Hinj_y = self.get_injected_2D_slice(
                solver, solver.z[self.j_start], t + solver.dt / 2, side="low"
            )
            solver.H_trans[:, :, self.j_start, "x"] = -Hinj_x
            solver.H_trans[:, :, self.j_start, "y"] = -Hinj_y

        if self.at_high_boundary:
            Einj_x, Einj_y, _, _ = self.get_injected_2D_slice(
                solver, solver.grid.z[self.j_stop], t, side="high"
            )
            solver.E_trans[:, :, self.j_stop, "x"] = -Einj_x
            solver.E_trans[:, :, self.j_stop, "y"] = -Einj_y
            _, _, Hinj_x, Hinj_y = self.get_injected_2D_slice(
                solver, solver.z[self.j_stop], t + solver.dt / 2, side="high"
            )
            solver.H_trans[:, :, self.j_stop, "x"] = Hinj_x
            solver.H_trans[:, :, self.j_stop, "y"] = Hinj_y

        # Truncate injection after the beam has passed the plane by 5 sigma.
        if s0 - (self.high_plane - self.v * (t + solver.dt / 2)) > 5 * self.sigmaz:
            solver.injection_done = True
            del solver.E_trans, solver.H_trans
            for side in ("low", "high"):
                for component in ("E2D_x", "E2D_y", "H2D_x", "H2D_y"):
                    name = f"{component}_{side}"
                    if hasattr(self, name):
                        delattr(self, name)
            del solver.tf_dxz, solver.tf_dyz, solver.tf_dtxz, solver.tf_dtyz
            if solver.verbose > 1:
                print(
                    f"[!] Total-Field/Scattered-Field injection done at t={t:.3e}s, switching to regular field updates"
                )

    def _set_tfsf_planes(self, solver):
        """Set local current limits and the physical injection planes."""
        self.at_low_boundary = not solver.use_mpi or solver.rank == 0
        self.at_high_boundary = not solver.use_mpi or solver.rank == solver.size - 1
        low_pml = solver.bc_low[2].lower() in ("cpml", "pml")
        high_pml = solver.bc_high[2].lower() in ("cpml", "pml")

        # Apply the PML offset only on the corresponding physical z face.
        self.j_start = solver.n_pml + 1 if self.at_low_boundary and low_pml else 1
        high_offset = solver.n_pml + 2 if high_pml else 2
        if self.at_high_boundary:
            self.j_stop = solver.Nz - high_offset
        else:
            # Current slices exclude the stop; Nz - 1 includes the last physical cell.
            self.j_stop = solver.Nz - 1

        # All MPI ranks use the same global high plane to end injection together.
        if solver.use_mpi:
            global_stop = solver.NZ - high_offset
            self.high_plane = solver.Z[global_stop]
        else:
            self.high_plane = solver.z[self.j_stop]

    def _calculate_injected_fields(self, solver, z_pos, side):
        """
        Pre-calculates the normalized 2D TEM transverse field templates (E and H)
        on the injection plane using a discrete FIT/Yee 2D Poisson solver.

        Guarantees machine-zero discrete divergence on staggered grids.
        """
        if self.beta != 1.0:
            raise NotImplementedError(
                "Only relativistic beta=1 is currently implemented."
            )

        def k(i, j):
            """Maps 2D grid coordinates to 1D flat index."""
            return i * Ny + j

        Nx, Ny = solver.Nx, solver.Ny
        N = Nx * Ny

        b = np.zeros((N), dtype=solver.dtype)
        rho_source = 1.0 / c_light  # Normalized source charge density
        b[k(self.ixs, self.iys)] = -rho_source / epsilon_0

        row, col, data = [], [], []

        for i in range(Nx):
            for j in range(Ny):
                row_idx = k(i, j)

                is_pec_node = (
                    i == 0
                    or i == Nx - 1
                    or j == 0
                    or j == Ny - 1
                    or solver.ieps[i, j, z_pos, "x"] == 0
                    or solver.ieps[i - 1, j, z_pos, "x"] == 0
                    or solver.ieps[i, j, z_pos, "y"] == 0
                    or solver.ieps[i, j - 1, z_pos, "y"] == 0
                )

                # Enforce Dirichlet boundary condition (phi = 0) at PEC nodes
                if is_pec_node:
                    row.append(row_idx)
                    col.append(row_idx)
                    data.append(1.0)
                    b[row_idx] = 0.0
                    continue

                a_E = 1.0 / (solver.dx[i] * solver.tdx[i])
                a_W = 1.0 / (solver.dx[i - 1] * solver.tdx[i])
                a_N = 1.0 / (solver.dy[j] * solver.tdy[j])
                a_S = 1.0 / (solver.dy[j - 1] * solver.tdy[j])
                a_C = -(a_E + a_W + a_N + a_S)

                row.append(row_idx)
                col.append(row_idx)
                data.append(a_C)
                row.append(row_idx)
                col.append(k(i + 1, j))
                data.append(a_E)
                row.append(row_idx)
                col.append(k(i - 1, j))
                data.append(a_W)
                row.append(row_idx)
                col.append(k(i, j + 1))
                data.append(a_N)
                row.append(row_idx)
                col.append(k(i, j - 1))
                data.append(a_S)

        A = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
        phi_vec = spsolve(A, b)
        phi = phi_vec.reshape((Nx, Ny))

        E2D_x = np.zeros((Nx, Ny), dtype=solver.dtype)
        E2D_y = np.zeros((Nx, Ny), dtype=solver.dtype)

        for i in range(Nx - 1):
            for j in range(Ny):
                E2D_x[i, j] = -(phi[i + 1, j] - phi[i, j]) / solver.dx[i]

        for i in range(Nx):
            for j in range(Ny - 1):
                E2D_y[i, j] = -(phi[i, j + 1] - phi[i, j]) / solver.dy[j]

        H2D_x = -E2D_y / Z0
        H2D_y = E2D_x / Z0

        if side == "low":
            self.E2D_x_low = E2D_x
            self.E2D_y_low = E2D_y
            self.H2D_x_low = H2D_x
            self.H2D_y_low = H2D_y

        elif side == "high":
            self.E2D_x_high = E2D_x
            self.E2D_y_high = E2D_y
            self.H2D_x_high = H2D_x
            self.H2D_y_high = H2D_y

    def get_injected_2D_slice(self, solver, z_pos, t, side):
        """
        Evaluates the analytical 2D transverse E and H fields at a specific
        scalar z-coordinate and time t.

        Parameters
        ----------
        solver : object
        z_pos : float
            The exact staggered z-coordinate of the boundary plane [m].
        t : float
            The exact staggered simulation time [s].

        Returns
        -------
        E_inj_x, E_inj_y, H_inj_x, H_inj_y : 2D numpy arrays of shape (Nx, Ny)
        """

        # Calculate the relative position in the bunch frame
        s0 = self.zmin - self.v * self.ti
        s = z_pos - self.v * t

        # Evaluate the Gaussian profile at this z-position
        profile = (
            1
            / np.sqrt(2 * np.pi * self.sigmaz**2)
            * np.exp(-((s - s0) ** 2) / (2 * self.sigmaz**2))
        )
        Jz_pos = self.q * self.v * profile / solver.tdx[self.ixs] / solver.tdy[self.iys]

        # Scale the 2D templates by the current density at this z-position and time step
        if side == "low":
            E_inj_x = self.E2D_x_low * Jz_pos
            E_inj_y = self.E2D_y_low * Jz_pos
            H_inj_x = self.H2D_x_low * Jz_pos
            H_inj_y = self.H2D_y_low * Jz_pos
        elif side == "high":
            E_inj_x = self.E2D_x_high * Jz_pos
            E_inj_y = self.E2D_y_high * Jz_pos
            H_inj_x = self.H2D_x_high * Jz_pos
            H_inj_y = self.H2D_y_high * Jz_pos

        return E_inj_x, E_inj_y, H_inj_x, H_inj_y

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
