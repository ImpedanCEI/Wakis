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
        profile = self._compute_longitudinal_profile(
            solver.z, t + solver.dt / 2, self.zmin
        )
        if solver.source_type == "tfsf":
            self._update_tfsf(solver, t, profile)
        else:
            self._update_direct(solver, profile)

    def _compute_longitudinal_profile(self, z_pos, t, zmin):
        """Return the normalized Gaussian bunch profile at ``z_pos`` and ``t``."""
        z_source = zmin + self.v * (t - self.ti)
        s = z_pos - z_source
        return (
            1
            / np.sqrt(2 * np.pi * self.sigmaz**2)
            * np.exp(-(s**2) / (2 * self.sigmaz**2))
        )

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
        if solver.verbose and solver.activate_cpml:
            print("[!] Beam uses Total-Field/Scattered-Field injection with CPML")
        elif solver.verbose and solver.activate_pml:
            print("[!] Beam uses Total-Field/Scattered-Field injection with PML")
        solver.injection_done = False
        self._set_tfsf_planes(solver)
        self._tfsf_field_templates = {}
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
            self._calculate_tfsf_injected_fields(
                solver, z_index=self.j_start, side="low"
            )
        if self.at_high_boundary:
            self._calculate_tfsf_injected_fields(
                solver, z_index=self.j_stop, side="high"
            )

    def _update_direct(self, solver, profile):
        """Update the longitudinal current over the complete local domain."""
        Jprofile = self.q * self.v * profile  # [C] [m/s] [1/m] = [A]
        Jprofile /= solver.tdx[self.ixs] * solver.tdy[self.iys]  # [A/m²]
        dJ = Jprofile - self.Jold
        solver.J[self.ixs, self.iys, :, "z"] += dJ
        self.Jold = Jprofile

    def _update_tfsf(self, solver, t, profile):
        """Update the current and transverse fields inside the TF/SF region."""
        Jprofile = self.q * self.v * profile[self.j_start : self.j_stop]  # [A]
        Jprofile /= solver.tdx[self.ixs] * solver.tdy[self.iys]  # [A/m²]
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
        z_source = self.zmin + self.v * (t + solver.dt / 2 - self.ti)
        if z_source - self.high_plane > 5 * self.sigmaz:
            solver.injection_done = True
            del solver.E_trans, solver.H_trans
            del self._tfsf_field_templates
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

    @staticmethod
    def _get_tfsf_pec_node_mask(solver, z_index):
        """Return the transverse potential nodes fixed to zero by PEC."""
        # The outer boundary is PEC, so only the interior nodes need inspection.
        pec_nodes = np.ones((solver.Nx, solver.Ny), dtype=bool)
        inverse_epsilon_x = solver.ieps[:, :, z_index, "x"]
        inverse_epsilon_y = solver.ieps[:, :, z_index, "y"]

        # A potential node is PEC when any of its four adjacent field edges is PEC.
        pec_nodes[1:-1, 1:-1] = (
            (inverse_epsilon_x[1:-1, 1:-1] == 0)
            | (inverse_epsilon_x[:-2, 1:-1] == 0)
            | (inverse_epsilon_y[1:-1, 1:-1] == 0)
            | (inverse_epsilon_y[1:-1, :-2] == 0)
        )
        return pec_nodes

    def _solve_tfsf_transverse_potential(self, solver, z_index):
        """Solve the normalized transverse Poisson problem for TF/SF injection.

        The relativistic beam is represented by a line charge at ``(ixs, iys)``.
        Its normalized scalar potential satisfies the two-dimensional Poisson
        equation on the selected z-plane. The outer boundary and any nodes next
        to PEC field edges use the Dirichlet condition ``potential = 0``.

        Every remaining node contributes one row of a five-point finite-
        difference stencil, ordered as::

            [center, east, west, north, south]

        with the center coefficient equal to the negative sum of the four
        directional coefficients. Rather than visiting the grid with nested
        Python loops, the method finds all free nodes first and calculates these
        five entries for every row as NumPy arrays. The row indices, column
        indices, and coefficients are then flattened into a COO sparse matrix
        and converted to CSR for the linear solve.

        Returns
        -------
        numpy.ndarray
            Normalized scalar potential with shape ``(Nx, Ny)``.
        """
        nx, ny = solver.Nx, solver.Ny
        node_count = nx * ny

        # Map each (x, y) node to its row in the flattened Poisson system.
        # PEC nodes use a Dirichlet row; the five-point stencil applies elsewhere.
        node_indices = np.arange(node_count).reshape(nx, ny)
        pec_nodes = self._get_tfsf_pec_node_mask(solver, z_index)
        free_x, free_y = np.nonzero(~pec_nodes)
        free_rows = node_indices[free_x, free_y]

        # Compute the four directional coefficients for every free node at once.
        # Each free row contains [center, east, west, north, south].
        east = 1.0 / (solver.dx[free_x] * solver.tdx[free_x])
        west = 1.0 / (solver.dx[free_x - 1] * solver.tdx[free_x])
        north = 1.0 / (solver.dy[free_y] * solver.tdy[free_y])
        south = 1.0 / (solver.dy[free_y - 1] * solver.tdy[free_y])

        stencil_columns = np.column_stack(
            (
                free_rows,
                node_indices[free_x + 1, free_y],
                node_indices[free_x - 1, free_y],
                node_indices[free_x, free_y + 1],
                node_indices[free_x, free_y - 1],
            )
        )
        stencil_coefficients = np.column_stack(
            (-(east + west + north + south), east, west, north, south)
        )

        # PEC rows contain only a unit diagonal; append all free-node stencils.
        pec_rows = node_indices[pec_nodes]
        pec_columns = pec_rows
        pec_coefficients = np.ones(pec_rows.size, dtype=solver.dtype)

        free_matrix_rows = np.repeat(free_rows, 5)
        free_matrix_columns = stencil_columns.ravel()
        free_matrix_coefficients = stencil_coefficients.ravel()

        rows = np.concatenate((pec_rows, free_matrix_rows))
        columns = np.concatenate((pec_columns, free_matrix_columns))
        coefficients = np.concatenate((pec_coefficients, free_matrix_coefficients))

        # Excite the beam node with a normalized line charge, unless it lies on PEC.
        source_term = np.zeros(node_count, dtype=solver.dtype)
        source_node = node_indices[self.ixs, self.iys]
        source_term[source_node] = -1.0 / (c_light * epsilon_0)
        source_term[pec_nodes.ravel()] = 0.0

        # Assemble once in COO format, then convert to CSR for the sparse solve.
        matrix = sp.coo_matrix(
            (coefficients, (rows, columns)),
            shape=(node_count, node_count),
            dtype=solver.dtype,
        ).tocsr()
        potential = spsolve(matrix, source_term)
        return potential.reshape((nx, ny))

    def _calculate_tfsf_injected_fields(self, solver, z_index, side):
        """Precompute normalized transverse TEM fields on an injection plane."""
        if self.beta != 1.0:
            raise NotImplementedError(
                "Only relativistic beta=1 is currently implemented."
            )
        if side not in ("low", "high"):
            raise ValueError("side must be either 'low' or 'high'")

        potential = self._solve_tfsf_transverse_potential(solver, z_index)
        electric_x = np.zeros((solver.Nx, solver.Ny), dtype=solver.dtype)
        electric_y = np.zeros_like(electric_x)
        electric_x[:-1, :] = -np.diff(potential, axis=0) / solver.dx[:-1, None]
        electric_y[:, :-1] = -np.diff(potential, axis=1) / solver.dy[None, :-1]

        magnetic_x = -electric_y / Z0
        magnetic_y = electric_x / Z0
        self._tfsf_field_templates[side] = (
            electric_x,
            electric_y,
            magnetic_x,
            magnetic_y,
        )

    def get_injected_2D_slice(self, solver, z_pos, t, side):
        """
        Evaluates the analytical 2D transverse E and H fields at a specific
        scalar z-coordinate and time t.

        Parameters
        ----------
        solver : object
            The solver instance containing the simulation grid and parameters.
        z_pos : float
            The exact staggered z-coordinate of the boundary plane [m].
        t : float
            The exact staggered simulation time [s].
        side : {'low', 'high'}
            Specifies which boundary plane to evaluate ('low' or 'high').

        Returns
        -------
        E_inj_x, E_inj_y, H_inj_x, H_inj_y : 2D numpy arrays of shape (Nx, Ny)
        """

        profile = self._compute_longitudinal_profile(z_pos, t, self.zmin)
        Jz_pos = self.q * self.v * profile  # [A]
        Jz_pos /= solver.tdx[self.ixs] * solver.tdy[self.iys]  # [A/m²]

        electric_x, electric_y, magnetic_x, magnetic_y = self._tfsf_field_templates[
            side
        ]

        return (
            electric_x * Jz_pos,
            electric_y * Jz_pos,
            magnetic_x * Jz_pos,
            magnetic_y * Jz_pos,
        )

    def plot(self, t):
        """
        Plot the time evolution of the beam current profile.

        Parameters
        ----------
        t : array_like
            Array of time values [s].
        """
        profile = self._compute_longitudinal_profile(0.0, t, 0.0)
        source = self.q * self.v * profile  # line current [A]

        fig, ax = plt.subplots()
        ax.plot(t, source, "darkorange")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Beam current [A]", color="darkorange")
        ax.set_ylim(0.0, +np.abs(source).max() * 1.3)

        fig.tight_layout()
        plt.show()
