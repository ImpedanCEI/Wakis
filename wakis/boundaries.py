# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np
from scipy.constants import mu_0 as mu_0
from scipy.sparse import diags

from .field import Field


class BCsMixin:
    def _apply_bc_to_C(self):
        """
        Apply boundary conditions by modifying curl and metric matrices.

        Adjusts rows/columns of the curl operator ``C`` and the metric-diagonal
        matrices (``tDs``, ``itDa``) according to the low/high boundary
        condition lists ``bc_low`` and ``bc_high``. Handles periodic, PEC/PMC,
        ABC and PML options and also configures MPI-internal faces when the
        grid is subdivided.
        """
        xlo, ylo, zlo = 1.0, 1.0, 1.0
        xhi, yhi, zhi = 1.0, 1.0, 1.0

        # Check BCs for internal MPI subdomains
        if self.use_mpi and self.grid.use_mpi:
            if self.rank > 0:
                self.bc_low = ["pec", "pec", "mpi"]

            if self.rank < self.size - 1:
                self.bc_high = ["pec", "pec", "mpi"]

        # Perodic: out == in
        if any(True for x in self.bc_low if x.lower() == "periodic"):
            if (
                self.bc_low[0].lower() == "periodic"
                and self.bc_high[0].lower() == "periodic"
            ):
                self.tL[-1, :, :, "x"] = self.L[0, :, :, "x"]
                self.itA[-1, :, :, "y"] = self.iA[0, :, :, "y"]
                self.itA[-1, :, :, "z"] = self.iA[0, :, :, "z"]

            if (
                self.bc_low[1].lower() == "periodic"
                and self.bc_high[1].lower() == "periodic"
            ):
                self.tL[:, -1, :, "y"] = self.L[:, 0, :, "y"]
                self.itA[:, -1, :, "x"] = self.iA[:, 0, :, "x"]
                self.itA[:, -1, :, "z"] = self.iA[:, 0, :, "z"]

            if (
                self.bc_low[2].lower() == "periodic"
                and self.bc_high[2].lower() == "periodic"
            ):
                self.tL[:, :, -1, "z"] = self.L[:, :, 0, "z"]
                self.itA[:, :, -1, "x"] = self.iA[:, :, 0, "x"]
                self.itA[:, :, -1, "y"] = self.iA[:, :, 0, "y"]

            self.tDs = diags(
                self.tL.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
            self.itDa = diags(
                self.itA.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )

        # Dirichlet PEC: tangential E field = 0 at boundary
        if any(
            True
            for x in self.bc_low
            if x.lower() in ("electric", "pec", "pml")
        ) or any(
            True
            for x in self.bc_high
            if x.lower() in ("electric", "pec", "pml")
        ):
            if self.bc_low[0].lower() in ("electric", "pec", "pml"):
                xlo = 0
            if self.bc_low[1].lower() in ("electric", "pec", "pml"):
                ylo = 0
            if self.bc_low[2].lower() in ("electric", "pec", "pml"):
                zlo = 0
            if self.bc_high[0].lower() in ("electric", "pec", "pml"):
                xhi = 0
            if self.bc_high[1].lower() in ("electric", "pec", "pml"):
                yhi = 0
            if self.bc_high[2].lower() in ("electric", "pec", "pml"):
                zhi = 0

            # Assemble matrix
            self.BC = Field(
                self.Nx, self.Ny, self.Nz, dtype=np.int8, use_ones=True
            )

            for d in ["x", "y", "z"]:  # tangential to zero
                if d != "x":
                    self.BC[0, :, :, d] = xlo
                    self.BC[-1, :, :, d] = xhi
                if d != "y":
                    self.BC[:, 0, :, d] = ylo
                    self.BC[:, -1, :, d] = yhi
                if d != "z":
                    self.BC[:, :, 0, d] = zlo
                    self.BC[:, :, -1, d] = zhi

            self.Dbc = diags(
                self.BC.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=np.int8,
            )

            # Update C (columns)
            self.C = self.C * self.Dbc

        # Dirichlet PMC: tangential H field = 0 at boundary
        if any(
            True for x in self.bc_low if x.lower() in ("magnetic", "pmc")
        ) or any(
            True for x in self.bc_high if x.lower() in ("magnetic", "pmc")
        ):
            if self.bc_low[0].lower() == "magnetic" or self.bc_low[0] == "pmc":
                xlo = 0
            if self.bc_low[1].lower() == "magnetic" or self.bc_low[1] == "pmc":
                ylo = 0
            if self.bc_low[2].lower() == "magnetic" or self.bc_low[2] == "pmc":
                zlo = 0
            if (
                self.bc_high[0].lower() == "magnetic"
                or self.bc_high[0] == "pmc"
            ):
                xhi = 0
            if (
                self.bc_high[1].lower() == "magnetic"
                or self.bc_high[1] == "pmc"
            ):
                yhi = 0
            if (
                self.bc_high[2].lower() == "magnetic"
                or self.bc_high[2] == "pmc"
            ):
                zhi = 0

            # Assemble matrix
            self.BC = Field(
                self.Nx, self.Ny, self.Nz, dtype=np.int8, use_ones=True
            )

            for d in ["x", "y", "z"]:  # tangential to zero
                if d != "x":
                    self.BC[0, :, :, d] = xlo
                    self.BC[-1, :, :, d] = xhi
                if d != "y":
                    self.BC[:, 0, :, d] = ylo
                    self.BC[:, -1, :, d] = yhi
                if d != "z":
                    self.BC[:, :, 0, d] = zlo
                    self.BC[:, :, -1, d] = zhi

            self.Dbc = diags(
                self.BC.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=np.int8,
            )

            # Update C (rows)
            self.C = self.Dbc * self.C

        # Absorbing boundary conditions ABC
        if any(True for x in self.bc_low if x.lower() == "abc") or any(
            True for x in self.bc_high if x.lower() == "abc"
        ):
            if self.bc_high[0].lower() == "abc":
                self.tL[-1, :, :, "x"] = self.L[0, :, :, "x"]
                self.itA[-1, :, :, "y"] = self.iA[0, :, :, "y"]
                self.itA[-1, :, :, "z"] = self.iA[0, :, :, "z"]

            if self.bc_high[1].lower() == "abc":
                self.tL[:, -1, :, "y"] = self.L[:, 0, :, "y"]
                self.itA[:, -1, :, "x"] = self.iA[:, 0, :, "x"]
                self.itA[:, -1, :, "z"] = self.iA[:, 0, :, "z"]

            if self.bc_high[2].lower() == "abc":
                self.tL[:, :, -1, "z"] = self.L[:, :, 0, "z"]
                self.itA[:, :, -1, "x"] = self.iA[:, :, 0, "x"]
                self.itA[:, :, -1, "y"] = self.iA[:, :, 0, "y"]

            self.tDs = diags(
                self.tL.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
            self.itDa = diags(
                self.itA.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
            self.activate_abc = True

        # Perfect Matching Layers (PML)
        if any(True for x in self.bc_low if x.lower() == "pml") or any(
            True for x in self.bc_high if x.lower() == "pml"
        ):
            self.activate_pml = True
            self.use_conductivity = True

    def _initialize_PML(self):
        """
        Compute and apply PML sigma profiles to the solver conductivity tensor.

        Uses configured PML settings (number of layers, profile function and
        scaling) to set per-component conductivity in the PML regions. This is
        used to absorb outgoing waves and reduce reflections at domain edges.
        """

        # Initialize
        sx, sy, sz = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        # pml_exp = 2
        self.pml_lo = 5.0e-3
        self.pml_hi = 10.0
        self.pml_func = np.geomspace
        self.pml_eps_r = 1.0

        # Fill
        if self.bc_low[0].lower() == "pml":
            # sx[0:self.n_pml] = eps_0/(2*self.dt)*((self.x[self.n_pml] - self.x[:self.n_pml])/(self.n_pml*self.dx))**pml_exp
            sx[0 : self.n_pml] = self.pml_func(
                self.pml_hi, self.pml_lo, self.n_pml
            )
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the yz plane
                ieps_0_pml = self.ieps[
                    self.n_pml + 1, self.Ny // 2, self.Nz // 2, d
                ]
                sigma_0_pml = self.sigma[
                    self.n_pml + 1, self.Ny // 2, self.Nz // 2, d
                ]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for i in range(self.n_pml):
                    self.ieps[i, :, :, d] = ieps_0_pml
                    self.sigma[i, :, :, d] = (
                        sigma_0_pml + sigma_mult_pml * sx[i]
                    )
                    # if sx[i] > 0 : self.ieps[i, :, :, d] = 1/(eps_0+sx[i]*(2*self.dt))

        if self.bc_low[1].lower() == "pml":
            # sy[0:self.n_pml] = 1/(2*self.dt)*((self.y[self.n_pml] - self.y[:self.n_pml])/(self.n_pml*self.dy))**pml_exp
            sy[0 : self.n_pml] = self.pml_func(
                self.pml_hi, self.pml_lo, self.n_pml
            )
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the xz plane
                ieps_0_pml = self.ieps[
                    self.Nx // 2, self.n_pml + 1, self.Nz // 2, d
                ]
                sigma_0_pml = self.sigma[
                    self.Nx // 2, self.n_pml + 1, self.Nz // 2, d
                ]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for j in range(self.n_pml):
                    self.ieps[:, j, :, d] = ieps_0_pml
                    self.sigma[:, j, :, d] = (
                        sigma_0_pml + sigma_mult_pml * sy[j]
                    )
                    # if sy[j] > 0 : self.ieps[:, j, :, d] = 1/(eps_0+sy[j]*(2*self.dt))

        if self.bc_low[2].lower() == "pml":
            # sz[0:self.n_pml] = eps_0/(2*self.dt)*((self.z[self.n_pml] - self.z[:self.n_pml])/(self.n_pml*self.dz))**pml_exp
            sz[0 : self.n_pml] = self.pml_func(
                self.pml_hi, self.pml_lo, self.n_pml
            )
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the xy plane
                ieps_0_pml = self.ieps[
                    self.Nx // 2, self.Ny // 2, self.n_pml + 1, d
                ]
                sigma_0_pml = self.sigma[
                    self.Nx // 2, self.Ny // 2, self.n_pml + 1, d
                ]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for k in range(self.n_pml):
                    self.ieps[:, :, k, d] = ieps_0_pml
                    self.sigma[:, :, k, d] = (
                        sigma_0_pml + sigma_mult_pml * sz[k]
                    )
                    # if sz[k] > 0. : self.ieps[:, :, k, d] = 1/(np.mean(sz[:self.n_pml])*eps_0)

        if self.bc_high[0].lower() == "pml":
            # sx[-self.n_pml:] = 1/(2*self.dt)*((self.x[-self.n_pml:] - self.x[-self.n_pml])/(self.n_pml*self.dx))**pml_exp
            sx[-self.n_pml :] = self.pml_func(
                self.pml_lo, self.pml_hi, self.n_pml
            )
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the yz plane
                ieps_0_pml = self.ieps[
                    -(self.n_pml + 1), self.Ny // 2, self.Nz // 2, d
                ]
                sigma_0_pml = self.sigma[
                    -(self.n_pml + 1), self.Ny // 2, self.Nz // 2, d
                ]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for i in range(self.n_pml):
                    i += 1
                    self.ieps[-i, :, :, d] = ieps_0_pml
                    self.sigma[-i, :, :, d] = (
                        sigma_0_pml + sigma_mult_pml * sx[-i]
                    )
                    # if sx[-i] > 0 : self.ieps[-i, :, :, d] = 1/(eps_0+sx[-i]*(2*self.dt))

        if self.bc_high[1].lower() == "pml":
            # sy[-self.n_pml:] = 1/(2*self.dt)*((self.y[-self.n_pml:] - self.y[-self.n_pml])/(self.n_pml*self.dy))**pml_exp
            sy[-self.n_pml :] = self.pml_func(
                self.pml_lo, self.pml_hi, self.n_pml
            )
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the xz plane
                ieps_0_pml = self.ieps[
                    self.Nx // 2, -(self.n_pml + 1), self.Nz // 2, d
                ]
                sigma_0_pml = self.sigma[
                    self.Nx // 2, -(self.n_pml + 1), self.Nz // 2, d
                ]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for j in range(self.n_pml):
                    j += 1
                    self.ieps[:, -j, :, d] = ieps_0_pml
                    self.sigma[:, -j, :, d] = (
                        sigma_0_pml + sigma_mult_pml * sy[-j]
                    )
                    # if sy[-j] > 0 : self.ieps[:, -j, :, d] = 1/(eps_0+sy[-j]*(2*self.dt))

        if self.bc_high[2].lower() == "pml":
            # sz[-self.n_pml:] = eps_0/(2*self.dt)*((self.z[-self.n_pml:] - self.z[-self.n_pml])/(self.n_pml*self.dz))**pml_exp
            sz[-self.n_pml :] = self.pml_func(
                self.pml_lo, self.pml_hi, self.n_pml
            )
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the xy plane
                ieps_0_pml = self.ieps[
                    self.Nx // 2, self.Ny // 2, -(self.n_pml + 1), d
                ]
                sigma_0_pml = self.sigma[
                    self.Nx // 2, self.Ny // 2, -(self.n_pml + 1), d
                ]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for k in range(self.n_pml):
                    k += 1
                    self.ieps[:, :, -k, d] = ieps_0_pml
                    self.sigma[:, :, -k, d] = (
                        sigma_0_pml + sigma_mult_pml * sz[-k]
                    )
                    # self.ieps[:, :, -k, d] = 1/(np.mean(sz[-self.n_pml:])*eps_0)

    def get_abc(self):
        """
        Save boundary field snapshots needed by the Absorbing Boundary
        Condition (ABC) update.

        Extracts the necessary boundary layers for electric and magnetic
        fields for those faces configured with ABC and returns two
        dictionaries holding the saved arrays. Those dictionaries are later
        consumed by ``update_abc`` to restore boundary values.
        """
        E_abc, H_abc = {}, {}

        if self.bc_low[0].lower() == "abc":
            E_abc[0] = {}
            H_abc[0] = {}
            for d in ["x", "y", "z"]:
                E_abc[0][d + "lo"] = self.E[1, :, :, d]
                H_abc[0][d + "lo"] = self.H[1, :, :, d]

        if self.bc_low[1].lower() == "abc":
            E_abc[1] = {}
            H_abc[1] = {}
            for d in ["x", "y", "z"]:
                E_abc[1][d + "lo"] = self.E[:, 1, :, d]
                H_abc[1][d + "lo"] = self.H[:, 1, :, d]

        if self.bc_low[2].lower() == "abc":
            E_abc[2] = {}
            H_abc[2] = {}
            for d in ["x", "y", "z"]:
                E_abc[2][d + "lo"] = self.E[:, :, 1, d]
                H_abc[2][d + "lo"] = self.H[:, :, 1, d]

        if self.bc_high[0].lower() == "abc":
            E_abc[0] = {}
            H_abc[0] = {}
            for d in ["x", "y", "z"]:
                E_abc[0][d + "hi"] = self.E[-1, :, :, d]
                H_abc[0][d + "hi"] = self.H[-1, :, :, d]

        if self.bc_high[1].lower() == "abc":
            E_abc[1] = {}
            H_abc[1] = {}
            for d in ["x", "y", "z"]:
                E_abc[1][d + "hi"] = self.E[:, -1, :, d]
                H_abc[1][d + "hi"] = self.H[:, -1, :, d]

        if self.bc_high[2].lower() == "abc":
            E_abc[2] = {}
            H_abc[2] = {}
            for d in ["x", "y", "z"]:
                E_abc[2][d + "hi"] = self.E[:, :, -1, d]
                H_abc[2][d + "hi"] = self.H[:, :, -1, d]

        return E_abc, H_abc

    def update_abc(self, E_abc, H_abc):
        """
        Apply the Absorbing Boundary Condition (ABC) using previously saved
        snapshots.

        Parameters
        ----------
        E_abc, H_abc : dict
            Dictionaries produced by ``get_abc`` that contain boundary-layer
            field arrays. Each dictionary maps face indices to arrays used to
            overwrite the exterior cell values after a timestep.
        """

        if self.bc_low[0].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[0, :, :, d] = E_abc[0][d + "lo"]
                self.H[0, :, :, d] = H_abc[0][d + "lo"]

        if self.bc_low[1].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[:, 0, :, d] = E_abc[1][d + "lo"]
                self.H[:, 0, :, d] = H_abc[1][d + "lo"]

        if self.bc_low[2].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[:, :, 0, d] = E_abc[2][d + "lo"]
                self.H[:, :, 0, d] = H_abc[2][d + "lo"]

        if self.bc_high[0].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[-1, :, :, d] = E_abc[0][d + "hi"]
                self.H[-1, :, :, d] = H_abc[0][d + "hi"]

        if self.bc_high[1].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[:, -1, :, d] = E_abc[1][d + "hi"]
                self.H[:, -1, :, d] = H_abc[1][d + "hi"]

        if self.bc_high[2].lower() == "abc":
            for d in ["x", "y", "z"]:
                self.E[:, :, -1, d] = E_abc[2][d + "hi"]
                self.H[:, :, -1, d] = H_abc[2][d + "hi"]
