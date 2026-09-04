# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np
from scipy.constants import mu_0
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import c
from scipy.sparse import diags

from .field import Field


class BCsMixin:
    def _apply_bc_to_C(self):
        """
        Apply boundary conditions by modifying curl and metric matrices.

        Adjusts rows/columns of the curl operator ``C`` and the metric-diagonal
        matrices (``tDs``, ``itDa``) according to the low/high boundary
        condition lists ``bc_low`` and ``bc_high``. Handles periodic, PEC/PMC,
        ABC and PML/CPML options and also configures MPI-internal faces when the
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
            True for x in self.bc_low if x.lower() in ("electric", "pec", "pml", "cpml")
        ) or any(True for x in self.bc_high if x.lower() in ("electric", "pec", "pml", "cpml")):
            if self.bc_low[0].lower() in ("electric", "pec", "pml", "cpml"):
                xlo = 0
            if self.bc_low[1].lower() in ("electric", "pec", "pml", "cpml"):
                ylo = 0
            if self.bc_low[2].lower() in ("electric", "pec", "pml", "cpml"):
                zlo = 0
            if self.bc_high[0].lower() in ("electric", "pec", "pml", "cpml"):
                xhi = 0
            if self.bc_high[1].lower() in ("electric", "pec", "pml", "cpml"):
                yhi = 0
            if self.bc_high[2].lower() in ("electric", "pec", "pml", "cpml"):
                zhi = 0

            # Assemble matrix
            self.BC = Field(self.Nx, self.Ny, self.Nz, dtype=np.int8, use_ones=True)

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
            self.Dbc_x = diags(
                self.BC.field_x,
                shape=(self.N, self.N),
                dtype=np.int8,
            )
            self.Dbc_y = diags(
                self.BC.field_y,
                shape=(self.N, self.N),
                dtype=np.int8,
            )
            self.Dbc_z = diags(
                self.BC.field_z,
                shape=(self.N, self.N),
                dtype=np.int8,
            )

            # Update C (columns)
            self.C = self.C * self.Dbc

        # Dirichlet PMC: tangential H field = 0 at boundary
        if any(True for x in self.bc_low if x.lower() in ("magnetic", "pmc")) or any(
            True for x in self.bc_high if x.lower() in ("magnetic", "pmc")
        ):
            if self.bc_low[0].lower() == "magnetic" or self.bc_low[0] == "pmc":
                xlo = 0
            if self.bc_low[1].lower() == "magnetic" or self.bc_low[1] == "pmc":
                ylo = 0
            if self.bc_low[2].lower() == "magnetic" or self.bc_low[2] == "pmc":
                zlo = 0
            if self.bc_high[0].lower() == "magnetic" or self.bc_high[0] == "pmc":
                xhi = 0
            if self.bc_high[1].lower() == "magnetic" or self.bc_high[1] == "pmc":
                yhi = 0
            if self.bc_high[2].lower() == "magnetic" or self.bc_high[2] == "pmc":
                zhi = 0

            # Assemble matrix
            self.BC = Field(self.Nx, self.Ny, self.Nz, dtype=np.int8, use_ones=True)

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
            self.Dbc_x = diags(
                self.BC.field_x,
                shape=(self.N, self.N),
                dtype=np.int8,
            )
            self.Dbc_y = diags(
                self.BC.field_y,
                shape=(self.N, self.N),
                dtype=np.int8,
            )
            self.Dbc_z = diags(
                self.BC.field_z,
                shape=(self.N, self.N),
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
            
        if any(True for x in self.bc_low if x.lower() == "cpml") or any(
            True for x in self.bc_high if x.lower() == "cpml"
        ):
            self.activate_cpml = True

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
        self.pml_hi = 1.0
        self.pml_func = np.geomspace
        self.pml_eps_r = 1.0

        # Fill
        if self.bc_low[0].lower() == "pml":
            # sx[0:self.n_pml] = eps_0/(2*self.dt)*((self.x[self.n_pml] - self.x[:self.n_pml])/(self.n_pml*self.dx))**pml_exp
            sx[0 : self.n_pml] = self.pml_func(self.pml_hi, self.pml_lo, self.n_pml)
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the yz plane
                ieps_0_pml = 1/eps_0 #self.ieps[self.n_pml + 1, self.Ny // 2, self.Nz // 2, d]
                sigma_0_pml = 0. #self.sigma[self.n_pml + 1, self.Ny // 2, self.Nz // 2, d]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for i in range(self.n_pml):
                    self.ieps[i, :, :, d] = ieps_0_pml
                    self.sigma[i, :, :, d] = sigma_0_pml + sigma_mult_pml * sx[i]
                    # if sx[i] > 0 : self.ieps[i, :, :, d] = 1/(eps_0+sx[i]*(2*self.dt))

        if self.bc_low[1].lower() == "pml":
            # sy[0:self.n_pml] = 1/(2*self.dt)*((self.y[self.n_pml] - self.y[:self.n_pml])/(self.n_pml*self.dy))**pml_exp
            sy[0 : self.n_pml] = self.pml_func(self.pml_hi, self.pml_lo, self.n_pml)
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the xz plane
                ieps_0_pml = 1/eps_0 #self.ieps[self.Nx // 2, self.n_pml + 1, self.Nz // 2, d]
                sigma_0_pml = 0. #self.sigma[self.Nx // 2, self.n_pml + 1, self.Nz // 2, d]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for j in range(self.n_pml):
                    self.ieps[:, j, :, d] = ieps_0_pml
                    self.sigma[:, j, :, d] = sigma_0_pml + sigma_mult_pml * sy[j]
                    # if sy[j] > 0 : self.ieps[:, j, :, d] = 1/(eps_0+sy[j]*(2*self.dt))

        if self.bc_low[2].lower() == "pml":
            # sz[0:self.n_pml] = eps_0/(2*self.dt)*((self.z[self.n_pml] - self.z[:self.n_pml])/(self.n_pml*self.dz))**pml_exp
            sz[0 : self.n_pml] = self.pml_func(self.pml_hi, self.pml_lo, self.n_pml)
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the xy plane
                ieps_0_pml = 1/eps_0 #self.ieps[self.Nx // 2, self.Ny // 2, self.n_pml + 1, d]
                sigma_0_pml = 0. #self.sigma[self.Nx // 2, self.Ny // 2, self.n_pml + 1, d]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for k in range(self.n_pml):
                    self.ieps[:, :, k, d] = ieps_0_pml
                    self.sigma[:, :, k, d] = sigma_0_pml + sigma_mult_pml * sz[k]
                    # if sz[k] > 0. : self.ieps[:, :, k, d] = 1/(np.mean(sz[:self.n_pml])*eps_0)

        if self.bc_high[0].lower() == "pml":
            # sx[-self.n_pml:] = 1/(2*self.dt)*((self.x[-self.n_pml:] - self.x[-self.n_pml])/(self.n_pml*self.dx))**pml_exp
            sx[-self.n_pml :] = self.pml_func(self.pml_lo, self.pml_hi, self.n_pml)
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the yz plane
                ieps_0_pml = 1/eps_0 #self.ieps[-(self.n_pml + 1), self.Ny // 2, self.Nz // 2, d]
                sigma_0_pml = 0. #self.sigma[ -(self.n_pml + 1), self.Ny // 2, self.Nz // 2, d]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for i in range(self.n_pml):
                    i += 1
                    self.ieps[-i, :, :, d] = ieps_0_pml
                    self.sigma[-i, :, :, d] = sigma_0_pml + sigma_mult_pml * sx[-i]
                    # if sx[-i] > 0 : self.ieps[-i, :, :, d] = 1/(eps_0+sx[-i]*(2*self.dt))

        if self.bc_high[1].lower() == "pml":
            # sy[-self.n_pml:] = 1/(2*self.dt)*((self.y[-self.n_pml:] - self.y[-self.n_pml])/(self.n_pml*self.dy))**pml_exp
            sy[-self.n_pml :] = self.pml_func(self.pml_lo, self.pml_hi, self.n_pml)
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the xz plane
                ieps_0_pml = 1/eps_0 #self.ieps[self.Nx // 2, -(self.n_pml + 1), self.Nz // 2, d]
                sigma_0_pml = 0. #self.sigma[self.Nx // 2, -(self.n_pml + 1), self.Nz // 2, d]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for j in range(self.n_pml):
                    j += 1
                    self.ieps[:, -j, :, d] = ieps_0_pml
                    self.sigma[:, -j, :, d] = sigma_0_pml + sigma_mult_pml * sy[-j]
                    # if sy[-j] > 0 : self.ieps[:, -j, :, d] = 1/(eps_0+sy[-j]*(2*self.dt))

        if self.bc_high[2].lower() == "pml":
            # sz[-self.n_pml:] = eps_0/(2*self.dt)*((self.z[-self.n_pml:] - self.z[-self.n_pml])/(self.n_pml*self.dz))**pml_exp
            sz[-self.n_pml :] = self.pml_func(self.pml_lo, self.pml_hi, self.n_pml)
            for d in ["x", "y", "z"]:
                # Get the properties from the layer before the PML
                # Take the values at the center of the xy plane
                ieps_0_pml = 1/eps_0 #self.ieps[self.Nx // 2, self.Ny // 2, -(self.n_pml + 1), d]
                sigma_0_pml = 0. #self.sigma[self.Nx // 2, self.Ny // 2, -(self.n_pml + 1), d]
                sigma_mult_pml = (
                    1 if sigma_0_pml < 1 else sigma_0_pml
                )  # avoid null sigma in PML for relaxation time computation
                for k in range(self.n_pml):
                    k += 1
                    self.ieps[:, :, -k, d] = ieps_0_pml
                    self.sigma[:, :, -k, d] = sigma_0_pml + sigma_mult_pml * sz[-k]
                    # self.ieps[:, :, -k, d] = 1/(np.mean(sz[-self.n_pml:])*eps_0)
                    
    def _initialize_CPML(self):
        """
        Compute the CPML parameters for the low and high boundaries in each
        direction. The CPML parameters are stored in the ``sigmaPml``, ``kappa``, and ``alpha`` fields, which are used to calculate the b and c parameters of the CPML for
        the electric and magnetic fields. The CPML parameters are computed based on the distance into the CPML region and the specified CPML profile functions. The target reflection coefficient is set to 1e-8
        """

        # Initialize
        eta_0 = mu_0 * c  # Characteristic impedance of free space
        sx, sy, sz = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        ax, ay, az = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        tsx, tsy, tsz = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        tax, tay, taz = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        R0 = 1e-8  # Target reflection coefficient for PML design
        self.kappa = Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)
        self.tkappa = Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)
        self.alpha = Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        self.talpha = Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        self.sigmaPml = Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        self.tsigmaPml = Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)

        # Fill
        if self.bc_low[0].lower() == "cpml":
            interface = self.grid.x[self.n_pml]
            L = interface - self.grid.x[0]
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(self.n_pml):
                # Compute distance into PML for primal and dual grid points, then scale to [0,1] for profile functions
                dist = interface - self.grid.x[i]   # distance into PML
                tdist = interface - (self.grid.x[i] + self.dx[i]/2)   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sx[i] = (dist / L)**self.pml_exp
                ax[i] = (dist / L)
                tax[i] = (tdist / L)
                tsx[i] = (tdist / L)**self.pml_exp
                # Compute the PML parameters for this layer
                self.sigmaPml[i, :, :, 'x'] = sigma_max * sx[i]
                self.kappa[i, :, :, 'x'] = 1 + (self.kappa_max - 1) * sx[i]
                self.alpha[i, :, :, 'x'] = self.alpha_max * (1 - ax[i])
                self.tsigmaPml[i, :, :, 'x'] = sigma_max * tsx[i]
                self.tkappa[i, :, :, 'x'] = 1 + (self.kappa_max - 1) * tsx[i]
                self.talpha[i, :, :, 'x'] = self.alpha_max * (1 - tax[i])

        if self.bc_low[1].lower() == "cpml":
            interface = self.grid.y[self.n_pml]
            L = interface - self.grid.y[0]
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(self.n_pml):
                # Compute distance into PML for primal and dual grid points, then scale to [0,1] for profile functions
                dist = interface - self.grid.y[i]   # distance into PML
                tdist = interface - (self.grid.y[i] + self.dy[i]/2)   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sy[i] = (dist / L)**self.pml_exp
                tsy[i] = (tdist / L)**self.pml_exp
                ay[i] = (dist / L)
                tay[i] = (tdist / L)
                # Compute the PML parameters for this layer
                self.sigmaPml[:, i, :, 'y'] = sigma_max * sy[i]
                self.kappa[:, i, :, 'y'] = 1 + (self.kappa_max - 1) * sy[i]
                self.alpha[:, i, :, 'y'] = self.alpha_max * (1 - ay[i])
                self.tsigmaPml[:, i, :, 'y'] = sigma_max * tsy[i]
                self.tkappa[:, i, :, 'y'] = 1 + (self.kappa_max - 1) * tsy[i]
                self.talpha[:, i, :, 'y'] = self.alpha_max * (1 - tay[i])

        if self.bc_low[2].lower() == "cpml":
            interface = self.grid.z[self.n_pml]
            L = interface - self.grid.z[0]
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(self.n_pml):
                # Compute distance into PML for primal and dual grid points, then scale to [0,1] for profile functions
                dist = interface - self.grid.z[i]   # distance into PML
                tdist = interface - (self.grid.z[i] + self.dz[i]/2)   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sz[i] = (dist / L)**self.pml_exp
                tsz[i] = (tdist / L)**self.pml_exp
                az[i] = (dist / L)
                taz[i] = (tdist / L)
                # Compute the PML parameters for this layer
                self.sigmaPml[:, :, i, 'z'] = sigma_max * sz[i]
                self.kappa[:, :, i, 'z'] = 1 + (self.kappa_max - 1) * sz[i]
                self.alpha[:, :, i, 'z'] = self.alpha_max * (1 - az[i])
                self.tsigmaPml[:, :, i, 'z'] = sigma_max * tsz[i]
                self.tkappa[:, :, i, 'z'] = 1 + (self.kappa_max - 1) * tsz[i]
                self.talpha[:, :, i, 'z'] = self.alpha_max * (1 - taz[i])

        if self.bc_high[0].lower() == "cpml":
            interface = self.grid.x[-1-self.n_pml]
            L = self.grid.x[-1] - interface
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(-self.n_pml, 0):
                # Compute distance into PML for primal and dual grid points, then scale to [0,1] for profile functions
                dist = self.grid.x[i] - interface   # distance into PML
                tdist = (self.grid.x[i] + self.dx[i]/2) - interface   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sx[i] = (dist / L)**self.pml_exp
                tsx[i] = (tdist / L)**self.pml_exp
                ax[i] = (dist / L)
                tax[i] = (tdist / L)
                # Compute the PML parameters for this layer
                self.sigmaPml[i, :, :, 'x'] = sigma_max * sx[i]
                self.kappa[i, :, :, 'x'] = 1 + (self.kappa_max - 1) * sx[i]
                self.alpha[i, :, :, 'x'] = self.alpha_max * (1 - ax[i])
                self.tsigmaPml[i, :, :, 'x'] = sigma_max * tsx[i]
                self.tkappa[i, :, :, 'x'] = 1 + (self.kappa_max - 1) * tsx[i]
                self.talpha[i, :, :, 'x'] = self.alpha_max * (1 - tax[i])

        if self.bc_high[1].lower() == "cpml":
            interface = self.grid.y[-1-self.n_pml]
            L = self.grid.y[-1] - interface
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(-self.n_pml, 0):
                # Compute distance into PML for primal and dual grid points, then scale to [0,1] for profile functions
                dist = self.grid.y[i] - interface   # distance into PML
                tdist = (self.grid.y[i] + self.dy[i]/2) - interface   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sy[i] = (dist / L)**self.pml_exp
                tsy[i] = (tdist / L)**self.pml_exp
                ay[i] = (dist / L)
                tay[i] = (tdist / L)
                # Compute the PML parameters for this layer
                self.sigmaPml[:, i, :, 'y'] = sigma_max * sy[i]
                self.kappa[:, i, :, 'y'] = 1 + (self.kappa_max - 1) * sy[i]
                self.alpha[:, i, :, 'y'] = self.alpha_max * (1 - ay[i])
                self.tsigmaPml[:, i, :, 'y'] = sigma_max * tsy[i]
                self.tkappa[:, i, :, 'y'] = 1 + (self.kappa_max - 1) * tsy[i]
                self.talpha[:, i, :, 'y'] = self.alpha_max * (1 - tay[i])

        if self.bc_high[2].lower() == "cpml":
            interface = self.grid.z[-1-self.n_pml]
            L = self.grid.z[-1] - interface
            sigma_max = -self.sigma_factor * (self.pml_exp + 1) * np.log(R0) / (2 * L * eta_0)
            for i in range(-self.n_pml, 0):
                # Compute distance into PML for primal and dual grid points, then scale to [0,1] for profile functions
                dist = self.grid.z[i] - interface   # distance into PML
                tdist = self.grid.z[i] + self.dz[i]/2 - interface   # distance into PML for half-grid points
                tdist = max(0.0, min(tdist, L))
                sz[i] = (dist / L)**self.pml_exp
                tsz[i] = (tdist / L)**self.pml_exp
                az[i] = (dist / L)
                taz[i] = (tdist / L)
                # Compute the PML parameters for this layer
                self.sigmaPml[:, :, i, 'z'] = sigma_max * sz[i]
                self.kappa[:, :, i, 'z'] = 1 + (self.kappa_max - 1) * sz[i]
                self.alpha[:, :, i, 'z'] = self.alpha_max * (1 - az[i])
                self.tsigmaPml[:, :, i, 'z'] = sigma_max * tsz[i]
                self.tkappa[:, :, i, 'z'] = 1 + (self.kappa_max - 1) * tsz[i]
                self.talpha[:, :, i, 'z'] = self.alpha_max * (1 - taz[i])

    def _initialize_CPML_matrices(self):
        N = self.N
        self.tLx = diags(
            self.tL.field_x, shape=(N, N), dtype=self.dtype
        )
        self.tLy = diags(
            self.tL.field_y, shape=(N, N), dtype=self.dtype
        )
        self.tLz = diags(
            self.tL.field_z, shape=(N, N), dtype=self.dtype
        )
        self.iAx = diags(
            self.iA.field_x, shape=(N, N), dtype=self.dtype
        )
        self.iAy = diags(
            self.iA.field_y, shape=(N, N), dtype=self.dtype
        )
        self.iAz = diags(
            self.iA.field_z, shape=(N, N), dtype=self.dtype
        )
        self.Lx = diags(
            self.L.field_x, shape=(N, N), dtype=self.dtype
        )
        self.Ly = diags(
            self.L.field_y, shape=(N, N), dtype=self.dtype
        )
        self.Lz = diags(
            self.L.field_z, shape=(N, N), dtype=self.dtype
        )
        self.itAx = diags(
            self.itA.field_x, shape=(N, N), dtype=self.dtype
        )
        self.itAy = diags(
            self.itA.field_y, shape=(N, N), dtype=self.dtype
        )
        self.itAz = diags(
            self.itA.field_z, shape=(N, N), dtype=self.dtype
        )
        self.ikapx = diags(
            1.0 / self.kappa.field_x, shape=(N, N), dtype=self.dtype
        )
        self.ikapy = diags(
            1.0 / self.kappa.field_y, shape=(N, N), dtype=self.dtype
        )
        self.ikapz = diags(
            1.0 / self.kappa.field_z, shape=(N, N), dtype=self.dtype
        )
        self.itkapx = diags(
            1.0 / self.tkappa.field_x, shape=(N, N), dtype=self.dtype
        )
        self.itkapy = diags(
            1.0 / self.tkappa.field_y, shape=(N, N), dtype=self.dtype
        )
        self.itkapz = diags(
            1.0 / self.tkappa.field_z, shape=(N, N), dtype=self.dtype
        )
        # Compute combined derivative operators for CPML update equations
        self.dxy = self.iAx * self.itkapy * self.Py * self.Dbc_z * self.Lz
        self.dxz = self.iAx * self.itkapz * self.Pz * self.Dbc_y * self.Ly
        self.dyz = self.iAy * self.itkapz * self.Pz * self.Dbc_x * self.Lx
        self.dyx = self.iAy * self.itkapx * self.Px * self.Dbc_z * self.Lz
        self.dzx = self.iAz * self.itkapx * self.Px * self.Dbc_y * self.Ly
        self.dzy = self.iAz * self.itkapy * self.Py * self.Dbc_x * self.Lx

        self.dtxy = self.itAx * self.ikapy * self.Dbc_x * -self.Py.transpose() * self.tLz
        self.dtxz = self.itAx * self.ikapz * self.Dbc_x * -self.Pz.transpose() * self.tLy
        self.dtyz = self.itAy * self.ikapz * self.Dbc_y * -self.Pz.transpose() * self.tLx
        self.dtyx = self.itAy * self.ikapx * self.Dbc_y * -self.Px.transpose() * self.tLz
        self.dtzx = self.itAz * self.ikapx * self.Dbc_z * -self.Px.transpose() * self.tLy
        self.dtzy = self.itAz * self.ikapy * self.Dbc_z * -self.Py.transpose() * self.tLx

        if self.source_type.lower() == "tfsf":
            self.tf_dxz = self.iAx * self.itkapz * self.Ly
            self.tf_dyz = self.iAy * self.itkapz * self.Lx
            self.tf_dtxz = self.itAx * self.ikapz * self.tLy
            self.tf_dtyz = self.itAy * self.ikapz * self.tLx

        del self.iAx, self.iAy, self.iAz, self.itAx, self.itAy, self.itAz, self.Lx, self.Ly, self.Lz, self.tLx, self.tLy, self.tLz, self.ikapx, self.ikapy, self.ikapz, self.itkapx, self.itkapy, self.itkapz

        self.pml_b_H = (
        Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        )
        self.pml_c_H = (
        Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        )
        self.pml_b_E = (
            Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        )
        self.pml_c_E = (
            Field(self.Nx, self.Ny, self.Nz, dtype=self.dtype)
        )

        # Convolution Parameter computation, only valid if sigma is zero in physical domain
        self.pml_b_E.fromarray(np.exp(
            -(self.sigmaPml.toarray() / (self.kappa.toarray()) + self.alpha.toarray()) * self.dt / eps_0))
        denom = self.sigmaPml.toarray() + self.kappa.toarray() * self.alpha.toarray()
        ratio = np.divide(self.sigmaPml.toarray(), denom, out=np.zeros_like(self.sigmaPml.toarray()), where=denom != 0)
        self.pml_c_E.fromarray(ratio * (self.pml_b_E.toarray() - 1.0))

        self.pml_b_H.fromarray(np.exp(
            -(self.tsigmaPml.toarray() / (self.tkappa.toarray()) + self.talpha.toarray()) * self.dt / eps_0))
        denom = self.tsigmaPml.toarray() + self.tkappa.toarray() * self.talpha.toarray()
        ratio = np.divide(self.tsigmaPml.toarray(), denom, out=np.zeros_like(self.tsigmaPml.toarray()), where=denom != 0)
        self.pml_c_H.fromarray(ratio * (self.pml_b_H.toarray() - 1.0))

        del self.kappa, self.alpha, self.sigmaPml, self.tkappa, self.talpha, self.tsigmaPml

        def _get_f_order_indices(x_range, y_range, z_range, Nx, Ny, Nz):
            """Helper to generate 1D F-order indices for given 3D slice ranges."""
            X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            # Apply the Fortran-order memory stride formula
            indices_3d = X + (Y * Nx) + (Z * Nx * Ny)
            return indices_3d.ravel(order='F')
        
        # Initialize CPML auxiliary fields and flattened PML parameters for low and high boundaries
        if self.bc_low[0].lower() == "cpml":
            self.psiHa_z_low = np.zeros((self.n_pml * self.Ny * self.Nz), dtype=self.dtype)
            self.psiHb_y_low = np.zeros((self.n_pml * self.Ny * self.Nz), dtype=self.dtype)
            self.psiEa_z_low = np.zeros((self.n_pml * self.Ny * self.Nz), dtype=self.dtype)
            self.psiEb_y_low = np.zeros((self.n_pml * self.Ny * self.Nz), dtype=self.dtype)
            self.pml_b_E_x_low = (self.pml_b_E[:self.n_pml, :, :, 'x'].copy()).flatten(order='F')
            self.pml_b_H_x_low = (self.pml_b_H[:self.n_pml, :, :, 'x'].copy()).flatten(order='F')
            self.pml_c_E_x_low = (self.pml_c_E[:self.n_pml, :, :, 'x'].copy()).flatten(order='F')
            self.pml_c_H_x_low = (self.pml_c_H[:self.n_pml, :, :, 'x'].copy()).flatten(order='F')
            self.idx_x_low = _get_f_order_indices(np.arange(self.n_pml), np.arange(self.Ny), np.arange(self.Nz), self.Nx, self.Ny, self.Nz)
        if self.bc_low[1].lower() == "cpml":
            self.psiHa_x_low = np.zeros((self.Nx *self.n_pml * self.Nz), dtype=self.dtype)
            self.psiHb_z_low = np.zeros((self.Nx * self.n_pml * self.Nz), dtype=self.dtype)
            self.psiEa_x_low = np.zeros((self.Nx * self.n_pml * self.Nz), dtype=self.dtype)
            self.psiEb_z_low = np.zeros((self.Nx * self.n_pml * self.Nz), dtype=self.dtype)
            self.pml_c_E_y_low = (self.pml_c_E[:, :self.n_pml, :, 'y'].copy()).flatten(order='F')
            self.pml_c_H_y_low = (self.pml_c_H[:, :self.n_pml, :, 'y'].copy()).flatten(order='F')
            self.pml_b_E_y_low = (self.pml_b_E[:, :self.n_pml, :, 'y'].copy()).flatten(order='F')
            self.pml_b_H_y_low = (self.pml_b_H[:, :self.n_pml, :, 'y'].copy()).flatten(order='F')
            self.idx_y_low = _get_f_order_indices(np.arange(self.Nx), np.arange(self.n_pml), np.arange(self.Nz), self.Nx, self.Ny, self.Nz)
        if self.bc_low[2].lower() == "cpml":
            self.psiHa_y_low = np.zeros((self.Nx * self.Ny * self.n_pml), dtype=self.dtype)
            self.psiHb_x_low = np.zeros((self.Nx * self.Ny * self.n_pml), dtype=self.dtype)
            self.psiEa_y_low = np.zeros((self.Nx * self.Ny * self.n_pml), dtype=self.dtype)
            self.psiEb_x_low = np.zeros((self.Nx * self.Ny * self.n_pml), dtype=self.dtype)
            self.pml_c_E_z_low = (self.pml_c_E[:, :, :self.n_pml, 'z'].copy()).flatten(order='F')
            self.pml_c_H_z_low = (self.pml_c_H[:, :, :self.n_pml, 'z'].copy()).flatten(order='F')
            self.pml_b_E_z_low = (self.pml_b_E[:, :, :self.n_pml, 'z'].copy()).flatten(order='F')
            self.pml_b_H_z_low = (self.pml_b_H[:, :, :self.n_pml, 'z'].copy()).flatten(order='F')
            self.idx_z_low = _get_f_order_indices(np.arange(self.Nx), np.arange(self.Ny), np.arange(self.n_pml), self.Nx, self.Ny, self.Nz)
        if self.bc_high[0].lower() == "cpml":
            self.psiHa_z_high = np.zeros((self.n_pml * self.Ny * self.Nz), dtype=self.dtype)
            self.psiHb_y_high = np.zeros((self.n_pml * self.Ny * self.Nz), dtype=self.dtype)
            self.psiEa_z_high = np.zeros((self.n_pml * self.Ny * self.Nz), dtype=self.dtype)
            self.psiEb_y_high = np.zeros((self.n_pml * self.Ny * self.Nz), dtype=self.dtype)
            self.pml_c_E_x_high = (self.pml_c_E[-self.n_pml:, :, :, 'x'].copy()).flatten(order='F')
            self.pml_c_H_x_high = (self.pml_c_H[-self.n_pml:, :, :, 'x'].copy()).flatten(order='F')
            self.pml_b_E_x_high = (self.pml_b_E[-self.n_pml:, :, :, 'x'].copy()).flatten(order='F')
            self.pml_b_H_x_high = (self.pml_b_H[-self.n_pml:, :, :, 'x'].copy()).flatten(order='F')
            self.idx_x_high = _get_f_order_indices(np.arange(self.Nx - self.n_pml, self.Nx), np.arange(self.Ny), np.arange(self.Nz), self.Nx, self.Ny, self.Nz)
        if self.bc_high[1].lower() == "cpml":
            self.psiHa_x_high = np.zeros((self.Nx * self.n_pml * self.Nz), dtype=self.dtype)
            self.psiHb_z_high = np.zeros((self.Nx * self.n_pml * self.Nz), dtype=self.dtype)
            self.psiEa_x_high = np.zeros((self.Nx * self.n_pml * self.Nz), dtype=self.dtype)
            self.psiEb_z_high = np.zeros((self.Nx * self.n_pml * self.Nz), dtype=self.dtype)
            self.pml_c_E_y_high = (self.pml_c_E[:, -self.n_pml:, :, 'y'].copy()).flatten(order='F')
            self.pml_c_H_y_high = (self.pml_c_H[:, -self.n_pml:, :, 'y'].copy()).flatten(order='F')
            self.pml_b_E_y_high = (self.pml_b_E[:, -self.n_pml:, :, 'y'].copy()).flatten(order='F')
            self.pml_b_H_y_high = (self.pml_b_H[:, -self.n_pml:, :, 'y'].copy()).flatten(order='F')
            self.idx_y_high = _get_f_order_indices(np.arange(self.Nx), np.arange(self.Ny - self.n_pml, self.Ny), np.arange(self.Nz), self.Nx, self.Ny, self.Nz)
        if self.bc_high[2].lower() == "cpml":
            self.psiHa_y_high = np.zeros((self.Nx * self.Ny * self.n_pml), dtype=self.dtype)
            self.psiHb_x_high = np.zeros((self.Nx * self.Ny * self.n_pml), dtype=self.dtype)
            self.psiEa_y_high = np.zeros((self.Nx * self.Ny * self.n_pml), dtype=self.dtype)
            self.psiEb_x_high = np.zeros((self.Nx * self.Ny * self.n_pml), dtype=self.dtype)
            self.pml_c_E_z_high = (self.pml_c_E[:, :, -self.n_pml:, 'z'].copy()).flatten(order='F')
            self.pml_c_H_z_high = (self.pml_c_H[:, :, -self.n_pml:, 'z'].copy()).flatten(order='F')
            self.pml_b_E_z_high = (self.pml_b_E[:, :, -self.n_pml:, 'z'].copy()).flatten(order='F')
            self.pml_b_H_z_high = (self.pml_b_H[:, :, -self.n_pml:, 'z'].copy()).flatten(order='F')
            self.idx_z_high = _get_f_order_indices(np.arange(self.Nx), np.arange(self.Ny), np.arange(self.Nz - self.n_pml, self.Nz), self.Nx, self.Ny, self.Nz)

        del self.pml_b_E, self.pml_c_E, self.pml_b_H, self.pml_c_H

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
