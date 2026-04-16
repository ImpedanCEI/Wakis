import numpy as np
from numba import jit
from pmlBlock3D import PmlBlock3D
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0
from solver2D import EMSolver2D


def eq(a, b, tol=1e-8):
    return abs(a - b) < tol


def neq(a, b, tol=1e-8):
    return not eq(a, b, tol)


class EMSolver3D:
    def __init__(
        self,
        grid,
        sol_type,
        cfln=0.5,
        bc_low=["Dirichlet", "Dirichlet", "Dirichlet"],
        bc_high=["Dirichlet", "Dirichlet", "Dirichlet"],
        i_s=0,
        j_s=0,
        k_s=0,
        N_pml_low=None,
        N_pml_high=None,
    ):
        self.grid = grid
        self.type = type
        self.cfln = cfln

        self.dt = cfln / (
            c_light
            * np.sqrt(1 / self.grid.dx**2 + 1 / self.grid.dy**2 + 1 / self.grid.dz**2)
        )
        self.dx = self.grid.dx
        self.dy = self.grid.dy
        self.dz = self.grid.dz
        self.Nx = self.grid.nx
        self.Ny = self.grid.ny
        self.Nz = self.grid.nz
        self.sol_type = sol_type

        self.N_pml_low = np.zeros(3, dtype=int)
        self.N_pml_high = np.zeros(3, dtype=int)
        self.bc_low = bc_low
        self.bc_high = bc_high

        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz

        self.sigma_x = np.zeros((Nx + 1, Ny + 1, Nz + 1))
        self.sigma_y = np.zeros((Nx + 1, Ny + 1, Nz + 1))
        self.sigma_z = np.zeros((Nx + 1, Ny + 1, Nz + 1))
        self.sigma_star_x = np.zeros((Nx, Ny + 1, Nz + 1))
        self.sigma_star_y = np.zeros((Nx + 1, Ny, Nz + 1))
        self.sigma_star_z = np.zeros((Nx + 1, Ny + 1, Nz))

        if bc_low[0] == "pml":
            self.N_pml_low[0] = 10 if N_pml_low is None else N_pml_low[0]
        if bc_low[1] == "pml":
            self.N_pml_low[1] = 10 if N_pml_low is None else N_pml_low[1]
        if bc_low[2] == "pml":
            self.N_pml_low[2] = 10 if N_pml_low is None else N_pml_low[2]
        if bc_high[0] == "pml":
            self.N_pml_high[0] = 10 if N_pml_high is None else N_pml_high[0]
        if bc_high[1] == "pml":
            self.N_pml_high[1] = 10 if N_pml_high is None else N_pml_high[1]
        if bc_high[2] == "pml":
            self.N_pml_high[2] = 10 if N_pml_high is None else N_pml_high[2]

        self.blocks = []

        self.blocks_mat = np.full((3, 3, 3), None)
        self.blocks_mat[1, 1, 1] = self

        self.connect_pmls()

        self.alpha_pml = 3
        self.R0_pml = 0.001

        self.assemble_conductivities_pmls()
        self.assemble_coeffs_pmls()

        self.Ex = np.zeros((self.Nx, self.Ny + 1, self.Nz + 1))
        self.Ey = np.zeros((self.Nx + 1, self.Ny, self.Nz + 1))
        self.Ez = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz))
        self.Hx = np.zeros((self.Nx + 1, self.Ny, self.Nz))
        self.Hy = np.zeros((self.Nx, self.Ny + 1, self.Nz))
        self.Hz = np.zeros((self.Nx, self.Ny, self.Nz + 1))
        self.Jx = np.zeros((self.Nx, self.Ny + 1, self.Nz + 1))
        self.Jy = np.zeros((self.Nx + 1, self.Ny, self.Nz + 1))
        self.Jz = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz))
        self.rho_xy = np.zeros((self.Nx, self.Ny, self.Nz + 1))
        self.rho_yz = np.zeros((self.Nx + 1, self.Ny, self.Nz))
        self.rho_zx = np.zeros((self.Nx, self.Ny + 1, self.Nz))

        if (sol_type != "FDTD") and (sol_type != "DM") and (sol_type != "ECT"):
            raise ValueError(
                "sol_type must be:\n"
                + "\t'FDTD' for standard staircased FDTD\n"
                + "\t'DM' for Dey-Mittra conformal FDTD\n"
                + "\t'ECT' for Enlarged Cell Technique conformal FDTD"
            )

        if sol_type == "DM" or sol_type == "ECT":
            self.Vxy = np.zeros((self.Nx, self.Ny, self.Nz + 1))
            self.Vyz = np.zeros((self.Nx + 1, self.Ny, self.Nz))
            self.Vzx = np.zeros((self.Nx, self.Ny + 1, self.Nz))
        if sol_type == "ECT":
            self.Vxy_enl = np.zeros((self.Nx, self.Ny, self.Nz + 1))
            self.Vyz_enl = np.zeros((self.Nx + 1, self.Ny, self.Nz))
            self.Vzx_enl = np.zeros((self.Nx, self.Ny + 1, self.Nz))

        self.C1 = self.dt / (self.dx * mu_0)
        self.C2 = self.dt / (self.dy * mu_0)
        self.C7 = self.dt / (self.dz * mu_0)
        self.C4 = self.dt / (self.dy * eps_0)
        self.C5 = self.dt / (self.dx * eps_0)
        self.C8 = self.dt / (self.dz * eps_0)
        self.C3 = self.dt / eps_0
        self.C6 = self.dt / eps_0

        self.CN = self.dt / mu_0

        # indices for the source
        self.i_s = i_s
        self.j_s = j_s
        self.k_s = k_s

        self.time = 0

    def connect_pmls(self):
        bc_low = self.bc_low
        bc_high = self.bc_high
        if bc_low[0] == "pml":
            i_block = 0
            j_block = 1
            k_block = 1
            self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                self.N_pml_low[0],
                self.Ny,
                self.Nz,
                self.dt,
                self.dx,
                self.dy,
                self.dz,
                i_block,
                j_block,
                k_block,
            )
            self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_low[1] == "pml":
                i_block = 0
                j_block = 0
                k_block = 1
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.N_pml_low[0],
                    self.N_pml_low[1],
                    self.Nz,
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
                if bc_low[2] == "pml":
                    i_block = 0
                    j_block = 0
                    k_block = 0
                    self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                        self.N_pml_low[0],
                        self.N_pml_low[1],
                        self.N_pml_low[2],
                        self.dt,
                        self.dx,
                        self.dy,
                        self.dz,
                        i_block,
                        j_block,
                        k_block,
                    )
                    self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
                if bc_high[2] == "pml":
                    i_block = 0
                    j_block = 0
                    k_block = 2
                    self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                        self.N_pml_low[0],
                        self.N_pml_low[1],
                        self.N_pml_high[2],
                        self.dt,
                        self.dx,
                        self.dy,
                        self.dz,
                        i_block,
                        j_block,
                        k_block,
                    )
                    self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_high[1] == "pml":
                i_block = 0
                j_block = 2
                k_block = 1
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.N_pml_low[0],
                    self.N_pml_high[1],
                    self.Nz,
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
                if bc_low[2] == "pml":
                    i_block = 0
                    j_block = 2
                    k_block = 0
                    self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                        self.N_pml_low[0],
                        self.N_pml_high[1],
                        self.N_pml_low[2],
                        self.dt,
                        self.dx,
                        self.dy,
                        self.dz,
                        i_block,
                        j_block,
                        k_block,
                    )
                    self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
                if bc_high[2] == "pml":
                    i_block = 0
                    j_block = 2
                    k_block = 2
                    self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                        self.N_pml_low[0],
                        self.N_pml_high[1],
                        self.N_pml_high[2],
                        self.dt,
                        self.dx,
                        self.dy,
                        self.dz,
                        i_block,
                        j_block,
                        k_block,
                    )
                    self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_low[2] == "pml":
                i_block = 0
                j_block = 1
                k_block = 0
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.N_pml_low[0],
                    self.Ny,
                    self.N_pml_low[2],
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_high[2] == "pml":
                i_block = 0
                j_block = 1
                k_block = 2
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.N_pml_low[0],
                    self.Ny,
                    self.N_pml_high[2],
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])

        if bc_high[0] == "pml":
            i_block = 2
            j_block = 1
            k_block = 1
            self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                self.N_pml_high[0],
                self.Ny,
                self.Nz,
                self.dt,
                self.dx,
                self.dy,
                self.dz,
                i_block,
                j_block,
                k_block,
            )
            self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_low[1] == "pml":
                i_block = 2
                j_block = 0
                k_block = 1
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.N_pml_high[0],
                    self.N_pml_low[1],
                    self.Nz,
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
                if bc_low[2] == "pml":
                    i_block = 2
                    j_block = 0
                    k_block = 0
                    self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                        self.N_pml_high[0],
                        self.N_pml_low[1],
                        self.N_pml_low[2],
                        self.dt,
                        self.dx,
                        self.dy,
                        self.dz,
                        i_block,
                        j_block,
                        k_block,
                    )
                    self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
                if bc_high[2] == "pml":
                    i_block = 2
                    j_block = 0
                    k_block = 2
                    self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                        self.N_pml_high[0],
                        self.N_pml_low[1],
                        self.N_pml_high[2],
                        self.dt,
                        self.dx,
                        self.dy,
                        self.dz,
                        i_block,
                        j_block,
                        k_block,
                    )
                    self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_high[1] == "pml":
                i_block = 2
                j_block = 2
                k_block = 1
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.N_pml_high[0],
                    self.N_pml_high[1],
                    self.Nz,
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
                if bc_low[2] == "pml":
                    i_block = 2
                    j_block = 2
                    k_block = 0
                    self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                        self.N_pml_high[0],
                        self.N_pml_high[1],
                        self.N_pml_low[2],
                        self.dt,
                        self.dx,
                        self.dy,
                        self.dz,
                        i_block,
                        j_block,
                        k_block,
                    )
                    self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
                if bc_high[2] == "pml":
                    i_block = 2
                    j_block = 2
                    k_block = 2
                    self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                        self.N_pml_high[0],
                        self.N_pml_high[1],
                        self.N_pml_high[2],
                        self.dt,
                        self.dx,
                        self.dy,
                        self.dz,
                        i_block,
                        j_block,
                        k_block,
                    )
                    self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_low[2] == "pml":
                i_block = 2
                j_block = 1
                k_block = 0
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.N_pml_high[0],
                    self.Ny,
                    self.N_pml_low[2],
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_high[2] == "pml":
                i_block = 2
                j_block = 1
                k_block = 2
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.N_pml_high[0],
                    self.Ny,
                    self.N_pml_high[2],
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])

        if bc_low[1] == "pml":
            i_block = 1
            j_block = 0
            k_block = 1
            self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                self.Nx,
                self.N_pml_low[1],
                self.Nz,
                self.dt,
                self.dx,
                self.dy,
                self.dz,
                i_block,
                j_block,
                k_block,
            )
            self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_low[2] == "pml":
                i_block = 1
                j_block = 0
                k_block = 0
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.Nx,
                    self.N_pml_low[1],
                    self.N_pml_low[2],
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_high[2] == "pml":
                i_block = 1
                j_block = 0
                k_block = 2
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.Nx,
                    self.N_pml_low[1],
                    self.N_pml_high[2],
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])

        if bc_high[1] == "pml":
            i_block = 1
            j_block = 2
            k_block = 1
            self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                self.Nx,
                self.N_pml_high[1],
                self.Nz,
                self.dt,
                self.dx,
                self.dy,
                self.dz,
                i_block,
                j_block,
                k_block,
            )
            self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_low[2] == "pml":
                i_block = 1
                j_block = 2
                k_block = 0
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.Nx,
                    self.N_pml_low[1],
                    self.N_pml_low[2],
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])
            if bc_high[2] == "pml":
                i_block = 1
                j_block = 2
                k_block = 2
                self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                    self.Nx,
                    self.N_pml_low[1],
                    self.N_pml_high[2],
                    self.dt,
                    self.dx,
                    self.dy,
                    self.dz,
                    i_block,
                    j_block,
                    k_block,
                )
                self.blocks.append(self.blocks_mat[i_block, j_block, k_block])

        if bc_low[2] == "pml":
            i_block = 1
            j_block = 1
            k_block = 0
            self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                self.Nx,
                self.Ny,
                self.N_pml_low[2],
                self.dt,
                self.dx,
                self.dy,
                self.dz,
                i_block,
                j_block,
                k_block,
            )
            self.blocks.append(self.blocks_mat[i_block, j_block, k_block])

        if bc_high[2] == "pml":
            i_block = 1
            j_block = 1
            k_block = 2
            self.blocks_mat[i_block, j_block, k_block] = PmlBlock3D(
                self.Nx,
                self.Ny,
                self.N_pml_high[2],
                self.dt,
                self.dx,
                self.dy,
                self.dz,
                i_block,
                j_block,
                k_block,
            )
            self.blocks.append(self.blocks_mat[i_block, j_block, k_block])

        for block in self.blocks:
            block.blocks_mat = self.blocks_mat

    def assemble_conductivities_pmls(self):
        sigma_m_low_x = 0
        sigma_m_high_x = 0
        sigma_m_low_y = 0
        sigma_m_high_y = 0
        sigma_m_low_z = 0
        sigma_m_high_z = 0
        Z_0_2 = mu_0 / eps_0

        if self.bc_low[0] == "pml":
            sigma_m_low_x = (
                -(self.alpha_pml + 1)
                * eps_0
                * c_light
                / (2 * (self.N_pml_low[0] - 1) * self.dx)
                * np.log(self.R0_pml)
            )
        if self.bc_low[1] == "pml":
            sigma_m_low_y = (
                -(self.alpha_pml + 1)
                * eps_0
                * c_light
                / (2 * (self.N_pml_low[1] - 1) * self.dy)
                * np.log(self.R0_pml)
            )
        if self.bc_low[2] == "pml":
            sigma_m_low_z = (
                -(self.alpha_pml + 1)
                * eps_0
                * c_light
                / (2 * (self.N_pml_low[2] - 1) * self.dy)
                * np.log(self.R0_pml)
            )
        if self.bc_high[0] == "pml":
            sigma_m_high_x = (
                -(self.alpha_pml + 1)
                * eps_0
                * c_light
                / (2 * (self.N_pml_high[0] - 1) * self.dy)
                * np.log(self.R0_pml)
            )
        if self.bc_high[1] == "pml":
            sigma_m_high_y = (
                -(self.alpha_pml + 1)
                * eps_0
                * c_light
                / (2 * (self.N_pml_high[1] - 1) * self.dy)
                * np.log(self.R0_pml)
            )
        if self.bc_high[2] == "pml":
            sigma_m_high_z = (
                -(self.alpha_pml + 1)
                * eps_0
                * c_light
                / (2 * (self.N_pml_high[2] - 1) * self.dy)
                * np.log(self.R0_pml)
            )

        if self.bc_low[0] == "pml":
            (i_block, j_block, k_block) = (0, 1, 1)
            for n in range(self.N_pml_low[0]):
                self.blocks_mat[i_block, j_block, k_block].sigma_x[-(n + 1), :, :] = (
                    sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                )
                self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                    -(n + 1), :, :
                ] = sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml * Z_0_2
            if self.bc_low[1] == "pml":
                (i_block, j_block, k_block) = (0, 0, 1)
                for n in range((self.N_pml_low[0])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_x[
                        -(n + 1), :, :
                    ] = sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                        -(n + 1), :, :
                    ] = (
                        sigma_m_low_x
                        * (n / (self.N_pml_low[0])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range((self.N_pml_low[1])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_y[
                        :, -(n + 1), :
                    ] = sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                        :, -(n + 1), :
                    ] = (
                        sigma_m_low_y
                        * (n / (self.N_pml_low[1])) ** self.alpha_pml
                        * Z_0_2
                    )
                if self.bc_low[2] == "pml":
                    (i_block, j_block, k_block) = (0, 0, 0)
                    for n in range((self.N_pml_low[0])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_x[
                            -(n + 1), :, :
                        ] = sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                            -(n + 1), :, :
                        ] = (
                            sigma_m_low_x
                            * (n / (self.N_pml_low[0])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_low[1])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_y[
                            :, -(n + 1), :
                        ] = sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                            :, -(n + 1), :
                        ] = (
                            sigma_m_low_y
                            * (n / (self.N_pml_low[1])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_low[2])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_z[
                            :, :, -(n + 1)
                        ] = sigma_m_low_z * (n / (self.N_pml_low[2])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                            :, :, -(n + 1)
                        ] = (
                            sigma_m_low_z
                            * (n / (self.N_pml_low[2])) ** self.alpha_pml
                            * Z_0_2
                        )
                if self.bc_low[2] == "pml":
                    (i_block, j_block, k_block) = (0, 0, 2)
                    for n in range((self.N_pml_low[0])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_x[
                            -(n + 1), :, :
                        ] = sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                            -(n + 1), :, :
                        ] = (
                            sigma_m_low_x
                            * (n / (self.N_pml_low[0])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_low[1])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_y[
                            :, -(n + 1), :
                        ] = sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                            :, -(n + 1), :
                        ] = (
                            sigma_m_low_y
                            * (n / (self.N_pml_low[1])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_high[2])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_z[:, :, n] = (
                            sigma_m_high_z
                            * (n / (self.N_pml_high[2])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                            :, :, n
                        ] = (
                            sigma_m_high_z
                            * (n / (self.N_pml_high[2])) ** self.alpha_pml
                            * Z_0_2
                        )
            if self.bc_high[1] == "pml":
                (i_block, j_block, k_block) = (0, 2, 1)
                for n in range(self.N_pml_low[0]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_x[
                        -(n + 1), :, :
                    ] = sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                        -(n + 1), :, :
                    ] = (
                        sigma_m_low_x
                        * (n / (self.N_pml_low[0])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range(self.N_pml_high[1]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_y[:, n, :] = (
                        sigma_m_high_y * (n / (self.N_pml_high[1])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_y[:, n, :] = (
                        sigma_m_high_y
                        * (n / (self.N_pml_high[1])) ** self.alpha_pml
                        * Z_0_2
                    )

                if self.bc_low[2] == "pml":
                    (i_block, j_block, k_block) = (0, 2, 0)
                    for n in range((self.N_pml_low[0])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_x[
                            -(n + 1), :, :
                        ] = sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                            -(n + 1), :, :
                        ] = (
                            sigma_m_low_x
                            * (n / (self.N_pml_low[0])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_high[1])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_y[:, n, :] = (
                            sigma_m_high_y
                            * (n / (self.N_pml_high[1])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                            :, n, :
                        ] = (
                            sigma_m_high_y
                            * (n / (self.N_pml_high[1])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_low[2])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_z[
                            :, :, -(n + 1)
                        ] = sigma_m_low_z * (n / (self.N_pml_low[2])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                            :, :, -(n + 1)
                        ] = (
                            sigma_m_low_z
                            * (n / (self.N_pml_low[2])) ** self.alpha_pml
                            * Z_0_2
                        )
                if self.bc_low[2] == "pml":
                    (i_block, j_block, k_block) = (0, 2, 2)
                    for n in range((self.N_pml_low[0])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_x[
                            -(n + 1), :, :
                        ] = sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                            -(n + 1), :, :
                        ] = (
                            sigma_m_low_x
                            * (n / (self.N_pml_low[0])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_high[1])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_y[:, n, :] = (
                            sigma_m_high_y
                            * (n / (self.N_pml_high[1])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                            :, n, :
                        ] = (
                            sigma_m_high_y
                            * (n / (self.N_pml_high[1])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_high[2])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_z[:, :, n] = (
                            sigma_m_high_z
                            * (n / (self.N_pml_high[2])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                            :, :, n
                        ] = (
                            sigma_m_high_z
                            * (n / (self.N_pml_high[2])) ** self.alpha_pml
                            * Z_0_2
                        )
            if self.bc_low[2] == "pml":
                (i_block, j_block, k_block) = (0, 1, 0)
                for n in range((self.N_pml_low[0])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_x[
                        -(n + 1), :, :
                    ] = sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                        -(n + 1), :, :
                    ] = (
                        sigma_m_low_x
                        * (n / (self.N_pml_low[0])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range((self.N_pml_low[2])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_z[
                        :, :, -(n + 1)
                    ] = sigma_m_low_z * (n / (self.N_pml_low[2])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                        :, :, -(n + 1)
                    ] = (
                        sigma_m_low_z
                        * (n / (self.N_pml_low[2])) ** self.alpha_pml
                        * Z_0_2
                    )
            if self.bc_high[2] == "pml":
                (i_block, j_block, k_block) = (0, 1, 2)
                for n in range((self.N_pml_low[0])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_x[
                        -(n + 1), :, :
                    ] = sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                        -(n + 1), :, :
                    ] = (
                        sigma_m_low_x
                        * (n / (self.N_pml_low[0])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range((self.N_pml_high[2])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_z[:, :, n] = (
                        sigma_m_high_z * (n / (self.N_pml_high[2])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_z[:, :, n] = (
                        sigma_m_high_z
                        * (n / (self.N_pml_high[2])) ** self.alpha_pml
                        * Z_0_2
                    )

        if self.bc_high[0] == "pml":
            (i_block, j_block, k_block) = (2, 1, 1)
            for n in range(self.N_pml_high[0]):
                self.blocks_mat[i_block, j_block, k_block].sigma_x[n, :, :] = (
                    sigma_m_high_x * (n / (self.N_pml_high[0])) ** self.alpha_pml
                )
                self.blocks_mat[i_block, j_block, k_block].sigma_star_x[n, :, :] = (
                    sigma_m_high_x
                    * (n / (self.N_pml_high[0])) ** self.alpha_pml
                    * Z_0_2
                )
            if self.bc_low[1] == "pml":
                (i_block, j_block, k_block) = (2, 0, 1)
                for n in range(self.N_pml_high[0]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_x[n, :, :] = (
                        sigma_m_high_x * (n / (self.N_pml_high[0])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_x[n, :, :] = (
                        sigma_m_high_x
                        * (n / (self.N_pml_high[0])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range((self.N_pml_low[1])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_y[
                        :, -(n + 1), :
                    ] = sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                        :, -(n + 1), :
                    ] = (
                        sigma_m_low_y
                        * (n / (self.N_pml_low[1])) ** self.alpha_pml
                        * Z_0_2
                    )
                if self.bc_low[2] == "pml":
                    (i_block, j_block, k_block) = (2, 0, 0)
                    for n in range(self.N_pml_high[0]):
                        self.blocks_mat[i_block, j_block, k_block].sigma_x[n, :, :] = (
                            sigma_m_high_x
                            * (n / (self.N_pml_high[0])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                            n, :, :
                        ] = (
                            sigma_m_high_x
                            * (n / (self.N_pml_high[0])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_low[1])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_y[
                            :, -(n + 1), :
                        ] = sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                            :, -(n + 1), :
                        ] = (
                            sigma_m_low_y
                            * (n / (self.N_pml_low[1])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_low[2])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_z[
                            :, :, -(n + 1)
                        ] = sigma_m_low_z * (n / (self.N_pml_low[2])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                            :, :, -(n + 1)
                        ] = (
                            sigma_m_low_z
                            * (n / (self.N_pml_low[2])) ** self.alpha_pml
                            * Z_0_2
                        )
                if self.bc_low[2] == "pml":
                    (i_block, j_block, k_block) = (2, 0, 2)
                    for n in range(self.N_pml_high[0]):
                        self.blocks_mat[i_block, j_block, k_block].sigma_x[n, :, :] = (
                            sigma_m_high_x
                            * (n / (self.N_pml_high[0])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                            n, :, :
                        ] = (
                            sigma_m_high_x
                            * (n / (self.N_pml_high[0])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_low[1])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_y[
                            :, -(n + 1), :
                        ] = sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                            :, -(n + 1), :
                        ] = (
                            sigma_m_low_y
                            * (n / (self.N_pml_low[1])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_high[2])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_z[:, :, n] = (
                            sigma_m_high_z
                            * (n / (self.N_pml_high[2])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                            :, :, n
                        ] = (
                            sigma_m_high_z
                            * (n / (self.N_pml_high[2])) ** self.alpha_pml
                            * Z_0_2
                        )
            if self.bc_high[1] == "pml":
                (i_block, j_block, k_block) = (2, 2, 1)
                for n in range(self.N_pml_high[0]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_x[n, :, :] = (
                        sigma_m_high_x * (n / (self.N_pml_high[0])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_x[n, :, :] = (
                        sigma_m_high_x
                        * (n / (self.N_pml_high[0])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range(self.N_pml_high[1]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_y[:, n, :] = (
                        sigma_m_high_y * (n / (self.N_pml_high[1])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_y[:, n, :] = (
                        sigma_m_high_y
                        * (n / (self.N_pml_high[1])) ** self.alpha_pml
                        * Z_0_2
                    )
                if self.bc_low[2] == "pml":
                    (i_block, j_block, k_block) = (2, 2, 0)
                    for n in range(self.N_pml_high[0]):
                        self.blocks_mat[i_block, j_block, k_block].sigma_x[n, :, :] = (
                            sigma_m_high_x
                            * (n / (self.N_pml_high[0])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                            n, :, :
                        ] = (
                            sigma_m_high_x
                            * (n / (self.N_pml_high[0])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_high[1])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_y[:, n, :] = (
                            sigma_m_high_y
                            * (n / (self.N_pml_high[1])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                            :, n, :
                        ] = (
                            sigma_m_high_y
                            * (n / (self.N_pml_high[1])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_low[2])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_z[
                            :, :, -(n + 1)
                        ] = sigma_m_low_z * (n / (self.N_pml_low[2])) ** self.alpha_pml
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                            :, :, -(n + 1)
                        ] = (
                            sigma_m_low_z
                            * (n / (self.N_pml_low[2])) ** self.alpha_pml
                            * Z_0_2
                        )
                if self.bc_low[2] == "pml":
                    (i_block, j_block, k_block) = (2, 2, 2)
                    for n in range(self.N_pml_high[0]):
                        self.blocks_mat[i_block, j_block, k_block].sigma_x[n, :, :] = (
                            sigma_m_high_x
                            * (n / (self.N_pml_high[0])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_x[
                            n, :, :
                        ] = (
                            sigma_m_high_x
                            * (n / (self.N_pml_high[0])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_high[1])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_y[:, n, :] = (
                            sigma_m_high_y
                            * (n / (self.N_pml_high[1])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                            :, n, :
                        ] = (
                            sigma_m_high_y
                            * (n / (self.N_pml_high[1])) ** self.alpha_pml
                            * Z_0_2
                        )
                    for n in range((self.N_pml_high[2])):
                        self.blocks_mat[i_block, j_block, k_block].sigma_z[:, :, n] = (
                            sigma_m_high_z
                            * (n / (self.N_pml_high[2])) ** self.alpha_pml
                        )
                        self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                            :, :, n
                        ] = (
                            sigma_m_high_z
                            * (n / (self.N_pml_high[2])) ** self.alpha_pml
                            * Z_0_2
                        )
            if self.bc_low[2] == "pml":
                (i_block, j_block, k_block) = (2, 1, 0)
                for n in range(self.N_pml_high[0]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_x[n, :, :] = (
                        sigma_m_high_x * (n / (self.N_pml_high[0])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_x[n, :, :] = (
                        sigma_m_high_x
                        * (n / (self.N_pml_high[0])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range((self.N_pml_low[2])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_z[
                        :, :, -(n + 1)
                    ] = sigma_m_low_z * (n / (self.N_pml_low[2])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                        :, :, -(n + 1)
                    ] = (
                        sigma_m_low_z
                        * (n / (self.N_pml_low[2])) ** self.alpha_pml
                        * Z_0_2
                    )
            if self.bc_high[2] == "pml":
                (i_block, j_block, k_block) = (2, 1, 2)
                for n in range(self.N_pml_high[0]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_x[n, :, :] = (
                        sigma_m_high_x * (n / (self.N_pml_high[0])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_x[n, :, :] = (
                        sigma_m_high_x
                        * (n / (self.N_pml_high[0])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range((self.N_pml_high[2])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_z[:, :, n] = (
                        sigma_m_high_z * (n / (self.N_pml_high[2])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_z[:, :, n] = (
                        sigma_m_high_z
                        * (n / (self.N_pml_high[2])) ** self.alpha_pml
                        * Z_0_2
                    )

        if self.bc_low[1] == "pml":
            (i_block, j_block, k_block) = (1, 0, 1)
            for n in range(self.N_pml_low[1]):
                self.blocks_mat[i_block, j_block, k_block].sigma_y[:, -(n + 1), :] = (
                    sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                )
                self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                    :, -(n + 1), :
                ] = sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml * Z_0_2
            if self.bc_low[2] == "pml":
                (i_block, j_block, k_block) = (1, 0, 0)
                for n in range(self.N_pml_low[1]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_y[
                        :, -(n + 1), :
                    ] = sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                        :, -(n + 1), :
                    ] = (
                        sigma_m_low_y
                        * (n / (self.N_pml_low[1])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range((self.N_pml_low[2])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_z[
                        :, :, -(n + 1)
                    ] = sigma_m_low_z * (n / (self.N_pml_low[2])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                        :, :, -(n + 1)
                    ] = (
                        sigma_m_low_z
                        * (n / (self.N_pml_low[2])) ** self.alpha_pml
                        * Z_0_2
                    )
            if self.bc_high[2] == "pml":
                (i_block, j_block, k_block) = (1, 0, 2)
                for n in range(self.N_pml_low[1]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_y[
                        :, -(n + 1), :
                    ] = sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_y[
                        :, -(n + 1), :
                    ] = (
                        sigma_m_low_y
                        * (n / (self.N_pml_low[1])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range((self.N_pml_high[2])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_z[:, :, n] = (
                        sigma_m_high_z * (n / (self.N_pml_high[2])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_z[:, :, n] = (
                        sigma_m_high_z
                        * (n / (self.N_pml_high[2])) ** self.alpha_pml
                        * Z_0_2
                    )

        if self.bc_high[1] == "pml":
            (i_block, j_block, k_block) = (1, 2, 1)
            for n in range(self.N_pml_high[1]):
                self.blocks_mat[i_block, j_block, k_block].sigma_y[:, n, :] = (
                    sigma_m_high_y * (n / (self.N_pml_high[1])) ** self.alpha_pml
                )
                self.blocks_mat[i_block, j_block, k_block].sigma_star_y[:, n, :] = (
                    sigma_m_high_y
                    * (n / (self.N_pml_high[1])) ** self.alpha_pml
                    * Z_0_2
                )
            if self.bc_low[2] == "pml":
                (i_block, j_block, k_block) = (1, 2, 0)
                for n in range(self.N_pml_high[1]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_y[:, n, :] = (
                        sigma_m_high_y * (n / (self.N_pml_high[1])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_y[:, n, :] = (
                        sigma_m_high_y
                        * (n / (self.N_pml_high[1])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range((self.N_pml_low[2])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_z[
                        :, :, -(n + 1)
                    ] = sigma_m_low_z * (n / (self.N_pml_low[2])) ** self.alpha_pml
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                        :, :, -(n + 1)
                    ] = (
                        sigma_m_low_z
                        * (n / (self.N_pml_low[2])) ** self.alpha_pml
                        * Z_0_2
                    )
            if self.bc_high[2] == "pml":
                (i_block, j_block, k_block) = (1, 2, 2)
                for n in range(self.N_pml_high[1]):
                    self.blocks_mat[i_block, j_block, k_block].sigma_y[:, n, :] = (
                        sigma_m_high_y * (n / (self.N_pml_high[1])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_y[:, n, :] = (
                        sigma_m_high_y
                        * (n / (self.N_pml_high[1])) ** self.alpha_pml
                        * Z_0_2
                    )
                for n in range((self.N_pml_high[2])):
                    self.blocks_mat[i_block, j_block, k_block].sigma_z[:, :, n] = (
                        sigma_m_high_z * (n / (self.N_pml_high[2])) ** self.alpha_pml
                    )
                    self.blocks_mat[i_block, j_block, k_block].sigma_star_z[:, :, n] = (
                        sigma_m_high_z
                        * (n / (self.N_pml_high[2])) ** self.alpha_pml
                        * Z_0_2
                    )

        if self.bc_low[2] == "pml":
            (i_block, j_block, k_block) = (1, 1, 0)
            for n in range((self.N_pml_low[2])):
                self.blocks_mat[i_block, j_block, k_block].sigma_z[:, :, -(n + 1)] = (
                    sigma_m_low_z * (n / (self.N_pml_low[2])) ** self.alpha_pml
                )
                self.blocks_mat[i_block, j_block, k_block].sigma_star_z[
                    :, :, -(n + 1)
                ] = sigma_m_low_z * (n / (self.N_pml_low[2])) ** self.alpha_pml * Z_0_2

        if self.bc_high[2] == "pml":
            (i_block, j_block, k_block) = (1, 1, 2)
            for n in range((self.N_pml_high[2])):
                self.blocks_mat[i_block, j_block, k_block].sigma_z[:, :, n] = (
                    sigma_m_high_z * (n / (self.N_pml_high[2])) ** self.alpha_pml
                )
                self.blocks_mat[i_block, j_block, k_block].sigma_star_z[:, :, n] = (
                    sigma_m_high_z
                    * (n / (self.N_pml_high[2])) ** self.alpha_pml
                    * Z_0_2
                )

        # for i_block in range(3):
        #    for j_block in range(3):
        #        for k_block in range(3):
        #            block = self.blocks_mat[i_block, j_block, k_block]
        #            if block is not None:
        #                if not (i_block == 1 and j_block == 1 and k_block == 1):
        #                    block.sigma_star_x = block.sigma_x * mu_0 / eps_0
        #                    block.sigma_star_y = block.sigma_y * mu_0 / eps_0
        #                    block.sigma_star_z = block.sigma_z * mu_0 / eps_0

    def assemble_coeffs_pmls(self):
        for i_block in range(3):
            for j_block in range(3):
                for k_block in range(3):
                    if self.blocks_mat[i_block, j_block, k_block] is not None:
                        if not (i_block == 1 and j_block == 1 and k_block == 1):
                            self.blocks_mat[i_block, j_block, k_block].assemble_coeffs()

    def update_e_boundary(self):
        Ex = self.Ex
        Ey = self.Ey
        Ez = self.Ez
        Hx = self.Hx
        Hy = self.Hy
        Hz = self.Hz

        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz

        # Update E on "lower" faces
        if self.blocks_mat[0, 1, 1] is not None:
            for jj in range(self.Ny):
                for kk in range(1, self.Nz):
                    Ey[0, jj, kk] = (
                        Ey[0, jj, kk]
                        - self.C3 * self.Jy[0, jj, kk]
                        + self.C8 * (Hx[0, jj, kk] - Hx[0, jj, kk - 1])
                        - self.C5
                        * (Hz[0, jj, kk] - self.blocks_mat[0, 1, 1].Hz[-1, jj, kk])
                    )

            for jj in range(1, self.Ny):
                for kk in range(self.Nz):
                    Ez[0, jj, kk] = (
                        Ez[0, jj, kk]
                        - self.C3 * self.Jz[0, jj, kk]
                        + self.C5
                        * (Hy[0, jj, kk] - self.blocks_mat[0, 1, 1].Hy[-1, jj, kk])
                        - self.C4 * (Hx[0, jj, kk] - Hx[0, jj - 1, kk])
                    )

        if self.blocks_mat[1, 0, 1] is not None:
            for ii in range(self.Nx):
                for kk in range(1, self.Nz):
                    Ex[ii, 0, kk] = (
                        Ex[ii, 0, kk]
                        - self.C3 * self.Jx[ii, 0, kk]
                        + self.C4
                        * (Hz[ii, 0, kk] - self.blocks_mat[1, 0, 1].Hz[ii, -1, kk])
                        - self.C8 * (Hy[ii, 0, kk] - Hy[ii, 0, kk - 1])
                    )
            for ii in range(1, self.Nx):
                for kk in range(self.Nz):
                    Ez[ii, 0, kk] = (
                        Ez[ii, 0, kk]
                        - self.C3 * self.Jz[ii, 0, kk]
                        + self.C5 * (Hy[ii, 0, kk] - Hy[ii - 1, 0, kk])
                        - self.C4
                        * (Hx[ii, 0, kk] - self.blocks_mat[1, 0, 1].Hx[ii, -1, kk])
                    )

        if self.blocks_mat[1, 1, 0] is not None:
            for ii in range(self.Nx):
                for jj in range(1, self.Ny):
                    Ex[ii, jj, 0] = (
                        Ex[ii, jj, 0]
                        - self.C3 * self.Jx[ii, jj, 0]
                        + self.C4 * (Hz[ii, jj, 0] - Hz[ii, jj - 1, 0])
                        - self.C8
                        * (Hy[ii, jj, 0] - self.blocks_mat[1, 1, 0].Hy[ii, jj, -1])
                    )
            for ii in range(1, self.Nx):
                for jj in range(self.Ny):
                    Ey[ii, jj, 0] = (
                        Ey[ii, jj, 0]
                        - self.C3 * self.Jy[ii, jj, 0]
                        + self.C8
                        * (Hx[ii, jj, 0] - self.blocks_mat[1, 1, 0].Hx[ii, jj, -1])
                        - self.C5 * (Hz[ii, jj, 0] - Hz[ii - 1, jj, 0])
                    )

        # Update E on "upper" faces
        if self.blocks_mat[2, 1, 1] is not None:
            for jj in range(self.Ny):
                for kk in range(1, self.Nz):
                    Ey[Nx, jj, kk] = (
                        Ey[Nx, jj, kk]
                        - self.C3 * self.Jy[Nx, jj, kk]
                        + self.C8 * (Hx[Nx, jj, kk] - Hx[Nx, jj, kk - 1])
                        - self.C5
                        * (self.blocks_mat[2, 1, 1].Hz[0, jj, kk] - Hz[Nx - 1, jj, kk])
                    )
            for jj in range(1, self.Ny):
                for kk in range(self.Nz):
                    Ez[Nx, jj, kk] = (
                        Ez[Nx, jj, kk]
                        - self.C3 * self.Jz[Nx, jj, kk]
                        + self.C5
                        * (self.blocks_mat[2, 1, 1].Hy[0, jj, kk] - Hy[Nx - 1, jj, kk])
                        - self.C4 * (Hx[Nx, jj, kk] - Hx[Nx, jj - 1, kk])
                    )

        if self.blocks_mat[1, 2, 1] is not None:
            for ii in range(self.Nx):
                for kk in range(1, self.Nz):
                    Ex[ii, Ny, kk] = (
                        Ex[ii, Ny, kk]
                        - self.C3 * self.Jx[ii, Ny, kk]
                        + self.C4
                        * (self.blocks_mat[1, 2, 1].Hz[ii, 0, kk] - Hz[ii, Ny - 1, kk])
                        - self.C8 * (Hy[ii, Ny, kk] - Hy[ii, Ny, kk - 1])
                    )
            for ii in range(1, self.Nx):
                for kk in range(self.Nz):
                    Ez[ii, Ny, kk] = (
                        Ez[ii, Ny, kk]
                        - self.C3 * self.Jz[ii, Ny, kk]
                        + self.C5 * (Hy[ii, Ny, kk] - Hy[ii - 1, Ny, kk])
                        - self.C4
                        * (self.blocks_mat[1, 2, 1].Hx[ii, 0, kk] - Hx[ii, Ny - 1, kk])
                    )

        if self.blocks_mat[1, 1, 2] is not None:
            for ii in range(self.Nx):
                for jj in range(1, self.Ny):
                    Ex[ii, jj, Nz] = (
                        Ex[ii, jj, Nz]
                        - self.C3 * self.Jx[ii, jj, Nz]
                        + self.C4 * (Hz[ii, jj, Nz] - Hz[ii, jj - 1, Nz])
                        - self.C8
                        * (self.blocks_mat[1, 1, 2].Hy[ii, jj, 0] - Hy[ii, jj, Nz - 1])
                    )
            for ii in range(1, self.Nx):
                for jj in range(self.Ny):
                    Ey[ii, jj, Nz] = (
                        Ey[ii, jj, Nz]
                        - self.C3 * self.Jy[ii, jj, Nz]
                        + self.C8
                        * (self.blocks_mat[1, 1, 2].Hx[ii, jj, 0] - Hx[ii, jj, Nz - 1])
                        - self.C5 * (Hz[ii, jj, Nz] - Hz[ii - 1, jj, Nz])
                    )
        # Update Ez on edges (xy)
        if (
            self.blocks_mat[0, 1, 1] is not None
            and self.blocks_mat[1, 0, 1] is not None
        ):
            for kk in range(Nz):
                Ez[0, 0, kk] = (
                    Ez[0, 0, kk]
                    - self.C3 * self.Jz[0, 0, kk]
                    + self.C5 * (Hy[0, 0, kk] - self.blocks_mat[0, 1, 1].Hy[-1, 0, kk])
                    - self.C4 * (Hx[0, 0, kk] - self.blocks_mat[1, 0, 1].Hx[0, -1, kk])
                )
        if (
            self.blocks_mat[0, 1, 1] is not None
            and self.blocks_mat[1, 2, 1] is not None
        ):
            for kk in range(Nz):
                Ez[0, Ny, kk] = (
                    Ez[0, Ny, kk]
                    - self.C3 * self.Jz[0, Ny, kk]
                    + self.C5
                    * (Hy[0, Ny, kk] - self.blocks_mat[0, 1, 1].Hy[-1, Ny, kk])
                    - self.C4
                    * (self.blocks_mat[1, 2, 1].Hx[0, 0, kk] - Hx[0, Ny - 1, kk])
                )
        if (
            self.blocks_mat[2, 1, 1] is not None
            and self.blocks_mat[1, 0, 1] is not None
        ):
            for kk in range(Nz):
                Ez[Nx, 0, kk] = (
                    Ez[Nx, 0, kk]
                    - self.C3 * self.Jz[Nx, 0, kk]
                    + self.C5
                    * (self.blocks_mat[2, 1, 1].Hy[0, 0, kk] - Hy[Nx - 1, 0, kk])
                    - self.C4
                    * (Hx[Nx, 0, kk] - self.blocks_mat[1, 0, 1].Hx[Nx, -1, kk])
                )
        if (
            self.blocks_mat[2, 1, 1] is not None
            and self.blocks_mat[1, 2, 1] is not None
        ):
            for kk in range(Nz):
                Ez[Nx, Ny, kk] = (
                    Ez[Nx, Ny, kk]
                    - self.C3 * self.Jz[Nx, Ny, kk]
                    + self.C5
                    * (self.blocks_mat[2, 1, 1].Hy[0, Ny, kk] - Hy[Nx - 1, Ny, kk])
                    - self.C4
                    * (self.blocks_mat[1, 2, 1].Hx[Nx, 0, kk] - Hx[Nx, Ny - 1, kk])
                )
        # Update Ex on edges (yz)
        if (
            self.blocks_mat[1, 0, 1] is not None
            and self.blocks_mat[1, 1, 0] is not None
        ):
            for ii in range(Nx):
                Ex[ii, 0, 0] = (
                    Ex[ii, 0, 0]
                    - self.C3 * self.Jx[ii, 0, 0]
                    + self.C4 * (Hz[ii, 0, 0] - self.blocks_mat[1, 0, 1].Hz[ii, -1, 0])
                    - self.C8 * (Hy[ii, 0, 0] - self.blocks_mat[1, 1, 0].Hy[ii, 0, -1])
                )
        if (
            self.blocks_mat[1, 0, 1] is not None
            and self.blocks_mat[1, 1, 2] is not None
        ):
            for ii in range(Nx):
                Ex[ii, 0, Nz] = (
                    Ex[ii, 0, Nz]
                    - self.C3 * self.Jx[ii, 0, Nz]
                    + self.C4
                    * (Hz[ii, 0, Nz] - self.blocks_mat[1, 0, 1].Hz[ii, -1, Nz])
                    - self.C8
                    * (self.blocks_mat[1, 1, 2].Hy[ii, 0, 0] - Hy[ii, 0, Nz - 1])
                )
        if (
            self.blocks_mat[1, 2, 1] is not None
            and self.blocks_mat[1, 1, 0] is not None
        ):
            for ii in range(Nx):
                Ex[ii, Ny, 0] = (
                    Ex[ii, Ny, 0]
                    - self.C3 * self.Jx[ii, Ny, 0]
                    + self.C4
                    * (self.blocks_mat[1, 2, 1].Hz[ii, 0, 0] - Hz[ii, Ny - 1, 0])
                    - self.C8
                    * (Hy[ii, Ny, 0] - self.blocks_mat[1, 1, 0].Hy[ii, Ny, -1])
                )
        if (
            self.blocks_mat[1, 2, 1] is not None
            and self.blocks_mat[1, 1, 2] is not None
        ):
            for ii in range(Nx):
                Ex[ii, Ny, Nz] = (
                    Ex[ii, Ny, Nz]
                    - self.C3 * self.Jx[ii, Ny, Nz]
                    + self.C4
                    * (self.blocks_mat[1, 2, 1].Hz[ii, 0, Nz] - Hz[ii, Ny - 1, Nz])
                    - self.C8
                    * (self.blocks_mat[1, 1, 2].Hy[ii, Ny, 0] - Hy[ii, Ny, Nz - 1])
                )

        # Update Ey on edges (xz)
        if (
            self.blocks_mat[0, 1, 1] is not None
            and self.blocks_mat[1, 1, 0] is not None
        ):
            for jj in range(Ny):
                Ey[0, jj, 0] = (
                    Ey[0, jj, 0]
                    - self.C3 * self.Jy[0, jj, 0]
                    + self.C8 * (Hx[0, jj, 0] - self.blocks_mat[1, 1, 0].Hx[0, jj, -1])
                    - self.C5 * (Hz[0, jj, 0] - self.blocks_mat[0, 1, 1].Hz[-1, jj, 0])
                )
        if (
            self.blocks_mat[0, 1, 1] is not None
            and self.blocks_mat[1, 1, 2] is not None
        ):
            for jj in range(Ny):
                Ey[0, jj, Nz] = (
                    Ey[0, jj, Nz]
                    - self.C3 * self.Jy[0, jj, Nz]
                    + self.C8
                    * (self.blocks_mat[1, 1, 2].Hx[0, jj, 0] - Hx[0, jj, Nz - 1])
                    - self.C5
                    * (Hz[0, jj, Nz] - self.blocks_mat[0, 1, 1].Hz[-1, jj, Nz])
                )
        if (
            self.blocks_mat[2, 1, 1] is not None
            and self.blocks_mat[1, 1, 0] is not None
        ):
            for jj in range(Ny):
                Ey[Nx, jj, 0] = (
                    Ey[Nx, jj, 0]
                    - self.C3 * self.Jy[Nx, jj, 0]
                    + self.C8
                    * (Hx[Nx, jj, 0] - self.blocks_mat[1, 1, 0].Hx[Nx, jj, -1])
                    - self.C5
                    * (self.blocks_mat[2, 1, 1].Hz[0, jj, 0] - Hz[Nx - 1, jj, 0])
                )
        if (
            self.blocks_mat[2, 1, 1] is not None
            and self.blocks_mat[1, 1, 2] is not None
        ):
            for jj in range(Ny):
                Ey[Nx, jj, Nz] = (
                    Ey[Nx, jj, Nz]
                    - self.C3 * self.Jy[Nx, jj, Nz]
                    + self.C8
                    * (self.blocks_mat[1, 1, 2].Hx[Nx, jj, 0] - Hx[Nx, jj, Nz - 1])
                    - self.C5
                    * (self.blocks_mat[2, 1, 1].Hz[0, jj, Nz] - Hz[Nx - 1, jj, Nz])
                )

    def gauss(self, t):
        tau = 10 * self.dt
        if t < 6 * tau:
            return 100 * np.exp(-((t - 3 * tau) ** 2) / tau**2)
        else:
            return 0.0

    def one_step(self):
        if self.sol_type == "ECT":
            self.one_step_ect()
        if self.sol_type == "FDTD":
            self.one_step_fdtd()
        if self.sol_type == "DM":
            self.one_step_dm()

        self.time += self.dt

    def one_step_ect(self):
        self.compute_v_and_rho()
        self.advance_h_ect()
        for block in self.blocks:
            block.advance_h_fdtd()
            block.sum_h_fields()
        self.advance_e_dm()
        self.update_e_boundary()
        for block in self.blocks:
            block.advance_e_fdtd()
            block.update_e_boundary()
            block.sum_e_fields()

    def one_step_fdtd(self):
        self.advance_h_fdtd(
            self.grid.Sxy,
            self.grid.Syz,
            self.grid.Szx,
            self.Ex,
            self.Ey,
            self.Ez,
            self.Hx,
            self.Hy,
            self.Hz,
            self.Nx,
            self.Ny,
            self.Nz,
            self.C1,
            self.C2,
            self.C7,
        )

        for block in self.blocks:
            block.advance_h_fdtd()

            block.sum_h_fields()
        self.advance_e_fdtd(
            self.grid.l_x,
            self.grid.l_y,
            self.grid.l_z,
            self.Ex,
            self.Ey,
            self.Ez,
            self.Hx,
            self.Hy,
            self.Hz,
            self.Jx,
            self.Jy,
            self.Jz,
            self.Nx,
            self.Ny,
            self.Nz,
            self.C3,
            self.C4,
            self.C5,
            self.C8,
        )
        self.update_e_boundary()
        for block in self.blocks:
            block.advance_e_fdtd()
            block.update_e_boundary()
            block.sum_e_fields()

    @staticmethod
    @jit(nopython=True)
    def advance_h_fdtd(Sxy, Syz, Szx, Ex, Ey, Ez, Hx, Hy, Hz, Nx, Ny, Nz, C1, C2, C7):
        # Compute cell voltages
        for ii in range(Nx + 1):
            for jj in range(Ny):
                for kk in range(Nz):
                    if Syz[ii, jj, kk] > 0:
                        Hx[ii, jj, kk] = (
                            Hx[ii, jj, kk]
                            - C2 * (Ez[ii, jj + 1, kk] - Ez[ii, jj, kk])
                            + C7 * (Ey[ii, jj, kk + 1] - Ey[ii, jj, kk])
                        )

        for ii in range(Nx):
            for jj in range(Ny + 1):
                for kk in range(Nz):
                    if Szx[ii, jj, kk] > 0:
                        Hy[ii, jj, kk] = (
                            Hy[ii, jj, kk]
                            - C7 * (Ex[ii, jj, kk + 1] - Ex[ii, jj, kk])
                            + C1 * (Ez[ii + 1, jj, kk] - Ez[ii, jj, kk])
                        )

        for ii in range(Nx):
            for jj in range(Ny):
                for kk in range(Nz + 1):
                    if Sxy[ii, jj, kk] > 0:
                        Hz[ii, jj, kk] = (
                            Hz[ii, jj, kk]
                            - C1 * (Ey[ii + 1, jj, kk] - Ey[ii, jj, kk])
                            + C2 * (Ex[ii, jj + 1, kk] - Ex[ii, jj, kk])
                        )

    @staticmethod
    @jit(nopython=True)
    def advance_e_fdtd(
        l_x, l_y, l_z, Ex, Ey, Ez, Hx, Hy, Hz, Jx, Jy, Jz, Nx, Ny, Nz, C3, C4, C5, C8
    ):
        for ii in range(Nx):
            for jj in range(1, Ny):
                for kk in range(1, Nz):
                    if l_x[ii, jj, kk] > 0:
                        Ex[ii, jj, kk] = (
                            Ex[ii, jj, kk]
                            - C3 * Jx[ii, jj, kk]
                            + C4 * (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk])
                            - C8 * (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1])
                        )

        for ii in range(1, Nx):
            for jj in range(Ny):
                for kk in range(1, Nz):
                    if l_y[ii, jj, kk] > 0:
                        Ey[ii, jj, kk] = (
                            Ey[ii, jj, kk]
                            - C3 * Jy[ii, jj, kk]
                            + C8 * (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1])
                            - C5 * (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk])
                        )

        for ii in range(1, Nx):
            for jj in range(1, Ny):
                for kk in range(Nz):
                    if l_z[ii, jj, kk] > 0:
                        Ez[ii, jj, kk] = (
                            Ez[ii, jj, kk]
                            - C3 * Jz[ii, jj, kk]
                            + C5 * (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk])
                            - C4 * (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk])
                        )

    def one_step_dm(self):
        self.compute_v_and_rho()

        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz + 1):
                    if self.grid.flag_int_cell_xy[i, j, k]:
                        self.Hz[i, j, k] = (
                            self.Hz[i, j, k]
                            - self.dt
                            / (mu_0 * self.grid.Sxy[i, j, k])
                            * self.Vxy[i, j, k]
                        )

        for i in range(self.Nx + 1):
            for j in range(self.Ny):
                for k in range(self.Nz):
                    if self.grid.flag_int_cell_yz[i, j, k]:
                        self.Hx[i, j, k] = (
                            self.Hx[i, j, k]
                            - self.dt
                            / (mu_0 * self.grid.Syz[i, j, k])
                            * self.Vyz[i, j, k]
                        )

        for i in range(self.Nx):
            for j in range(self.Ny + 1):
                for k in range(self.Nz):
                    if self.grid.flag_int_cell_zx[i, j, k]:
                        self.Hy[i, j, k] = (
                            self.Hy[i, j, k]
                            - self.dt
                            / (mu_0 * self.grid.Szx[i, j, k])
                            * self.Vzx[i, j, k]
                        )

        self.advance_e_dm()

    def advance_h_ect(self, dt=None):
        for ii in range(self.Nx + 1):
            EMSolver2D.advance_h_ect(
                Nx=self.Ny,
                Ny=self.Nz,
                V_enl=self.Vyz_enl[ii, :, :],
                rho=self.rho_yz[ii, :, :],
                Hz=self.Hx[ii, :, :],
                C1=self.CN,
                flag_int_cell=self.grid.flag_int_cell_yz[ii, :, :],
                flag_unst_cell=self.grid.flag_unst_cell_yz[ii, :, :],
                flag_intr_cell=self.grid.flag_intr_cell_yz[ii, :, :],
                S=self.grid.Syz[ii, :, :],
                borrowing=self.grid.borrowing_yz[ii, :, :],
                S_enl=self.grid.Syz_enl[ii, :, :],
                S_red=self.grid.Syz_red[ii, :, :],
                dt=dt,
                comp="x",
                kk=ii,
            )

        for jj in range(self.Ny + 1):
            EMSolver2D.advance_h_ect(
                Nx=self.Nx,
                Ny=self.Nz,
                V_enl=self.Vzx_enl[:, jj, :],
                rho=self.rho_zx[:, jj, :],
                Hz=self.Hy[:, jj, :],
                C1=self.CN,
                flag_int_cell=self.grid.flag_int_cell_zx[:, jj, :],
                flag_unst_cell=self.grid.flag_unst_cell_zx[:, jj, :],
                flag_intr_cell=self.grid.flag_intr_cell_zx[:, jj, :],
                S=self.grid.Szx[:, jj, :],
                borrowing=self.grid.borrowing_zx[:, jj, :],
                S_enl=self.grid.Szx_enl[:, jj, :],
                S_red=self.grid.Szx_red[:, jj, :],
                dt=dt,
                comp="y",
                kk=jj,
            )

        for kk in range(self.Nz + 1):
            EMSolver2D.advance_h_ect(
                Nx=self.Nx,
                Ny=self.Ny,
                V_enl=self.Vxy_enl[:, :, kk],
                rho=self.rho_xy[:, :, kk],
                Hz=self.Hz[:, :, kk],
                C1=self.CN,
                flag_int_cell=self.grid.flag_int_cell_xy[:, :, kk],
                flag_unst_cell=self.grid.flag_unst_cell_xy[:, :, kk],
                flag_intr_cell=self.grid.flag_intr_cell_xy[:, :, kk],
                S=self.grid.Sxy[:, :, kk],
                borrowing=self.grid.borrowing_xy[:, :, kk],
                S_enl=self.grid.Sxy_enl[:, :, kk],
                S_red=self.grid.Sxy_red[:, :, kk],
                dt=dt,
                comp="z",
                kk=kk,
            )

    def compute_v_and_rho(self):
        l_x = self.grid.l_x
        l_y = self.grid.l_y
        l_z = self.grid.l_z

        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz + 1):
                    if self.grid.flag_int_cell_xy[ii, jj, kk]:
                        self.Vxy[ii, jj, kk] = (
                            self.Ex[ii, jj, kk] * l_x[ii, jj, kk]
                            - self.Ex[ii, jj + 1, kk] * l_x[ii, jj + 1, kk]
                            + self.Ey[ii + 1, jj, kk] * l_y[ii + 1, jj, kk]
                            - self.Ey[ii, jj, kk] * l_y[ii, jj, kk]
                        )

                        if self.sol_type != "DM":
                            self.rho_xy[ii, jj, kk] = (
                                self.Vxy[ii, jj, kk] / self.grid.Sxy[ii, jj, kk]
                            )

        for ii in range(self.Nx + 1):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    if self.grid.flag_int_cell_yz[ii, jj, kk]:
                        self.Vyz[ii, jj, kk] = (
                            self.Ey[ii, jj, kk] * l_y[ii, jj, kk]
                            - self.Ey[ii, jj, kk + 1] * l_y[ii, jj, kk + 1]
                            + self.Ez[ii, jj + 1, kk] * l_z[ii, jj + 1, kk]
                            - self.Ez[ii, jj, kk] * l_z[ii, jj, kk]
                        )

                        if self.sol_type != "DM":
                            self.rho_yz[ii, jj, kk] = (
                                self.Vyz[ii, jj, kk] / self.grid.Syz[ii, jj, kk]
                            )

        for ii in range(self.Nx):
            for jj in range(self.Ny + 1):
                for kk in range(self.Nz):
                    if self.grid.flag_int_cell_zx[ii, jj, kk]:
                        self.Vzx[ii, jj, kk] = (
                            self.Ez[ii, jj, kk] * l_z[ii, jj, kk]
                            - self.Ez[ii + 1, jj, kk] * l_z[ii + 1, jj, kk]
                            + self.Ex[ii, jj, kk + 1] * l_x[ii, jj, kk + 1]
                            - self.Ex[ii, jj, kk] * l_x[ii, jj, kk]
                        )

                        if self.sol_type != "DM":
                            self.rho_zx[ii, jj, kk] = (
                                self.Vzx[ii, jj, kk] / self.grid.Szx[ii, jj, kk]
                            )

    def advance_e_dm(self, dt=None):
        if dt is None:
            dt = self.dt

        Ex = self.Ex
        Ey = self.Ey
        Ez = self.Ez
        Hx = self.Hx
        Hy = self.Hy
        Hz = self.Hz

        for ii in range(self.Nx):
            for jj in range(1, self.Ny):
                for kk in range(1, self.Nz):
                    if self.grid.l_x[ii, jj, kk] > 0:
                        Ex[ii, jj, kk] = (
                            Ex[ii, jj, kk]
                            - self.C3 * self.Jx[ii, jj, kk]
                            + self.C4 * (Hz[ii, jj, kk] - Hz[ii, jj - 1, kk])
                            - self.C8 * (Hy[ii, jj, kk] - Hy[ii, jj, kk - 1])
                        )

        for ii in range(1, self.Nx):
            for jj in range(self.Ny):
                for kk in range(1, self.Nz):
                    if self.grid.l_y[ii, jj, kk] > 0:
                        Ey[ii, jj, kk] = (
                            Ey[ii, jj, kk]
                            - self.C3 * self.Jy[ii, jj, kk]
                            + self.C8 * (Hx[ii, jj, kk] - Hx[ii, jj, kk - 1])
                            - self.C5 * (Hz[ii, jj, kk] - Hz[ii - 1, jj, kk])
                        )

        for ii in range(1, self.Nx):
            for jj in range(1, self.Ny):
                for kk in range(self.Nz):
                    if self.grid.l_z[ii, jj, kk] > 0:
                        Ez[ii, jj, kk] = (
                            Ez[ii, jj, kk]
                            - self.C3 * self.Jz[ii, jj, kk]
                            + self.C5 * (Hy[ii, jj, kk] - Hy[ii - 1, jj, kk])
                            - self.C4 * (Hx[ii, jj, kk] - Hx[ii, jj - 1, kk])
                        )
