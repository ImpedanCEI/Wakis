import numpy as np
from pmlBlock2D import PmlBlock2D
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0


def eq(a, b, tol=1e-8):
    return abs(a - b) < tol


def neq(a, b, tol=1e-8):
    return not eq(a, b, tol)


class EMSolver2D:
    def __init__(
        self,
        grid,
        sol_type,
        cfln,
        i_s,
        j_s,
        bc_low,
        bc_high,
        N_pml_low=None,
        N_pml_high=None,
    ):
        self.grid = grid
        self.type = type
        self.cfln = cfln

        self.dt = cfln / (c_light * np.sqrt(1 / self.grid.dx**2 + 1 / self.grid.dy**2))
        self.dx = self.grid.dx
        self.dy = self.grid.dy
        self.Nx = self.grid.nx
        self.Ny = self.grid.ny
        self.sol_type = sol_type

        self.N_pml_low = np.zeros(2, dtype=int)
        self.N_pml_high = np.zeros(2, dtype=int)
        self.bc_low = bc_low
        self.bc_high = bc_high

        if bc_low[0] == "pml":
            self.N_pml_low[0] = 10 if N_pml_low is None else N_pml_low[0]
        if bc_low[1] == "pml":
            self.N_pml_low[1] = 10 if N_pml_low is None else N_pml_low[1]
        if bc_high[0] == "pml":
            self.N_pml_high[0] = 10 if N_pml_high is None else N_pml_high[0]
        if bc_high[1] == "pml":
            self.N_pml_high[1] = 10 if N_pml_high is None else N_pml_high[1]

        self.blocks = []
        self.pml_ly = None
        self.pml_lx = None
        self.pml_rx = None
        self.pml_ry = None
        self.pml_lxly = None
        self.pml_rxly = None
        self.pml_lxry = None
        self.pml_rxry = None

        if bc_low[0] == "pml":
            self.pml_lx = PmlBlock2D(
                self.N_pml_low[0], self.Ny, self.dt, self.dx, self.dy
            )
            self.blocks.append(self.pml_lx)
            if bc_low[1] == "pml":
                self.pml_lxly = PmlBlock2D(
                    self.N_pml_low[0], self.N_pml_low[1], self.dt, self.dx, self.dy
                )
                self.blocks.append(self.pml_lxly)
            if bc_high[1] == "pml":
                self.pml_lxry = PmlBlock2D(
                    self.N_pml_low[0], self.N_pml_high[1], self.dt, self.dx, self.dy
                )
                self.blocks.append(self.pml_lxry)

        if bc_high[0] == "pml":
            self.pml_rx = PmlBlock2D(
                self.N_pml_high[0], self.Ny, self.dt, self.dx, self.dy
            )
            self.blocks.append(self.pml_rx)
            if bc_low[1] == "pml":
                self.pml_rxry = PmlBlock2D(
                    self.N_pml_high[0], self.N_pml_high[1], self.dt, self.dx, self.dy
                )
                self.blocks.append(self.pml_rxry)
            if bc_high[1] == "pml":
                self.pml_rxly = PmlBlock2D(
                    self.N_pml_high[0], self.N_pml_low[1], self.dt, self.dx, self.dy
                )
                self.blocks.append(self.pml_rxly)

        if bc_low[1] == "pml":
            self.pml_ly = PmlBlock2D(
                self.Nx, self.N_pml_low[1], self.dt, self.dx, self.dy
            )
            self.blocks.append(self.pml_ly)

        if bc_high[1] == "pml":
            self.pml_ry = PmlBlock2D(
                self.Nx, self.N_pml_high[1], self.dt, self.dx, self.dy
            )
            self.blocks.append(self.pml_ry)

        self.organize_pmls()
        self.alpha_pml = 3
        self.R0_pml = 0.001

        self.assemble_conductivities_pmls()

        self.assemble_coeffs_pmls()

        self.N_tot_x = self.Nx + self.N_pml_low[0] + self.N_pml_high[0]
        self.N_tot_y = self.Ny + self.N_pml_low[0] + self.N_pml_high[0]
        self.sol_type = sol_type

        self.Ex = np.zeros((self.Nx, self.Ny + 1))
        self.Ey = np.zeros((self.Nx + 1, self.Ny))
        self.Hz = np.zeros((self.Nx, self.Ny))
        self.V_new = np.zeros((self.Nx, self.Ny))
        self.Jx = np.zeros((self.Nx, self.Ny + 1))
        self.Jy = np.zeros((self.Nx + 1, self.Ny))

        self.rho = np.zeros((self.Nx, self.Ny))

        if (sol_type != "FDTD") and (sol_type != "DM") and (sol_type != "ECT"):
            raise ValueError(
                "sol_type must be:\n"
                + "\t'FDTD' for standard staircased FDTD\n"
                + "\t'DM' for Dey-Mittra conformal FDTD\n"
                + "\t'ECT' for Enlarged Cell Technique conformal FDTD"
            )

        if sol_type == "DM" or sol_type == "ECT":
            self.Vxy = np.zeros((self.Nx, self.Ny))
        if sol_type == "ECT":
            self.V_enl = np.zeros((self.Nx, self.Ny))

        if sol_type == "ECT" or sol_type == "DM":
            self.C1 = self.dt / mu_0
            self.C4 = self.dt / (eps_0 * self.dy)
            self.C5 = self.dt / (eps_0 * self.dx)
            self.C3 = self.dt / eps_0
            self.C6 = self.dt / eps_0

        if sol_type == "FDTD":
            self.C1 = self.dt / (self.dx * mu_0)
            self.C2 = self.dt / (self.dy * mu_0)
            self.C4 = self.dt / (self.dy * eps_0)
            self.C5 = self.dt / (self.dx * eps_0)
            self.C3 = self.dt / eps_0
            self.C6 = self.dt / eps_0

        # indices for the source
        self.i_s = i_s
        self.j_s = j_s

        self.time = 0

    def organize_pmls(self):
        if self.bc_low[0] == "pml":
            self.pml_lx.rx_block = self
            if self.bc_low[1] == "pml":
                self.pml_lx.ly_block = self.pml_lxly
                self.pml_lxly.ry_block = self.pml_lx
                self.pml_lxly.rx_block = self.pml_ly
            if self.bc_high[1] == "pml":
                self.pml_lx.ry_block = self.pml_lxry
                self.pml_lxry.ly_block = self.pml_lx
                self.pml_lxry.rx_block = self.pml_ry

        if self.bc_high[0] == "pml":
            self.pml_rx.lx_block = self
            if self.bc_high[1] == "pml":
                self.pml_rx.ry_block = self.pml_rxry
                self.pml_rxly.lx_block = self.pml_ly
                self.pml_rxly.ry_block = self.pml_rx
            if self.bc_low[1] == "pml":
                self.pml_rx.ly_block = self.pml_rxly
                self.pml_rxry.lx_block = self.pml_ry
                self.pml_rxry.ly_block = self.pml_rx

        if self.bc_low[1] == "pml":
            self.pml_ly.ry_block = self
            if self.bc_low[0] == "pml":
                self.pml_ly.lx_block = self.pml_lxly
            if self.bc_low[0] == "pml":
                self.pml_ly.rx_block = self.pml_rxly

        if self.bc_high[1] == "pml":
            self.pml_ry.ly_block = self
            if self.bc_low[0] == "pml":
                self.pml_ry.lx_block = self.pml_lxry
            if self.bc_high[0] == "pml":
                self.pml_ry.rx_block = self.pml_rxry

    def assemble_conductivities_pmls(self):
        sigma_m_low_x = 0
        sigma_m_high_x = 0
        sigma_m_low_y = 0
        sigma_m_high_y = 0
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

        if self.bc_low[0] == "pml":
            for n in range(self.N_pml_low[0]):
                self.pml_lx.sigma_x[-(n + 1), :] = (
                    sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                )
            if self.bc_low[1] == "pml":
                for n in range((self.N_pml_low[1])):
                    self.pml_lxly.sigma_y[:, -(n + 1)] = (
                        sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                    )
                for n in range((self.N_pml_low[0])):
                    self.pml_lxly.sigma_x[-(n + 1), :] = (
                        sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                    )
            if self.bc_high[1] == "pml":
                for n in range(self.N_pml_high[1]):
                    self.pml_lxry.sigma_y[:, n] = (
                        sigma_m_high_y * (n / (self.N_pml_high[1])) ** self.alpha_pml
                    )
                for n in range(self.N_pml_low[0]):
                    self.pml_lxry.sigma_x[-(n + 1), :] = (
                        sigma_m_low_x * (n / (self.N_pml_low[0])) ** self.alpha_pml
                    )

        if self.bc_high[0] == "pml":
            for n in range(self.N_pml_high[0]):
                self.pml_rx.sigma_x[n, :] = (
                    sigma_m_high_x * (n / (self.N_pml_high[0])) ** self.alpha_pml
                )
            if self.bc_high[1] == "pml":
                for n in range(self.N_pml_high[0]):
                    self.pml_rxry.sigma_x[n, :] = (
                        sigma_m_high_x * (n / (self.N_pml_high[0])) ** self.alpha_pml
                    )
                for n in range(self.N_pml_high[1]):
                    self.pml_rxry.sigma_y[:, n] = (
                        sigma_m_high_y * (n / (self.N_pml_high[1])) ** self.alpha_pml
                    )
            if self.bc_low[1] == "pml":
                for n in range(self.N_pml_low[1]):
                    self.pml_rxly.sigma_y[:, -(n + 1)] = (
                        sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                    )
                for n in range(self.N_pml_high[0]):
                    self.pml_rxly.sigma_x[n, :] = (
                        sigma_m_high_x * (n / (self.N_pml_high[0])) ** self.alpha_pml
                    )

        if self.bc_low[1] == "pml":
            for n in range(self.N_pml_low[1]):
                self.pml_ly.sigma_y[:, -(n + 1)] = (
                    sigma_m_low_y * (n / (self.N_pml_low[1])) ** self.alpha_pml
                )

        if self.bc_high[1] == "pml":
            for n in range(self.N_pml_high[1]):
                self.pml_ry.sigma_y[:, n] = (
                    sigma_m_high_y * (n / (self.N_pml_high[1])) ** self.alpha_pml
                )

        if self.pml_lx is not None:
            self.pml_lx.sigma_star_x = self.pml_lx.sigma_x * mu_0 / eps_0
            self.pml_lx.sigma_star_y = self.pml_lx.sigma_y * mu_0 / eps_0
        if self.pml_ly is not None:
            self.pml_ly.sigma_star_x = self.pml_ly.sigma_x * mu_0 / eps_0
            self.pml_ly.sigma_star_y = self.pml_ly.sigma_y * mu_0 / eps_0
        if self.pml_rx is not None:
            self.pml_rx.sigma_star_x = self.pml_rx.sigma_x * mu_0 / eps_0
            self.pml_rx.sigma_star_y = self.pml_rx.sigma_y * mu_0 / eps_0
        if self.pml_ry is not None:
            self.pml_ry.sigma_star_x = self.pml_ry.sigma_x * mu_0 / eps_0
            self.pml_ry.sigma_star_y = self.pml_ry.sigma_y * mu_0 / eps_0
        if self.pml_lxly is not None:
            self.pml_lxly.sigma_star_x = self.pml_lxly.sigma_x * mu_0 / eps_0
            self.pml_lxly.sigma_star_y = self.pml_lxly.sigma_y * mu_0 / eps_0
        if self.pml_lxry is not None:
            self.pml_lxry.sigma_star_x = self.pml_lxry.sigma_x * mu_0 / eps_0
            self.pml_lxry.sigma_star_y = self.pml_lxry.sigma_y * mu_0 / eps_0
        if self.pml_rxry is not None:
            self.pml_rxry.sigma_star_x = self.pml_rxry.sigma_x * mu_0 / eps_0
            self.pml_rxry.sigma_star_y = self.pml_rxry.sigma_y * mu_0 / eps_0
        if self.pml_rxly is not None:
            self.pml_rxly.sigma_star_x = self.pml_rxly.sigma_x * mu_0 / eps_0
            self.pml_rxly.sigma_star_y = self.pml_rxly.sigma_y * mu_0 / eps_0

    def assemble_coeffs_pmls(self):
        if self.bc_low[0] == "pml":
            self.pml_lx.assemble_coeffs()
            if self.bc_low[1] == "pml":
                self.pml_lxly.assemble_coeffs()
            if self.bc_high[1] == "pml":
                self.pml_lxry.assemble_coeffs()

        if self.bc_high[0] == "pml":
            self.pml_rx.assemble_coeffs()
            if self.bc_low[0] == "pml":
                self.pml_rxry.assemble_coeffs()
            if self.bc_high[1] == "pml":
                self.pml_rxly.assemble_coeffs()

        if self.bc_low[1] == "pml":
            self.pml_ly.assemble_coeffs()

        if self.bc_high[1] == "pml":
            self.pml_ry.assemble_coeffs()

    def update_e_boundary(self):
        Ex = self.Ex
        Ey = self.Ey
        Hz = self.Hz
        if self.pml_lx is not None:
            for jj in range(self.Ny):
                Ey[0, jj] = (
                    Ey[0, jj]
                    - self.C3 * self.Jy[0, jj]
                    - self.C5 * (Hz[0, jj] - self.pml_lx.Hz[-1, jj])
                )

        if self.pml_rx is not None:
            for jj in range(self.Ny):
                Ey[-1, jj] = (
                    Ey[-1, jj]
                    - self.C3 * self.Jy[-1, jj]
                    - self.C5 * (self.pml_rx.Hz[0, jj] - Hz[-1, jj])
                )

        if self.pml_ly is not None:
            for ii in range(self.Nx):
                Ex[ii, 0] = (
                    Ex[ii, 0]
                    - self.C3 * self.Jx[ii, 0]
                    + self.C4 * (Hz[ii, 0] - self.pml_ly.Hz[ii, -1])
                )

        if self.pml_ry is not None:
            for ii in range(self.Nx):
                Ex[ii, -1] = (
                    Ex[ii, -1]
                    - self.C3 * self.Jx[ii, -1]
                    + self.C4 * (self.pml_ry.Hz[ii, 0] - Hz[ii, -1])
                )

        for block in self.blocks:
            block.update_e_boundary()

    def gauss(self, t):
        tau = 10 * self.dt
        if t < 6 * tau:
            return 100 * np.exp(-((t - 3 * tau) ** 2) / tau**2)
        else:
            return 0.0

    def one_step(self):
        if self.sol_type == "ECT":
            self.compute_v_and_rho()
            self.one_step_ect(
                Nx=self.Nx,
                Ny=self.Ny,
                V_enl=self.V_enl,
                rho=self.rho,
                Hz=self.Hz,
                C1=self.C1,
                flag_int_cell=self.grid.flag_int_cell,
                flag_unst_cell=self.grid.flag_unst_cell,
                flag_intr_cell=self.grid.flag_intr_cell,
                S=self.grid.S,
                borrowing=self.grid.borrowing,
                S_enl=self.grid.S_enl,
                S_red=self.grid.S_red,
                V_new=self.V_new,
                dt=self.dt,
            )
            for block in self.blocks:
                block.advance_h_fdtd()

            self.advance_e_dm()

            for block in self.blocks:
                block.advance_e_fdtd()

            self.update_e_boundary()

            self.time += self.dt

        if self.sol_type == "FDTD":
            self.one_step_fdtd()
        if self.sol_type == "DM":
            self.one_step_dm()

    def one_step_fdtd(self):
        self.advance_h_fdtd()
        for block in self.blocks:
            block.advance_h_fdtd()
        self.advance_e_fdtd()
        for block in self.blocks:
            block.advance_e_fdtd()
        self.update_e_boundary()

        self.time += self.dt

    def one_step_dm(self):
        self.compute_v_and_rho()

        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.grid.flag_int_cell[i, j]:
                    self.Hz[i, j] = (
                        self.Hz[i, j]
                        - self.dt / (mu_0 * self.grid.S[i, j]) * self.Vxy[i, j]
                    )

        for block in self.blocks:
            block.advance_h_fdtd()

        self.advance_e_dm()

        for block in self.blocks:
            block.advance_e_fdtd()

        self.update_e_boundary_dm()

        self.time += self.dt

    def advance_h_fdtd(self):
        Ex = self.Ex
        Ey = self.Ey
        Hz = self.Hz
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                if self.grid.flag_int_cell[ii, jj]:
                    Hz[ii, jj] = (
                        Hz[ii, jj]
                        - self.C1 * (Ey[ii + 1, jj] - Ey[ii, jj])
                        + self.C2 * (Ex[ii, jj + 1] - Ex[ii, jj])
                    )

    def advance_e_fdtd(self):
        Ex = self.Ex
        Ey = self.Ey
        Hz = self.Hz
        for ii in range(self.Nx):
            for jj in range(1, self.Ny):
                if self.grid.flag_int_cell[ii, jj]:
                    if self.grid.l_x[ii, jj] > 0:
                        Ex[ii, jj] = (
                            Ex[ii, jj]
                            - self.C3 * self.Jx[ii, jj]
                            + self.C4 * (Hz[ii, jj] - Hz[ii, jj - 1])
                        )

        for ii in range(1, self.Nx):
            for jj in range(self.Ny):
                if self.grid.flag_int_cell[ii, jj]:
                    if self.grid.l_y[ii, jj] > 0:
                        Ey[ii, jj] = (
                            Ey[ii, jj]
                            - self.C3 * self.Jy[ii, jj]
                            - self.C5 * (Hz[ii, jj] - Hz[ii - 1, jj])
                        )

    @staticmethod
    def one_step_ect(
        Nx=None,
        Ny=None,
        V_enl=None,
        rho=None,
        Hz=None,
        C1=None,
        flag_int_cell=None,
        flag_unst_cell=None,
        flag_intr_cell=None,
        S=None,
        borrowing=None,
        S_enl=None,
        S_red=None,
        V_new=None,
        dt=None,
        comp=None,
        kk=None,
    ):
        # if dt==None: dt = self.dt

        V_enl = np.zeros((Nx, Ny))

        # take care of unstable cells
        for ii in range(Nx):
            for jj in range(Ny):
                if flag_int_cell[ii, jj] and flag_unst_cell[ii, jj]:
                    V_enl[ii, jj] = rho[ii, jj] * S[ii, jj]

                    if len(borrowing[ii, jj]) == 0:
                        print("error in one_step_ect")
                    for ip, jp, patch, _ in borrowing[ii, jj]:
                        V_enl[ii, jj] += rho[ip, jp] * patch

                    rho_enl = V_enl[ii, jj] / S_enl[ii, jj]

                    # communicate to the intruded cell the intruding rho
                    for ip, jp, patch, _ in borrowing[ii, jj]:
                        V_enl[ip, jp] += rho_enl * patch

                    Hz[ii, jj] = Hz[ii, jj] - dt / mu_0 * rho_enl

        # take care of stable cells
        for ii in range(Nx):
            for jj in range(Ny):
                if flag_int_cell[ii, jj] and not flag_unst_cell[ii, jj]:
                    # stable cell which hasn't been intruded
                    if not flag_intr_cell[ii, jj]:
                        Hz[ii, jj] = Hz[ii, jj] - dt / mu_0 * rho[ii, jj]
                    # stable cell which has been intruded
                    else:
                        V_enl[ii, jj] += rho[ii, jj] * S_red[ii, jj]
                        Hz[ii, jj] = Hz[ii, jj] - dt / mu_0 * V_enl[ii, jj] / S[ii, jj]

    def compute_v_and_rho(self):
        l_y = self.grid.l_y
        l_x = self.grid.l_x
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                if self.grid.flag_int_cell[ii, jj]:
                    self.Vxy[ii, jj] = (
                        self.Ey[ii + 1, jj] * l_y[ii + 1, jj]
                        - self.Ey[ii, jj] * l_y[ii, jj]
                        - self.Ex[ii, jj + 1] * l_x[ii, jj + 1]
                        + self.Ex[ii, jj] * l_x[ii, jj]
                    )
                    if self.sol_type != "DM":
                        self.rho[ii, jj] = self.Vxy[ii, jj] / self.grid.S[ii, jj]

    def advance_e_dm(self):
        for ii in range(self.Nx):
            for jj in range(1, self.Ny):
                if self.grid.l_x[ii, jj] > 0:
                    self.Ex[ii, jj] = (
                        self.Ex[ii, jj]
                        + self.dt
                        / (eps_0 * self.dy)
                        * (self.Hz[ii, jj] - self.Hz[ii, jj - 1])
                        - self.C3 * self.Jx[ii, jj]
                    )

        for ii in range(1, self.Nx):
            for jj in range(self.Ny):
                if self.grid.l_y[ii, jj] > 0:
                    self.Ey[ii, jj] = (
                        self.Ey[ii, jj]
                        - self.dt
                        / (eps_0 * self.dx)
                        * (self.Hz[ii, jj] - self.Hz[ii - 1, jj])
                        - self.C3 * self.Jy[ii, jj]
                    )

    def update_e_boundary_dm(self):
        Ex = self.Ex
        Ey = self.Ey
        Hz = self.Hz
        if self.pml_lx is not None:
            for jj in range(self.Ny):
                Ey[0, jj] = (
                    Ey[0, jj]
                    - self.C3 * self.Jy[0, jj]
                    - self.dt / (eps_0 * self.dy) * (Hz[0, jj] - self.pml_lx.Hz[-1, jj])
                )

        if self.pml_rx is not None:
            for jj in range(self.Ny):
                Ey[-1, jj] = (
                    Ey[-1, jj]
                    - self.C3 * self.Jy[-1, jj]
                    - self.dt / (eps_0 * self.dy) * (self.pml_rx.Hz[0, jj] - Hz[-1, jj])
                )

        if self.pml_ly is not None:
            for ii in range(self.Nx):
                Ex[ii, 0] = (
                    Ex[ii, 0]
                    - self.C3 * self.Jx[ii, 0]
                    + self.dt / (eps_0 * self.dy) * (Hz[ii, 0] - self.pml_ly.Hz[ii, -1])
                )

        if self.pml_ry is not None:
            for ii in range(self.Nx):
                Ex[ii, -1] = (
                    Ex[ii, -1]
                    - self.C3 * self.Jx[ii, -1]
                    + self.dt / (eps_0 * self.dy) * (self.pml_ry.Hz[ii, 0] - Hz[ii, -1])
                )

        for block in self.blocks:
            block.update_e_boundary()
