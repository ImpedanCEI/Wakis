import numpy as np
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0


class PmlBlock2D:
    def __init__(
        self,
        Nx,
        Ny,
        dt,
        dx,
        dy,
        lx_block=None,
        ly_block=None,
        rx_block=None,
        ry_block=None,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.dt = dt
        self.dx = dx
        self.dy = dy

        self.Ex = np.zeros((Nx, Ny + 1))
        self.Ey = np.zeros((Nx + 1, Ny))
        self.Exy = np.zeros((Nx, Ny + 1))
        self.Exz = np.zeros((Nx, Ny + 1))
        self.Eyx = np.zeros((Nx + 1, Ny))
        self.Eyz = np.zeros((Nx + 1, Ny))
        self.Hz = np.zeros((Nx, Ny))
        self.Hzx = np.zeros((Nx, Ny))
        self.Hzy = np.zeros((Nx, Ny))
        self.Jx = np.zeros((self.Nx, self.Ny + 1))
        self.Jy = np.zeros((self.Nx + 1, self.Ny))

        # the sigmas must be assembled by the solver
        self.sigma_x = np.zeros_like(self.Ex)
        self.sigma_y = np.zeros_like(self.Ey)
        self.sigma_z = np.zeros_like(self.Ex)
        self.sigma_star_x = np.zeros_like(self.Hz)
        self.sigma_star_y = np.zeros_like(self.Hz)
        self.sigma_star_z = np.zeros_like(self.Hz)

        # we assemble these after the sigmas
        self.Ax = np.zeros_like(self.sigma_x)
        self.Ay = np.zeros_like(self.sigma_y)
        self.Az = np.zeros_like(self.sigma_z)
        self.Bx = np.zeros_like(self.sigma_x)
        self.By = np.zeros_like(self.sigma_y)
        self.Bz = np.zeros_like(self.sigma_z)
        self.Cx = np.zeros_like(self.sigma_star_x)
        self.Cy = np.zeros_like(self.sigma_star_y)
        self.Cz = np.zeros_like(self.sigma_star_z)
        self.Dx = np.zeros_like(self.sigma_star_x)
        self.Dy = np.zeros_like(self.sigma_star_y)
        self.Dz = np.zeros_like(self.sigma_star_z)

        self.lx_block = lx_block
        self.ly_block = ly_block
        self.rx_block = rx_block
        self.ry_block = ry_block

        self.C1 = self.dt / (self.dx * mu_0)
        self.C2 = self.dt / (self.dy * mu_0)
        self.C4 = self.dt / (self.dy * eps_0)
        self.C5 = self.dt / (self.dx * eps_0)
        self.C3 = self.dt / eps_0
        self.C6 = self.dt / eps_0

    def assemble_coeffs(self):
        self.Ax = (2 * eps_0 - self.dt * self.sigma_x) / (
            2 * eps_0 + self.dt * self.sigma_x
        )
        self.Ay = (2 * eps_0 - self.dt * self.sigma_y) / (
            2 * eps_0 + self.dt * self.sigma_y
        )
        self.Az = (2 * eps_0 - self.dt * self.sigma_z) / (
            2 * eps_0 + self.dt * self.sigma_z
        )
        self.Bx = 2 * self.dt / (2 * eps_0 + self.dt * self.sigma_x)
        self.By = 2 * self.dt / (2 * eps_0 + self.dt * self.sigma_y)
        self.Bz = 2 * self.dt / (2 * eps_0 + self.dt * self.sigma_z)
        self.Cx = (2 * mu_0 - self.dt * self.sigma_star_x) / (
            2 * mu_0 + self.dt * self.sigma_star_x
        )
        self.Cy = (2 * mu_0 - self.dt * self.sigma_star_y) / (
            2 * mu_0 + self.dt * self.sigma_star_y
        )
        self.Cz = (2 * mu_0 - self.dt * self.sigma_star_z) / (
            2 * mu_0 + self.dt * self.sigma_star_z
        )
        self.Dx = 2 * self.dt / (2 * mu_0 + self.dt * self.sigma_star_x)
        self.Dy = 2 * self.dt / (2 * mu_0 + self.dt * self.sigma_star_y)
        self.Dz = 2 * self.dt / (2 * mu_0 + self.dt * self.sigma_star_z)

    def advance_h_fdtd(self):
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                self.Hzx[ii, jj] = self.Cx[ii, jj] * self.Hzx[ii, jj] - self.Dx[
                    ii, jj
                ] / self.dx * (self.Ey[ii + 1, jj] - self.Ey[ii, jj])
                self.Hzy[ii, jj] = self.Cy[ii, jj] * self.Hzy[ii, jj] + self.Dy[
                    ii, jj
                ] / self.dy * (self.Ex[ii, jj + 1] - self.Ex[ii, jj])

        self.Hz = self.Hzx + self.Hzy

    def advance_e_fdtd(self):
        for ii in range(self.Nx):
            for jj in range(1, self.Ny):
                self.Exy[ii, jj] = self.Ay[ii, jj] * self.Exy[ii, jj] + self.By[
                    ii, jj
                ] / self.dy * (self.Hz[ii, jj] - self.Hz[ii, jj - 1])
                self.Exz[ii, jj] = self.Az[ii, jj] * self.Exz[ii, jj]

        for ii in range(1, self.Nx):
            for jj in range(self.Ny):
                self.Eyz[ii, jj] = self.Az[ii, jj] * self.Eyz[ii, jj]
                self.Eyx[ii, jj] = self.Ax[ii, jj] * self.Eyx[ii, jj] - self.Bx[
                    ii, jj
                ] / self.dx * (self.Hz[ii, jj] - self.Hz[ii - 1, jj])

        self.Ex = self.Exy + self.Exz
        self.Ey = self.Eyx + self.Eyz

    def update_e_boundary(self):
        if self.lx_block is not None:
            for jj in range(self.Ny):
                self.Eyz[0, jj] = self.Az[0, jj] * self.Eyz[0, jj]
                self.Eyx[0, jj] = self.Ax[0, jj] * self.Eyx[0, jj] - self.Bx[
                    0, jj
                ] / self.dx * (self.Hz[0, jj] - self.lx_block.Hz[-1, jj])

        if self.rx_block is not None:
            for jj in range(self.Ny):
                self.Eyz[-1, jj] = self.Az[-1, jj] * self.Eyz[-1, jj]
                self.Eyx[-1, jj] = self.Ax[-1, jj] * self.Eyx[-1, jj] - self.Bx[
                    -1, jj
                ] / self.dx * (self.rx_block.Hz[0, jj] - self.Hz[-1, jj])

        self.Ey = self.Eyx + self.Eyz

        if self.ly_block is not None:
            for ii in range(self.Nx):
                self.Exy[ii, 0] = self.Ay[ii, 0] * self.Exy[ii, 0] + self.By[
                    ii, 0
                ] / self.dy * (self.Hz[ii, 0] - self.ly_block.Hz[ii, -1])
                self.Exz[ii, 0] = self.Az[ii, 0] * self.Exz[ii, 0]

        if self.ry_block is not None:
            for ii in range(self.Nx):
                self.Exy[ii, -1] = self.Ay[ii, -1] * self.Exy[ii, -1] + self.By[
                    ii, -1
                ] / self.dy * (self.ry_block.Hz[ii, 0] - self.Hz[ii, -1])
                self.Exz[ii, -1] = self.Az[ii, -1] * self.Exz[ii, -1]

        self.Ex = self.Exy + self.Exz
