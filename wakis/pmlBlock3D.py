import numpy as np
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0


class PmlBlock3D:
    def __init__(
        self,
        Nx,
        Ny,
        Nz,
        dt,
        dx,
        dy,
        dz,
        i_block,
        j_block,
        k_block,
        lx_block=None,
        ly_block=None,
        lz_block=None,
        rx_block=None,
        ry_block=None,
        rz_block=None,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.i_block = i_block
        self.j_block = j_block
        self.k_block = k_block

        # Solver3D constructor takes care of populating this matrix (should we ensure this somehow?)
        self.blocks_mat = np.full((3, 3, 3), None)

        self.Ex = np.zeros((Nx, Ny + 1, Nz + 1))
        self.Ey = np.zeros((Nx + 1, Ny, Nz + 1))
        self.Ez = np.zeros((Nx + 1, Ny + 1, Nz))
        self.Exy = np.zeros((Nx, Ny + 1, Nz + 1))
        self.Exz = np.zeros((Nx, Ny + 1, Nz + 1))
        self.Eyx = np.zeros((Nx + 1, Ny, Nz + 1))
        self.Eyz = np.zeros((Nx + 1, Ny, Nz + 1))
        self.Ezx = np.zeros((Nx + 1, Ny + 1, Nz))
        self.Ezy = np.zeros((Nx + 1, Ny + 1, Nz))
        self.Hx = np.zeros((Nx + 1, Ny, Nz))
        self.Hy = np.zeros((Nx, Ny + 1, Nz))
        self.Hz = np.zeros((Nx, Ny, Nz + 1))
        self.Hzx = np.zeros((Nx, Ny, Nz + 1))
        self.Hzy = np.zeros((Nx, Ny, Nz + 1))
        self.Hxy = np.zeros((Nx + 1, Ny, Nz))
        self.Hxz = np.zeros((Nx + 1, Ny, Nz))
        self.Hyx = np.zeros((Nx, Ny + 1, Nz))
        self.Hyz = np.zeros((Nx, Ny + 1, Nz))
        self.Jx = np.zeros((self.Nx, self.Ny + 1, self.Nz + 1))
        self.Jy = np.zeros((self.Nx + 1, self.Ny, self.Nz + 1))
        self.Jz = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz))

        # the sigmas must be assembled by the solver
        self.sigma_x = np.zeros((Nx + 1, Ny + 1, Nz + 1))
        self.sigma_y = np.zeros((Nx + 1, Ny + 1, Nz + 1))
        self.sigma_z = np.zeros((Nx + 1, Ny + 1, Nz + 1))
        self.sigma_star_x = np.zeros((Nx, Ny + 1, Nz + 1))
        self.sigma_star_y = np.zeros((Nx + 1, Ny, Nz + 1))
        self.sigma_star_z = np.zeros((Nx + 1, Ny + 1, Nz))

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
        self.lz_block = lz_block
        self.rx_block = rx_block
        self.ry_block = ry_block
        self.rz_block = rz_block

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
                for kk in range(self.Nz + 1):
                    self.Hzx[ii, jj, kk] = self.Cx[ii, jj, kk] * self.Hzx[
                        ii, jj, kk
                    ] - self.Dx[ii, jj, kk] / self.dx * (
                        self.Ey[ii + 1, jj, kk] - self.Ey[ii, jj, kk]
                    )
                    self.Hzy[ii, jj, kk] = self.Cy[ii, jj, kk] * self.Hzy[
                        ii, jj, kk
                    ] + self.Dy[ii, jj, kk] / self.dy * (
                        self.Ex[ii, jj + 1, kk] - self.Ex[ii, jj, kk]
                    )

        for ii in range(self.Nx + 1):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    self.Hxy[ii, jj, kk] = self.Cy[ii, jj, kk] * self.Hxy[
                        ii, jj, kk
                    ] - self.Dy[ii, jj, kk] / self.dy * (
                        self.Ez[ii, jj + 1, kk] - self.Ez[ii, jj, kk]
                    )
                    self.Hxz[ii, jj, kk] = self.Cz[ii, jj, kk] * self.Hxz[
                        ii, jj, kk
                    ] + self.Dz[ii, jj, kk] / self.dz * (
                        self.Ey[ii, jj, kk + 1] - self.Ey[ii, jj, kk]
                    )

        for ii in range(self.Nx):
            for jj in range(self.Ny + 1):
                for kk in range(self.Nz):
                    self.Hyx[ii, jj, kk] = self.Cx[ii, jj, kk] * self.Hyx[
                        ii, jj, kk
                    ] + self.Dx[ii, jj, kk] / self.dx * (
                        self.Ez[ii + 1, jj, kk] - self.Ez[ii, jj, kk]
                    )
                    self.Hyz[ii, jj, kk] = self.Cz[ii, jj, kk] * self.Hyz[
                        ii, jj, kk
                    ] - self.Dz[ii, jj, kk] / self.dz * (
                        self.Ex[ii, jj, kk + 1] - self.Ex[ii, jj, kk]
                    )

    def advance_e_fdtd(self):
        for ii in range(self.Nx):
            for jj in range(1, self.Ny):
                for kk in range(self.Nz + 1):
                    self.Exy[ii, jj, kk] = self.Ay[ii, jj, kk] * self.Exy[
                        ii, jj, kk
                    ] + self.By[ii, jj, kk] / self.dy * (
                        self.Hz[ii, jj, kk] - self.Hz[ii, jj - 1, kk]
                    )

        for ii in range(self.Nx):
            for jj in range(self.Ny + 1):
                for kk in range(1, self.Nz):
                    self.Exz[ii, jj, kk] = self.Az[ii, jj, kk] * self.Exz[
                        ii, jj, kk
                    ] - self.Bz[ii, jj, kk] / self.dz * (
                        self.Hy[ii, jj, kk] - self.Hy[ii, jj, kk - 1]
                    )

        for ii in range(1, self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz + 1):
                    self.Eyx[ii, jj, kk] = self.Ax[ii, jj, kk] * self.Eyx[
                        ii, jj, kk
                    ] - self.Bx[ii, jj, kk] / self.dx * (
                        self.Hz[ii, jj, kk] - self.Hz[ii - 1, jj, kk]
                    )

        for ii in range(self.Nx + 1):
            for jj in range(self.Ny):
                for kk in range(1, self.Nz):
                    self.Eyz[ii, jj, kk] = self.Az[ii, jj, kk] * self.Eyz[
                        ii, jj, kk
                    ] + self.Bz[ii, jj, kk] / self.dz * (
                        self.Hx[ii, jj, kk] - self.Hx[ii, jj, kk - 1]
                    )

        for ii in range(1, self.Nx):
            for jj in range(self.Ny + 1):
                for kk in range(self.Nz):
                    self.Ezx[ii, jj, kk] = self.Ax[ii, jj, kk] * self.Ezx[
                        ii, jj, kk
                    ] + self.Bx[ii, jj, kk] / self.dx * (
                        self.Hy[ii, jj, kk] - self.Hy[ii - 1, jj, kk]
                    )

        for ii in range(self.Nx + 1):
            for jj in range(1, self.Ny):
                for kk in range(self.Nz):
                    self.Ezy[ii, jj, kk] = self.Ay[ii, jj, kk] * self.Ezy[
                        ii, jj, kk
                    ] - self.By[ii, jj, kk] / self.dy * (
                        self.Hx[ii, jj, kk] - self.Hx[ii, jj - 1, kk]
                    )

    def sum_e_fields(self):
        self.Ex = self.Exy + self.Exz
        self.Ey = self.Eyx + self.Eyz
        self.Ez = self.Ezx + self.Ezy

    def sum_h_fields(self):
        self.Hx = self.Hxy + self.Hxz
        self.Hy = self.Hyx + self.Hyz
        self.Hz = self.Hzx + self.Hzy

    def update_e_boundary(self):
        i_block = self.i_block
        j_block = self.j_block
        k_block = self.k_block
        blocks_mat = self.blocks_mat
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        Ax = self.Ax
        Ay = self.Ay
        Az = self.Az
        Bx = self.Bx
        By = self.By
        Bz = self.Bz
        Exy = self.Exy
        Exz = self.Exz
        Eyx = self.Eyx
        Eyz = self.Eyz
        Ezx = self.Ezx
        Ezy = self.Ezy
        Hx = self.Hx
        Hy = self.Hy
        Hz = self.Hz
        dx = self.dx
        dy = self.dy
        dz = self.dz

        # Separate update on edges doesn't seem to be needed as derivatives are taken in one direction per time
        # Update E on "lower" faces
        if i_block > 0 and blocks_mat[i_block - 1, j_block, k_block] is not None:
            for jj in range(Ny):
                for kk in range(Nz + 1):
                    Eyx[0, jj, kk] = Ax[0, jj, kk] * Eyx[0, jj, kk] - Bx[
                        0, jj, kk
                    ] / dx * (
                        Hz[0, jj, kk]
                        - blocks_mat[i_block - 1, j_block, k_block].Hz[-1, jj, kk]
                    )
            for jj in range(Ny + 1):
                for kk in range(Nz):
                    Ezx[0, jj, kk] = Ax[0, jj, kk] * Ezx[0, jj, kk] + Bx[
                        0, jj, kk
                    ] / dx * (
                        Hy[0, jj, kk]
                        - blocks_mat[i_block - 1, j_block, k_block].Hy[-1, jj, kk]
                    )

        if j_block > 0 and blocks_mat[i_block, j_block - 1, k_block] is not None:
            for ii in range(Nx):
                for kk in range(Nz + 1):
                    Exy[ii, 0, kk] = Ay[ii, 0, kk] * Exy[ii, 0, kk] + By[
                        ii, 0, kk
                    ] / dy * (
                        Hz[ii, 0, kk]
                        - blocks_mat[i_block, j_block - 1, k_block].Hz[ii, -1, kk]
                    )
            for ii in range(Nx + 1):
                for kk in range(Nz):
                    Ezy[ii, 0, kk] = Ay[ii, 0, kk] * Ezy[ii, 0, kk] - By[
                        ii, 0, kk
                    ] / dy * (
                        Hx[ii, 0, kk]
                        - blocks_mat[i_block, j_block - 1, k_block].Hx[ii, -1, kk]
                    )

        if k_block > 0 and blocks_mat[i_block, j_block, k_block - 1] is not None:
            for ii in range(Nx + 1):
                for jj in range(Ny):
                    Eyz[ii, jj, 0] = Az[ii, jj, 0] * Eyz[ii, jj, 0] + Bz[
                        ii, jj, 0
                    ] / dz * (
                        Hx[ii, jj, 0]
                        - blocks_mat[i_block, j_block, k_block - 1].Hx[ii, jj, -1]
                    )
            for ii in range(Nx):
                for jj in range(Ny + 1):
                    Exz[ii, jj, 0] = Az[ii, jj, 0] * Exz[ii, jj, 0] - Bz[
                        ii, jj, 0
                    ] / dz * (
                        Hy[ii, jj, 0]
                        - blocks_mat[i_block, j_block, k_block - 1].Hy[ii, jj, -1]
                    )

        # Update E on "upper" faces
        if i_block < 2 and blocks_mat[i_block + 1, j_block, k_block] is not None:
            for jj in range(Ny):
                for kk in range(Nz + 1):
                    Eyx[Nx, jj, kk] = Ax[Nx, jj, kk] * Eyx[Nx, jj, kk] - Bx[
                        Nx, jj, kk
                    ] / dx * (
                        blocks_mat[i_block + 1, j_block, k_block].Hz[0, jj, kk]
                        - Hz[-1, jj, kk]
                    )
            for jj in range(Ny + 1):
                for kk in range(Nz):
                    Ezx[Nx, jj, kk] = Ax[Nx, jj, kk] * Ezx[Nx, jj, kk] + Bx[
                        Nx, jj, kk
                    ] / dx * (
                        blocks_mat[i_block + 1, j_block, k_block].Hy[0, jj, kk]
                        - Hy[-1, jj, kk]
                    )

        if j_block < 2 and blocks_mat[i_block, j_block + 1, k_block] is not None:
            for ii in range(Nx):
                for kk in range(Nz + 1):
                    Exy[ii, Ny, kk] = Ay[ii, Ny, kk] * Exy[ii, Ny, kk] + By[
                        ii, Ny, kk
                    ] / dy * (
                        blocks_mat[i_block, j_block + 1, k_block].Hz[ii, 0, kk]
                        - Hz[ii, -1, kk]
                    )
            for ii in range(Nx + 1):
                for kk in range(Nz):
                    Ezy[ii, Ny, kk] = Ay[ii, Ny, kk] * Ezy[ii, Ny, kk] - By[
                        ii, Ny, kk
                    ] / dy * (
                        blocks_mat[i_block, j_block + 1, k_block].Hx[ii, 0, kk]
                        - Hx[ii, -1, kk]
                    )

        if k_block < 2 and blocks_mat[i_block, j_block, k_block + 1] is not None:
            for ii in range(Nx + 1):
                for jj in range(Ny):
                    Eyz[ii, jj, Nz] = Az[ii, jj, Nz] * Eyz[ii, jj, Nz] + Bz[
                        ii, jj, Nz
                    ] / dz * (
                        blocks_mat[i_block, j_block, k_block + 1].Hx[ii, jj, 0]
                        - Hx[ii, jj, -1]
                    )
            for ii in range(Nx):
                for jj in range(Ny + 1):
                    Exz[ii, jj, Nz] = Az[ii, jj, Nz] * Exz[ii, jj, Nz] - Bz[
                        ii, jj, Nz
                    ] / dz * (
                        blocks_mat[i_block, j_block, k_block + 1].Hy[ii, jj, 0]
                        - Hy[ii, jj, -1]
                    )
