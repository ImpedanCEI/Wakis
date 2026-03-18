import numpy as np
from field import Field
from grid2D import Grid2D
from grid2D import compute_areas as compute_areas_2D
from grid2D import mark_cells as mark_cells_2D
from numba import jit


def seg_length(x_1, y_1, z_1, x_2, y_2, z_2):
    return np.linalg.norm(np.array([x_1 - x_2, y_1 - y_2, z_1 - z_2]))


def eq(a, b, tol=1e-8):
    return abs(a - b) < tol


def neq(a, b, tol=1e-8):
    return not eq(a, b, tol)


class Grid3D:
    """
    Class holding the grid info and the routines for cell extensions.
      Constructor arguments:
          - xmin, xmax, ymin, ymax, zmin, zmax: extent of the domain.
          - nx, ny, nz: number of cells per direction
          - conductors: conductor object
          - sol_type: type of solver. 'FDTD' for staircased FDTD, 'DM' for Conformal Dey-Mittra FDTD,
                      'ECT' for Enlarged Cell Technique conformal FDTD
    """

    def __init__(
        self, xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz, conductors, sol_type
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = (xmax - xmin) / nx
        self.dy = (ymax - ymin) / ny
        self.dz = (ymax - ymin) / nz
        self.conductors = conductors

        self.l_x = np.zeros((nx, ny + 1, nz + 1))
        self.l_y = np.zeros((nx + 1, ny, nz + 1))
        self.l_z = np.zeros((nx + 1, ny + 1, nz))
        self.Sxy = np.zeros((nx, ny, nz + 1))
        self.Syz = np.zeros((nx + 1, ny, nz))
        self.Szx = np.zeros((nx, ny + 1, nz))
        self.Sxy_stab = np.zeros_like(self.Sxy)
        self.Sxy_enl = np.zeros_like(self.Sxy)
        self.Sxy_red = np.zeros_like(self.Sxy)
        self.Syz_stab = np.zeros_like(self.Syz)
        self.Syz_enl = np.zeros_like(self.Syz)
        self.Syz_red = np.zeros_like(self.Syz)
        self.Szx_stab = np.zeros_like(self.Szx)
        self.Szx_enl = np.zeros_like(self.Szx)
        self.Szx_red = np.zeros_like(self.Szx)
        self.flag_unst_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_intr_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_int_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_bound_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_avail_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_ext_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_unst_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_intr_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_int_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_bound_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_avail_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_ext_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_unst_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.flag_intr_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.flag_int_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.flag_bound_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.flag_avail_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.flag_ext_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.broken_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.broken_yz = np.zeros_like(self.Syz, dtype=bool)
        self.broken_zx = np.zeros_like(self.Szx, dtype=bool)

        if (
            (sol_type != "FDTD")
            and (sol_type != "DM")
            and (sol_type != "ECT")
            and (sol_type != "FIT")
        ):
            raise ValueError(
                "sol_type must be:\n"
                + "\t'FDTD' for standard staircased FDTD\n"
                + "\t'DM' for Dey-Mittra conformal FDTD\n"
                + "\t'ECT' for Enlarged Cell Technique conformal FDTD"
            )

        self.compute_edges()
        if sol_type == "DM" or sol_type == "FDTD":  # or sol_type is 'FIT':
            self.compute_areas(
                self.l_x,
                self.l_y,
                self.l_z,
                self.Sxy,
                self.Syz,
                self.Szx,
                self.Sxy_red,
                self.Syz_red,
                self.Szx_red,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.dy,
                self.dz,
            )
            self.mark_cells(
                self.l_x,
                self.l_y,
                self.l_z,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.dy,
                self.dz,
                self.Sxy,
                self.Syz,
                self.Szx,
                self.flag_int_cell_xy,
                self.flag_int_cell_yz,
                self.flag_int_cell_zx,
                self.Sxy_stab,
                self.Syz_stab,
                self.Szx_stab,
                self.flag_unst_cell_xy,
                self.flag_unst_cell_yz,
                self.flag_unst_cell_zx,
                self.flag_bound_cell_xy,
                self.flag_bound_cell_yz,
                self.flag_bound_cell_zx,
                self.flag_avail_cell_xy,
                self.flag_avail_cell_yz,
                self.flag_avail_cell_zx,
            )
        elif sol_type == "ECT":
            self.compute_areas(
                self.l_x,
                self.l_y,
                self.l_z,
                self.Sxy,
                self.Syz,
                self.Szx,
                self.Sxy_red,
                self.Syz_red,
                self.Szx_red,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.dy,
                self.dz,
            )
            self.mark_cells(
                self.l_x,
                self.l_y,
                self.l_z,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.dy,
                self.dz,
                self.Sxy,
                self.Syz,
                self.Szx,
                self.flag_int_cell_xy,
                self.flag_int_cell_yz,
                self.flag_int_cell_zx,
                self.Sxy_stab,
                self.Syz_stab,
                self.Szx_stab,
                self.flag_unst_cell_xy,
                self.flag_unst_cell_yz,
                self.flag_unst_cell_zx,
                self.flag_bound_cell_xy,
                self.flag_bound_cell_yz,
                self.flag_bound_cell_zx,
                self.flag_avail_cell_xy,
                self.flag_avail_cell_yz,
                self.flag_avail_cell_zx,
            )

            # info about intruded cells (i,j,[(i_borrowing,j_borrowing,area_borrowing, )])
            self.borrowing_xy = np.empty((nx, ny, nz + 1), dtype=object)
            self.borrowing_yz = np.empty((nx + 1, ny, nz), dtype=object)
            self.borrowing_zx = np.empty((nx, ny + 1, nz), dtype=object)

            for ii in range(nx):
                for jj in range(ny):
                    for kk in range(nz + 1):
                        self.borrowing_xy[ii, jj, kk] = []
            for ii in range(nx + 1):
                for jj in range(ny):
                    for kk in range(nz):
                        self.borrowing_yz[ii, jj, kk] = []
            for ii in range(nx):
                for jj in range(ny + 1):
                    for kk in range(nz):
                        self.borrowing_zx[ii, jj, kk] = []

            self.flag_ext_cell_xy = self.flag_unst_cell_xy.copy()
            self.flag_ext_cell_yz = self.flag_unst_cell_yz.copy()
            self.flag_ext_cell_zx = self.flag_unst_cell_zx.copy()
            self.mark_cells(
                self.l_x,
                self.l_y,
                self.l_z,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.dy,
                self.dz,
                self.Sxy,
                self.Syz,
                self.Szx,
                self.flag_int_cell_xy,
                self.flag_int_cell_yz,
                self.flag_int_cell_zx,
                self.Sxy_stab,
                self.Syz_stab,
                self.Szx_stab,
                self.flag_unst_cell_xy,
                self.flag_unst_cell_yz,
                self.flag_unst_cell_zx,
                self.flag_bound_cell_xy,
                self.flag_bound_cell_yz,
                self.flag_bound_cell_zx,
                self.flag_avail_cell_xy,
                self.flag_avail_cell_yz,
                self.flag_avail_cell_zx,
            )
            self.compute_extensions()

        elif sol_type == "FIT":
            # primal Grid G
            self.x = np.linspace(self.xmin, self.xmax, self.nx + 1)
            self.y = np.linspace(self.ymin, self.ymax, self.ny + 1)
            self.z = np.linspace(self.zmin, self.zmax, self.nz + 1)

            Y, X, Z = np.meshgrid(self.y, self.x, self.z)

            self.L = Field(self.nx, self.ny, self.nz)
            self.L.field_x = X[1:, 1:, 1:] - X[:-1, :-1, :-1]
            self.L.field_y = Y[1:, 1:, 1:] - Y[:-1, :-1, :-1]
            self.L.field_z = Z[1:, 1:, 1:] - Z[:-1, :-1, :-1]

            self.iA = Field(self.nx, self.ny, self.nz)
            self.iA.field_x = np.divide(1.0, self.L.field_y * self.L.field_z)
            self.iA.field_y = np.divide(1.0, self.L.field_x * self.L.field_z)
            self.iA.field_z = np.divide(1.0, self.L.field_x * self.L.field_y)

            # tilde grid ~G
            # self.itA = self.iA
            # self.tL = self.L

            self.tx = (self.x[1:] + self.x[:-1]) / 2
            self.ty = (self.y[1:] + self.y[:-1]) / 2
            self.tz = (self.z[1:] + self.z[:-1]) / 2

            self.tx = np.append(self.tx, self.tx[-1])
            self.ty = np.append(self.ty, self.ty[-1])
            self.tz = np.append(self.tz, self.tz[-1])

            tY, tX, tZ = np.meshgrid(self.ty, self.tx, self.tz)

            self.tL = Field(self.nx, self.ny, self.nz)
            self.tL.field_x = tX[1:, 1:, 1:] - tX[:-1, :-1, :-1]
            self.tL.field_y = tY[1:, 1:, 1:] - tY[:-1, :-1, :-1]
            self.tL.field_z = tZ[1:, 1:, 1:] - tZ[:-1, :-1, :-1]

            self.itA = Field(self.nx, self.ny, self.nz)
            aux = self.tL.field_y * self.tL.field_z
            self.itA.field_x = np.divide(
                1.0, aux, out=np.zeros_like(aux), where=aux != 0
            )
            aux = self.tL.field_x * self.tL.field_z
            self.itA.field_y = np.divide(
                1.0, aux, out=np.zeros_like(aux), where=aux != 0
            )
            aux = self.tL.field_x * self.tL.field_y
            self.itA.field_z = np.divide(
                1.0, aux, out=np.zeros_like(aux), where=aux != 0
            )
            del aux

            self.compute_areas(
                self.l_x,
                self.l_y,
                self.l_z,
                self.Sxy,
                self.Syz,
                self.Szx,
                self.Sxy_red,
                self.Syz_red,
                self.Szx_red,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.dy,
                self.dz,
            )
            self.mark_cells(
                self.l_x,
                self.l_y,
                self.l_z,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                self.dy,
                self.dz,
                self.Sxy,
                self.Syz,
                self.Szx,
                self.flag_int_cell_xy,
                self.flag_int_cell_yz,
                self.flag_int_cell_zx,
                self.Sxy_stab,
                self.Syz_stab,
                self.Szx_stab,
                self.flag_unst_cell_xy,
                self.flag_unst_cell_yz,
                self.flag_unst_cell_zx,
                self.flag_bound_cell_xy,
                self.flag_bound_cell_yz,
                self.flag_bound_cell_zx,
                self.flag_avail_cell_xy,
                self.flag_avail_cell_yz,
                self.flag_avail_cell_zx,
            )

    """
  Function to compute the length of the edges of the conformal grid.
    Inputs:
        - tol: an edge shorter than tol will be considered as of zero length
    """

    def compute_edges(self, tol=1e-8):
        # edges xy
        for kk in range(self.nz + 1):
            for ii in range(self.nx):
                for jj in range(self.ny + 1):
                    x_1 = ii * self.dx + self.xmin
                    y_1 = jj * self.dy + self.ymin
                    x_2 = (ii + 1) * self.dx + self.xmin
                    y_2 = jj * self.dy + self.ymin
                    z = kk * self.dz + self.zmin
                    # if point 1 is in conductor
                    if self.conductors.in_conductor(x_1, y_1, z):
                        # if point 2 is in conductor, length of the l_x[i, j] is zero
                        if self.conductors.in_conductor(x_2, y_2, z):
                            self.l_x[ii, jj, kk] = 0
                        # if point 2 is not in conductor, length of l_x[i, j]
                        # is the fractional length
                        else:
                            self.l_x[ii, jj, kk] = seg_length(
                                self.conductors.intersec_x(x_2, y_2, z),
                                y_2,
                                z,
                                x_2,
                                y_2,
                                z,
                            )
                    # if point 1 is not in conductor
                    else:
                        # if point 2 is in conductor, length of l_x[i, j] is the fractional length
                        if self.conductors.in_conductor(x_2, y_2, z):
                            self.l_x[ii, jj, kk] = seg_length(
                                x_1,
                                y_1,
                                z,
                                self.conductors.intersec_x(x_1, y_1, z),
                                y_1,
                                z,
                            )
                        # if point 2 is not in conductor, length of l_x[i, j] is dx
                        else:
                            self.l_x[ii, jj, kk] = self.dx

        for kk in range(self.nz + 1):
            for ii in range(self.nx + 1):
                for jj in range(self.ny):
                    x_1 = ii * self.dx + self.xmin
                    y_1 = jj * self.dy + self.ymin
                    x_3 = ii * self.dx + self.xmin
                    y_3 = (jj + 1) * self.dy + self.ymin
                    z = kk * self.dz + self.zmin
                    # if point 1 is in conductor
                    if self.conductors.in_conductor(x_1, y_1, z):
                        # if point 3 to the right is in conductor, length of the l_y[i, j] is zero
                        if self.conductors.in_conductor(x_3, y_3, z):
                            self.l_y[ii, jj, kk] = 0
                        # if point 3 is not in conductor, length of l_y[i, j]
                        # is the fractional length
                        else:
                            self.l_y[ii, jj, kk] = seg_length(
                                x_3,
                                self.conductors.intersec_y(x_3, y_3, z),
                                z,
                                x_3,
                                y_3,
                                z,
                            )
                    # if point 1 is not in conductor
                    else:
                        # if point 3 is in conductor, length of the l_y[i, j]
                        # is the fractional length
                        if self.conductors.in_conductor(x_3, y_3, z):
                            self.l_y[ii, jj, kk] = seg_length(
                                x_1,
                                y_1,
                                z,
                                x_1,
                                self.conductors.intersec_y(x_1, y_1, z),
                                z,
                            )
                        # if point 3 is not in conductor, length of l_y[i, j] is dy
                        else:
                            self.l_y[ii, jj, kk] = self.dy

        for jj in range(self.ny + 1):
            for ii in range(self.nx + 1):
                for kk in range(self.nz):
                    y = jj * self.dy + self.ymin
                    x_1 = ii * self.dx + self.xmin
                    z_1 = kk * self.dz + self.zmin
                    x_3 = ii * self.dx + self.xmin
                    z_3 = (kk + 1) * self.dz + self.zmin
                    # if point 1 is in conductor
                    if self.conductors.in_conductor(x_1, y, z_1):
                        # if point 3 to the right is in conductor, length of the l_y[i, j] is zero
                        if self.conductors.in_conductor(x_3, y, z_3):
                            self.l_z[ii, jj, kk] = 0
                        # if point 3 is not in conductor, length of l_y[i, j]
                        # is the fractional length
                        else:
                            self.l_z[ii, jj, kk] = seg_length(
                                x_3,
                                y,
                                self.conductors.intersec_z(x_3, y, z_3),
                                x_3,
                                y,
                                z_3,
                            )
                    # if point 1 is not in conductor
                    else:
                        # if point 3 is in conductor, length of the l_y[i, j]
                        # is the fractional length
                        if self.conductors.in_conductor(x_3, y, z_3):
                            self.l_z[ii, jj, kk] = seg_length(
                                x_1,
                                y,
                                z_1,
                                x_1,
                                y,
                                self.conductors.intersec_z(x_1, y, z_1),
                            )
                        # if point 3 is not in conductor, length of l_y[i, j] is dy
                        else:
                            self.l_z[ii, jj, kk] = self.dz

        # set to zero the length of very small cells
        if tol > 0.0:
            self.l_x /= self.dx
            self.l_y /= self.dy
            self.l_z /= self.dz

            low_values_flags = abs(self.l_x) < tol
            high_values_flags = abs(self.l_x - 1.0) < tol
            self.l_x[low_values_flags] = 0
            self.l_x[high_values_flags] = 1.0

            low_values_flags = abs(self.l_y) < tol
            high_values_flags = abs(self.l_y - 1.0) < tol
            self.l_y[low_values_flags] = 0
            self.l_y[high_values_flags] = 1.0

            low_values_flags = abs(self.l_z) < tol
            high_values_flags = abs(self.l_z - 1.0) < tol
            self.l_z[low_values_flags] = 0
            self.l_z[high_values_flags] = 1.0

            self.l_x *= self.dx
            self.l_y *= self.dy
            self.l_z *= self.dz

    """
  Function to compute the area of the cells of the conformal grid.
    """

    @staticmethod
    @jit(
        "(f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], i4, i4, i4, f8, f8, f8)",
        nopython=True,
    )
    def compute_areas(
        l_x, l_y, l_z, Sxy, Syz, Szx, Sxy_red, Syz_red, Szx_red, nx, ny, nz, dx, dy, dz
    ):
        for kk in range(nz + 1):
            compute_areas_2D(
                l_x[:, :, kk],
                l_y[:, :, kk],
                Sxy[:, :, kk],
                Sxy_red[:, :, kk],
                nx,
                ny,
                dx,
                dy,
            )

        for ii in range(nx + 1):
            compute_areas_2D(
                l_y[ii, :, :],
                l_z[ii, :, :],
                Syz[ii, :, :],
                Syz_red[ii, :, :],
                ny,
                nz,
                dy,
                dz,
            )

        for jj in range(ny + 1):
            compute_areas_2D(
                l_x[:, jj, :],
                l_z[:, jj, :],
                Szx[:, jj, :],
                Szx_red[:, jj, :],
                nx,
                nz,
                dx,
                dz,
            )

    """
  Function to mark wich cells are interior (int), require extension (unst),
  are on the boundary(bound), are available for intrusion (avail)
    """

    @staticmethod
    @jit(
        "(f8[:,:,:], f8[:,:,:], f8[:,:,:], i4, i4, i4, f8, f8, f8, f8[:,:,:], f8[:,:,:], f8[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:])",
        nopython=True,
    )
    def mark_cells(
        l_x,
        l_y,
        l_z,
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        Sxy,
        Syz,
        Szx,
        flag_int_cell_xy,
        flag_int_cell_yz,
        flag_int_cell_zx,
        Sxy_stab,
        Syz_stab,
        Szx_stab,
        flag_unst_cell_xy,
        flag_unst_cell_yz,
        flag_unst_cell_zx,
        flag_bound_cell_xy,
        flag_bound_cell_yz,
        flag_bound_cell_zx,
        flag_avail_cell_xy,
        flag_avail_cell_yz,
        flag_avail_cell_zx,
    ):
        for kk in range(nz + 1):
            mark_cells_2D(
                l_x[:, :, kk],
                l_y[:, :, kk],
                nx,
                ny,
                dx,
                dy,
                Sxy[:, :, kk],
                flag_int_cell_xy[:, :, kk],
                Sxy_stab[:, :, kk],
                flag_unst_cell_xy[:, :, kk],
                flag_bound_cell_xy[:, :, kk],
                flag_avail_cell_xy[:, :, kk],
            )

        for ii in range(nx + 1):
            mark_cells_2D(
                l_y[ii, :, :],
                l_z[ii, :, :],
                ny,
                nz,
                dy,
                dz,
                Syz[ii, :, :],
                flag_int_cell_yz[ii, :, :],
                Syz_stab[ii, :, :],
                flag_unst_cell_yz[ii, :, :],
                flag_bound_cell_yz[ii, :, :],
                flag_avail_cell_yz[ii, :, :],
            )

        for jj in range(ny + 1):
            mark_cells_2D(
                l_x[:, jj, :],
                l_z[:, jj, :],
                nx,
                nz,
                dx,
                dz,
                Szx[:, jj, :],
                flag_int_cell_zx[:, jj, :],
                Szx_stab[:, jj, :],
                flag_unst_cell_zx[:, jj, :],
                flag_bound_cell_zx[:, jj, :],
                flag_avail_cell_zx[:, jj, :],
            )

    """
  Function to compute the extension of the unstable cells
    """

    def compute_extensions(self):
        # breakpoint()
        for kk in range(self.nz + 1):
            # breakpoint()
            Grid2D.compute_extensions(
                nx=self.nx,
                ny=self.ny,
                S=self.Sxy[:, :, kk],
                flag_int_cell=self.flag_int_cell_xy[:, :, kk],
                S_stab=self.Sxy_stab[:, :, kk],
                S_enl=self.Sxy_enl[:, :, kk],
                S_red=self.Sxy_red[:, :, kk],
                flag_unst_cell=self.flag_unst_cell_xy[:, :, kk],
                flag_avail_cell=self.flag_avail_cell_xy[:, :, kk],
                flag_ext_cell=self.flag_ext_cell_xy[:, :, kk],
                flag_intr_cell=self.flag_intr_cell_xy[:, :, kk],
                borrowing=self.borrowing_xy[:, :, kk],
                kk=kk,
                l_verbose=False,
            )

        for ii in range(self.nx + 1):
            Grid2D.compute_extensions(
                nx=self.ny,
                ny=self.nz,
                S=self.Syz[ii, :, :],
                flag_int_cell=self.flag_int_cell_yz[ii, :, :],
                S_stab=self.Syz_stab[ii, :, :],
                S_enl=self.Syz_enl[ii, :, :],
                S_red=self.Syz_red[ii, :, :],
                flag_unst_cell=self.flag_unst_cell_yz[ii, :, :],
                flag_avail_cell=self.flag_avail_cell_yz[ii, :, :],
                flag_ext_cell=self.flag_ext_cell_yz[ii, :, :],
                borrowing=self.borrowing_yz[ii, :, :],
                flag_intr_cell=self.flag_intr_cell_yz[ii, :, :],
                kk=ii,
                l_verbose=False,
            )

        for jj in range(self.ny + 1):
            Grid2D.compute_extensions(
                nx=self.nx,
                ny=self.nz,
                S=self.Szx[:, jj, :],
                flag_int_cell=self.flag_int_cell_zx[:, jj, :],
                S_stab=self.Szx_stab[:, jj, :],
                S_enl=self.Szx_enl[:, jj, :],
                S_red=self.Szx_red[:, jj, :],
                flag_unst_cell=self.flag_unst_cell_zx[:, jj, :],
                flag_avail_cell=self.flag_avail_cell_zx[:, jj, :],
                flag_ext_cell=self.flag_ext_cell_zx[:, jj, :],
                flag_intr_cell=self.flag_intr_cell_zx[:, jj, :],
                borrowing=self.borrowing_zx[:, jj, :],
                kk=jj,
                l_verbose=False,
            )
