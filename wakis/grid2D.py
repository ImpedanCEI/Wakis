import numpy as np
from numba import jit


def seg_length(x_1, y_1, x_2, y_2):
    return np.linalg.norm(np.array([x_1 - x_2, y_1 - y_2]))


@jit("b1(f8, f8)", nopython=True)
def eq(a, b):
    tol = 1e-8
    return abs(a - b) < tol


@jit("b1(f8, f8)", nopython=True)
def neq(a, b):
    return not eq(a, b)


class Grid2D:
    """
    Class holding the grid info and the routines for cell extensions.
      Constructor arguments:
          - xmin,xmax,ymin,ymax: extent of the domain.
          - nx, ny: number of cells per direction
          - conductors: conductor object
          - sol_type: type of solver. 'FDTD' for staircased FDTD, 'DM' for Conformal Dey-Mittra FDTD,
                      'ECT' for Extended Cell Technique conformal FDTD
    """

    def __init__(self, xmin, xmax, ymin, ymax, nx, ny, conductors, sol_type):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny
        self.dx = (xmax - xmin) / nx
        self.dy = (ymax - ymin) / ny
        self.conductors = conductors

        self.l_x = np.zeros((nx, ny + 1))
        self.l_y = np.zeros((nx + 1, ny))
        self.S = np.zeros((nx, ny))
        self.S_stab = np.zeros((nx, ny))
        self.S_enl = np.zeros((nx, ny))
        self.S_red = np.zeros((nx, ny))
        self.flag_unst_cell = np.zeros((nx, ny), dtype=bool)
        self.flag_int_cell = np.zeros((nx, ny), dtype=bool)
        self.flag_bound_cell = np.zeros((nx, ny), dtype=bool)
        self.flag_avail_cell = np.zeros((nx, ny), dtype=bool)
        self.flag_ext_cell = np.zeros((nx, ny), dtype=bool)
        self.flag_intr_cell = np.zeros((nx, ny), dtype=bool)
        self.broken = np.zeros((self.nx, self.ny), dtype=bool)

        if (sol_type != "FDTD") and (sol_type != "DM") and (sol_type != "ECT"):
            raise ValueError(
                "sol_type must be:\n"
                + "\t'FDTD' for standard staircased FDTD\n"
                + "\t'DM' for Dey-Mittra conformal FDTD\n"
                + "\t'ECT' for Enlarged Cell Technique conformal FDTD"
            )

        if sol_type == "DM" or sol_type == "FDTD":
            self.compute_edges(
                in_conductor=self.conductors.in_conductor,
                intersec_x=self.conductors.intersec_x,
                intersec_y=self.conductors.intersec_y,
                l_x=self.l_x,
                l_y=self.l_y,
                dx=self.dx,
                dy=self.dy,
                nx=self.nx,
                ny=self.ny,
                xmin=self.xmin,
                ymin=self.ymin,
            )
            compute_areas(
                l_x=self.l_x,
                l_y=self.l_y,
                S=self.S,
                S_red=self.S_red,
                nx=self.nx,
                ny=self.ny,
                dx=self.dx,
                dy=self.dy,
            )
            mark_cells(
                l_x=self.l_x,
                l_y=self.l_y,
                nx=self.nx,
                ny=self.ny,
                dx=self.dx,
                dy=self.dy,
                S=self.S,
                flag_int_cell=self.flag_int_cell,
                S_stab=self.S_stab,
                flag_unst_cell=self.flag_unst_cell,
                flag_bound_cell=self.flag_bound_cell,
                flag_avail_cell=self.flag_avail_cell,
            )
        elif sol_type == "ECT":
            self.compute_edges(
                in_conductor=self.conductors.in_conductor,
                intersec_x=self.conductors.intersec_x,
                intersec_y=self.conductors.intersec_y,
                l_x=self.l_x,
                l_y=self.l_y,
                dx=self.dx,
                dy=self.dy,
                nx=self.nx,
                ny=self.ny,
                xmin=self.xmin,
                ymin=self.ymin,
            )
            compute_areas(
                l_x=self.l_x,
                l_y=self.l_y,
                S=self.S,
                S_red=self.S_red,
                nx=self.nx,
                ny=self.ny,
                dx=self.dx,
                dy=self.dy,
            )
            mark_cells(
                l_x=self.l_x,
                l_y=self.l_y,
                nx=self.nx,
                ny=self.ny,
                dx=self.dx,
                dy=self.dy,
                S=self.S,
                flag_int_cell=self.flag_int_cell,
                S_stab=self.S_stab,
                flag_unst_cell=self.flag_unst_cell,
                flag_bound_cell=self.flag_bound_cell,
                flag_avail_cell=self.flag_avail_cell,
            )
            # info about intruded cells (i,j,[(i_borrowing,j_borrowing,area_borrowing, )])
            self.borrowing = np.empty((nx, ny), dtype=object)

            for i in range(nx):
                for j in range(ny):
                    self.borrowing[i, j] = []
            self.flag_ext_cell = self.flag_unst_cell.copy()
            self.compute_extensions(
                nx=nx,
                ny=ny,
                S=self.S,
                flag_int_cell=self.flag_int_cell,
                S_stab=self.S_stab,
                S_enl=self.S_enl,
                S_red=self.S_red,
                flag_unst_cell=self.flag_unst_cell,
                flag_avail_cell=self.flag_avail_cell,
                flag_ext_cell=self.flag_ext_cell,
                flag_intr_cell=self.flag_intr_cell,
                borrowing=self.borrowing,
            )

    """
  Function to compute the length of the edges of the conformal grid.
    Inputs:
        - tol: an edge shorter than tol will be considered as of zero length
    """

    @staticmethod
    def compute_edges(
        tol=1e-8,
        in_conductor=None,
        intersec_x=None,
        intersec_y=None,
        l_x=None,
        l_y=None,
        dx=None,
        dy=None,
        nx=None,
        ny=None,
        xmin=None,
        ymin=None,
    ):
        """
        Notation:

              (x_3, y_3)----- l_x[i, j + 1] ---------
                  |                                  |
                  |                                  |
                 l_y[i, j]        (i, j)            l_y[i + 1, j]
                  |                                  |
                  |                                  |
              (x_1, y_1)------- l_x[i, j] -------(x_2, y_2)

        """
        for ii in range(nx):
            for jj in range(ny + 1):
                x_1 = ii * dx + xmin
                y_1 = jj * dy + ymin
                x_2 = (ii + 1) * dx + xmin
                y_2 = jj * dy + ymin
                # if point 1 is in conductor
                if in_conductor(x_1, y_1):
                    # if point 2 is in conductor, length of the l_x[i, j] is zero
                    if in_conductor(x_2, y_2):
                        l_x[ii, jj] = 0
                    # if point 2 is not in conductor, length of l_x[i, j] is the fractional length
                    else:
                        l_x[ii, jj] = seg_length(intersec_x(x_2, y_2), y_1, x_2, y_2)
                # if point 1 is not in conductor
                else:
                    # if point 2 is in conductor, length of l_x[i, j] is the fractional length
                    if in_conductor(x_2, y_2):
                        l_x[ii, jj] = seg_length(x_1, y_1, intersec_x(x_1, y_1), y_2)
                    # if point 2 is not in conductor, length of l_x[i, j] is dx
                    else:
                        l_x[ii, jj] = dx

        for ii in range(nx + 1):
            for jj in range(ny):
                x_1 = ii * dx + xmin
                y_1 = jj * dy + ymin
                x_3 = ii * dx + xmin
                y_3 = (jj + 1) * dy + ymin
                # if point 1 is in conductor
                if in_conductor(x_1, y_1):
                    # if point 3 to the right is in conductor, length of the l_y[i, j] is zero
                    if in_conductor(x_3, y_3):
                        l_y[ii, jj] = 0
                    # if point 3 is not in conductor, length of l_y[i, j] is the fractional length
                    else:
                        l_y[ii, jj] = seg_length(x_1, intersec_y(x_3, y_3), x_3, y_3)
                # if point 1 is not in conductor
                else:
                    # if point 3 is in conductor, length of the l_y[i, j] is the fractional length
                    if in_conductor(x_3, y_3):
                        l_y[ii, jj] = seg_length(x_1, y_1, x_3, intersec_y(x_1, y_1))
                    # if point 3 is not in conductor, length of l_y[i, j] is dy
                    else:
                        l_y[ii, jj] = dy

        # set to zero the length of very small cells
        if tol > 0.0:
            l_x /= dx
            l_y /= dy

            low_values_flags = abs(l_x) < tol
            high_values_flags = abs(l_x - 1.0) < tol
            l_x[low_values_flags] = 0
            l_x[high_values_flags] = 1.0

            low_values_flags = abs(l_y) < tol
            high_values_flags = abs(l_y - 1.0) < tol
            l_y[low_values_flags] = 0
            l_y[high_values_flags] = 1.0

            l_x *= dx
            l_y *= dy

    """
  Function to compute the extension of the unstable cells
    """

    @staticmethod
    def compute_extensions(
        nx=None,
        ny=None,
        S=None,
        flag_int_cell=None,
        S_stab=None,
        S_enl=None,
        S_red=None,
        flag_unst_cell=None,
        flag_avail_cell=None,
        flag_ext_cell=None,
        flag_intr_cell=None,
        borrowing=None,
        l_verbose=True,
        kk=0,
    ):
        N = np.sum(flag_ext_cell)

        if l_verbose:
            print(kk)
            print("ext cells: %d" % N)
        # Do the simple one-cell extension
        Grid2D._compute_extensions_one_cell(
            nx=nx,
            ny=ny,
            S=S,
            S_stab=S_stab,
            S_enl=S_enl,
            S_red=S_red,
            flag_avail_cell=flag_avail_cell,
            flag_ext_cell=flag_ext_cell,
            flag_intr_cell=flag_intr_cell,
            borrowing=borrowing,
        )

        N_one_cell = N - np.sum(flag_ext_cell)
        if l_verbose:
            print("one cell exts: %d" % N_one_cell)
        # If any cell could not be extended do the four-cell extension
        """
        if np.sum(flag_ext_cell) > 0:
            N = np.sum(flag_ext_cell)
            Grid2D._compute_extensions_four_cells(nx=nx, ny=ny, S=S, flag_int_cell=flag_int_cell,
                                                  S_stab=S_stab, S_enl=S_enl, S_red=S_red,
                                                  flag_unst_cell=flag_unst_cell,
                                                  flag_avail_cell=flag_avail_cell,
                                                  flag_ext_cell=flag_ext_cell,
                                                  flag_intr_cell = flag_intr_cell,
                                                  borrowing=borrowing)
            N_four_cells = (N - np.sum(flag_ext_cell))
            if 1: #l_verbose:
                print('four cell exts: %d' % N_four_cells)
        """
        # If any cell could not be extended do the eight-cell extension
        if np.sum(flag_ext_cell) > 0:
            N = np.sum(flag_ext_cell)
            Grid2D._compute_extensions_eight_cells(
                nx=nx,
                ny=ny,
                S=S,
                flag_int_cell=flag_int_cell,
                S_stab=S_stab,
                S_enl=S_enl,
                S_red=S_red,
                flag_unst_cell=flag_unst_cell,
                flag_avail_cell=flag_avail_cell,
                flag_ext_cell=flag_ext_cell,
                flag_intr_cell=flag_intr_cell,
                borrowing=borrowing,
            )
            N_eight_cells = N - np.sum(flag_ext_cell)
            if 1:  # l_verbose:
                print("eight cell exts: %d" % N_eight_cells)
        # If any cell could not be extended the algorithm failed
        if np.sum(flag_ext_cell) > 0:
            N = np.sum(flag_ext_cell)
            raise RuntimeError(
                str(N) + " cells could not be extended.\n" + "Please refine the mesh"
            )

    """
  Function to compute the one-cell extension of the unstable cells
    """

    @staticmethod
    def _compute_extensions_one_cell(
        nx=None,
        ny=None,
        S=None,
        S_stab=None,
        S_enl=None,
        S_red=None,
        flag_avail_cell=None,
        flag_ext_cell=None,
        flag_intr_cell=None,
        borrowing=None,
    ):
        for ii in range(0, nx):
            for jj in range(0, ny):
                if flag_ext_cell[ii, jj]:
                    S_ext = S_stab[ii, jj] - S[ii, jj]
                    if S[ii - 1, jj] > S_ext and flag_avail_cell[ii - 1, jj]:
                        denom = S[ii - 1, jj]
                        patch = S_ext * S[ii - 1, jj] / denom
                        if S_red[ii - 1, jj] - patch > 0:
                            S_red[ii - 1, jj] -= patch
                            borrowing[ii, jj].append([ii - 1, jj, patch, None])
                            flag_intr_cell[ii - 1, jj] = True
                            S_enl[ii, jj] = S[ii, jj] + patch
                            flag_ext_cell[ii, jj] = False
                    if (
                        S[ii, jj - 1] > S_ext
                        and flag_avail_cell[ii, jj - 1]
                        and flag_ext_cell[ii, jj]
                    ):
                        denom = S[ii, jj - 1]
                        patch = S_ext * S[ii, jj - 1] / denom
                        if S_red[ii, jj - 1] - patch > 0:
                            S_red[ii, jj - 1] -= patch
                            borrowing[ii, jj].append([ii, jj - 1, patch, None])
                            flag_intr_cell[ii, jj - 1] = True
                            S_enl[ii, jj] = S[ii, jj] + patch
                            flag_ext_cell[ii, jj] = False
                    if (
                        S[ii, jj + 1] > S_ext
                        and flag_avail_cell[ii, jj + 1]
                        and flag_ext_cell[ii, jj]
                    ):
                        denom = S[ii, jj + 1]
                        patch = S_ext * S[ii, jj + 1] / denom
                        if S_red[ii, jj + 1] - patch > 0:
                            S_red[ii, jj + 1] -= patch
                            borrowing[ii, jj].append([ii, jj + 1, patch, None])
                            flag_intr_cell[ii, jj + 1] = True
                            S_enl[ii, jj] = S[ii, jj] + patch
                            flag_ext_cell[ii, jj] = False
                    if (
                        S[ii + 1, jj] > S_ext
                        and flag_avail_cell[ii + 1, jj]
                        and flag_ext_cell[ii, jj]
                    ):
                        denom = S[ii + 1, jj]
                        patch = S_ext * S[ii + 1, jj] / denom
                        if S_red[ii + 1, jj] - patch > 0:
                            S_red[ii + 1, jj] -= patch
                            borrowing[ii, jj].append([ii + 1, jj, patch, None])
                            flag_intr_cell[ii + 1, jj] = True
                            S_enl[ii, jj] = S[ii, jj] + patch
                            flag_ext_cell[ii, jj] = False

    """
  Function to compute the four-cell extension of the unstable cells
    """

    @staticmethod
    def _compute_extensions_four_cells(
        nx=None,
        ny=None,
        S=None,
        flag_int_cell=None,
        S_stab=None,
        S_enl=None,
        S_red=None,
        flag_unst_cell=None,
        flag_avail_cell=None,
        flag_ext_cell=None,
        flag_intr_cell=None,
        borrowing=None,
    ):
        for ii in range(0, nx):
            for jj in range(0, ny):
                local_avail = flag_avail_cell.copy()
                if (
                    flag_unst_cell[ii, jj]
                    and flag_int_cell[ii, jj]
                    and flag_ext_cell[ii, jj]
                ):
                    denom = (
                        (flag_avail_cell[ii - 1, jj]) * S[ii - 1, jj]
                        + (flag_avail_cell[ii + 1, jj]) * S[ii + 1, jj]
                        + (flag_avail_cell[ii, jj - 1]) * S[ii, jj - 1]
                        + (flag_avail_cell[ii, jj + 1]) * S[ii, jj + 1]
                    )
                    S_ext = S_stab[ii, jj] - S[ii, jj]
                    neg_cell = True
                    # idea: if any cell would reach negative area it is locally not available.
                    #       then denom has to be recomputed from scratch

                    while denom >= S_ext and neg_cell:
                        neg_cell = False
                        if local_avail[ii - 1, jj]:
                            patch = S_ext * S[ii - 1, jj] / denom
                            if S_red[ii - 1, jj] - patch <= 0:
                                neg_cell = True
                                local_avail[ii - 1, jj] = False
                        if local_avail[ii + 1, jj]:
                            patch = S_ext * S[ii + 1, jj] / denom
                            if S_red[ii + 1, jj] - patch <= 0:
                                neg_cell = True
                                local_avail[ii + 1, jj] = False
                        if local_avail[ii, jj - 1]:
                            patch = S_ext * S[ii, jj - 1] / denom
                            if S_red[ii, jj - 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii, jj - 1] = False
                        if local_avail[ii, jj + 1]:
                            patch = S_ext * S[ii, jj + 1] / denom
                            if S_red[ii, jj + 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii, jj + 1] = False
                        denom = (
                            (local_avail[ii - 1, jj]) * S[ii - 1, jj]
                            + (local_avail[ii + 1, jj]) * S[ii + 1, jj]
                            + (local_avail[ii, jj - 1]) * S[ii, jj - 1]
                            + (local_avail[ii, jj + 1]) * S[ii, jj + 1]
                        )

                    # If possible, do 4-cell extension
                    if denom >= S_ext:
                        S_enl[ii, jj] = S[ii, jj]
                        if local_avail[ii - 1, jj]:
                            patch = S_ext * S[ii - 1, jj] / denom
                            borrowing[ii, jj].append([ii - 1, jj, patch, None])
                            flag_intr_cell[ii - 1, jj] = True
                            S_enl[ii, jj] += patch
                            S_red[ii - 1, jj] -= patch
                        if local_avail[ii + 1, jj]:
                            patch = S_ext * S[ii + 1, jj] / denom
                            borrowing[ii, jj].append([ii + 1, jj, patch, None])
                            flag_intr_cell[ii + 1, jj] = True
                            S_enl[ii, jj] += patch
                            S_red[ii + 1, jj] -= patch
                        if local_avail[ii, jj - 1]:
                            patch = S_ext * S[ii, jj - 1] / denom
                            borrowing[ii, jj].append([ii, jj - 1, patch, None])
                            flag_intr_cell[ii, jj - 1] = True
                            S_enl[ii, jj] += patch
                            S_red[ii, jj - 1] -= patch
                        if local_avail[ii, jj + 1]:
                            patch = S_ext * S[ii, jj + 1] / denom
                            borrowing[ii, jj].append([ii, jj + 1, patch, None])
                            flag_intr_cell[ii, jj + 1] = True
                            S_enl[ii, jj] += patch
                            S_red[ii, jj + 1] -= patch

                        flag_ext_cell[ii, jj] = False

    """
  Function to compute the eight-cell extension of the unstable cells
    """

    @staticmethod
    def _compute_extensions_eight_cells(
        nx=None,
        ny=None,
        S=None,
        flag_int_cell=None,
        S_stab=None,
        S_enl=None,
        S_red=None,
        flag_unst_cell=None,
        flag_avail_cell=None,
        flag_ext_cell=None,
        flag_intr_cell=None,
        borrowing=None,
    ):
        for ii in range(0, nx):
            for jj in range(0, ny):
                local_avail = flag_avail_cell.copy()
                if (
                    flag_unst_cell[ii, jj]
                    and flag_int_cell[ii, jj]
                    and flag_ext_cell[ii, jj]
                ):
                    S_enl[ii, jj] = S[ii, jj]
                    S_ext = S_stab[ii, jj] - S[ii, jj]

                    denom = (
                        (flag_avail_cell[ii - 1, jj]) * S[ii - 1, jj]
                        + (flag_avail_cell[ii + 1, jj]) * S[ii + 1, jj]
                        + (flag_avail_cell[ii, jj - 1]) * S[ii, jj - 1]
                        + (flag_avail_cell[ii, jj + 1]) * S[ii, jj + 1]
                        + (flag_avail_cell[ii - 1, jj - 1]) * S[ii - 1, jj - 1]
                        + (flag_avail_cell[ii + 1, jj - 1]) * S[ii + 1, jj - 1]
                        + (flag_avail_cell[ii - 1, jj + 1]) * S[ii - 1, jj + 1]
                        + (flag_avail_cell[ii + 1, jj + 1]) * S[ii + 1, jj + 1]
                    )

                    neg_cell = True
                    while denom >= S_ext and neg_cell:
                        neg_cell = False
                        if local_avail[ii - 1, jj]:
                            patch = S_ext * S[ii - 1, jj] / denom
                            if S_red[ii - 1, jj] - patch <= 0:
                                neg_cell = True
                                local_avail[ii - 1, jj] = False
                        if local_avail[ii + 1, jj]:
                            patch = S_ext * S[ii + 1, jj] / denom
                            if S_red[ii + 1, jj] - patch <= 0:
                                neg_cell = True
                                local_avail[ii + 1, jj] = False
                        if local_avail[ii, jj - 1]:
                            patch = S_ext * S[ii, jj - 1] / denom
                            if S_red[ii, jj - 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii, jj - 1] = False
                        if local_avail[ii, jj + 1]:
                            patch = S_ext * S[ii, jj + 1] / denom
                            if S_red[ii, jj + 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii, jj + 1] = False
                        if local_avail[ii - 1, jj - 1]:
                            patch = S_ext * S[ii - 1, jj - 1] / denom
                            if S_red[ii - 1, jj - 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii - 1, jj - 1] = False
                        if local_avail[ii + 1, jj - 1]:
                            patch = S_ext * S[ii + 1, jj - 1] / denom
                            if S_red[ii + 1, jj - 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii + 1, jj - 1] = False
                        if local_avail[ii - 1, jj + 1]:
                            patch = S_ext * S[ii - 1, jj + 1] / denom
                            if S_red[ii - 1, jj + 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii - 1, jj + 1] = False
                        if local_avail[ii + 1, jj + 1]:
                            patch = S_ext * S[ii + 1, jj + 1] / denom
                            if S_red[ii + 1, jj + 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii + 1, jj + 1] = False

                        denom = (
                            (local_avail[ii - 1, jj]) * S[ii - 1, jj]
                            + (local_avail[ii + 1, jj]) * S[ii + 1, jj]
                            + (local_avail[ii, jj - 1]) * S[ii, jj - 1]
                            + (local_avail[ii, jj + 1]) * S[ii, jj + 1]
                            + (local_avail[ii - 1, jj - 1]) * S[ii - 1, jj - 1]
                            + (local_avail[ii + 1, jj - 1]) * S[ii + 1, jj - 1]
                            + (local_avail[ii - 1, jj + 1]) * S[ii - 1, jj + 1]
                            + (local_avail[ii + 1, jj + 1]) * S[ii + 1, jj + 1]
                        )

                    if denom >= S_ext:
                        S_enl[ii, jj] = S[ii, jj]
                        if local_avail[ii - 1, jj]:
                            patch = S_ext * S[ii - 1, jj] / denom
                            borrowing[ii, jj].append([ii - 1, jj, patch, None])
                            flag_intr_cell[ii - 1, jj] = True
                            S_enl[ii, jj] += patch
                            S_red[ii - 1, jj] -= patch
                        if local_avail[ii + 1, jj]:
                            patch = S_ext * S[ii + 1, jj] / denom
                            borrowing[ii, jj].append([ii + 1, jj, patch, None])
                            flag_intr_cell[ii + 1, jj] = True
                            S_enl[ii, jj] += patch
                            S_red[ii + 1, jj] -= patch
                        if local_avail[ii, jj - 1]:
                            patch = S_ext * S[ii, jj - 1] / denom
                            borrowing[ii, jj].append([ii, jj - 1, patch, None])
                            flag_intr_cell[ii, jj - 1] = True
                            S_enl[ii, jj] += patch
                            S_red[ii, jj - 1] -= patch
                        if local_avail[ii, jj + 1]:
                            patch = S_ext * S[ii, jj + 1] / denom
                            borrowing[ii, jj].append([ii, jj + 1, patch, None])
                            flag_intr_cell[ii, jj + 1] = True
                            S_enl[ii, jj] += patch
                            S_red[ii, jj + 1] -= patch
                        if local_avail[ii - 1, jj - 1]:
                            patch = S_ext * S[ii - 1, jj - 1] / denom
                            borrowing[ii, jj].append([ii - 1, jj - 1, patch, None])
                            flag_intr_cell[ii - 1, jj - 1] = True
                            S_enl[ii, jj] += patch
                            S_red[ii - 1, jj - 1] -= patch
                        if local_avail[ii + 1, jj - 1]:
                            patch = S_ext * S[ii + 1, jj - 1] / denom
                            borrowing[ii, jj].append([ii + 1, jj - 1, patch, None])
                            flag_intr_cell[ii + 1, jj - 1] = True
                            S_enl[ii, jj] += patch
                            S_red[ii + 1, jj - 1] -= patch
                        if local_avail[ii - 1, jj + 1]:
                            patch = S_ext * S[ii - 1, jj + 1] / denom
                            borrowing[ii, jj].append([ii - 1, jj + 1, patch, None])
                            flag_intr_cell[ii - 1, jj + 1] = True
                            S_enl[ii, jj] += patch
                            S_red[ii - 1, jj + 1] -= patch
                        if local_avail[ii + 1, jj + 1]:
                            patch = S_ext * S[ii + 1, jj + 1] / denom
                            borrowing[ii, jj].append([ii + 1, jj + 1, patch, None])
                            flag_intr_cell[ii + 1, jj + 1] = True
                            S_enl[ii, jj] += patch
                            S_red[ii + 1, jj + 1] -= patch

                        flag_ext_cell[ii, jj] = False


"""
Function to compute the area of the cells of the conformal grid.
"""


@jit("(f8[:,:], f8[:,:], f8[:,:], f8[:,:], i4, i4, f8, f8)", nopython=True)
def compute_areas(l_x, l_y, S, S_red, nx, ny, dx, dy):
    # Normalize the grid lengths for robustness
    l_x /= dx
    l_y /= dy

    # Loop over the cells
    for ii in range(nx):
        for jj in range(ny):
            # If at least 3 edges have full length consider the area as full
            # (this also takes care of the case in which an edge lies exactly on the boundary)
            # count_cane = eq(l_x[ii, jj], 1.0) + eq(l_y[ii, jj], 1.0) + eq(l_x[ii, jj + 1], 1.0) + eq(l_y[ii + 1, jj], 1.0)
            if (
                np.sum(
                    np.array(
                        [
                            eq(l_x[ii, jj], 1.0),
                            eq(l_y[ii, jj], 1.0),
                            eq(l_x[ii, jj + 1], 1.0),
                            eq(l_y[ii + 1, jj], 1.0),
                        ]
                    )
                )
                >= 3
            ):
                # if count_cane >=3:
                S[ii, jj] = 1.0
            elif (
                eq(l_x[ii, jj], 0.0)
                and neq(l_y[ii, jj], 0.0)
                and neq(l_x[ii, jj + 1], 0.0)
                and neq(l_y[ii + 1, jj], 0)
            ):
                S[ii, jj] = 0.5 * (l_y[ii, jj] + l_y[ii + 1, jj]) * l_x[ii, jj + 1]
            elif (
                eq(l_x[ii, jj + 1], 0.0)
                and neq(l_y[ii, jj], 0.0)
                and neq(l_x[ii, jj], 0.0)
                and neq(l_y[ii + 1, jj], 0.0)
            ):
                S[ii, jj] = 0.5 * (l_y[ii, jj] + l_y[ii + 1, jj]) * l_x[ii, jj]
            elif (
                eq(l_y[ii, jj], 0.0)
                and neq(l_x[ii, jj], 0.0)
                and neq(l_x[ii, jj + 1], 0.0)
                and neq(l_y[ii + 1, jj], 0.0)
            ):
                S[ii, jj] = 0.5 * (l_x[ii, jj] + l_x[ii, jj + 1]) * l_y[ii + 1, jj]
            elif (
                eq(l_y[ii + 1, jj], 0.0)
                and neq(l_x[ii, jj], 0.0)
                and neq(l_y[ii, jj], 0.0)
                and neq(l_x[ii, jj + 1], 0.0)
            ):
                S[ii, jj] = 0.5 * (l_x[ii, jj] + l_x[ii, jj + 1]) * l_y[ii, jj]
            elif (
                eq(l_x[ii, jj], 0.0)
                and eq(l_y[ii, jj], 0.0)
                and neq(l_x[ii, jj + 1], 0.0)
                and neq(l_y[ii + 1, jj], 0.0)
            ):
                S[ii, jj] = 0.5 * l_x[ii, jj + 1] * l_y[ii + 1, jj]
            elif (
                eq(l_x[ii, jj], 0.0)
                and eq(l_y[ii + 1, jj], 0.0)
                and neq(l_x[ii, jj + 1], 0.0)
                and neq(l_y[ii, jj], 0.0)
            ):
                S[ii, jj] = 0.5 * l_x[ii, jj + 1] * l_y[ii, jj]
            elif (
                eq(l_x[ii, jj + 1], 0.0)
                and eq(l_y[ii + 1, jj], 0.0)
                and neq(l_x[ii, jj], 0.0)
                and neq(l_y[ii, jj], 0.0)
            ):
                S[ii, jj] = 0.5 * l_x[ii, jj] * l_y[ii, jj]
            elif (
                eq(l_x[ii, jj + 1], 0.0)
                and eq(l_y[ii, jj], 0.0)
                and neq(l_x[ii, jj], 0.0)
                and neq(l_y[ii + 1, jj], 0.0)
            ):
                S[ii, jj] = 0.5 * l_x[ii, jj] * l_y[ii + 1, jj]
            elif (
                0.0 < l_x[ii, jj] <= 1.0
                and 0.0 < l_y[ii, jj] <= 1.0
                and eq(l_x[ii, jj + 1], 1.0)
                and eq(l_y[ii + 1, jj], 1.0)
            ):
                S[ii, jj] = 1.0 - 0.5 * (1.0 - l_x[ii, jj]) * (1.0 - l_y[ii, jj])
            elif (
                0.0 < l_x[ii, jj] <= 1.0
                and 0.0 < l_y[ii + 1, jj] <= 1.0
                and eq(l_x[ii, jj + 1], 1.0)
                and eq(l_y[ii, jj], 1.0)
            ):
                S[ii, jj] = 1.0 - 0.5 * (1.0 - l_x[ii, jj]) * (1.0 - l_y[ii + 1, jj])
            elif (
                0.0 < l_x[ii, jj + 1] <= 1.0
                and 0.0 < l_y[ii + 1, jj] <= 1.0
                and eq(l_x[ii, jj], 1.0)
                and eq(l_y[ii, jj], 1.0)
            ):
                S[ii, jj] = 1.0 - 0.5 * (1.0 - l_x[ii, jj + 1]) * (
                    1.0 - l_y[ii + 1, jj]
                )
            elif (
                0.0 < l_x[ii, jj + 1] <= 1.0
                and 0.0 < l_y[ii, jj] <= 1.0
                and eq(l_x[ii, jj], 1.0)
                and eq(l_y[ii + 1, jj], 1.0)
            ):
                S[ii, jj] = 1.0 - 0.5 * (1.0 - l_x[ii, jj + 1]) * (1.0 - l_y[ii, jj])

    l_x *= dx
    l_y *= dy


"""
Function to mark wich cells are interior (int), require extension (unst),
are on the boundary(bound), are available for intrusion (avail)
"""


@jit(
    "(f8[:,:], f8[:,:], i4, i4, f8, f8, f8[:,:], b1[:,:], f8[:,:], b1[:,:], b1[:,:], b1[:,:])",
    nopython=True,
)
def mark_cells(
    l_x,
    l_y,
    nx,
    ny,
    dx,
    dy,
    S,
    flag_int_cell,
    S_stab,
    flag_unst_cell,
    flag_bound_cell,
    flag_avail_cell,
):
    for ii in range(nx):
        for jj in range(ny):
            flag_int_cell[ii, jj] = S[ii, jj] > 0
            if flag_int_cell[ii, jj]:
                S_stab[ii, jj] = 0.5 * np.max(
                    np.array(
                        [
                            l_x[ii, jj] * dy,
                            l_x[ii, jj + 1] * dy,
                            l_y[ii, jj] * dx,
                            l_y[ii + 1, jj] * dx,
                        ]
                    )
                )
            flag_unst_cell[ii, jj] = S[ii, jj] < S_stab[ii, jj]
            flag_bound_cell[ii, jj] = (
                (0 < l_x[ii, jj] < dx)
                or (0 < l_y[ii, jj] < dy)
                or (0 < l_x[ii, jj + 1] < dx)
                or (0 < l_y[ii + 1, jj] < dy)
            )
            flag_avail_cell[ii, jj] = flag_int_cell[ii, jj] and (
                not flag_unst_cell[ii, jj]
            )
