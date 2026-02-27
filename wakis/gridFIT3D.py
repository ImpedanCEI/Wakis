# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import time

import h5py
import numpy as np
import pyvista as pv
from scipy.optimize import least_squares

from .field import Field
from .logger import Logger
from .materials import material_colors

try:
    from mpi4py import MPI

    imported_mpi = True
except ImportError:
    imported_mpi = False


class GridFIT3D:
    """
    Class holding the grid information and
    stl importing handling using PyVista
    """

    def __init__(
        self,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        zmin=None,
        zmax=None,
        Nx=None,
        Ny=None,
        Nz=None,
        x=None,
        y=None,
        z=None,
        use_mpi=False,
        use_mesh_refinement=False,
        refinement_method="insert",
        refinement_tol=1e-8,
        snap_points=None,
        snap_tol=1e-5,
        snap_solids=None,
        stl_solids=None,
        stl_materials=None,
        stl_rotate=[0.0, 0.0, 0.0],
        stl_translate=[0.0, 0.0, 0.0],
        stl_scale=1.0,
        stl_colors=None,
        stl_tol=1e-3,
        load_from_h5=None,
        verbose=1,
    ):
        """
        Class holding the grid information and STL importing/handling using PyVista.

        Parameters
        ----------
        xmin, xmax, ymin, ymax, zmin, zmax : float, optional
            Extent of the simulation domain. If None, must provide x, y, z arrays.
        Nx, Ny, Nz : int, optional
            Number of cells per direction. If None, must provide x, y, z arrays.
        x, y, z : array_like, optional
            Custom grid axis arrays to be used in the meshgrid generation.
            Non-uniform grids are supported.
        use_mpi : bool, optional
            Enable MPI domain decomposition in the z direction. Default is False.
        use_mesh_refinement : bool, optional
            Enable mesh refinement based on snap points extracted from the STL solids.
            Default is False.
        refinement_method : str, optional
            Refinement algorithm for mesh refinement. Default is "insert".
        refinement_tol : float, optional
            Tolerance for mesh refinement. Default is 1e-8.
        snap_points : array_like, optional
            Points to snap the mesh to. Default is None.
        snap_tol : float, optional
            Tolerance for snap point detection. Default is 1e-5.
        snap_solids : list or None, optional
            STL solids to use for snap point extraction. Default is None (all).
        stl_solids : dict or str, optional
            STL files to import in the domain. {'Solid 1': stl_1, ...}
        stl_materials : dict, optional
            Material properties associated with STL solids.
            {'Solid 1': [eps1, mu1], ...}
        stl_rotate : list or dict, optional
            Angle of rotation to apply to the STL models: [rot_x, rot_y, rot_z].
            If dict, must contain the same keys as stl_solids.
        stl_translate : list or dict, optional
            Translation to apply to the STL models: [dx, dy, dz].
            If dict, must contain the same keys as stl_solids.
        stl_scale : float or dict, optional
            Scaling value to apply to the STL model to convert to [m].
            If dict, must contain the same keys as stl_solids.
        stl_colors : str, list, dict, or None, optional
            Color(s) for STL solids. If None, assigned automatically.
        stl_tol : float, optional
            Tolerance factor for STL import, used in grid.select_enclosed_points.
            Default is 1e-3.
        load_from_h5 : str, optional
            Load grid from an h5 file previously saved with `save_to_h5`.
        verbose : int or bool, optional
            Enable verbose output on the terminal. Use `verbose=2` for more detail.

        Attributes
        ----------
        x, y, z : ndarray
            Grid axis arrays.
        Nx, Ny, Nz : int
            Number of cells in each direction.
        dx, dy, dz : ndarray
            Cell sizes in each direction.
        grid : pyvista.StructuredGrid
            PyVista grid object.
        stl_solids, stl_materials, stl_colors : dict
            STL solid file paths, materials, and colors.
        (...)
        """

        t0 = time.time()
        self.logger = Logger()
        self.verbose = verbose
        self.use_mpi = use_mpi

        # Grid data
        # generate from file
        if load_from_h5 is not None:
            self.load_from_h5(load_from_h5)
            return  # TODO: support MPI decomposition

        # generate from custom x,y,z arrays
        elif x is not None and y is not None and z is not None:
            # allow user to set the grid axis manually
            self.x = x
            self.y = y
            self.z = z
            self.Nx = len(self.x) - 1
            self.Ny = len(self.y) - 1
            self.Nz = len(self.z) - 1
            self.xmin, self.xmax = self.x[0], self.x[-1]
            self.ymin, self.ymax = self.y[0], self.y[-1]
            self.zmin, self.zmax = self.z[0], self.z[-1]
            if self.use_mpi:
                raise ValueError(
                    "[!] Error: use_mpi=True is not compatible with custom x,y,z arrays."
                )

        # generate from domain extents and number of cells [LEGACY]
        elif all(v is not None for v in [xmin, xmax, ymin, ymax, zmin, zmax]):
            # uniform grid from domain extents and number of cells
            self.xmin, self.xmax = xmin, xmax
            self.ymin, self.ymax = ymin, ymax
            self.zmin, self.zmax = zmin, zmax
            self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
            self.x = np.linspace(self.xmin, self.xmax, self.Nx + 1)
            self.y = np.linspace(self.ymin, self.ymax, self.Ny + 1)
            self.z = np.linspace(self.zmin, self.zmax, self.Nz + 1)

        else:
            raise ValueError(
                "[!] Error initializing GridFIT3D:\n"
                "  - Provide grid axis arrays: x, y, z\n"
                "  - OR domain extents and number of cells: \
                    xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz\n"
                "  - OR load from a HDF5 file using load_from_h5"
            )

        # TODO: allow non uniform dx, dy, dz
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)
        self.update_logger(["Nx", "Ny", "Nz", "dx", "dy", "dz"])
        self.update_logger(["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])

        # stl info
        self.stl_solids = stl_solids
        self.stl_materials = stl_materials
        self.stl_rotate = stl_rotate
        self.stl_translate = stl_translate
        self.stl_scale = stl_scale
        self.stl_colors = stl_colors
        self.update_logger(["stl_solids", "stl_materials"])
        if stl_rotate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_rotate"])
        if stl_translate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_translate"])
        if stl_scale != 1.0:
            self.update_logger(["stl_scale"])

        if stl_solids is not None:
            self._prepare_stl_dicts()

        # refine self.x, self.y, self.z using snap points
        self.use_mesh_refinement = use_mesh_refinement
        self.refinement_method = refinement_method
        self.snap_points = snap_points
        self.snap_tol = snap_tol
        self.snap_solids = snap_solids  # if None, use all stl_solids
        self.update_logger(["use_mesh_refinement"])

        if self.use_mesh_refinement:
            if verbose:
                print("Applying mesh refinement...")
            if self.snap_points is None and stl_solids is not None:
                self._compute_snap_points(
                    snap_solids=snap_solids, snap_tol=snap_tol
                )
            self._refine_xyz_axis(method=refinement_method, tol=refinement_tol)

        if verbose:
            print(
                f"Generating grid with {self.Nx * self.Ny * self.Nz} mesh cells..."
            )
            if verbose > 1:
                print(
                    f" * Simulation domain bounds: \n\
                    x:[{xmin:.3f}, {xmax:.3f}],\n\
                    y:[{ymin:.3f}, {ymax:.3f}],\n\
                    z:[{zmin:.3f}, {zmax:.3f}]"
                )

        # MPI subdivide domain
        if self.use_mpi:
            self.ZMIN = None
            self.ZMAX = None
            self.NZ = None
            self.Z = None
            if imported_mpi:
                self._mpi_initialize()
                if self.verbose:
                    print(f"MPI initialized for {self.rank} of {self.size}")
            else:
                raise ImportError(
                    "[!] mpi4py is required when use_mpi=True but was not found"
                )

        # grid G and tilde grid ~G, lengths and inverse areas
        self._compute_grid()

        # tolerance for stl import tol*min(dx,dy,dz)
        if verbose:
            print("Importing STL solids...")
        self.stl_tol = stl_tol
        if stl_solids is not None:
            self._mark_cells_in_stl()

        if verbose:
            print(f"Total grid initialization time: {time.time() - t0} s")

        self.gridInitializationTime = time.time() - t0
        self.update_logger(["gridInitializationTime"])

    def _compute_grid(self):
        """
        Compute the PyVista grid and related geometric quantities.

        Sets up the structured grid, cell lengths, inverse areas, and tilde grid.
        """
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.grid = pv.StructuredGrid(
            X.transpose(), Y.transpose(), Z.transpose()
        )

        self.L = Field(self.Nx, self.Ny, self.Nz)
        self.L.field_x = X[1:, 1:, 1:] - X[:-1, :-1, :-1]
        self.L.field_y = Y[1:, 1:, 1:] - Y[:-1, :-1, :-1]
        self.L.field_z = Z[1:, 1:, 1:] - Z[:-1, :-1, :-1]

        self.iA = Field(self.Nx, self.Ny, self.Nz)
        self.iA.field_x = np.divide(1.0, self.L.field_y * self.L.field_z)
        self.iA.field_y = np.divide(1.0, self.L.field_x * self.L.field_z)
        self.iA.field_z = np.divide(1.0, self.L.field_x * self.L.field_y)

        # tilde grid ~G
        self.tx = (self.x[1:] + self.x[:-1]) / 2
        self.ty = (self.y[1:] + self.y[:-1]) / 2
        self.tz = (self.z[1:] + self.z[:-1]) / 2

        self.tx = np.append(self.tx, self.tx[-1])
        self.ty = np.append(self.ty, self.ty[-1])
        self.tz = np.append(self.tz, self.tz[-1])

        tX, tY, tZ = np.meshgrid(self.tx, self.ty, self.tz, indexing="ij")

        self.tL = Field(self.Nx, self.Ny, self.Nz)
        self.tL.field_x = tX[1:, 1:, 1:] - tX[:-1, :-1, :-1]
        self.tL.field_y = tY[1:, 1:, 1:] - tY[:-1, :-1, :-1]
        self.tL.field_z = tZ[1:, 1:, 1:] - tZ[:-1, :-1, :-1]

        self.itA = Field(self.Nx, self.Ny, self.Nz)
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

    def _mpi_initialize(self):
        """
        Initialize MPI domain decomposition in the z direction.

        Sets up MPI communicator, rank, size, and subdomain grid quantities.
        """
        comm = MPI.COMM_WORLD  # Get MPI communicator
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Error handling for Nz < size
        if self.Nz < self.size:
            raise ValueError(
                f"Nz ({self.Nz}) must be greater than or equal to the number of \
                MPI processes ({self.size})."
            )

        # global z quantities [ALLCAPS]
        self.ZMIN = self.zmin
        self.ZMAX = self.zmax
        self.NZ = self.Nz - self.Nz % (
            self.size
        )  # ensure multiple of MPI size
        self.Z = np.linspace(self.ZMIN, self.ZMAX, self.NZ + 1)[:-1]
        self.Z += (self.ZMAX - self.ZMIN) / (2 * self.NZ)

        if self.verbose and self.rank == 0:
            print(
                f" * Global grid ZMIN={self.ZMIN}, ZMAX={self.ZMAX}, NZ={self.NZ}"
            )

        # MPI subdomain quantities [TODO: support non-uniform dz with MPI]
        self.Nz = self.NZ // (self.size)
        self.dz = (self.ZMAX - self.ZMIN) / self.NZ
        self.zmin = self.rank * self.Nz * self.dz + self.ZMIN
        self.zmax = (self.rank + 1) * self.Nz * self.dz + self.ZMIN

        if self.verbose:
            print(
                f"MPI rank {self.rank} of {self.size} initialized with \
                        zmin={self.zmin}, zmax={self.zmax}, Nz={self.Nz}"
            )
        # Add ghost cells
        self.n_ghosts = 1
        if self.rank > 0:
            self.zmin += -self.n_ghosts * self.dz
            self.Nz += self.n_ghosts
        if self.rank < (self.size - 1):
            self.zmax += self.n_ghosts * self.dz
            self.Nz += self.n_ghosts

        # Support for single core
        if self.rank == 0 and self.size == 1:
            self.zmax += self.n_ghosts * self.dz
            self.Nz += self.n_ghosts

        self.z = np.linspace(self.zmin, self.zmax, self.Nz + 1)
        self.dz = np.diff(self.z)  # only uniform grid possible with MPI

    def mpi_gather_asGrid(self):
        """
        Gather the global grid from all MPI subdomains and return a new GridFIT3D.

        Returns
        -------
        _grid : GridFIT3D or None
            The global grid object (only on rank 0).
        """
        _grid = None
        if self.rank == 0:
            print(f"Generating global grid from {self.ZMIN} to {self.ZMAX}")
            _grid = GridFIT3D(
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax,
                self.ZMIN,
                self.ZMAX,
                self.Nx,
                self.Ny,
                self.NZ,
                use_mpi=False,
                stl_solids=self.stl_solids,
                stl_materials=self.stl_materials,
                stl_scale=self.stl_scale,
                stl_rotate=self.stl_rotate,
                stl_translate=self.stl_translate,
                stl_colors=self.stl_colors,
                verbose=self.verbose,
                stl_tol=self.stl_tol,
            )
        return _grid

    def _prepare_stl_dicts(self):
        """
        Prepare STL-related dictionaries for rotation, scale, translation, and colors.

        Ensures all STL solids have corresponding entries for rotation, scale,
        translation, and color, converting single values to dicts as needed.
        """
        if type(self.stl_solids) is not dict:
            if type(self.stl_solids) is str:
                self.stl_solids = {"Solid 1": self.stl_solids}
            else:
                raise Exception(
                    "Attribute `stl_solids` must contain a string or a dictionary"
                )

        if type(self.stl_rotate) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_rotate = {}
            for key in self.stl_solids.keys():
                stl_rotate[key] = self.stl_rotate
            self.stl_rotate = stl_rotate

        if type(self.stl_scale) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_scale = {}
            for key in self.stl_solids.keys():
                stl_scale[key] = self.stl_scale
            self.stl_scale = stl_scale

        if type(self.stl_translate) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_translate = {}
            for key in self.stl_solids.keys():
                stl_translate[key] = self.stl_translate
            self.stl_translate = stl_translate

        if type(self.stl_colors) is not dict:
            if self.stl_colors is None:
                self._assign_colors()
            elif self.stl_colors is str:  # single color for all solids
                stl_colors = {}
                for key in self.stl_solids.keys():
                    stl_colors[key] = self.stl_colors
                self.stl_colors = stl_colors
            elif type(self.stl_colors) is list:
                stl_colors = {}
                try:
                    for i, key in enumerate(self.stl_solids.keys()):
                        stl_colors[key] = self.stl_colors[i]
                    self.stl_colors = stl_colors
                except IndexError:
                    self._assign_colors()
                    print(
                        "[!] If `stl_colors` is a list, it must have \
                        the same length as `stl_solids`."
                    )

    def _mark_cells_in_stl(self):
        """
        Mark grid cells that are inside each STL solid.

        Uses PyVista's select_enclosed_points to create boolean masks for each solid.
        """
        # Obtain masks with grid cells inside each stl solid
        stl_tolerance = (
            np.min([np.min(self.dx), np.min(self.dy), np.min(self.dz)])
            * self.stl_tol
        )
        progress_bar = False
        if self.Nx * self.Ny * self.Nz > 5e6 and self.verbose:
            progress_bar = True
        for key in self.stl_solids.keys():
            surf = self.read_stl(key)

            # mark cells in stl [True == in stl, False == out stl]
            try:
                select = self.grid.select_enclosed_points(
                    surf, tolerance=stl_tolerance, progress_bar=progress_bar
                )
            except Exception:
                select = self.grid.select_enclosed_points(
                    surf,
                    tolerance=stl_tolerance,
                    check_surface=False,
                    progress_bar=progress_bar,
                )
                if self.verbose > 1:
                    print(
                        f"[!] Warning: stl solid {key} may have issues with closed surfaces. \
                        Consider checking the STL file."
                    )

            self.grid[key] = (
                select.point_data_to_cell_data()["SelectedPoints"]
                > stl_tolerance
            )

            if self.verbose and np.sum(self.grid[key]) == 0:
                print(
                    f"[!] Warning: no cells were marked inside stl solid {key}. \
                    Consider increasing the tolerance factor (currently {self.stl_tol})."
                )

            if self.verbose > 1:
                print(
                    f" * STL solid {key}: {np.sum(self.grid[key])} cells marked inside the solid."
                )

    def read_stl(self, key):
        """
        Read and transform an STL solid by key.

        Parameters
        ----------
        key : str
            Key of the STL solid to read.

        Returns
        -------
        surf : pyvista.PolyData
            The transformed STL surface.
        """
        # import stl
        surf = pv.read(self.stl_solids[key])

        # rotate
        surf = surf.rotate_x(self.stl_rotate[key][0])
        surf = surf.rotate_y(self.stl_rotate[key][1])
        surf = surf.rotate_z(self.stl_rotate[key][2])

        # translate
        surf = surf.translate(self.stl_translate[key])

        # scale
        surf = surf.scale(self.stl_scale[key])

        return surf

    def _compute_snap_points(self, snap_solids=None, snap_tol=1e-8):
        """
        Compute snap points from STL feature edges for mesh refinement.

        Parameters
        ----------
        snap_solids : list or None, optional
            STL solids to use for snap point extraction. Default is all.
        snap_tol : float, optional
            Tolerance for snap point detection.
        """
        if self.verbose > 1:
            print(" * Calculating snappy points...")
        # Support for user-defined stl_keys as list
        if snap_solids is None:
            snap_solids = self.stl_solids.keys()

        # Union of all the surfaces
        model = None
        for key in snap_solids:
            solid = self.read_stl(key)
            if model is None:
                model = solid
            else:
                model = model + solid

        edges = model.extract_feature_edges(
            boundary_edges=True, manifold_edges=False
        )

        # Extract points lying in the X-Z plane (Y ≈ 0)
        xz_plane_points = edges.points[np.abs(edges.points[:, 1]) < snap_tol]
        # Extract points lying in the Y-Z plane (X ≈ 0)
        yz_plane_points = edges.points[np.abs(edges.points[:, 0]) < snap_tol]
        # Extract points lying in the X-Y plane (Z ≈ 0)
        xy_plane_points = edges.points[np.abs(edges.points[:, 2]) < snap_tol]

        self.snap_points = np.r_[
            xz_plane_points, yz_plane_points, xy_plane_points
        ]

        # get the unique x, y, z coordinates
        x_snaps = np.unique(np.round(self.snap_points[:, 0], 5))
        y_snaps = np.unique(np.round(self.snap_points[:, 1], 5))
        z_snaps = np.unique(np.round(self.snap_points[:, 2], 5))

        # Include simulation domain bounds
        self.x_snaps = np.unique(
            np.concatenate(([self.xmin], x_snaps, [self.xmax]))
        )
        self.y_snaps = np.unique(
            np.concatenate(([self.ymin], y_snaps, [self.ymax]))
        )
        self.z_snaps = np.unique(
            np.concatenate(([self.zmin], z_snaps, [self.zmax]))
        )

    def plot_snap_points(self, snap_solids=None, snap_tol=1e-8):
        """
        Plot snap points extracted from STL feature edges for mesh refinement.

        Parameters
        ----------
        snap_solids : list or None, optional
            STL solids to use for snap point extraction. Default is all.
        snap_tol : float, optional
            Tolerance for snap point detection.
        """
        # Support for user-defined stl_keys as list
        if snap_solids is None:
            snap_solids = self.stl_solids.keys()

        # Union of all the surfaces
        model = None
        for key in snap_solids:
            solid = self.read_stl(key)
            if model is None:
                model = solid
            else:
                model = model + solid

        edges = model.extract_feature_edges(
            boundary_edges=True, manifold_edges=False
        )

        # Extract points lying in the X-Z plane (Y ≈ 0)
        xz_plane_points = edges.points[np.abs(edges.points[:, 1]) < snap_tol]
        # Extract points lying in the Y-Z plane (X ≈ 0)
        yz_plane_points = edges.points[np.abs(edges.points[:, 0]) < snap_tol]
        # Extract points lying in the X-Y plane (Z ≈ 0)
        xy_plane_points = edges.points[np.abs(edges.points[:, 2]) < snap_tol]

        xz_cloud = pv.PolyData(xz_plane_points)
        yz_cloud = pv.PolyData(yz_plane_points)
        xy_cloud = pv.PolyData(xy_plane_points)

        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter()
        pl.add_mesh(model, color="white", opacity=0.5, label="base STL")
        pl.add_mesh(
            edges,
            color="black",
            line_width=5,
            opacity=0.8,
        )
        pl.add_mesh(
            xz_cloud,
            color="green",
            point_size=20,
            render_points_as_spheres=True,
            label="XZ plane points",
        )
        pl.add_mesh(
            yz_cloud,
            color="orange",
            point_size=20,
            render_points_as_spheres=True,
            label="YZ plane points",
        )
        pl.add_mesh(
            xy_cloud,
            color="magenta",
            point_size=20,
            render_points_as_spheres=True,
            label="XY plane points",
        )
        pl.add_legend()
        pl.show()

    def refine_axis(self, xmin, xmax, Nx, x_snaps, method="insert", tol=1e-12):
        """
        Refine a grid axis using snap points and a chosen method.

        Parameters
        ----------
        xmin, xmax : float
            Axis bounds.
        Nx : int
            Number of grid points.
        x_snaps : array_like
            Snap points to include in the axis.
        method : str, optional
            Refinement algorithm ('insert', 'neighbor', 'subdivision').
        tol : float, optional
            Convergence tolerance for optimization.

        Returns
        -------
        x : ndarray
            Refined axis array.
        """

        # Loss function to minimize cell size spread
        def loss_function(x, x0, is_snap):
            # avoid moving snap points
            penalty_snap = np.sum((x[is_snap] - x0[is_snap]) ** 2) * 1000
            # avoid gaps < uniform gap
            dx = np.diff(x)
            threshold = 1 / (len(x) - 1)  # or a hardcoded `min_spacing`
            penalty_small_gaps = np.sum((threshold - dx[dx < threshold]) ** 2)
            # avoid large spread in gap length
            dx = np.diff(x)
            penalty_variance = np.std(dx) * 10
            # return penalty_snap + penalty_small_gaps + penalty_variance
            return np.hstack(
                [penalty_snap, penalty_small_gaps, penalty_variance]
            )

        # Uniformly distributed points as initial guess
        x_snaps = (x_snaps - xmin) / (xmax - xmin)  # normalize to [0,1]

        if method == "insert":
            x0 = np.unique(
                np.append(x_snaps, np.linspace(0, 1, Nx - len(x_snaps)))
            )

        elif method == "neighbor":
            x = np.linspace(0, 1, Nx)
            dx = np.diff(x)[0]
            for s in x_snaps:
                m = np.isclose(x, s, rtol=0.0, atol=dx / 2)
                if np.sum(m) > 0:
                    x[np.argmax(m)] = s
            x0 = x.copy()

        elif method == "subdivision":
            # x = snaps
            while len(x) < Nx:
                # idx of segments sorted min -> max
                idx_max_diffs = np.argsort(np.diff(x))[-1]  # take bigger

                # print(f"Bigger segment starts at {x[idx_max_diffs]}")
                # compute new point in the middle of the segment
                val = (
                    x[idx_max_diffs]
                    + (x[idx_max_diffs + 1] - x[idx_max_diffs]) / 2
                )

                # insert the new point
                x = np.insert(x, idx_max_diffs + 1, val)
                x = np.unique(x)
                # print(f"Inserted point {val} at index {idx_max_diffs}")
            x0 = x.copy()
        else:
            raise ValueError(
                f"Method {method} not supported. Use 'insert', 'neighbor' or 'subdivision'."
            )

        # minimize segment length spread for the test points
        is_snap = np.isin(x0, x_snaps)
        result = least_squares(
            loss_function,
            x0=x0.copy(),
            bounds=(0, 1),  # (zmin, zmax),
            jac="3-point",
            method="dogbox",
            loss="arctan",
            gtol=tol,
            ftol=tol,
            xtol=tol,
            verbose=0,
            args=(x0.copy(), is_snap.copy()),
        )
        # transform back to [xmin, xmax]
        return result.x * (xmax - xmin) + xmin

    def _refine_xyz_axis(self, method="insert", tol=1e-6):
        """
        Refine grid axes using snap points extracted from STL solids.

        Uses the stored snap points (``self.x_snaps``, ``self.y_snaps``,
        ``self.z_snaps``) to refine the axis arrays ``self.x``, ``self.y``,
        ``self.z``. The refinement method and convergence tolerance control
        how new grid nodes are inserted.

        Parameters
        ----------
        method : {'insert','neighbor','subdivision'}, optional
            Refinement algorithm to use when inserting snap points. Default is
            'insert'.
        tol : float, optional
            Convergence tolerance passed to the refinement routine.
        """

        if self.verbose > 1:
            print(f" * Refining x axis with {len(self.x_snaps)} snaps...")
        self.x = self.refine_axis(
            self.xmin,
            self.xmax,
            self.Nx + 1,
            self.x_snaps,
            method=method,
            tol=tol,
        )

        if self.verbose > 1:
            print(f" * Refining y axis with {len(self.y_snaps)} snaps...")
        self.y = self.refine_axis(
            self.ymin,
            self.ymax,
            self.Ny + 1,
            self.y_snaps,
            method=method,
            tol=tol,
        )

        if self.verbose > 1:
            print(f" * Refining z axis with {len(self.z_snaps)} snaps...")
        self.z = self.refine_axis(
            self.zmin,
            self.zmax,
            self.Nz + 1,
            self.z_snaps,
            method=method,
            tol=tol,
        )

        self.Nx = len(self.x) - 1
        self.Ny = len(self.y) - 1
        self.Nz = len(self.z) - 1
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)

        if self.verbose > 1:
            print(
                f"Refined grid: Nx = {self.Nx}, Ny ={self.Ny}, Nz = {self.Nz}"
            )

    def _assign_colors(self):
        """
        Assign colors for each STL solid based on material categories.

        Maps entries in ``self.stl_materials`` to color names using the
        ``material_colors`` lookup. Supports string keys referencing the
        material library or explicit material tuples (eps_r, mu_r[, sigma]).
        The resulting mapping is stored in ``self.stl_colors``.
        """
        self.stl_colors = {}

        for key in self.stl_solids:
            mat = self.stl_materials[key]
            if type(mat) is str:
                _color = material_colors.get(mat, material_colors["other"])
                self.stl_colors[key] = _color
            elif len(mat) == 2:
                if mat[0] is np.inf:  # eps_r
                    self.stl_colors[key] = material_colors["pec"]
                elif mat[0] > 1.0:  # eps_r
                    self.stl_colors[key] = material_colors["dielectric"]
                else:
                    self.stl_colors[key] = material_colors["vacuum"]
            elif len(mat) == 3:
                self.stl_colors[key] = material_colors["lossy metal"]
            else:
                self.stl_colors[key] = material_colors["other"]

    def _add_logo_widget(self, pl):
        """
        Add packaged logo to a PyVista plotter via importlib.resources.

        Attempts to load a packaged logo image from the installed package
        resources. Falls back to a local development path when resources are
        not available (typical in editable/dev installs).
        """
        try:
            from importlib import resources

            # resource inside the installed package (use current package)
            logo_res = resources.files(__package__).joinpath(
                "static", "img", "wakis-logo-pink.png"
            )
            with resources.as_file(logo_res) as logo_path:
                pl.add_logo_widget(str(logo_path))
                return
        except Exception as e:
            # fallback to the legacy relative path for dev installs
            try:
                pl.add_logo_widget("../docs/img/wakis-logo-pink.png")
            except Exception:
                if self.verbose > 1:
                    print(f"[!] Could not add logo widget: {e}")

    def plot_solids(
        self,
        bounding_box=False,
        show_grid=False,
        anti_aliasing=None,
        opacity=1.0,
        specular=0.5,
        smooth_shading=False,
        off_screen=False,
        **kwargs,
    ):
        """
        Generate a 3D visualization of imported STL geometries using PyVista.

        Parameters
        ----------
        bounding_box : bool, optional
            If True, adds a bounding box around the plotted geometry (default False).
        show_grid : bool, optional
            If True, overlays the grid wireframe on the scene (default False).
        anti_aliasing : str or None, optional
            Anti-aliasing mode passed to PyVista (default: None).
        opacity : float, optional
            Opacity for solids (1.0 opaque, 0.0 transparent). Default 1.0.
        specular : float, optional
            Specular lighting strength, higher is shinier (default 0.5).
        smooth_shading : bool, optional
            Enable smooth shading for surface rendering (default False).
        off_screen : bool, optional
            If True, export to HTML instead of opening an interactive window.
        **kwargs : dict
            Additional keyword args forwarded to ``pyvista.add_mesh``.

        Notes
        -----
        - Colors come from ``self.stl_colors`` when available.
        - Solids labeled 'vacuum' are rendered with reduced opacity by default.
        """
        pl = pv.Plotter()
        pl.add_mesh(self.grid, opacity=0.0, name="grid", show_scalar_bar=False)
        for key in self.stl_solids:
            color = self.stl_colors[key]
            if self.stl_materials[key] == "vacuum":
                _opacity = 0.3
            else:
                _opacity = opacity
            pl.add_mesh(
                self.read_stl(key),
                color=color,
                opacity=_opacity,
                specular=specular,
                smooth_shading=smooth_shading,
                **kwargs,
            )

        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.camera_position = "zx"
        pl.camera.azimuth += 30
        pl.camera.elevation += 30
        pl.add_axes()

        if anti_aliasing is not None:
            pl.enable_anti_aliasing(anti_aliasing)

        if bounding_box:
            pl.add_bounding_box()

        if show_grid:
            pl.add_mesh(
                self.grid,
                style="wireframe",
                color="grey",
                opacity=0.3,
                name="grid",
            )

        if off_screen:
            return pl
            # pl.export_html("grid_plot_solids.html")
        else:
            pl.show()

    def plot_stl_mask(
        self,
        stl_solid,
        cmap="viridis",
        bounding_box=True,
        show_grid=True,
        add_stl="all",
        stl_opacity=0.0,
        stl_colors=None,
        xmax=None,
        ymax=None,
        zmax=None,
        anti_aliasing="ssaa",
        smooth_shading=False,
        off_screen=False,
    ):
        """
        Interactive 3D visualization of the structured grid mask and imported STL
        geometries.

        This routine uses PyVista to display the grid scalar field corresponding to a
        chosen STL mask. It provides interactive slider widgets to clip the domain
        along the X, Y, and Z directions. At each slider position, the clipped
        scalar field is shown with a colormap while the grid structure is shown
        as a 2D slice in wireframe. Optionally, one or more STL geometries can
        be added to the scene, along with a bounding box of the simulation domain.

        Parameters
        ----------
        stl_solid : str
            Key name of the `stl_solids` dictionary to retrieve the mask for
            visualization (used as the scalar field).
        cmap : str, optional
            Colormap used to visualize the clipped scalar values. Default 'viridis'.
        bounding_box : bool, optional
            If True, add a static wireframe bounding box of the simulation domain.
        show_grid : bool, optional
            If True, adds the computational grid overlay on the clipped slice.
        add_stl : {'all', str, list[str]}, optional
            STL geometries to add. Default 'all'.
        stl_opacity : float, optional
            Opacity of the STL surfaces (0 = fully transparent, 1 = fully opaque).
        stl_colors : str, list[str], dict, or None, optional
            Color(s) of the STL surfaces.
        xmax, ymax, zmax : float, optional
            Initial clipping positions along each axis. If None, use the maximum domain
            extent.
        anti_aliasing : {'ssaa', 'fxaa', None}, optional
            Anti-aliasing mode passed to `pl.enable_anti_aliasing`.
        smooth_shading : bool, optional
            Enable smooth shading for STL surfaces. Default False.
        off_screen : bool, optional
            If True, render off-screen and export the scene to HTML.

        Notes
        -----
        - Three sliders (X, Y, Z) control clipping of the scalar field by a box.
        - STL solids can be visualized in transparent mode.
        - A static domain bounding box can be added for reference.
        """
        if stl_colors is None:
            stl_colors = self.stl_colors

        if xmax is None:
            xmax = self.xmax
        if ymax is None:
            ymax = self.ymax
        if zmax is None:
            zmax = self.zmax

        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter()
        vals = {"x": xmax, "y": ymax, "z": zmax}

        # --- Update function ---
        def update_clip(val, axis="x"):
            vals[axis] = val
            # define bounds dynamically
            if axis == "x":
                slice_obj = self.grid.slice(normal="x", origin=(val, 0, 0))
            elif axis == "y":
                slice_obj = self.grid.slice(normal="y", origin=(0, val, 0))
            else:  # z
                slice_obj = self.grid.slice(normal="z", origin=(0, 0, val))

            # add clipped volume (scalars)
            pl.add_mesh(
                self.grid.clip_box(
                    bounds=(
                        self.xmin,
                        vals["x"],
                        self.ymin,
                        vals["y"],
                        self.zmin,
                        vals["z"],
                    ),
                    invert=False,
                ),
                scalars=stl_solid,
                cmap=cmap,
                name="clip",
            )

            # add slice wireframe (grid structure)
            if show_grid:
                pl.add_mesh(
                    slice_obj, style="wireframe", color="grey", name="slice"
                )

        # Plot stl surface(s)
        if add_stl is not None:
            if type(add_stl) is str:  # add all stl solids
                if add_stl.lower() == "all":
                    for i, key in enumerate(self.stl_solids):
                        surf = self.read_stl(key)
                        if type(stl_colors) is dict:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[key],
                                opacity=stl_opacity,
                                silhouette=dict(color=stl_colors[key]),
                                name=key,
                            )
                        elif type(stl_colors) is list:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[i],
                                opacity=stl_opacity,
                                silhouette=dict(color=stl_colors[i]),
                                name=key,
                            )
                        else:
                            pl.add_mesh(
                                surf,
                                color="white",
                                opacity=stl_opacity,
                                silhouette=True,
                                name=key,
                            )
                else:  # add 1 selected stl solid
                    key = add_stl
                    surf = self.read_stl(key)
                    pl.add_mesh(
                        surf,
                        color=stl_colors[key],
                        opacity=stl_opacity,
                        silhouette=dict(color=stl_colors[key]),
                        name=key,
                    )

            elif type(add_stl) is list:  # add selected list of stl solids
                for i, key in enumerate(add_stl):
                    surf = self.read_stl(key)
                    if type(stl_colors[key]) is dict:
                        pl.add_mesh(
                            surf,
                            color=stl_colors[key],
                            opacity=stl_opacity,
                            silhouette=dict(color=stl_colors[key]),
                            name=key,
                        )
                    elif type(stl_colors) is list:
                        pl.add_mesh(
                            surf,
                            color=stl_colors[i],
                            opacity=stl_opacity,
                            silhouette=dict(color=stl_colors[i]),
                            name=key,
                        )
                    else:
                        pl.add_mesh(
                            surf,
                            color="white",
                            opacity=stl_opacity,
                            silhouette=True,
                            name=key,
                        )

        # --- Sliders (placed side-by-side vertically) ---
        pl.add_slider_widget(
            lambda val: update_clip(val, "x"),
            [self.xmin, self.xmax],
            value=xmax,
            title="X Clip",
            pointa=(0.8, 0.8),
            pointb=(0.95, 0.8),  # top-right
            style="modern",
        )

        pl.add_slider_widget(
            lambda val: update_clip(val, "y"),
            [self.ymin, self.ymax],
            value=ymax,
            title="Y Clip",
            pointa=(0.8, 0.6),
            pointb=(0.95, 0.6),  # middle-right
            style="modern",
        )

        pl.add_slider_widget(
            lambda val: update_clip(val, "z"),
            [self.zmin, self.zmax],
            value=zmax,
            title="Z Clip",
            pointa=(0.8, 0.4),
            pointb=(0.95, 0.4),  # lower-right
            style="modern",
        )

        # Camera orientation
        pl.camera_position = "zx"
        pl.camera.azimuth += 30
        pl.camera.elevation += 30
        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.add_axes()
        pl.enable_3_lights()
        pl.enable_anti_aliasing(anti_aliasing)

        if bounding_box:
            pl.add_mesh(
                pv.Box(
                    bounds=(
                        self.xmin,
                        self.xmax,
                        self.ymin,
                        self.ymax,
                        self.zmin,
                        self.zmax,
                    )
                ),
                style="wireframe",
                color="black",
                line_width=2,
                name="domain_box",
            )

        if off_screen:
            return pl
            # pl.export_html(f"grid_stl_mask_{stl_solid}.html")
        else:
            pl.show()

    def inspect(
        self,
        add_stl=None,
        stl_opacity=0.5,
        stl_colors=None,
        anti_aliasing="ssaa",
        smooth_shading=True,
        off_screen=False,
    ):
        """
        Interactive 3D inspector showing grid and STL geometries.

        Parameters
        ----------
        add_stl : str or list, optional
            Key or list of keys of STL solids to include. If None, all solids are shown.
        stl_opacity : float, optional
            Opacity for STL surfaces (0 transparent, 1 opaque). Default 0.5.
        stl_colors : str, list, or dict, optional
            Color specification for STL surfaces; defaults to ``self.stl_colors``.
        anti_aliasing : str or None, optional
            Anti-aliasing mode to enable in the plotter (default 'ssaa').
        smooth_shading : bool, optional
            Enable smooth shading for surfaces (default True).
        off_screen : bool, optional
            If True, return the off-screen plotter object instead of showing an
            interactive window.

        Returns
        -------
        pl : pyvista.Plotter or None
            The plotter object if off_screen is True, otherwise None.
        """
        if stl_colors is None:
            stl_colors = self.stl_colors

        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter()
        pl.add_mesh(
            self.grid, show_edges=True, cmap=["white", "white"], name="grid"
        )

        def clip(widget):
            # Plot structured grid
            b = widget.bounds
            x = self.x[np.logical_and(self.x >= b[0], self.x <= b[1])]
            y = self.y[np.logical_and(self.y >= b[2], self.y <= b[3])]
            z = self.z[np.logical_and(self.z >= b[4], self.z <= b[5])]
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            grid = pv.StructuredGrid(
                X.transpose(), Y.transpose(), Z.transpose()
            )

            pl.add_mesh(
                grid, show_edges=True, cmap=["white", "white"], name="grid"
            )
            # Plot stl surface(s)
            if add_stl is not None:  # add 1 selected stl solid
                if type(add_stl) is str:
                    key = add_stl
                    surf = self.read_stl(key)
                    surf = surf.clip_box(widget.bounds, invert=False)
                    pl.add_mesh(
                        surf,
                        color=stl_colors[key],
                        opacity=stl_opacity,
                        silhouette=True,
                        smooth_shading=smooth_shading,
                        name=key,
                    )

                elif type(add_stl) is list:  # add selected list of stl solids
                    for i, key in enumerate(add_stl):
                        surf = self.read_stl(key)
                        surf = surf.clip_box(widget.bounds, invert=False)
                        if type(stl_colors) is dict:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[key],
                                opacity=stl_opacity,
                                silhouette=True,
                                smooth_shading=smooth_shading,
                                name=key,
                            )
                        elif type(stl_colors) is list:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[i],
                                opacity=stl_opacity,
                                silhouette=True,
                                smooth_shading=smooth_shading,
                                name=key,
                            )
                        else:
                            pl.add_mesh(
                                surf,
                                color="white",
                                opacity=stl_opacity,
                                silhouette=True,
                                smooth_shading=smooth_shading,
                                name=key,
                            )

            else:  # add all stl solids
                for i, key in enumerate(self.stl_solids):
                    surf = self.read_stl(key)
                    surf = surf.clip_box(widget.bounds, invert=False)
                    if type(stl_colors) is dict:
                        pl.add_mesh(
                            surf,
                            color=stl_colors[key],
                            opacity=stl_opacity,
                            silhouette=True,
                            smooth_shading=smooth_shading,
                            name=key,
                        )
                    elif type(stl_colors) is list:
                        pl.add_mesh(
                            surf,
                            color=stl_colors[i],
                            opacity=stl_opacity,
                            silhouette=True,
                            smooth_shading=smooth_shading,
                            name=key,
                        )
                    else:
                        pl.add_mesh(
                            surf,
                            color="white",
                            opacity=stl_opacity,
                            silhouette=True,
                            smooth_shading=smooth_shading,
                            name=key,
                        )

        _ = pl.add_box_widget(callback=clip, rotation_enabled=False)

        # Camera orientation
        pl.camera_position = "zx"
        pl.camera.azimuth += 30
        pl.camera.elevation += 30
        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.add_axes()
        pl.enable_3_lights()
        pl.enable_anti_aliasing(anti_aliasing)

        if off_screen:
            pl.off_screen = True
            return pl
            # pl.export_html('grid_inspect.html')
        else:
            pl.show()
            return None

    def save_to_h5(self, filename="grid.h5"):
        """
        Save the generated grid and STL metadata to an HDF5 file.

        The file contains axis arrays, STL masks suitable for ``grid.cell_data``,
        and all ``stl_`` related variables (materials, colors, transforms).

        Parameters
        ----------
        filename : str, optional
            Output filename for the HDF5 file. Default 'grid.h5'.
        """

        if not filename.endswith(".h5"):
            filename += ".h5"

        if self.verbose:
            print("Saving grid to HDF5 file:", filename)

        with h5py.File(filename, "w") as hf:
            hf.create_dataset("x", data=np.array(self.x))
            hf.create_dataset("y", data=np.array(self.y))
            hf.create_dataset("z", data=np.array(self.z))

            # Save stl_ variables as groups
            for attr in [
                "stl_solids",
                "stl_materials",
                "stl_colors",
                "stl_scale",
                "stl_rotate",
                "stl_translate",
            ]:
                grp = hf.create_group(attr)
                dct = getattr(self, attr)
                for key, val in dct.items():
                    # Use dtype='S' for strings, otherwise np.array
                    if isinstance(val, str):
                        grp.create_dataset(str(key), data=np.bytes_(val))
                    else:
                        grp.create_dataset(str(key), data=np.array(val))

            for key in self.stl_solids.keys():
                hf.create_dataset("grid_" + key, data=np.array(self.grid[key]))

    def load_from_h5(self, filename):
        """
        Load grid axis arrays and STL metadata from an HDF5 file.

        The function restores axis arrays, recomputes grid metrics and fills
        ``self.grid`` cell_data with imported STL masks saved previously by
        ``save_to_h5``.

        Parameters
        ----------
        filename : str
            HDF5 filename to read. The '.h5' suffix is appended if missing.
        """
        if not filename.endswith(".h5"):
            filename += ".h5"

        if self.verbose:
            print("Loading grid from HDF5 file:", filename)

        with h5py.File(filename, "r") as hf:
            # reconstruct stl dicts
            self.x = hf["x"][()]
            self.y = hf["y"][()]
            self.z = hf["z"][()]

            # Load stl_ variables from groups
            for attr in [
                "stl_solids",
                "stl_materials",
                "stl_colors",
                "stl_scale",
                "stl_rotate",
                "stl_translate",
            ]:
                dct = {}
                grp = hf[attr]
                for key in grp.keys():
                    val = grp[key][()]
                    # Decode bytes to string if needed
                    if isinstance(val, bytes):
                        val = val.decode()
                    dct[key] = val
                setattr(self, attr, dct)

        # recompute dx, dy, dz, Nx, Ny, Nz
        self.Nx = len(self.x) - 1
        self.Ny = len(self.y) - 1
        self.Nz = len(self.z) - 1
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.dz = np.diff(self.z)
        self.xmin, self.xmax = self.x[0], self.x[-1]
        self.ymin, self.ymax = self.y[0], self.y[-1]
        self.zmin, self.zmax = self.z[0], self.z[-1]

        # recommpute grid and L, iA, tL, itA
        self._compute_grid()

        # asign masks to grid.cell_data
        with h5py.File(filename, "r") as hf:
            for key in self.stl_solids.keys():
                self.grid[key] = hf["grid_" + key][()]

        # add verbosity
        if self.verbose > 1:
            print(
                f"Loaded grid with {self.Nx * self.Ny * self.Nz} mesh cells:"
            )
            print(
                f" * Number of cells: Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}"
            )
            print(
                f" * Simulation domain bounds: \n\
                x:[{self.xmin:.3f}, {self.xmax:.3f}],\n\
                y:[{self.ymin:.3f}, {self.ymax:.3f}],\n\
                z:[{self.zmin:.3f}, {self.zmax:.3f}]"
            )
            print(
                f" * STL solids imported:\n\
                {list(self.stl_solids.keys())}"
            )
            print(
                f" * STL solids assigned materials [eps_r, mu_r, sigma]:\n\
                {list(self.stl_materials.values())}"
            )

        # update logger
        self.update_logger(["Nx", "Ny", "Nz", "dx", "dy", "dz"])
        self.update_logger(["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])
        self.update_logger(["stl_solids", "stl_materials"])
        if self.stl_rotate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_rotate"])
        if self.stl_translate != [0.0, 0.0, 0.0]:
            self.update_logger(["stl_translate"])
        if self.stl_scale != 1.0:
            self.update_logger(["stl_scale"])

    def update_logger(self, attrs):
        """
        Copy selected Grid attributes into the internal ``Logger``.

        Parameters
        ----------
        attrs : iterable of str
            Names of attributes to copy into ``self.logger.grid``.
        """
        for atr in attrs:
            self.logger.grid[atr] = getattr(self, atr)
