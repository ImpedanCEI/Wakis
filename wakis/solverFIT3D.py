# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import time

import h5py
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0
from scipy.sparse import csc_matrix as sparse_mat
from scipy.sparse import diags, hstack, vstack

from .boundaries import BCsMixin
from .field import Field
from .logger import Logger
from .materials import material_lib
from .plotting import PlotMixin
from .routines import RoutinesMixin

try:
    from cupyx.scipy.sparse import csc_matrix as gpu_sparse_mat

    imported_cupyx = True
except ImportError:
    imported_cupyx = False

try:
    from sparse_dot_mkl import csr_matrix as mkl_sparse_mat
    from sparse_dot_mkl import dot_product_mkl

    imported_mkl = True
except ImportError:
    imported_mkl = False


class SolverFIT3D(PlotMixin, RoutinesMixin, BCsMixin):
    def __init__(
        self,
        grid,
        wake=None,
        cfln=0.5,
        dt=None,
        bc_low=["Periodic", "Periodic", "Periodic"],
        bc_high=["Periodic", "Periodic", "Periodic"],
        use_stl=False,
        use_conductors=False,
        use_gpu=False,
        use_mpi=False,
        use_sibc=True,
        fmax=None,
        dtype=np.float64,
        n_pml=10,
        bg=[1.0, 1.0],
        verbose=1,
    ):
        """
        3D time-domain electromagnetic solver based on the Finite Integration
        Technique (FIT).

        Handles mesh and geometry, material assignment, boundary conditions and
        time-stepping. Supports CPU, optional GPU acceleration (cupyx) and MPI
        domain decomposition. Provides utilities for importing conductors and
        STL solids, applying PML/ABC boundaries, and saving/restoring solver
        state.

        Parameters
        ----------
        grid : GridFIT3D
            Instance providing mesh, coordinate arrays and geometry flags.
        wake : WakeSolver, optional
            Wakefield object with beam parameters used for wake computations.
        cfln : float, optional
            CFL number used to compute a stable timestep when ``dt`` is None.
        dt : float, optional
            Explicit timestep. If provided, it overrides the CFL-based value.
        bc_low, bc_high : list of str, optional
            Boundary conditions for low/high faces in (x, y, z) order.
        use_stl : bool, optional
            If True, apply solids and materials provided in the ``grid`` object.
        use_conductors : bool, optional
            [LEGACY] will be removed in future releases.
            If True, import conductor geometry from ``conductors.py`` masks.
        use_gpu : bool, optional
            Enable GPU acceleration via ``cupyx`` (if available).
        use_mpi : bool, optional
            Enable MPI execution for a subdivided grid.
        dtype : numpy dtype, optional
            Numeric dtype for solver arrays (default ``np.float64``).
        n_pml : int, optional
            Number of PML cells for PML boundary regions.
        bg : sequence or str, optional
            Background material [eps_r, mu_r, sigma] or a material key from
            the library. If a sigma value is provided conductivity handling is
            enabled.
        verbose : int or bool, optional
            Verbosity flag for initialization messages.

        Attributes
        ----------
        E, H, J : wakis.Field
            Electric field, magnetic field and current density containers.
            Access components via labels 'x','y','z'. Example:
            ``solver.E[:, :, n, 'z']`` gives Ez at z-index n.
        ieps, imu, sigma : wakis.Field
            Material tensors (inverse permittivity, inverse permeability and
            conductivity) stored per field component.
        grid : GridFIT3D
            Reference to the input grid object.
        dt : float
            Time-step used for time integration.
        cfln : float
            CFL number used when computing dt from grid spacing.
        """

        self.verbose = verbose
        t0 = time.time()
        self.logger = Logger()

        # Flags
        self.step_0 = True
        self.nstep = int(0)
        self.plotter_active = False
        self.use_conductors = use_conductors
        self.use_stl = use_stl
        self.use_gpu = use_gpu
        self.use_mpi = use_mpi
        self.use_sibc = use_sibc  # surface impedance boundary condition
        self.fmax = (
            fmax  # maximum frequency for SIBC, used to compute surface impedance
        )
        self.activate_abc = False  # Will turn true if abc BCs are chosen
        self.activate_pml = False  # Will turn true if pml BCs are chosen
        self.use_conductivity = False  # Will turn true with conductive material or pml
        self.imported_mkl = imported_mkl  # Use MKL backend when available
        self.one_step = self._one_step

        if wake is not None and fmax is None:
            self.fmax = self.wake.fmax
        if verbose > 1:
            print(f"* Maximum frequency set to fmax={self.fmax / 1e9} GHz")

        if use_stl:
            self.use_conductors = False
        self.update_logger(["use_gpu", "use_mpi"])

        # Grid
        self.grid = grid
        self.background = bg
        self.Nx = self.grid.Nx
        self.Ny = self.grid.Ny
        self.Nz = self.grid.Nz
        self.N = self.Nx * self.Ny * self.Nz

        self.dx = self.grid.dx
        self.dy = self.grid.dy
        self.dz = self.grid.dz

        self.x = self.grid.x[:-1] + self.dx / 2
        self.y = self.grid.y[:-1] + self.dy / 2
        self.z = self.grid.z[:-1] + self.dz / 2

        self.L = self.grid.L
        self.iA = self.grid.iA
        self.tL = self.grid.tL
        self.itA = self.grid.itA
        self.update_logger(["grid", "background"])

        # Wake computation
        self.wake = wake
        if self.wake is not None:
            self.logger.wakeSolver = self.wake.logger.wakeSolver

        # Fields
        self.dtype = dtype
        self.E = Field(
            self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype
        )
        self.H = Field(
            self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype
        )
        self.J = Field(
            self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype
        )

        # MPI init
        if self.use_mpi:
            if self.grid.use_mpi:
                self._mpi_initialize()
                self.one_step = self._mpi_one_step
            else:
                print(
                    "[!] Grid not subdivided for MPI, set `use_mpi`=True also in \
                    `GridFIT3D` to enable MPI"
                )

        # Matrices
        if verbose:
            print("Assembling operator matrices...")
        N = self.N
        self.Px = diags([-1, 1], [0, 1], shape=(N, N), dtype=np.int8)
        self.Py = diags([-1, 1], [0, self.Nx], shape=(N, N), dtype=np.int8)
        self.Pz = diags([-1, 1], [0, self.Nx * self.Ny], shape=(N, N), dtype=np.int8)

        # original grid
        self.Ds = diags(self.L.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)
        self.iDa = diags(self.iA.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)

        # tilde grid
        self.tDs = diags(self.tL.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)
        self.itDa = diags(self.itA.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)

        # Curl matrix
        self.C = vstack(
            [
                hstack([sparse_mat((N, N)), -self.Pz, self.Py]),
                hstack([self.Pz, sparse_mat((N, N)), -self.Px]),
                hstack([-self.Py, self.Px, sparse_mat((N, N))]),
            ],
            dtype=np.int8,
        )

        # Boundaries
        if verbose:
            print("Applying boundary conditions...")
        self.bc_low = bc_low
        self.bc_high = bc_high
        self.update_logger(["bc_low", "bc_high"])
        self._apply_bc_to_C()

        # Materials
        if verbose:
            print("Adding material tensors...")
        if type(bg) is str:
            bg = material_lib[bg.lower()]

        if len(bg) == 3:
            self.eps_bg, self.mu_bg, self.sigma_bg = (
                bg[0] * eps_0,
                bg[1] * mu_0,
                bg[2],
            )
            self.use_conductivity = True
        else:
            self.eps_bg, self.mu_bg, self.sigma_bg = (
                bg[0] * eps_0,
                bg[1] * mu_0,
                0.0,
            )

        # fmt: off
        self.ieps = (
            Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)
            * (1.0 / self.eps_bg)
        )
        self.imu = (
            Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)
            * (1.0 / self.mu_bg)
        )
        self.sigma = (
            Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)
            * self.sigma_bg
        )
        # fmt: on

        if self.use_stl:
            self._apply_stl_materials()

        # Fill PML BCs
        if self.activate_pml:
            if verbose:
                print("Filling PML sigmas...")
            self.n_pml = n_pml
            self._initialize_PML()
            self.update_logger(["n_pml"])

        # Timestep calculation
        if verbose:
            print("Calculating maximal stable timestep...")
        self.cfln = cfln
        if dt is None:
            self.dt = cfln / (
                c_light
                * np.sqrt(
                    1 / np.min(self.grid.dx) ** 2
                    + 1 / np.min(self.grid.dy) ** 2
                    + 1 / np.min(self.grid.dz) ** 2
                )
            )
        else:
            self.dt = dt
        self.dt = self.dtype(self.dt)
        self.update_logger(["dt"])

        if self.use_conductivity:  # relaxation time criterion tau
            mask = np.logical_and(
                self.sigma.toarray() != 0,  # for non-conductive
                self.ieps.toarray() != 0,
            )  # for PEC eps=inf

            self.tau = (1 / self.ieps.toarray()[mask]) / self.sigma.toarray()[mask]

            if self.dt > self.tau.min():
                self.dt = self.tau.min()

        # Pre-computing
        if verbose:
            print("Pre-computing...")
        self.iDeps = diags(self.ieps.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)
        self.iDmu = diags(self.imu.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype)
        self.Dsigma = diags(
            self.sigma.toarray(), shape=(3 * N, 3 * N), dtype=self.dtype
        )

        self.tDsiDmuiDaC = self.iDa * self.iDmu * self.C * self.Ds
        self.itDaiDepsDstC = self.iDeps * self.itDa * self.C.transpose() * self.tDs

        if imported_mkl and not self.use_gpu:  # MKL backend for CPU
            if verbose:
                print("Using MKL backend for time-stepping...")
            self.tDsiDmuiDaC = mkl_sparse_mat(self.tDsiDmuiDaC)
            self.itDaiDepsDstC = mkl_sparse_mat(self.itDaiDepsDstC)
            self.one_step = (
                self._mpi_one_step_mkl if self.use_mpi else self._one_step_mkl
            )

        # Move to GPU
        if use_gpu:
            if verbose:
                print("Moving to GPU...")
            if imported_cupyx:
                self.tDsiDmuiDaC = gpu_sparse_mat(self.tDsiDmuiDaC)
                self.itDaiDepsDstC = gpu_sparse_mat(self.itDaiDepsDstC)
                self.ieps.to_gpu()
                self.sigma.to_gpu()
            else:
                raise ImportError(
                    "[!] cupyx could not be imported, please check CUDA installation"
                )

        if verbose:
            print(f"Total solver initialization time: {time.time() - t0} s")

        self.solverInitializationTime = time.time() - t0
        self.update_logger(["solverInitializationTime"])

        self.solverInitializationTime = time.time() - t0
        self.update_logger(["solverInitializationTime"])

    def update_tensors(self, tensor="all"):
        """
        Update tensor matrices after material Field changes and precompute
        combined operators used for time-stepping.

        When ``ieps``, ``imu`` or ``sigma`` are modified this routine
        reconstructs the corresponding sparse diagonal matrices and the
        composite operator products used in the update equations. Use the
        ``tensor`` argument to restrict work to a single tensor for efficiency.

        Parameters
        ----------
        tensor : {'ieps','imu','sigma','all'}, optional
            Which tensor to update. Default is 'all' which recomputes every
            tensor and refreshes the precomputed time-stepping matrices.
        """
        if self.verbose:
            print(f'Re-computing tensor "{tensor}"...')

        if tensor == "ieps":
            self.iDeps = diags(
                self.ieps.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
        elif tensor == "imu":
            self.iDmu = diags(
                self.imu.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
        elif tensor == "sigma":
            self.Dsigma = diags(
                self.sigma.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
        elif tensor == "all":
            self.iDeps = diags(
                self.ieps.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
            self.iDmu = diags(
                self.imu.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )
            self.Dsigma = diags(
                self.sigma.toarray(),
                shape=(3 * self.N, 3 * self.N),
                dtype=self.dtype,
            )

        if self.verbose:
            print("Re-Pre-computing ...")
        self.tDsiDmuiDaC = self.iDa * self.iDmu * self.C * self.Ds
        self.itDaiDepsDstC = self.iDeps * self.itDa * self.C.transpose() * self.tDs
        self.step_0 = False

    def _apply_stl_materials(self):
        """
        Mask STL solids in the grid and assign user-defined materials.

        Iterates over STL solids imported in the grid and updates ``ieps``,
        ``imu`` and ``sigma`` according to the material provided for each
        solid. Materials may be referenced by a library key (string) or given
        as explicit tuples (eps_r, mu_r[, sigma]). Inverse permittivity and
        inverse permeability values are stored in the corresponding Fields.

        Notes
        -----
        - STL material values must be relative (eps_r, mu_r).
        - Supply conductivity explicitly to enable conductive behaviour.
        """
        grid = self.grid.grid
        self.stl_solids = self.grid.stl_solids
        self.stl_materials = self.grid.stl_materials
        self.stl_colors = self.grid.stl_colors

        for key in self.stl_solids.keys():
            # TODO: adapt for subpixel smoothing

            # Retrieve mask and materials from grid
            mask = np.reshape(grid[key], (self.Nx, self.Ny, self.Nz)).astype(int)
            eps = self.stl_materials[key][0] * eps_0
            mu = self.stl_materials[key][1] * mu_0

            # Conductivity
            # Max conductivity that can be resolved without SIBC
            dn = np.sqrt(2) * min(self.dx.min(), self.dy.min(), self.dz.min())
            sigma_max = 10 / (np.pi * self.fmax * mu * dn**2)
            if self.verbose > 1:
                print(f"* Max resolved conductivity without SIBC: {sigma_max} S/m")

            if len(self.stl_materials[key]) == 3:
                sigma = self.stl_materials[key][2]

                # Mark surface cells for SIBC if conductivity is high
                if self.use_sibc and self.stl_materials[key][2] > sigma_max:
                    if self.verbose > 1:
                        print(
                            f'* Applying SIBC for solid "{key}" with sigma={sigma} S/m'
                        )
                    self.grid._mark_cells_in_surface(key)
                    imp = np.sqrt(np.pi * self.fmax * mu / sigma)
                    sigma = 1 / imp  # SIBC surface conductivity [S]
                    eps = 1 / imp

                # Update sigma tensor
                self.sigma += self.sigma * (-1.0 * mask)
                self.sigma += mask * sigma
                self.use_conductivity = True

            # Update ieps and imu tensors
            self.ieps += self.ieps * (-1.0 * mask)
            self.imu += self.imu * (-1.0 * mask)
            self.ieps += mask * 1.0 / eps
            self.imu += mask * 1.0 / mu

    def _one_step(self):
        if self.step_0:
            self._set_ghosts_to_0()
            self.step_0 = False
            self._attrcleanup()

        self.H.fromarray(
            self.H.toarray() - self.dt * self.tDsiDmuiDaC * self.E.toarray()
        )

        self.E.fromarray(
            self.E.toarray()
            + self.dt
            * (
                self.itDaiDepsDstC * self.H.toarray()
                - self.ieps.toarray() * self.J.toarray()
            )
        )

        # include current computation
        if self.use_conductivity:
            self.J.fromarray(self.sigma.toarray() * self.E.toarray())

    def _one_step_mkl(self):
        if self.step_0:
            self._set_ghosts_to_0()
            self.step_0 = False
            self._attrcleanup()

        self.H.fromarray(
            self.H.toarray()
            - self.dt * dot_product_mkl(self.tDsiDmuiDaC, self.E.toarray())
        )

        self.E.fromarray(
            self.E.toarray()
            + self.dt
            * (
                dot_product_mkl(self.itDaiDepsDstC, self.H.toarray())
                - self.ieps.toarray() * self.J.toarray()
            )
        )

        # include current computation
        if self.use_conductivity:
            self.J.fromarray(self.sigma.toarray() * self.E.toarray())

    def _mpi_initialize(self):
        self.comm = self.grid.comm
        self.rank = self.grid.rank
        self.size = self.grid.size

        self.NZ = self.grid.NZ
        self.ZMIN = self.grid.ZMIN
        self.ZMAX = self.grid.ZMAX
        self.Z = self.grid.Z

    def _mpi_one_step(self):
        if self.step_0:
            self._set_ghosts_to_0()
            self.step_0 = False
            self._attrcleanup()

        self.H.fromarray(
            self.H.toarray() - self.dt * self.tDsiDmuiDaC * self.E.toarray()
        )

        self._mpi_communicate(self.H)
        self._mpi_communicate(self.J)
        self.E.fromarray(
            self.E.toarray()
            + self.dt
            * (
                self.itDaiDepsDstC * self.H.toarray()
                - self.ieps.toarray() * self.J.toarray()
            )
        )

        self._mpi_communicate(self.E)
        # include current computation
        if self.use_conductivity:
            self.J.fromarray(self.sigma.toarray() * self.E.toarray())

    def _mpi_one_step_mkl(self):
        if self.step_0:
            self._set_ghosts_to_0()
            self.step_0 = False
            self._attrcleanup()

        self.H.fromarray(
            self.H.toarray()
            - self.dt * dot_product_mkl(self.tDsiDmuiDaC, self.E.toarray())
        )

        self._mpi_communicate(self.H)
        self._mpi_communicate(self.J)

        self.E.fromarray(
            self.E.toarray()
            + self.dt
            * (
                dot_product_mkl(self.itDaiDepsDstC, self.H.toarray())
                - self.ieps.toarray() * self.J.toarray()
            )
        )

        self._mpi_communicate(self.E)
        # include current computation
        if self.use_conductivity:
            self.J.fromarray(self.sigma.toarray() * self.E.toarray())

    def _mpi_communicate(self, field):
        if self.use_gpu:
            field.from_gpu()

        # ghosts lo
        if self.rank > 0:
            for d in ["x", "y", "z"]:
                self.comm.Sendrecv(
                    field[:, :, 1, d],
                    recvbuf=field[:, :, 0, d],
                    dest=self.rank - 1,
                    sendtag=0,
                    source=self.rank - 1,
                    recvtag=1,
                )
        # ghosts hi
        if self.rank < self.size - 1:
            for d in ["x", "y", "z"]:
                self.comm.Sendrecv(
                    field[:, :, -2, d],
                    recvbuf=field[:, :, -1, d],
                    dest=self.rank + 1,
                    sendtag=1,
                    source=self.rank + 1,
                    recvtag=0,
                )

        if self.use_gpu:
            field.to_gpu()

    def mpi_gather(self, field, x=None, y=None, z=None, component=None):
        """
        Gather a component or slice of a distributed Field from all MPI ranks.

        Assumes the field is split along the z-axis among ranks. The function
        collects local buffers, removes ghost cells and concatenates rank
        contributions to build a global NumPy array on the root rank (rank 0).

        Parameters
        ----------
        field : str or wakis.Field
            Field identifier ('E','H','J') optionally with a component suffix
            (e.g. 'Ex'), or a ``wakis.Field`` object. If no component is given
            the 'z' component is used by default.
        x, y, z : int or slice, optional
            Index or slice for each axis to gather. Defaults to the full range.
        component : {'x','y','z'} or slice, optional
            Component to gather when ``field`` is a Field object.

        Returns
        -------
        numpy.ndarray or None
            Assembled global array on rank 0; returns ``None`` on non-root ranks.
        """

        if x is None:
            x = slice(0, self.Nx)
        if y is None:
            y = slice(0, self.Ny)
        if z is None:
            z = slice(0, self.NZ)

        if type(field) is str:
            if len(field) == 2:  # support for e.g. field='Ex'
                component = field[1]
                field = field[0]
            elif len(field) == 4:  # support for Abs
                component = field[1:]
                field = field[0]
            elif component is None:
                component = "z"
                print("[!] `component` not specified, using default component='z'")

            if field == "E":
                local = self.E[x, y, :, component].ravel()
            elif field == "H":
                local = self.H[x, y, :, component].ravel()
            elif field == "J":
                local = self.J[x, y, :, component].ravel()
        else:
            if component is None:
                component = "z"
                print("[!] `component` not specified, using default component='z'")
            local = field[x, y, :, component].ravel()

        buffer = self.comm.gather(local, root=0)
        _field = None

        if self.rank == 0:
            if type(x) is int and type(y) is int:  # 1d array at x=a, y=b
                nz = self.NZ // self.size
                _field = np.zeros((self.NZ))
                for r in range(self.size):
                    zz = np.s_[r * nz : (r + 1) * nz]
                    if r == 0:
                        _field[zz] = np.reshape(buffer[r], (nz + self.grid.n_ghosts))[
                            :-1
                        ]
                    elif r == (self.size - 1):
                        _field[zz] = np.reshape(buffer[r], (nz + self.grid.n_ghosts))[
                            1:
                        ]
                    else:
                        _field[zz] = np.reshape(
                            buffer[r], (nz + 2 * self.grid.n_ghosts)
                        )[1:-1]
                _field = _field[z]

            elif type(x) is int:  # 2d slice at x=a
                ny = y.stop - y.start
                nz = self.NZ // self.size
                _field = np.zeros((ny, self.NZ))
                for r in range(self.size):
                    zz = np.s_[r * nz : (r + 1) * nz]
                    if r == 0:
                        _field[:, zz] = np.reshape(
                            buffer[r], (ny, nz + self.grid.n_ghosts)
                        )[:, :-1]
                    elif r == (self.size - 1):
                        _field[:, zz] = np.reshape(
                            buffer[r], (ny, nz + self.grid.n_ghosts)
                        )[:, 1:]
                    else:
                        _field[:, zz] = np.reshape(
                            buffer[r], (ny, nz + 2 * self.grid.n_ghosts)
                        )[:, 1:-1]
                _field = _field[:, z]

            elif type(y) is int:  # 2d slice at y=a
                nx = x.stop - x.start
                nz = self.NZ // self.size
                _field = np.zeros((nx, self.NZ))
                for r in range(self.size):
                    zz = np.s_[r * nz : (r + 1) * nz]
                    if r == 0:
                        _field[:, zz] = np.reshape(
                            buffer[r], (nx, nz + self.grid.n_ghosts)
                        )[:, :-1]
                    elif r == (self.size - 1):
                        _field[:, zz] = np.reshape(
                            buffer[r], (nx, nz + self.grid.n_ghosts)
                        )[:, 1:]
                    else:
                        _field[:, zz] = np.reshape(
                            buffer[r], (nx, nz + 2 * self.grid.n_ghosts)
                        )[:, 1:-1]
                _field = _field[:, z]

            else:  # both type slice -> 3d array
                nx = x.stop - x.start
                ny = y.stop - y.start
                nz = self.NZ // self.size
                _field = np.zeros((nx, ny, self.NZ))
                for r in range(self.size):
                    zz = np.s_[r * nz : (r + 1) * nz]
                    if r == 0:
                        _field[:, :, zz] = np.reshape(
                            buffer[r], (nx, ny, nz + self.grid.n_ghosts)
                        )[:, :, :-1]
                    elif r == (self.size - 1):
                        _field[:, :, zz] = np.reshape(
                            buffer[r], (nx, ny, nz + self.grid.n_ghosts)
                        )[:, :, 1:]
                    else:
                        _field[:, :, zz] = np.reshape(
                            buffer[r], (nx, ny, nz + 2 * self.grid.n_ghosts)
                        )[:, :, 1:-1]
                _field = _field[:, :, z]

        return _field

    def mpi_gather_asField(self, field):
        """
        Gather distributed field data from MPI ranks and return a global Field.

        Collects the full 3-component field (E, H or J) from each rank and
        reconstructs a single ``wakis.Field`` on the root rank. Ghost cells are
        removed when reassembling the per-rank buffers.

        Parameters
        ----------
        field : str or wakis.Field
            Identifier ('E','H','J') or a Field-like object to gather.

        Returns
        -------
        wakis.Field or None
            Global Field object assembled on rank 0. Returns ``None`` on other
            ranks.
        """

        _field = Field(self.Nx, self.Ny, self.NZ)

        for d in ["x", "y", "z"]:
            if type(field) is str:
                if field == "E":
                    local = self.E[:, :, :, d].ravel()
                elif field == "H":
                    local = self.H[:, :, :, d].ravel()
                elif field == "J":
                    local = self.J[:, :, :, d].ravel()
            else:
                local = field[:, :, :, d].ravel()

            buffer = self.comm.gather(local, root=0)
            if self.rank == 0:
                nz = self.NZ // self.size
                for r in range(self.size):
                    zz = np.s_[r * nz : (r + 1) * nz]
                    if r == 0:
                        _field[:, :, zz, d] = np.reshape(
                            buffer[r],
                            (self.Nx, self.Ny, nz + self.grid.n_ghosts),
                        )[:, :, :-1]
                    elif r == (self.size - 1):
                        _field[:, :, zz, d] = np.reshape(
                            buffer[r],
                            (self.Nx, self.Ny, nz + self.grid.n_ghosts),
                        )[:, :, 1:]
                    else:
                        _field[:, :, zz, d] = np.reshape(
                            buffer[r],
                            (self.Nx, self.Ny, nz + 2 * self.grid.n_ghosts),
                        )[:, :, 1:-1]

        return _field

    def _set_ghosts_to_0(self):
        """
        Zero-out ghost-cell field values used for MPI and boundary exchange.

        Clears any initial condition values that were accidentally placed in
        ghost cells so that subsequent MPI sends/receives and boundary updates
        behave correctly.
        """
        # Set H ghost quantities to 0
        for d in ["x", "y", "z"]:  # tangential to zero
            if d != "x":
                self.H[-1, :, :, d] = 0.0
            if d != "y":
                self.H[:, -1, :, d] = 0.0
            if d != "z":
                self.H[:, :, -1, d] = 0.0

        # Set E ghost quantities to 0
        self.E[-1, :, :, "x"] = 0.0
        self.E[:, -1, :, "y"] = 0.0
        self.E[:, :, -1, "z"] = 0.0

    def _apply_conductors(self):
        """
        Apply PEC conductor masking by zeroing inverse-permittivity inside
        conductor volumes.

        This enforces tangential electric field cancellation inside conductor
        regions by setting the local 1/epsilon to zero.
        """
        self.flag_in_conductors = (
            self.grid.flag_int_cell_yz[:-1, :, :]
            + self.grid.flag_int_cell_zx[:, :-1, :]
            + self.grid.flag_int_cell_xy[:, :, :-1]
        )

        self.ieps *= self.flag_in_conductors

    def _set_field_in_conductors_to_0(self):
        """
        Zero dynamic fields inside conductor masks.

        Ensures that any initial E/H fields mapped into conductor regions are
        removed before time-stepping, avoiding non-physical behaviour.
        """
        self.flag_cleanup = (
            self.grid.flag_int_cell_yz[:-1, :, :]
            + self.grid.flag_int_cell_zx[:, :-1, :]
            + self.grid.flag_int_cell_xy[:, :, :-1]
        )

        self.H *= self.flag_cleanup
        self.E *= self.flag_cleanup

    def _attrcleanup(self):
        # Fields
        del self.L, self.tL, self.iA, self.itA
        if hasattr(self, "BC"):
            del self.BC
            del self.Dbc

        # Matrices
        del self.Px, self.Py, self.Pz
        del self.Ds, self.iDa, self.tDs, self.itDa
        del self.C

    def save_state(self, filename="solver_state.h5", close=True):
        """
        Save dynamic solver state (H, E, J) to an HDF5 file.

        Writes the core dynamic fields to ``filename``. When running under MPI
        the distributed fields are gathered to the root rank before saving.

        Parameters
        ----------
        filename : str, optional
            Output HDF5 filename. Default is "solver_state.h5".
        close : bool, optional
            If True (default) the file is closed before returning. If False an
            open ``h5py.File`` is returned for caller-managed operations.

        Returns
        -------
        h5py.File or None
            Open file object when ``close`` is False, otherwise None.
        """

        if self.use_mpi:  # MPI savestate
            H = self.mpi_gather_asField("H")
            E = self.mpi_gather_asField("E")
            J = self.mpi_gather_asField("J")
            state = None

            if self.rank == 0:
                state = h5py.File(filename, "w")
                state.create_dataset("H", data=H.toarray())
                state.create_dataset("E", data=E.toarray())
                state.create_dataset("J", data=J.toarray())
            # TODO: check for MPI-GPU

        elif self.use_gpu:  # GPU savestate
            state = h5py.File(filename, "w")
            state.create_dataset("H", data=self.H.toarray().get())
            state.create_dataset("E", data=self.E.toarray().get())
            state.create_dataset("J", data=self.J.toarray().get())

        else:  # CPU savestate
            state = h5py.File(filename, "w")
            state.create_dataset("H", data=self.H.toarray())
            state.create_dataset("E", data=self.E.toarray())
            state.create_dataset("J", data=self.J.toarray())

        if close and state is not None:
            state.close()
        else:
            return state  # Caller must close this manually

    def load_state(self, filename="solver_state.h5"):
        """
        Load dynamic solver state (H, E, J) from an HDF5 file and restore them.

        Parameters
        ----------
        filename : str, optional
            Input HDF5 filename. Default is "solver_state.h5".

        Notes
        -----
        Currently performs a simple load from a single-file state. MPI-aware
        redistribution of loaded arrays to worker ranks is TODO.
        """

        if self.use_mpi:  # TODO: test
            if self.rank == 0:
                with h5py.File(filename, "r") as f:
                    state = {"E": f["E"][:], "H": f["H"][:], "J": f["J"][:]}
                zz = np.s_[: self.Nz]
            elif self.rank == self.size - 1:
                state = None
                zz = np.s_[(self.NZ - self.Nz) :]
            else:
                state = None
                zlo = (self.NZ // self.size + 1) + +(self.NZ // self.size + 2) * (
                    self.rank - 1
                )
                zz = np.s_[zlo : zlo + self.Nz]

            state = self.comm.bcast(state, root=0)
            for d in [0, 1, 2]:  # x,y,z
                self.E[:, :, :, d] = state["E"].reshape(
                    (self.Nx, self.Ny, self.NZ, 3), order="F"
                )[:, :, zz, d]
                self.H[:, :, :, d] = state["H"].reshape(
                    (self.Nx, self.Ny, self.NZ, 3), order="F"
                )[:, :, zz, d]
                self.J[:, :, :, d] = state["J"].reshape(
                    (self.Nx, self.Ny, self.NZ, 3), order="F"
                )[:, :, zz, d]

        else:  # CPU/GPU loadstate
            state = h5py.File(filename, "r")
            self.E.fromarray(state["E"][:])
            self.H.fromarray(state["H"][:])
            self.J.fromarray(state["J"][:])
        state.close()

    def read_state(self, filename="solver_state.h5"):
        """
        Open an HDF5 file for read-only access without loading its contents.

        Returns an open ``h5py.File`` object that the caller must close when
        finished. This is useful for inspecting saved state without restoring
        it into the solver.

        Parameters
        ----------
        filename : str, optional
            Input HDF5 filename. Default is "solver_state.h5".

        Returns
        -------
        h5py.File
            Open HDF5 file object in read mode.
        """
        return h5py.File(filename, "r")

    def reset_fields(self):
        """
        Reset dynamic field arrays (E, H, J) to zero across the simulation.

        Useful when reusing a ``SolverFIT3D`` instance for a new run without
        reconstructing the entire object.
        """
        for d in ["x", "y", "z"]:
            self.E[:, :, :, d] = 0.0
            self.H[:, :, :, d] = 0.0
            self.J[:, :, :, d] = 0.0

    def update_logger(self, attrs):
        """
        Copy selected solver attributes into the internal ``Logger`` object.

        Parameters
        ----------
        attrs : iterable of str
            Names of attributes to copy to ``self.logger.solver``. Special case
            'grid' copies the grid logger reference instead of a value.
        """
        for atr in attrs:
            if atr == "grid":
                self.logger.grid = self.grid.logger.grid
            else:
                self.logger.solver[atr] = getattr(self, atr)
