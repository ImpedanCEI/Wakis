# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #


import copy

import numpy as xp

try:
    import cupy as xp_gpu

    imported_cupy = True
except ImportError:
    imported_cupy = False


class Field:
    """
    Class to handle 3D vector fields stored in a flattened 1D array.
    Uses lexicographic numbering::

        n = 1 + (i-1) + (j-1)*Nx + (k-1)*Nx*Ny
        len(n) = Nx*Ny*Nz

    Parameters
    ----------
    Nx : int
        Number of grid points in the x direction.
    Ny : int
        Number of grid points in the y direction.
    Nz : int
        Number of grid points in the z direction.
    dtype : type, optional
        Data type of the field array. Default is float.
    use_ones : bool, optional
        If True, initialize the field array with ones. Otherwise, zeros. Default is False.
    use_gpu : bool, optional
        If True, use CuPy for GPU arrays. Default is False.
    coordinates : dict, optional
        Physical sample axes in metres, keyed by component ('x', 'y', 'z'). E.g.:
        primal_coordinates = {
            "x": (tx, y, z),  # Ex: x midpoints, y/z nodes
            "y": (x, ty, z),  # Ey: y midpoints, x/z nodes
            "z": (x, y, tz),  # Ez: z midpoints, x/y nodes
        }


    Attributes
    ----------
    Nx, Ny, Nz : int
        Grid dimensions.
    N : int
        Total number of grid points (Nx * Ny * Nz).
    dtype : type
        Data type of the field array.
    on_gpu : bool
        Whether the field is stored on GPU.
    xp : module
        Numpy or cupy module, depending on use_gpu.
    array : ndarray
        Flattened 1D field array of shape (N*3,).
    """

    def __init__(
        self, Nx, Ny, Nz, dtype=float, use_ones=False, use_gpu=False, coordinates=None
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.N = Nx * Ny * Nz
        self.dtype = dtype
        self.on_gpu = use_gpu
        self.coordinates = coordinates

        if self.on_gpu:
            if imported_cupy:
                self.xp = xp_gpu
            else:
                raise ImportError(
                    "[!] CuPy could not be imported, please check CUDA installation"
                )
        else:
            self.xp = xp

        if use_ones:
            self.array = self.xp.ones(self.N * 3, dtype=self.dtype, order="F")
        else:
            self.array = self.xp.zeros(self.N * 3, dtype=self.dtype, order="F")

    @property
    def field_x(self):
        """Return the x-component of the field as a 1D array."""
        return self.array[0 : self.N]

    @property
    def field_y(self):
        """Return the y-component of the field as a 1D array."""
        return self.array[self.N : 2 * self.N]

    @property
    def field_z(self):
        """Return the z-component of the field as a 1D array."""
        return self.array[2 * self.N : 3 * self.N]

    @field_x.setter
    def field_x(self, value):
        """Set the x-component of the field."""
        if len(value.shape) > 1:
            self.from_matrix(value, "x")
        else:
            self.array[0 : self.N] = value

    @field_y.setter
    def field_y(self, value):
        """Set the y-component of the field."""
        if len(value.shape) > 1:
            self.from_matrix(value, "y")
        else:
            self.array[self.N : 2 * self.N] = value

    @field_z.setter
    def field_z(self, value):
        """Set the z-component of the field."""
        if len(value.shape) > 1:
            self.from_matrix(value, "z")
        else:
            self.array[2 * self.N : 3 * self.N] = value

    def toarray(self):
        """
        Return the flattened field array.

        Returns
        -------
        array : ndarray
            Flattened field array of shape (N*3,).
        """
        return self.array

    def fromarray(self, array):
        """
        Set the field array from a flattened array.

        Parameters
        ----------
        array : ndarray
            Flattened field array of shape (N*3,).
        """
        self.array[:] = array

    def to_matrix(self, key):
        """
        Return the specified component as a 3D matrix.

        Parameters
        ----------
        key : int or str
            Component to return: 0 or 'x', 1 or 'y', 2 or 'z'.

        Returns
        -------
        mat : ndarray
            3D array of shape (Nx, Ny, Nz) for the selected component.
        """
        if key == 0 or key == "x":
            return self.xp.reshape(
                self.array[0 : self.N], (self.Nx, self.Ny, self.Nz), order="F"
            )
        if key == 1 or key == "y":
            return self.xp.reshape(
                self.array[self.N : 2 * self.N],
                (self.Nx, self.Ny, self.Nz),
                order="F",
            )
        if key == 2 or key == "z":
            return self.xp.reshape(
                self.array[2 * self.N : 3 * self.N],
                (self.Nx, self.Ny, self.Nz),
                order="F",
            )

    def from_matrix(self, mat, key):
        """
        Set the specified component from a 3D matrix.

        Parameters
        ----------
        mat : ndarray
            3D array of shape (Nx, Ny, Nz).
        key : int or str
            Component to set: 0 or 'x', 1 or 'y', 2 or 'z'.
        """
        if key == 0 or key == "x":
            self.array[0 : self.N] = self.xp.reshape(mat, self.N, order="F")
        elif key == 1 or key == "y":
            self.array[self.N : 2 * self.N] = self.xp.reshape(mat, self.N, order="F")
        elif key == 2 or key == "z":
            self.array[2 * self.N : 3 * self.N] = self.xp.reshape(
                mat, self.N, order="F"
            )
        else:
            raise IndexError("Component id not valid")

    def to_gpu(self):
        """
        Move the field array to GPU (CuPy).

        Returns
        -------
        None
        """
        if imported_cupy:
            self.xp = xp_gpu
            self.array = self.xp.asarray(self.array)  # to cupy arr
            self.on_gpu = True
        else:
            print("*** CuPy is not imported")
            pass

    def from_gpu(self):
        """
        Move the field array from GPU (CuPy) to CPU (NumPy).

        Returns
        -------
        None
        """
        if self.on_gpu:
            self.array = self.array.get()  # to numpy arr
            self.on_gpu = False
        else:
            print("*** GPU is not enabled")
            pass

    def __getitem__(self, key):
        """
        Get a field value or slice.

        Parameters
        ----------
        key : tuple, int, or slice
            - (ix, iy, iz, component): 3D index and component ('x', 'y', 'z', or 'Abs')
            - int: lexico-graphic index
            - slice: slice of the flattened array

        Returns
        -------
        value : float or ndarray
            Field value(s) at the specified location.
        """
        if type(key) is tuple:
            if len(key) != 4:
                raise IndexError("Need 3 indexes and component to access the field")
            if key[3] == 0 or key[3] == "x":
                if self.on_gpu:
                    field = self.xp.reshape(
                        self.array[0 : self.N],
                        (self.Nx, self.Ny, self.Nz),
                        order="F",
                    )
                    return field[key[0], key[1], key[2]].get()
                else:
                    field = self.xp.reshape(
                        self.array[0 : self.N],
                        (self.Nx, self.Ny, self.Nz),
                        order="F",
                    )
                    return field[key[0], key[1], key[2]]
            elif key[3] == 1 or key[3] == "y":
                if self.on_gpu:
                    field = self.xp.reshape(
                        self.array[self.N : 2 * self.N],
                        (self.Nx, self.Ny, self.Nz),
                        order="F",
                    )
                    return field[key[0], key[1], key[2]].get()
                else:
                    field = self.xp.reshape(
                        self.array[self.N : 2 * self.N],
                        (self.Nx, self.Ny, self.Nz),
                        order="F",
                    )
                    return field[key[0], key[1], key[2]]
            elif key[3] == 2 or key[3] == "z":
                if self.on_gpu:
                    field = self.xp.reshape(
                        self.array[2 * self.N : 3 * self.N],
                        (self.Nx, self.Ny, self.Nz),
                        order="F",
                    )
                    return field[key[0], key[1], key[2]].get()
                else:
                    field = self.xp.reshape(
                        self.array[2 * self.N : 3 * self.N],
                        (self.Nx, self.Ny, self.Nz),
                        order="F",
                    )
                    return field[key[0], key[1], key[2]]
            elif type(key[3]) is str and key[3].lower() == "abs":
                field = self.get_abs()
                return field[key[0], key[1], key[2]]
            else:
                raise IndexError("Component id not valid")

        elif type(key) is int:
            if key <= self.N:
                if self.on_gpu:
                    return self.array[key].get()
                else:
                    return self.array[key]
            else:
                raise IndexError(
                    "Lexico-graphic index cannot be higher than product of dimensions"
                )

        elif type(key) is slice:
            if self.on_gpu:
                return self.array[key].get()
            else:
                return self.array[key]

        else:
            raise ValueError("key must be a 3-tuple or an integer")

    def __setitem__(self, key, value):
        """
        Set a field value or slice.

        Parameters
        ----------
        key : tuple, int, or slice
            - (ix, iy, iz, component): 3D index and component
            - int: lexico-graphic index
            - slice: slice of the flattened array
        value : float or ndarray
            Value(s) to set.
        """
        if self.on_gpu:
            value = self.xp.asarray(value)

        if type(key) is tuple:
            if len(key) != 4:
                raise IndexError("Need 3 indexes and component to access the field")
            else:
                field = self.to_matrix(key[3])
                field[key[0], key[1], key[2]] = value
                self.from_matrix(field, key[3])

        elif type(key) is int:
            if key <= self.N:
                self.array[key] = value
            else:
                raise IndexError(
                    "Lexico-graphic index cannot be higher than product of dimensions"
                )

        elif type(key) is slice:
            self.array[key] = value
        else:
            raise IndexError("key must be a 3-tuple or an integer")

    def __mul__(self, other, dtype=None):
        """
        Multiply the field by a scalar, array, or matrix.

        Parameters
        ----------
        other : float, int, ndarray
            Scalar, 1D array, or 3D matrix to multiply.
        dtype : type, optional
            Data type for the result.

        Returns
        -------
        Field
            New Field object with the result.
        """
        if dtype is None:
            dtype = self.dtype

        # other is number
        if type(other) is float or type(other) is int:
            mulField = Field(
                self.Nx, self.Ny, self.Nz, dtype=dtype, coordinates=self.coordinates
            )
            mulField.array = self.array * other

        # other is matrix
        elif len(other.shape) > 1:
            mulField = Field(
                self.Nx, self.Ny, self.Nz, dtype=dtype, coordinates=self.coordinates
            )
            for d in ["x", "y", "z"]:
                mulField.from_matrix(self.to_matrix(d) * other, d)

        # other is 1d array
        else:
            mulField = Field(
                self.Nx, self.Ny, self.Nz, dtype=dtype, coordinates=self.coordinates
            )
            mulField.array = self.array * other

        return mulField

    def __div__(self, other, dtype=None):
        """
        Divide the field by a scalar, array, or matrix.

        Parameters
        ----------
        other : float, int, ndarray
            Scalar, 1D array, or 3D matrix to divide by.
        dtype : type, optional
            Data type for the result.

        Returns
        -------
        Field
            New Field object with the result.
        """
        if dtype is None:
            dtype = self.dtype

        # other is number
        if type(other) is float or type(other) is int:
            divField = Field(
                self.Nx, self.Ny, self.Nz, dtype=dtype, coordinates=self.coordinates
            )
            divField.array = self.array / other

        # other is matrix
        if len(other.shape) > 1:
            divField = Field(
                self.Nx, self.Ny, self.Nz, dtype=dtype, coordinates=self.coordinates
            )
            for d in ["x", "y", "z"]:
                divField.from_matrix(self.to_matrix(d) / other, d)

        # other is constant or 1d array
        else:
            divField = Field(
                self.Nx, self.Ny, self.Nz, dtype=dtype, coordinates=self.coordinates
            )
            divField.array = self.array / other

        return divField

    def __add__(self, other, dtype=None):
        """
        Add another field, scalar, array, or matrix to this field.

        Parameters
        ----------
        other : Field, float, int, ndarray
            Field, scalar, 1D array, or 3D matrix to add.
        dtype : type, optional
            Data type for the result.

        Returns
        -------
        Field
            New Field object with the result.
        """
        if dtype is None:
            dtype = self.dtype

        if type(other) is Field:
            addField = Field(
                self.Nx, self.Ny, self.Nz, dtype=dtype, coordinates=self.coordinates
            )
            addField.field_x = self.field_x + other.field_x
            addField.field_y = self.field_y + other.field_y
            addField.field_z = self.field_z + other.field_z

        # other is matrix
        elif len(other.shape) > 1:
            addField = Field(
                self.Nx, self.Ny, self.Nz, dtype=dtype, coordinates=self.coordinates
            )
            for d in ["x", "y", "z"]:
                addField.from_matrix(self.to_matrix(d) + other, d)

        # other is constant or 1d array
        else:
            addField = Field(
                self.Nx, self.Ny, self.Nz, dtype=dtype, coordinates=self.coordinates
            )
            addField.array = self.array + other

        return addField

    def __repr__(self):
        """String representation of the field (for debugging)."""
        return (
            "x:\n"
            + self.field_x.__repr__()
            + "\n"
            + "y:\n"
            + self.field_y.__repr__()
            + "\n"
            + "z:\n"
            + self.field_z.__repr__()
        )

    def __str__(self):
        """String representation of the field."""
        return (
            "x:\n"
            + self.field_x.__str__()
            + "\n"
            + "y:\n"
            + self.field_y.__str__()
            + "\n"
            + "z:\n"
            + self.field_z.__str__()
        )

    def copy(self):
        """
        Return a deep copy of the Field object.

        Returns
        -------
        Field
            Deep copy of the field.
        """
        obj = type(self).__new__(self.__class__)  # Create empty instance

        for key, value in self.__dict__.items():
            if key == "xp":
                obj.xp = self.xp  # Just copy reference, no need for deepcopy
            elif key == "array" and self.on_gpu:
                obj.array = self.xp.array(
                    self.array
                )  # Ensure CuPy array is copied properly
            else:
                obj.__dict__[key] = copy.deepcopy(value)

        return obj

    def compute_ijk(self, n):
        """
        Compute (i, j, k) indices from a lexico-graphic index.

        Parameters
        ----------
        n : int
            Lexico-graphic index.

        Returns
        -------
        i, j, k : int
            3D indices corresponding to the lexico-graphic index.
        """
        if n > (self.N):
            raise IndexError(
                "Lexico-graphic index cannot be higher than product of dimensions"
            )

        k = n // (self.Nx * self.Ny)
        i = (n - k * self.Nx * self.Ny) % self.Nx
        j = (n - k * self.Nx * self.Ny) // self.Nx

        return i, j, k

    def get_abs(self, as_matrix=True):
        """
        Compute the magnitude of the field.

        Parameters
        ----------
        as_matrix : bool, optional
            If True, return as a 3D matrix. If False, return as a 1D array.

        Returns
        -------
        abs_field : ndarray
            Magnitude of the field.
        """
        if as_matrix:
            if self.on_gpu:
                return xp.sqrt(
                    self.to_matrix("x") ** 2
                    + self.to_matrix("y") ** 2
                    + self.to_matrix("z") ** 2
                ).get()
            else:
                return xp.sqrt(
                    self.to_matrix("x") ** 2
                    + self.to_matrix("y") ** 2
                    + self.to_matrix("z") ** 2
                )

        else:  # 1d array
            if self.on_gpu:
                return xp.sqrt(
                    self.field_x**2 + self.field_y**2 + self.field_z**2
                ).get()
            else:
                return xp.sqrt(self.field_x**2 + self.field_y**2 + self.field_z**2)

    def inspect(
        self,
        plane="ZY",
        cmap=None,
        backend="matplotlib",
        component="z",
        grid=None,
        position=None,
        bounding_box=False,
        show_grid=False,
        dpi=100,
        figsize=[10, 6],
        x=None,
        y=None,
        z=None,
        off_screen=False,
        handles=False,
        **kwargs,
    ):
        """
        Visualize 2D slices of the field components.

        Supports two backends: ``'matplotlib'`` (default, static 2D plots of
        all three components) and ``'pyvista'`` (interactive 3D slice with a
        position slider for a single selected component).

        Parameters
        ----------
        plane : {'XY', 'YX', 'XZ', 'ZX', 'YZ', 'ZY'}, optional
            First letter is horizontal, second is vertical. Default is 'ZY'.
        cmap : str, optional
            Colormap for the plot. Default is 'Reds' for positive values and
            "bwr" for positive/negative values.
        backend : {'matplotlib', 'pyvista'}, optional
            Visualization backend. Default is 'matplotlib'.
        component : {'x', 'y', 'z', 'abs'}, optional
            Field component to display in PyVista mode. Ignored by the
            matplotlib backend (which always shows all three). Default 'z'.
        grid : GridFIT3D or None, optional
            Structured grid object providing real-space coordinates and STL
            solids. When provided the PyVista backend uses physical units and
            can overlay STL surface outlines. Falls back to cell-index
            coordinates when ``None``. Attached component coordinates take
            precedence; grid may still supply STL overlays. Ignored by Matplotlib.
        position : float or None, optional
            Initial slice position along the axis normal to ``plane``.
            Defaults to the domain centre. With coordinate metadata, Matplotlib
            selects the nearest physical sample separately for each component.
        bounding_box : bool, optional
            If True, add a wireframe bounding box of the domain. PyVista
            backend only. Default False.
        show_grid : bool, optional
            If True, overlay a wireframe grid slice at each slider update.
            PyVista backend only. Default False.
        dpi : int, optional
            Figure DPI (matplotlib backend only). Default is 100.
        figsize : list, optional
            Figure size (matplotlib backend only). Default is [8, 6].
        x, y, z : int, slice, or None, optional
            Custom slice indices (matplotlib backend only). If all are not
            None, use as custom slice.
        off_screen : bool, optional
            Matplotlib: if True, return ``(fig, axs)`` instead of showing.
            PyVista: if True, return the ``pyvista.Plotter`` instead of
            opening an interactive window. Default is False.
        handles : bool, optional
            If True, return (fig, axs) instead of showing (matplotlib only).
            Default is False.
        **kwargs
            Additional keyword arguments forwarded to ``pcolormesh`` when
            coordinates are attached, otherwise to ``imshow`` (Matplotlib only).

        Returns
        -------
        fig, axs : tuple, optional
            Returned when ``backend='matplotlib'`` and ``handles=True`` or
            ``off_screen=True``.
        pyvista.Plotter, optional
            Returned when ``backend='pyvista'`` and ``off_screen=True``.
        None
            Otherwise.

        Notes
        -----
        With coordinates attached, plots use metres and exclude storage beyond
        each component's physical axes. Matplotlib uses pcolormesh to preserve
        nonuniform spacing. PyVista interpolates a slice of the selected
        component. Absolute values use the current index-wise norm and the
        x-component axes for display, pending interpolation to common locations.
        """
        # ------------------------------------------------------------------ #
        # PyVista backend                                                      #
        # ------------------------------------------------------------------ #
        if backend.lower() == "pyvista":
            import pyvista as pv

            component = component.lower()

            # --- Build / retrieve the structured grid ---
            if self.coordinates is not None:
                coords = self.coordinates["x" if component == "abs" else component]
                pv_grid = pv.RectilinearGrid(*coords)
                xlo, xhi = coords[0][0], coords[0][-1]
                ylo, yhi = coords[1][0], coords[1][-1]
                zlo, zhi = coords[2][0], coords[2][-1]
            elif grid is not None and hasattr(grid, "grid"):
                pv_grid = grid.grid
                xlo, xhi = grid.xmin, grid.xmax
                ylo, yhi = grid.ymin, grid.ymax
                zlo, zhi = grid.zmin, grid.zmax
            else:
                _x = xp.linspace(0, self.Nx, self.Nx + 1)
                _y = xp.linspace(0, self.Ny, self.Ny + 1)
                _z = xp.linspace(0, self.Nz, self.Nz + 1)
                xlo, xhi = 0, self.Nx
                ylo, yhi = 0, self.Ny
                zlo, zhi = 0, self.Nz
                X, Y, Z = xp.meshgrid(_x, _y, _z, indexing="ij")
                pv_grid = pv.StructuredGrid(X.transpose(), Y.transpose(), Z.transpose())

            # --- Assign scalar data ---
            scalars = "Field Abs" if component == "abs" else f"Field {component}"
            values = self.get_abs() if component == "abs" else self.to_matrix(component)
            if self.on_gpu and hasattr(values, "get"):
                values = values.get()
            if self.coordinates is not None:
                physical = tuple(slice(0, len(axis)) for axis in coords)
                pv_grid.point_data[scalars] = values[physical].ravel(order="F")
            else:
                pv_grid[scalars] = values.reshape(self.N)

            pv_grid.set_active_scalars(scalars)

            # --- Plane → normal mapping ---
            plane_up = plane.upper()
            plane_to_normal = {
                "XY": "z",
                "YX": "z",
                "YZ": "x",
                "ZY": "x",
                "XZ": "y",
                "ZX": "y",
            }
            if plane_up not in plane_to_normal:
                raise ValueError(
                    f"plane must be one of 'XY', 'YX', 'XZ', 'YZ', 'ZX', 'ZY'; got '{plane}'"
                )
            normal = plane_to_normal[plane_up]
            unit_axes = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}
            view_up = unit_axes[plane_up[1]]
            view_direction = xp.cross(unit_axes[plane_up[0]], view_up)

            if normal == "x":
                axis_min, axis_max = xlo, xhi
                slider_title = "X Position"
                cy = (ylo + yhi) / 2
                cz = (zlo + zhi) / 2

                def origin_fn(v):
                    return (v, cy, cz)

            elif normal == "y":
                axis_min, axis_max = ylo, yhi
                slider_title = "Y Position"
                cx = (xlo + xhi) / 2
                cz = (zlo + zhi) / 2

                def origin_fn(v):
                    return (cx, v, cz)

            else:  # z
                axis_min, axis_max = zlo, zhi
                slider_title = "Z Position"
                cx = (xlo + xhi) / 2
                cy = (ylo + yhi) / 2

                def origin_fn(v):
                    return (cx, cy, v)

            if position is None:
                position = (axis_min + axis_max) / 2

            # --- Build plotter ---
            pv.global_theme.allow_empty_mesh = True
            pl = pv.Plotter(off_screen=off_screen)

            # Initial slice
            initial_slice = (
                pv_grid
                if axis_min == axis_max
                else pv_grid.slice(normal=normal, origin=origin_fn(position))
            )
            slice_actor = pl.add_mesh(initial_slice, cmap=cmap, name="slice")

            # Optional STL solid outlines (only when grid is a GridFIT3D)
            outline_actors = {}
            if (
                grid is not None
                and hasattr(grid, "stl_solids")
                and hasattr(grid, "read_stl")
            ):
                stl_colors_map = getattr(grid, "stl_colors", {})
                for key in grid.stl_solids:
                    surf = grid.read_stl(key)
                    if surf is not None:
                        init_outline = surf.slice(
                            normal=normal, origin=origin_fn(position)
                        )
                        color = "black"
                        outline_actors[key] = (
                            pl.add_mesh(
                                init_outline, color=color, name=f"outline_{key}"
                            ),
                            surf,
                        )

            # --- Update function ---
            def update_slice(val):
                new_slice = (
                    pv_grid
                    if axis_min == axis_max
                    else pv_grid.slice(normal=normal, origin=origin_fn(val))
                )
                slice_actor.mapper.SetInputData(new_slice)

                for key, (actor, surf) in outline_actors.items():
                    new_outline = surf.slice(normal=normal, origin=origin_fn(val))
                    actor.mapper.SetInputData(new_outline)

                if show_grid:
                    pl.add_mesh(
                        new_slice,
                        style="wireframe",
                        color="grey",
                        opacity=0.3,
                        name="grid_wire",
                    )

                pl.view_vector(view_direction, viewup=view_up)
                pl.render()

            # --- Slider ---
            if axis_min != axis_max:
                pl.add_slider_widget(
                    update_slice,
                    [axis_min, axis_max],
                    value=position,
                    title=slider_title,
                    pointa=(0.8, 0.6),
                    pointb=(0.95, 0.6),
                    style="modern",
                )

            # --- Optional bounding box ---
            if bounding_box:
                pl.add_mesh(
                    pv.Box(bounds=(xlo, xhi, ylo, yhi, zlo, zhi)),
                    style="wireframe",
                    color="black",
                    line_width=2,
                    name="domain_box",
                )

            # --- Camera / aesthetics ---
            # Use view_*() methods to explicitly set both position and viewup
            # so that the first letter of the plane is horizontal and the
            # second letter is vertical (e.g. 'YZ' → Y horizontal, Z vertical).
            pl.view_vector(view_direction, viewup=view_up)
            pl.set_background("mistyrose", top="white")
            pl.add_axes()
            pl.enable_anti_aliasing()
            pl.enable_3_lights()

            if off_screen:
                return pl
            else:
                pl.show()
                return None

        # ------------------------------------------------------------------ #
        # Matplotlib backend (original implementation)                        #
        # ------------------------------------------------------------------ #
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        plane = plane.upper()
        if plane not in ("XY", "YX", "XZ", "ZX", "YZ", "ZY"):
            raise ValueError("plane must be XY, YX, XZ, ZX, YZ, or ZY.")

        # Handle custom slices
        custom_slice = None
        if x is not None and y is not None and z is not None:
            custom_slice = (x, y, z)
            if sum(isinstance(index, slice) for index in custom_slice) != 2:
                raise ValueError(
                    "2D Inspection requires two slices and one fixed index."
                )
            if not isinstance(x, slice) and "X" in plane:
                plane = "YZ"
            elif not isinstance(y, slice) and "Y" in plane:
                plane = "XZ"
            elif not isinstance(z, slice) and "Z" in plane:
                plane = "XY"
        if plane in ("XY", "YX"):
            normal = "z"
        elif plane in ("XZ", "ZX"):
            normal = "y"
        else:
            normal = "x"
        horizontal, vertical = plane.lower()

        fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=figsize, dpi=dpi)

        # Support component metadata and plane orientation
        for component, ax in zip("xyz", axs):
            if self.coordinates is None:
                cx, cy, cz = xp.arange(self.Nx), xp.arange(self.Ny), xp.arange(self.Nz)
            else:
                cx, cy, cz = self.coordinates[component]
            field = self.to_matrix(component)
            if self.on_gpu:
                field = field.get()
            field = field[: len(cx), : len(cy), : len(cz)]

            if custom_slice is not None:
                ix, iy, iz = custom_slice
            else:
                normal_coords = {"x": cx, "y": cy, "z": cz}[normal]
                index = len(normal_coords) // 2
                if self.coordinates is not None:
                    location = position
                    if location is None:
                        location = (normal_coords[0] + normal_coords[-1]) / 2
                    index = int(xp.argmin(xp.abs(normal_coords - location)))
                ix, iy, iz = slice(None), slice(None), slice(None)
                if normal == "x":
                    ix = index
                elif normal == "y":
                    iy = index
                else:
                    iz = index

            # Start with ascending plane order: XY, XZ, or YZ.
            values = field[ix, iy, iz].T
            if normal == "z":
                hcoords, vcoords = cx[ix], cy[iy]
                location = cz[iz]
            elif normal == "y":
                hcoords, vcoords = cx[ix], cz[iz]
                location = cy[iy]
            else:
                hcoords, vcoords = cy[iy], cz[iz]
                location = cx[ix]
            if plane in ("YX", "ZX", "ZY"):
                values = values.T
                hcoords, vcoords = vcoords, hcoords
            options = {"vmin": field.min(), "vmax": field.max(), **kwargs}

            # pcolormesh for coordinates
            if self.coordinates is not None:
                image = ax.pcolormesh(
                    hcoords,
                    vcoords,
                    values,
                    cmap=cmap or "Reds",
                    **{"shading": "nearest", **options},
                )
                ax.set_xlabel(f"{horizontal} [m]")
                ax.set_ylabel(f"{vertical} [m]")

            # imshow for indexes
            else:
                extent = (
                    (0, len(hcoords), 0, len(vcoords)) if custom_slice is None else None
                )
                image = ax.imshow(
                    values,
                    cmap=cmap or "Reds",
                    extent=extent,
                    origin="lower",
                    **options,
                )
                ax.set_xlabel(f"n{horizontal}")
                ax.set_ylabel(f"n{vertical}")

            unit = " m" if self.coordinates is not None else ""
            ax.set_title(f"Field {component}, {normal}={location:g}{unit}")
            fig.colorbar(
                image,
                cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1),
            )

        if handles or off_screen:
            return fig, axs
        plt.show(block=False)
        return None

    def inspect3D(
        self,
        field="all",
        backend="pyvista",
        grid=None,
        xmax=None,
        ymax=None,
        zmax=None,
        bounding_box=True,
        show_grid=True,
        cmap="viridis",
        dpi=100,
        off_screen=False,
        stl_opacity=0.2,
    ):
        """Inspect components in 3D using physical coordinates when available.

        Parameters
        ----------
        field : {'x', 'y', 'z', 'abs', 'all'}, optional
            Component to display, or separate panels for all three components.
            Absolute values use the current index-wise norm; with coordinates
            attached they use the x-component axes for display.
        backend : {'pyvista', 'matplotlib'}, optional
            Interactive clipping (default) or static voxel rendering.
        grid : GridFIT3D, optional
            Supplies STL surface overlays for PyVista and physical mesh geometry
            when component coordinates are absent. Component coordinates take
            precedence. Ignored by the Matplotlib backend.
        xmax, ymax, zmax : float, optional
            Upper clipping positions in metres with physical geometry, otherwise
            index limits. Defaults to each component's full extent.
        bounding_box : bool, optional
            Show component bounds in PyVista. Default True.
        show_grid : bool, optional
            Show clipping-plane wireframes in PyVista. Default True.
        cmap : str, optional
            Scalar colormap. Default 'viridis'.
        dpi : int, optional
            Matplotlib figure resolution. Default 100.
        off_screen : bool, optional
            Return the plot without showing it. Default False.
        stl_opacity : float, optional
            STL surface opacity in PyVista, between 0 and 1. Default 0.2.
            Surfaces and silhouettes use the grid's STL colors.

        Returns
        -------
        pyvista.Plotter or tuple of (Figure, list of Axes), optional
            Returned when off_screen is True.

        Notes
        -----
        Physical axes omit inactive terminal storage. Matplotlib display boxes
        meet halfway between neighboring samples and extrapolate half a spacing
        at the ends; they are display regions, not physical FIT cells. A singleton
        physical axis is shown with points because its box width is undefined.
        Fields without coordinates retain index-based Matplotlib plotting and
        use the supplied grid geometry for PyVista when available.
        """
        field, backend = field.lower(), backend.lower()
        if field not in ("x", "y", "z", "abs", "all"):
            raise ValueError("field must be x, y, z, abs, or all.")
        if backend not in ("matplotlib", "pyvista"):
            raise ValueError("backend must be matplotlib or pyvista.")
        components = ("x", "y", "z") if field == "all" else (field,)
        limits = (xmax, ymax, zmax)

        # Resolve coordinates, physical extents, and CPU data for both backends.
        index_coords = tuple(xp.arange(n) + 0.5 for n in (self.Nx, self.Ny, self.Nz))
        component_coords = {}
        component_values = {}
        for component in components:
            coords = (
                index_coords
                if self.coordinates is None
                else self.coordinates["x" if component == "abs" else component]
            )  # TODO use cell centers for absolute values
            values = self.get_abs() if component == "abs" else self.to_matrix(component)
            if self.on_gpu and hasattr(values, "get"):
                values = values.get()
            component_coords[component] = coords
            component_values[component] = values[
                tuple(slice(0, len(axis)) for axis in coords)
            ]

        # ------------------------------------------------------------------ #
        # Matplotlib backend                                                 #
        # ------------------------------------------------------------------ #
        if backend == "matplotlib":
            import matplotlib.pyplot as plt

            if self.Nx > 50 or self.Ny > 50 or self.Nz > 50:
                print(
                    "[!] Matplotlib voxel rendering can be slow for large meshes. "
                    "Use backend='pyvista' for interactive inspection."
                )

            fig = plt.figure(
                tight_layout=True, dpi=dpi, figsize=(6 * len(components), 6)
            )
            axs = []
            for panel, component in enumerate(components):
                coords = component_coords[component]
                values = component_values[component]
                # Clip sample arrays using their own axes, including nonuniform grids.
                counts = [
                    len(axis)
                    if limit is None
                    else xp.searchsorted(axis, limit, side="right")
                    for axis, limit in zip(coords, limits)
                ]
                if any(count == 0 for count in counts):
                    raise ValueError(
                        "Clipping limits exclude all samples along an axis."
                    )
                coords = tuple(axis[:count] for axis, count in zip(coords, counts))
                values = values[tuple(slice(0, count) for count in counts)]
                ax = fig.add_subplot(1, len(components), panel + 1, projection="3d")
                axs.append(ax)
                scale = xp.max(xp.abs(values))
                norm = plt.Normalize(vmin=-float(scale), vmax=float(scale))
                colors = plt.get_cmap(cmap)(norm(values))
                if any(len(axis) == 1 for axis in coords):
                    xx, yy, zz = xp.meshgrid(*coords, indexing="ij")
                    ax.scatter(
                        xx.ravel(), yy.ravel(), zz.ravel(), c=colors.reshape(-1, 4)
                    )
                else:
                    edges = []
                    for axis in coords:
                        edges.append(
                            xp.r_[
                                axis[0] - (axis[1] - axis[0]) / 2,
                                (axis[:-1] + axis[1:]) / 2,
                                axis[-1] + (axis[-1] - axis[-2]) / 2,
                            ]
                        )
                    xx, yy, zz = xp.meshgrid(*edges, indexing="ij")
                    filled = (
                        values
                        if values.dtype == bool
                        else xp.ones(values.shape, dtype=bool)
                    )
                    ax.voxels(xx, yy, zz, filled=filled, facecolors=colors)
                if xmax is not None:
                    ax.set_xlim(right=xmax)
                if ymax is not None:
                    ax.set_ylim(top=ymax)
                if zmax is not None:
                    ax.set_zlim(top=zmax)
                unit = " [m]" if self.coordinates is not None else " index"
                ax.set(
                    xlabel="x" + unit,
                    ylabel="y" + unit,
                    zlabel="z" + unit,
                    title=f"Field {component}",
                )
                fig.colorbar(
                    plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5
                )

            if off_screen:
                return fig, axs
            plt.show(block=False)
            return None

        # ------------------------------------------------------------------ #
        # PyVista backend                                                    #
        # ------------------------------------------------------------------ #
        if backend == "pyvista":
            import pyvista as pv

            # STL overlays
            surfaces = {}
            if grid is not None:
                for key in getattr(grid, "stl_solids", None) or {}:
                    surfaces[key] = grid.read_stl(key)

            pl = pv.Plotter(shape=(1, len(components)), off_screen=off_screen)

            # Generate reference grid per panel
            for panel, component in enumerate(components):
                coords = component_coords[component]
                values = component_values[component]
                pl.subplot(0, panel)
                scalars = f"Field {component}"
                if self.coordinates is not None:
                    volume = pv.RectilinearGrid(*coords)
                    volume.point_data[scalars] = values.ravel(order="F")
                else:
                    volume = pv.ImageData(
                        dimensions=(self.Nx + 1, self.Ny + 1, self.Nz + 1)
                    )
                    volume.cell_data[scalars] = values.ravel(order="F")
                bounds = tuple(volume.bounds)
                clips = dict(zip("xyz", (bounds[1], bounds[3], bounds[5])))
                for axis, limit, low, high in zip(
                    "xyz", limits, bounds[::2], bounds[1::2]
                ):
                    if limit is not None:
                        if not low <= limit <= high:
                            raise ValueError(
                                f"{axis}max must lie within [{low}, {high}]."
                            )
                        clips[axis] = limit

                # Clip update and widget sliders per panel
                def update_clip(
                    value,
                    axis,
                    volume=volume,
                    clips=clips,
                    bounds=bounds,
                    panel=panel,
                    scalars=scalars,
                ):
                    pl.subplot(0, panel)
                    clips[axis] = value
                    clipped = volume
                    for clip_axis, upper in clips.items():
                        axis_index = "xyz".index(clip_axis)
                        if upper == bounds[2 * axis_index + 1]:
                            continue
                        origin = list(volume.center)
                        origin[axis_index] = upper
                        clipped = clipped.clip(
                            normal=clip_axis, origin=origin, invert=True
                        )
                    if clipped.n_points:
                        pl.add_mesh(clipped, scalars=scalars, cmap=cmap, name="clip")
                    else:
                        pl.remove_actor("clip")
                    if show_grid:
                        origin = list(volume.center)
                        origin["xyz".index(axis)] = value
                        section = volume.slice(normal=axis, origin=origin)
                        if section.n_points:
                            pl.add_mesh(
                                section, style="wireframe", color="grey", name="slice"
                            )
                        else:
                            pl.remove_actor("slice")

                update_clip(clips["x"], "x")
                for row, (axis, low, high) in enumerate(
                    zip("xyz", bounds[::2], bounds[1::2])
                ):
                    if low == high:
                        continue
                    pl.add_slider_widget(
                        lambda value, axis=axis, update=update_clip: update(
                            value, axis
                        ),
                        (low, high),
                        value=clips[axis],
                        title=f"{axis.upper()} Clip",
                        pointa=(0.8, 0.8 - row * 0.2),
                        pointb=(0.95, 0.8 - row * 0.2),
                        style="modern",
                    )
                stl_colors = getattr(grid, "stl_colors", None) or {}
                for key, surface in surfaces.items():
                    color = stl_colors.get(key, "grey")
                    pl.add_mesh(
                        surface.triangulate(),
                        color=color,
                        opacity=stl_opacity,
                        silhouette={"color": color, "line_width": 1.0},
                        name=f"stl_{key}",
                    )
                pl.add_text(f"Field {component}", font_size=10)
                pl.add_axes()
                if bounding_box:
                    pl.add_bounding_box(color="black")
                pl.view_isometric()
                pl.set_background("mistyrose", top="white")

            if off_screen:
                return pl
            pl.show()
            return None
