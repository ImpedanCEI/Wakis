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
    Uses lexicographic numbering:
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

    def __init__(self, Nx, Ny, Nz, dtype=float, use_ones=False, use_gpu=False):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.N = Nx * Ny * Nz
        self.dtype = dtype
        self.on_gpu = use_gpu

        if self.on_gpu:
            if imported_cupy:
                self.xp = xp_gpu
            else:
                print("*** cupy could not be imported, please CUDA check installation")
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
                self.array[self.N : 2 * self.N], (self.Nx, self.Ny, self.Nz), order="F"
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
                        self.array[0 : self.N], (self.Nx, self.Ny, self.Nz), order="F"
                    )
                    return field[key[0], key[1], key[2]].get()
                else:
                    field = self.xp.reshape(
                        self.array[0 : self.N], (self.Nx, self.Ny, self.Nz), order="F"
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
            mulField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            mulField.array = self.array * other

        # other is matrix
        elif len(other.shape) > 1:
            mulField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            for d in ["x", "y", "z"]:
                mulField.from_matrix(self.to_matrix(d) * other, d)

        # other is 1d array
        else:
            mulField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
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
            divField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            divField.array = self.array / other

        # other is matrix
        if len(other.shape) > 1:
            divField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            for d in ["x", "y", "z"]:
                divField.from_matrix(self.to_matrix(d) / other, d)

        # other is constant or 1d array
        else:
            divField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
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
            addField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            addField.field_x = self.field_x + other.field_x
            addField.field_y = self.field_y + other.field_y
            addField.field_z = self.field_z + other.field_z

        # other is matrix
        elif len(other.shape) > 1:
            addField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            for d in ["x", "y", "z"]:
                addField.from_matrix(self.to_matrix(d) + other, d)

        # other is constant or 1d array
        else:
            addField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
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
        plane="YZ",
        cmap="bwr",
        dpi=100,
        figsize=[8, 6],
        x=None,
        y=None,
        z=None,
        show=True,
        handles=False,
        **kwargs,
    ):
        """
        Visualize 2D slices of the field components using matplotlib.

        Parameters
        ----------
        plane : {'XY', 'XZ', 'YZ'}, optional
            Plane to visualize. Default is 'YZ'.
        cmap : str, optional
            Colormap for the plot. Default is 'bwr'.
        dpi : int, optional
            Figure DPI. Default is 100.
        figsize : list, optional
            Figure size. Default is [8, 6].
        x, y, z : int, slice, or None, optional
            Custom slice indices. If all are not None, use as custom slice.
        show : bool, optional
            If True, display the plot. Default is True.
        handles : bool, optional
            If True, return (fig, axs) instead of showing. Default is False.
        **kwargs
            Additional keyword arguments for imshow.

        Returns
        -------
        fig, axs : tuple, optional
            Returned if handles=True.
        None
            Otherwise.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if None not in (x, y, z):  # custom slice
            transpose = False
            extent = None
            xax, yax = "No. of cells", "No. of cells"

        elif plane == "XY":
            key = [slice(0, self.Nx), slice(0, self.Ny), int(self.Nz // 2)]
            x, y, z = key[0], key[1], key[2]
            extent = (0, self.Nx, 0, self.Ny)
            xax, yax = "nx", "ny"
            transpose = True

        elif plane == "XZ":
            key = [slice(0, self.Nx), int(self.Ny // 2), slice(0, self.Nz)]
            x, y, z = key[0], key[1], key[2]
            extent = (0, self.Nz, 0, self.Nx)
            xax, yax = "nz", "nx"
            transpose = False

        elif plane == "YZ":
            key = [int(self.Nx // 2), slice(0, self.Ny), slice(0, self.Nz)]
            x, y, z = key[0], key[1], key[2]
            extent = (0, self.Nz, 0, self.Ny)
            xax, yax = "nz", "ny"
            transpose = False

        fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=figsize, dpi=dpi)
        dims = {0: "x", 1: "y", 2: "z"}

        im = {}

        for d in [0, 1, 2]:
            field = self.to_matrix(d)

            if self.on_gpu and hasattr(field, "get"):
                field = field.get()

            if transpose:
                im[d] = axs[d].imshow(
                    field[x, y, z].T,
                    cmap=cmap,
                    vmin=-field.max(),
                    vmax=field.max(),
                    extent=extent,
                    origin="lower",
                    **kwargs,
                )

            else:
                im[d] = axs[d].imshow(
                    field[x, y, z],
                    cmap=cmap,
                    vmin=-field.max(),
                    vmax=field.max(),
                    extent=extent,
                    origin="lower",
                    **kwargs,
                )

        for i, ax in enumerate(axs):
            ax.set_title(f"Field {dims[i]}, plane {plane}")
            fig.colorbar(
                im[i],
                cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1),
            )
            ax.set_xlabel(xax)
            ax.set_ylabel(yax)

        if handles:
            return fig, axs

        if show:
            plt.show()
            return None

    def inspect3D(
        self,
        field="all",
        backend="pyista",
        grid=None,
        xmax=None,
        ymax=None,
        zmax=None,
        bounding_box=True,
        show_grid=True,
        cmap="viridis",
        dpi=100,
        show=True,
        handles=False,
    ):
        """
        Visualize 3D field data on the structured grid using either Matplotlib
        (voxel rendering) or PyVista (interactive clipping and slicing).

        This method provides two complementary visualization backends:
        - **Matplotlib**: static voxel plots of the field components (x, y, z)
          or all combined, useful for quick inspection, but memory intensive.
        - **PyVista**: interactive 3D visualization with sliders to dynamically
          clip the volume along X, Y, and Z, and optional wireframe slices.

        Parameters
        ----------
        field : {'x', 'y', 'z', 'all'}, optional
            Which field component(s) to visualize. Default is 'all'.
        backend : {'matplotlib', 'pyvista'}, optional
            Visualization backend to use. Default is 'pyvista'.
        grid : object, optional
            Structured grid object to use for visualization. If None, a grid is
            constructed from the solver's internal dimensions.
        xmax, ymax, zmax : int or float, optional
            Maximum extents in each direction for visualization. Defaults to the
            full grid dimensions if not specified.
        bounding_box : bool, optional
            If True, draw a wireframe bounding box of the simulation domain
            (only used in PyVista backend). Default is True.
        show_grid : bool, optional
            If True, show wireframe slice planes of the grid during interactive
            visualization (PyVista backend). Default is True.
        cmap : str, optional
            Colormap to apply to the scalar field. Default is 'viridis'.
        dpi : int, optional
            Resolution of Matplotlib figures (only for Matplotlib backend).
            Default is 100.
        show : bool, optional
            Whether to display the figure/plot immediately. If False in PyVista,
            exports to `field.html` instead. Default is True.
        handles : bool, optional
            If True, return figure/axes (Matplotlib) or the Plotter object
            (PyVista) for further customization instead of showing directly.
            Default is False.

        Returns
        -------
        fig, axs : tuple, optional
            Returned when `backend='matplotlib'` and `handles=True`.
        pl : pyvista.Plotter, optional
            Returned when `backend='pyvista'` and `handles=True`.

        Notes
        -----
        - The PyVista backend provides interactive sliders to clip the
          volume along each axis independently and inspect internal
          structures of the 3D field.
        - The Matplotlib backend provides a quick static voxel rendering
          but is limited in interactivity and scalability.
        """

        field = field.lower()

        # ---------- matplotlib backend ---------------
        if backend.lower() == "matplotlib":
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            fig = plt.figure(tight_layout=True, dpi=dpi, figsize=[12, 6])

            plot_x, plot_y, plot_z = False, False, False

            if field == "all":
                plot_x = True
                plot_y = True
                plot_z = True

            elif field.lower() == "x":
                plot_x = True
            elif field.lower() == "y":
                plot_y = True
            elif field.lower() == "z":
                plot_z = True

            if xmax is None:
                xmax = self.Nx
            if ymax is None:
                ymax = self.Ny
            if zmax is None:
                zmax = self.Nz

            x, y, z = self.xp.mgrid[0 : xmax + 1, 0 : ymax + 1, 0 : zmax + 1]
            axs = []

            # field x
            if plot_x:
                arr = self.to_matrix("x")[0 : int(xmax), 0 : int(ymax), 0 : int(zmax)]
                if field == "all":
                    ax = fig.add_subplot(1, 3, 1, projection="3d")
                else:
                    ax = fig.add_subplot(1, 1, 1, projection="3d")

                vmin, vmax = (
                    -self.xp.max(self.xp.abs(arr)),
                    +self.xp.max(self.xp.abs(arr)),
                )
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                colors = mpl.colormaps[cmap](norm(arr))
                ax.voxels(x, y, z, filled=self.xp.ones_like(arr), facecolors=colors)

                m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                m.set_array([])
                fig.colorbar(m, ax=ax, shrink=0.5, aspect=10)
                ax.set_title("Field x")
                axs.append(ax)

            # field y
            if plot_y:
                arr = self.to_matrix("y")[0 : int(xmax), 0 : int(ymax), 0 : int(zmax)]
                if field == "all":
                    ax = fig.add_subplot(1, 3, 2, projection="3d")
                else:
                    ax = fig.add_subplot(1, 1, 1, projection="3d")

                vmin, vmax = (
                    -self.xp.max(self.xp.abs(arr)),
                    +self.xp.max(self.xp.abs(arr)),
                )
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                colors = mpl.colormaps[cmap](norm(arr))
                ax.voxels(x, y, z, filled=self.xp.ones_like(arr), facecolors=colors)

                m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                m.set_array([])
                fig.colorbar(m, ax=ax, shrink=0.5, aspect=10)
                ax.set_title("Field y")
                axs.append(ax)

            # field z
            if plot_z:
                arr = self.to_matrix("z")[0 : int(xmax), 0 : int(ymax), 0 : int(zmax)]
                if field == "all":
                    ax = fig.add_subplot(1, 3, 3, projection="3d")
                else:
                    ax = fig.add_subplot(1, 1, 1, projection="3d")

                vmin, vmax = (
                    -self.xp.max(self.xp.abs(arr)),
                    +self.xp.max(self.xp.abs(arr)),
                )
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                colors = mpl.colormaps[cmap](norm(arr))
                ax.voxels(x, y, z, filled=self.xp.ones_like(arr), facecolors=colors)

                m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                m.set_array([])
                fig.colorbar(m, ax=ax, shrink=0.5, aspect=10)
                ax.set_title("Field z")
                axs.append(ax)

            for i, ax in enumerate(axs):
                ax.set_xlabel("Nx")
                ax.set_ylabel("Ny")
                ax.set_zlabel("Nz")
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.set_xlim(self.Nx, 0)
                ax.set_ylim(self.Ny, 0)
                ax.set_zlim(self.Nz, 0)

            if handles:
                return fig, axs

            if show:
                plt.show()

        # ----------- pyvista backend ---------------
        else:
            import pyvista as pv

            if grid is not None and hasattr(grid, "grid"):
                xlo, xhi, ylo, yhi, zlo, zhi = (
                    grid.xmin,
                    grid.xmax,
                    grid.ymin,
                    grid.ymax,
                    grid.zmin,
                    grid.zmax,
                )
                grid = grid.grid
                if field == "x":
                    scalars = "Field " + field
                    grid[scalars] = xp.reshape(self.to_matrix(field), self.N)
                elif field == "y":
                    scalars = "Field " + field
                    grid[scalars] = xp.reshape(self.to_matrix(field), self.N)
                elif field == "z":
                    scalars = "Field " + field
                    grid[scalars] = xp.reshape(self.to_matrix(field), self.N)
                else:  # for all or abs
                    scalars = "Field " + "Abs"
                    grid[scalars] = xp.reshape(self.get_abs(), self.N)

                if xmax is None:
                    xmax = xhi
                if ymax is None:
                    ymax = yhi
                if zmax is None:
                    zmax = zhi

            else:
                print(
                    "[!] `grid` is not passed or is not a GridFIT3D object -> Using #N cells instead "
                )
                x = xp.linspace(0, self.Nx, self.Nx + 1)
                y = xp.linspace(0, self.Ny, self.Ny + 1)
                z = xp.linspace(0, self.Nz, self.Nz + 1)
                xlo, xhi, ylo, yhi, zlo, zhi = (
                    x.min(),
                    x.max(),
                    y.min(),
                    y.max(),
                    z.min(),
                    z.max(),
                )
                if xmax is None:
                    xmax = self.Nx
                if ymax is None:
                    ymax = self.Ny
                if zmax is None:
                    zmax = self.Nz
                X, Y, Z = xp.meshgrid(x, y, z, indexing="ij")
                grid = pv.StructuredGrid(X.transpose(), Y.transpose(), Z.transpose())

                if field == "x":
                    scalars = "Field " + field
                    grid[scalars] = xp.reshape(self.to_matrix(field), self.N)
                elif field == "y":
                    scalars = "Field " + field
                    grid[scalars] = xp.reshape(self.to_matrix(field), self.N)
                elif field == "z":
                    scalars = "Field " + field
                    grid[scalars] = xp.reshape(self.to_matrix(field), self.N)
                else:  # for all or abs
                    scalars = "Field " + "Abs"
                    grid[scalars] = xp.reshape(self.get_abs(), self.N)

            pv.global_theme.allow_empty_mesh = True
            pl = pv.Plotter()
            vals = {"x": xmax, "y": ymax, "z": zmax}

            # --- Update function ---
            def update_clip(val, axis="x"):
                vals[axis] = val
                # define bounds dynamically
                if axis == "x":
                    slice_obj = grid.slice(normal="x", origin=(val, 0, 0))
                elif axis == "y":
                    slice_obj = grid.slice(normal="y", origin=(0, val, 0))
                else:  # z
                    slice_obj = grid.slice(normal="z", origin=(0, 0, val))

                # add clipped volume (scalars)
                pl.add_mesh(
                    grid.clip_box(
                        bounds=(xlo, vals["x"], ylo, vals["y"], zlo, vals["z"]),
                        invert=False,
                    ),
                    scalars=scalars,
                    cmap=cmap,
                    name="clip",
                )

                # add slice wireframe (grid structure)
                if show_grid:
                    pl.add_mesh(
                        slice_obj, style="wireframe", color="grey", name="slice"
                    )

            # --- Sliders (placed side-by-side vertically) ---
            pl.add_slider_widget(
                lambda value: update_clip(value, "x"),
                [xlo, xhi],
                value=xmax,
                title="X Clip",
                pointa=(0.8, 0.8),
                pointb=(0.95, 0.8),  # top-right
                style="modern",
            )

            pl.add_slider_widget(
                lambda value: update_clip(value, "y"),
                [ylo, yhi],
                value=ymax,
                title="Y Clip",
                pointa=(0.8, 0.6),
                pointb=(0.95, 0.6),  # middle-right
                style="modern",
            )

            pl.add_slider_widget(
                lambda value: update_clip(value, "z"),
                [zlo, zhi],
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
            try:
                pl.add_logo_widget("../docs/img/wakis-logo-pink.png")
            except Exception:
                pass
            pl.add_axes()
            pl.enable_3_lights()
            pl.enable_anti_aliasing()

            if bounding_box:
                pl.add_mesh(
                    pv.Box(bounds=(xlo, xhi, ylo, yhi, zlo, zhi)),
                    style="wireframe",
                    color="black",
                    line_width=2,
                    name="domain_box",
                )

            if handles:
                return pl

            if not show:
                pl.export_html("Field.html")
            else:
                pl.show()
