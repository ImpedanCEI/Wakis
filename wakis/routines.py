# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import time

import h5py
import numpy as np
from tqdm import tqdm

from wakis.plotting import PlotMixin
from wakis.sources import Beam


class RoutinesMixin:
    def emsolve(
        self,
        Nt,
        source=None,
        callback=None,
        save=False,
        fields=["E"],
        components=["Abs"],
        save_every=1,
        subdomain=None,
        plot=False,
        plot_every=1,
        use_etd=False,
        plot3d=False,
        **kwargs,
    ):
        """
        Run the simulation and save the selected field components in HDF5 files for
        every timestep. Each field will be saved in a separate HDF5 file 'Xy.h5'
        where X is the field and y the component.

        Parameters
        ----------
        Nt : int
            Number of timesteps to run.
        source : object, optional
            Source object from `sources.py` defining the time-dependent source.
            It should have an update function `source.update(solver, t)`.
        callback : callable, optional
            Function to call at each timestep as `callback(solver, t)`.
        save : bool, optional
            Flag to enable saving the field in HDF5 format.
        fields : list of str, optional
            3D field magnitude ('E', 'H', or 'J') to save. 'Ex', 'Hy', etc., is also
            accepted and will override the `components` parameter. Default is ['E'].
        components : list of str, optional
            Field component ('x', 'y', 'z', 'Abs') to save. It will be overridden if a
            component is specified in the `field` parameter. Default is ['Abs'].
        save_every : int, optional
            Number of timesteps between saves. Default is 1.
        subdomain : list or None, optional
            Slice [x, y, z] of the domain to be saved. Default is None (full domain).
        plot : bool, optional
            Flag to enable 2D plotting. Default is False.
        plot_every : int, optional
            Number of timesteps between consecutive plots. Default is 1.
        use_etd : bool, optional
            If True, use exponential time differencing. Default is False.
        plot3d : bool, optional
            Flag to enable 3D plotting. Default is False.
        **kwargs
            Keyword arguments to be passed to the Plot2D or Plot3D function.

            * Default kwargs used for 2D plotting:
            ```
            plotkw = {
                "field": "E",
                "component": "z",
                "plane": "ZY",
                "pos": 0.5,
                "cmap": "rainbow",
                "patch_reverse": True,
                "title": "Ez",
                "off_screen": True,
                "interpolation": "spline36",
            }
            ```

            * Default kwargs used for 3D plotting:
            ```
            plotkw = {
                "field": "E",
                "component": "z",
                "add_stl": None,
                "stl_opacity": 0.0,
                "stl_colors": "white",
                "title": "Ez",
                "cmap": "jet",
                "clip_box": False,
                "clip_normal": "-y",
                "off_screen": True,
                "zoom": 1.0,
                "nan_opacity": 1.0,
            }
            ```

        Raises
        ------
        ImportError
            If the hdf5 dependency cannot be imported.

        Dependencies
        -----------
        h5py

        Notes
        -----
        - If `activate_abc` is True, absorbing boundary conditions are updated each step.
        - If `callback` is provided, it is called as `callback(self, t)` at each step.
        - The plotting keyword arguments are merged with defaults for 2D/3D plotting.
        """
        self.Nt = Nt
        if source is not None:
            self.source = source

        if save:
            hfs = {}
            for field in fields:
                if len(field) == 1:
                    for component in components:
                        hfs[field + component] = h5py.File(
                            field + component + ".h5", "w"
                        )
                else:
                    hfs[field] = h5py.File(field + ".h5", "w")

            for hf in hfs:
                hf["x"], hf["y"], hf["z"] = self.x, self.y, self.z
                hf["dx"], hf["dy"], hf["dz"] = self.grid.dx, self.grid.dy, self.grid.dz
                hf["t"] = np.arange(0, Nt * self.dt, save_every * self.dt)

            if subdomain is not None:
                xx, yy, zz = subdomain
            else:
                xx, yy, zz = slice(0, self.Nx), slice(0, self.Ny), slice(0, self.Nz)

        if plot:
            plotkw = {
                "field": "E",
                "component": "z",
                "plane": "ZY",
                "pos": 0.5,
                "cmap": "rainbow",
                "patch_reverse": True,
                "title": "Ez",
                "off_screen": True,
                "interpolation": "spline36",
            }
            plotkw.update(kwargs)

        if plot3d:
            plotkw = {
                "field": "E",
                "component": "z",
                "add_stl": None,
                "stl_opacity": 0.0,
                "stl_colors": "white",
                "title": "Ez",
                "cmap": "jet",
                "clip_box": False,
                "clip_normal": "-y",
                "off_screen": True,
                "zoom": 1.0,
                "nan_opacity": 1.0,
            }

            plotkw.update(kwargs)

        # get ABC values
        if self.activate_abc:
            E_abc_2, H_abc_2 = self.get_abc()
            E_abc_1, H_abc_1 = self.get_abc()

        # Time loop
        for n in tqdm(range(Nt)):
            if source is not None:
                source.update(self, n * self.dt)

            if save and n % save_every == 0:
                for field in hfs.keys():
                    try:
                        d = getattr(self, field[0])[xx, yy, zz, field[1:]]
                    except AttributeError:
                        raise (
                            f'Component {field} not valid. Input must have a \
                              field ["E", "H", "J"] and a component ["x", "y", "z", "Abs"]'
                        )

                    # Save timestep in HDF5
                    hfs[field]["#" + str(n).zfill(5)] = d

            # Advance
            self.one_step()

            # Plot
            if plot and n % plot_every == 0:
                self.plot2D(n=n, **plotkw)

            if plot3d and n % plot_every == 0:
                self.plot3D(n=n, **plotkw)

            # ABC BCs
            if self.activate_abc:
                self.update_abc(E_abc_2, H_abc_2)  # n-2
                E_abc_2, H_abc_2 = E_abc_1, H_abc_1  # n-1
                E_abc_1, H_abc_1 = self.get_abc()  # n

            # Callback func(solver, t)
            if callback is not None:
                callback(self, n * self.dt)

        # End
        if save:
            for hf in hfs.values():
                hf.close()

    def wakesolve(
        self,
        wakelength,
        wake=None,
        callback=None,
        compute_plane="both",
        plot=False,
        plot_func=PlotMixin.plot2D,
        plot_from=None,
        plot_every=1,
        plot_until=None,
        save_J=False,
        add_space=None,  # for legacy
        use_field_monitor=False,
        field_monitor=None,
        use_edt=None,  # deprecated
        **kwargs,
    ):
        """
        Run the EM simulation and compute the longitudinal (z) and transverse (x, y)
        wake potential WP(s) and impedance Z(s).

        The `Ez` field is saved every timestep in a subdomain (xtest, ytest, z) around
        the beam trajectory in HDF5 format file `Ez.h5`.

        The computed results are available as Solver class attributes:
            - wake potential: WP (longitudinal), WPx, WPy (transverse) [V/pC]
            - impedance: Z (longitudinal), Zx, Zy (transverse) [Ohm]
            - beam charge distribution: lambdas (distance) [C/m], lambdaf (spectrum) [C]

        Parameters
        ----------
        wakelength : float
            Desired length of the wake in [m] to be computed.

            Maximum simulation time in [s] can be computed from the wakelength parameter as:
            .. math::    t_{max} = t_{inj} + (wakelength + (z_{max}-z_{min}))/c
        wake : object, optional
            `WakeSolver` object containing the information needed to run the wake solver
            calculation. See Wake docstring for more information. Can be passed at
            `Solver` instantiation as parameter too.
        callback : callable, optional
            Function to call at each timestep as `callback(solver, t)`.
        compute_plane : str, optional
            Which planes to compute: 'both', 'longitudinal', or 'transverse'.
            Default is 'both'.
        plot : bool, optional
            Flag to enable 2D plotting. Default is False.
        plot_func: callable, optional
            Plotting function to be called during the timestepping
        plot_from : int, optional
            Timestep to start plotting from. Default is int(self.ti / self.dt).
        plot_every : int, optional
            Number of timesteps between consecutive plots. Default is 1.
        plot_until : int, optional
            Timestep to stop plotting at. Default is Nt.
        save_J : bool, optional
            Flag to enable saving the current J in a different HDF5 file 'Jz.h5'.
        add_space : int, optional
            Legacy support for skip_cells.
        use_field_monitor : bool, optional
            Flag to enable monitoring the field monitor during simulation.
        field_monitor : object, optional
            FieldMonitor class instance.
        use_edt : bool, optional
            Deprecated. If True, use exponential time differencing.
        **kwargs
            Keyword arguments to be passed to the Plot2D function.
            Default kwargs used:
                {'plane':'ZY', 'pos':0.5, 'title':'Ez',
                'cmap':'rainbow', 'patch_reverse':True,
                'off_screen': True, 'interpolation':'spline36'}

        Raises
        ------
        AttributeError
            If the Wake object is not provided.
        ImportError
            If the hdf5 dependency cannot be imported.

        Dependencies
        -----------
        h5py

        Notes
        -----
        - The plotting keyword arguments are merged with defaults for 2D plotting.
        - The wake solver results are available as attributes after the run.
        - If `callback` is provided, it is called as `callback(self, t)` at each step.
        """

        if wake is not None:
            self.wake = wake
        if self.wake is None:
            raise AttributeError(
                "Wake solver information not passed to the solver instantiation"
            )

        if add_space is not None:  # legacy support
            self.wake.skip_cells = add_space

        # plot params defaults
        if plot:
            plotkw = self.get_plotting_kwargs(plot_func.__name__)
            plotkw.update(kwargs)

        # integration path (test position)
        self.xtest, self.ytest = self.wake.xtest, self.wake.ytest
        self.ixt, self.iyt = (
            np.abs(self.x - self.xtest).argmin(),
            np.abs(self.y - self.ytest).argmin(),
        )
        if compute_plane.lower() == "longitudinal":
            xx, yy = self.ixt, self.iyt
        else:
            xx, yy = (
                slice(self.ixt - 1, self.ixt + 2),
                slice(self.iyt - 1, self.iyt + 2),
            )

        # Compute simulation time
        self.wake.wakelength = wakelength
        self.ti = self.wake.ti
        self.v = self.wake.v
        if self.use_mpi:  # E- should it be zmin, zmax instead?
            z = self.Z  # use global coords
            dz = np.diff(self.Z)
            zz = slice(0, self.NZ)
        else:
            z = self.z
            zz = slice(0, self.Nz)

        tmax = (wakelength + self.ti * self.v + (z.max() - z.min())) / self.v  # [s]
        Nt = int(tmax / self.dt)
        self.tmax, self.Nt = tmax, Nt

        # Add beam source
        beam = Beam(
            q=self.wake.q,
            sigmaz=self.wake.sigmaz,
            beta=self.wake.beta,
            xsource=self.wake.xsource,
            ysource=self.wake.ysource,
        )

        # hdf5
        self.Ez_file = self.wake.Ez_file
        hf = None  # needed for MPI
        if self.use_mpi:
            if self.rank == 0:
                hf = h5py.File(self.Ez_file, "w")
                hf["x"], hf["y"], hf["z"] = self.x[xx], self.y[yy], z[zz]
                hf["dx"], hf["dy"], hf["dz"] = self.grid.dx, self.grid.dy, dz
                hf["t"] = np.arange(0, Nt * self.dt, self.dt)

                if save_J:
                    hfJ = h5py.File("Jz.h5", "w")
                    hfJ["x"], hfJ["y"], hfJ["z"] = self.x[xx], self.y[yy], z[zz]
                    hfJ["dx"], hfJ["dy"], hfJ["dz"] = self.grid.dx, self.grid.dy, dz
                    hfJ["t"] = np.arange(0, Nt * self.dt, self.dt)
        else:
            hf = h5py.File(self.Ez_file, "w")
            hf["x"], hf["y"], hf["z"] = self.x[xx], self.y[yy], z[zz]
            hf["dx"], hf["dy"], hf["dz"] = self.grid.dx, self.grid.dy, self.grid.dz
            hf["t"] = np.arange(0, Nt * self.dt, self.dt)

            if save_J:
                hfJ = h5py.File("Jz.h5", "w")
                hfJ["x"], hfJ["y"], hfJ["z"] = self.x[xx], self.y[yy], z[zz]
                hfJ["dx"], hfJ["dy"], hfJ["dz"] = (
                    self.grid.dx,
                    self.grid.dy,
                    self.grid.dz,
                )
                hfJ["t"] = np.arange(0, Nt * self.dt, self.dt)

        def save_to_h5(self, hf, field, x, y, z, comp, n):
            if self.use_mpi:
                _field = self.mpi_gather(field, x, y, z, comp)
                if self.rank == 0:
                    hf["#" + str(n).zfill(5)] = _field
            else:
                hf["#" + str(n).zfill(5)] = getattr(self, field)[x, y, z, comp]

        if plot_until is None:
            plot_until = Nt
        if plot_from is None:
            plot_from = int(self.ti / self.dt)

        print("Running electromagnetic time-domain simulation...")

        t0 = time.time()
        for n in tqdm(range(Nt)):
            # Initial condition
            beam.update(self, n * self.dt)

            # Save
            save_to_h5(self, hf, "E", xx, yy, zz, "z", n)
            if save_J:
                save_to_h5(self, hfJ, "J", xx, yy, zz, "z", n)

            # Advance
            self.one_step()

            if use_field_monitor and field_monitor is not None:
                field_monitor.update(self.E, self.dt)

            # Plot
            if plot:
                if n % plot_every == 0 and n < plot_until and n > plot_from:
                    plot_func(n=n, **plotkw)
                else:
                    pass

            # Callback func(solver, t)
            if callback is not None:
                callback(self, n * self.dt)

        # End of time loop
        if self.use_mpi:
            if self.rank == 0:
                hf.close()
                if save_J:
                    hfJ.close()

                # Compute wakefield magnitudes is done inside WakeSolver
                self.wake.solve(compute_plane=compute_plane)
        else:
            hf.close()
            if save_J:
                hfJ.close()

            # Compute wakefield magnitudes is done inside WakeSolver
            self.wake.solve(compute_plane=compute_plane)

        # Forward parameters to logger
        self.logger.wakeSolver = self.wake.logger.wakeSolver
        self.logger.wakeSolver["wakelength"] = wakelength
        self.logger.wakeSolver["simulationTime"] = time.time() - t0
        self.logger.save_logs()

    def get_plotting_kwargs(self, name="plot2D"):
        if name == "plot1D":
            plotkw = {
                "field": "E",
                "component": "z",
                "line": "z",
                "pos": [0.8, 0.6, 0.5, 0.4, 0.2],
                "xscale": "linear",
                "yscale": "linear",
                "off_screen": True,
                "colors": [
                    "#5ccfe6",
                    "#fdb6d0",
                    "#ffae57",
                    "#bae67e",
                    "#ffd580",
                    "#a2aabc",
                ],
                "title": "plot1D",
            }

        if name == "plot2D":
            plotkw = {
                "field": "E",
                "component": "z",
                "plane": "ZY",
                "pos": 0.5,
                "cmap": "rainbow",
                "patch_reverse": True,
                "off_screen": True,
                "interpolation": "spline36",
                "title": "plot2D",
            }
        elif name == "plot3D":
            plotkw = {
                "field": "E",
                "component": "z",
                "add_stl": None,
                "stl_opacity": 0.0,
                "stl_colors": "white",
                "cmap": "jet",
                "clip_box": False,
                "clip_normal": "-y",
                "off_screen": True,
                "zoom": 1.0,
                "nan_opacity": 1.0,
                "title": "plot3D",
            }
        elif name == "plot3DonSTL":
            plotkw = {
                "field": "E",
                "component": "z",
                "cmap": "rainbow",
                "stl_with_field": list(self.grid.stl_solids.keys())[0],
                "field_opacity": 1.0,
                "stl_transparent": list(self.grid.stl_solids.keys()),
                "stl_opacity": 0.1,
                "stl_colors": list(self.grid.stl_colors.values()),
                "clip_plane": True,
                "clip_normal": "-y",
                "clip_origin": [0, 0, 0],
                "off_screen": True,
                "zoom": 1.2,
                "title": "plot3DonSTL",
            }

        return plotkw
