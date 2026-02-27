# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


class PlotMixin:
    """Plotting mixin providing 1D/2D/3D visualization helpers."""

    def plot3D(
        self,
        field="E",
        component="z",
        clim=None,
        hide_solids=None,
        show_solids=None,
        add_stl=None,
        stl_opacity=0.1,
        stl_colors="white",
        title=None,
        cmap="jet",
        clip_interactive=True,
        clip_normal="-y",
        clip_box=False,
        clip_bounds=None,
        off_screen=False,
        zoom=0.5,
        camera_position=None,
        nan_opacity=1.0,
        n=None,
    ):
        """Built-in 3D plotting using PyVista.

        Displays a cell-centered field (E, H or J) on the structured grid using
        PyVista. Optionally overlays STL solids, clipping widgets or a static
        clip box. When rendered off-screen the plot can be saved to a PNG file
        for later composition into animations.

        Parameters
        ----------
        field : str, optional
            3D field magnitude to plot: 'E', 'H', or 'J'. A two-letter value
            (e.g., 'Ex') is accepted to plot a single component; in that case
            ``component`` is overridden. Default 'E'.
        component : str, optional
            Component of the field to plot: 'x', 'y', 'z', or 'Abs'. Default
            'z'. Overridden when ``field`` contains a component.
        clim : sequence, optional
            Colorbar limits for the field plot as ``[min, max]``.
        hide_solids : str, list, or None, optional
            If provided, mask values inside the named STL solids to ``np.nan``.
            NaNs are shown in grey due to a known PyVista opacity limitation.
        show_solids : str, list, or None, optional
            If provided, mask values outside the named STL solids to ``np.nan``.
        add_stl : str or list, optional
            STL solid key or list of keys to add to the scene via ``pv.add_mesh``.
        stl_opacity : float, optional
            Opacity for STL surfaces (0 transparent, 1 opaque). Default 0.1.
        stl_colors : str or list, optional
            Color or list of colors for STL surfaces (default 'white').
        title : str, optional
            Basename used to save screenshots when ``off_screen=True``.
        cmap : str, optional
            Colormap name for the field display (default 'jet').
        clip_interactive : bool, optional
            Enable an interactive clipping plane widget (default False).
        clip_normal : str, optional
            Normal direction for the interactive clip plane (default '-y').
        clip_box : bool, optional
            Enable a static box clipping of the domain (default False).
        clip_bounds : sequence, optional
            Box bounds as ``[xmin,xmax,ymin,ymax,zmin,zmax]`` when ``clip_box``
            is active.
        off_screen : bool, optional
            Render off-screen and save screenshots instead of opening an
            interactive window (default False).
        zoom : float, optional
            Camera zoom factor (default 0.5).
        camera_position : str, sequence, or None, optional
            Camera position preset or coordinates (default None).
        nan_opacity : float, optional
            Opacity for NaN values produced by masking (default 1.0).
        n : int, optional
            Timestep index appended to saved screenshot filenames.

        Returns
        -------
        None

        Notes
        -----
        - When running with MPI (`use_mpi=True`), plotting is not supported.
        - The method modifies the PyVista plotter state and may save screenshots if
          `off_screen=True`.
        """
        if self.use_mpi:
            print("[!] plot3D is not supported when `use_mpi=True`")
            return

        if len(field) == 2:  # support for e.g. field='Ex'
            component = field[1]
            field = field[0]

        if title is None:
            title = field + component + "3d"

        if self.plotter_active and not off_screen:
            self.plotter_active = False

        if not self.plotter_active:
            pl = pv.Plotter(off_screen=off_screen)

            # Plot stl surface(s)
            if add_stl is not None:
                if type(add_stl) is str:
                    key = add_stl
                    surf = self.grid.read_stl(key)
                    pl.add_mesh(
                        surf,
                        color=stl_colors,
                        opacity=stl_opacity,
                        smooth_shading=True,
                    )

                elif type(add_stl) is list:
                    for i, key in enumerate(add_stl):
                        surf = self.grid.read_stl(key)
                        if type(stl_colors) is list:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[i],
                                opacity=stl_opacity,
                                smooth_shading=True,
                            )
                        else:
                            pl.add_mesh(
                                surf,
                                color=stl_colors,
                                opacity=stl_opacity,
                                smooth_shading=True,
                            )
                else:
                    key = self.grid.stl_solids.keys()[0]
                    surf = self.grid.read_stl(key)
                    pl.add_mesh(
                        surf,
                        color=stl_colors,
                        opacity=stl_opacity,
                        smooth_shading=True,
                    )

            if camera_position is None:
                pl.camera_position = "zx"
                pl.camera.azimuth += 30
                pl.camera.elevation += 30
            else:
                pl.camera_position = camera_position

            pl.set_background("mistyrose", top="white")
            self._add_logo_widget(pl)
            pl.camera.zoom(zoom)
            pl.add_axes()
            pl.enable_3_lights()

            if off_screen:
                self.plotter_active = True
        else:
            pl = self.pl

        # Plot field
        if field == "E":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.E[:, :, :, component], self.N
            )
        elif field == "H":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.H[:, :, :, component], self.N
            )
        elif field == "J":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.J[:, :, :, component], self.N
            )
        else:
            print("`field` value not valid")

        points = self.grid.grid.cell_data_to_point_data()  # interpolate

        # Mask the values inside solid to np.nan
        if hide_solids is not None:
            tol = np.min([self.dx, self.dy, self.dz]) * 1e-3
            if type(hide_solids) is str:
                surf = self.grid.read_stl(hide_solids)
                select = self.grid.grid.select_enclosed_points(
                    surf, tolerance=tol
                )
                mask = select["SelectedPoints"] > 0

            elif type(hide_solids) is list:
                for i, solid in enumerate(hide_solids):
                    surf = self.grid.read_stl(solid)
                    select = self.grid.grid.select_enclosed_points(
                        surf, tolerance=tol
                    )
                    if i == 0:
                        mask = select["SelectedPoints"] > 0
                    else:
                        mask += select["SelectedPoints"] > 0

            points[field + component][mask] = np.nan

        # Mask the values outside solid to np.nan
        if show_solids is not None:
            tol = np.min([self.dx, self.dy, self.dz]) * 1e-3
            if type(show_solids) is str:
                surf = self.grid.read_stl(show_solids)
                select = self.grid.grid.select_enclosed_points(
                    surf, tolerance=tol
                )
                mask = select["SelectedPoints"] > 0

            elif type(show_solids) is list:
                for i, solid in enumerate(show_solids):
                    surf = self.grid.read_stl(solid)
                    select = self.grid.grid.select_enclosed_points(
                        surf, tolerance=tol
                    )
                    if i == 0:
                        mask = select["SelectedPoints"] > 0
                    else:
                        mask += select["SelectedPoints"] > 0

            points[field + component][np.logical_not(mask)] = np.nan

        # Clip a rectangle of the domain
        if clip_box:
            if clip_bounds is None:
                Lx, Ly = (
                    (self.grid.xmax - self.grid.xmin),
                    (self.grid.ymax - self.grid.ymin),
                )
                clip_bounds = [
                    self.grid.xmax - Lx / 2,
                    self.grid.xmax,
                    self.grid.ymax - Ly / 2,
                    self.grid.ymax,
                    self.grid.zmin,
                    self.grid.zmax,
                ]

            ac1 = pl.add_mesh(
                points.clip_box(bounds=clip_bounds),
                opacity=nan_opacity,
                scalars=field + component,
                cmap=cmap,
                clim=clim,
            )

        # Enable an interactive widget to clip out part of the domain with a plane, with clip_normal
        elif clip_interactive:
            ac1 = pl.add_mesh_clip_plane(
                points,
                normal=clip_normal,
                opacity=1.0,
                scalars=field + component,
                cmap=cmap,
                clim=clim,
                normal_rotation=False,
                nan_opacity=nan_opacity,
            )
        else:
            print("Plotting option inconsistent")

        # Save
        if n is not None:
            pl.add_title(
                field + component + f" field, timestep={n}",
                font="times",
                font_size=12,
            )
            title += "_" + str(n).zfill(6)
        if off_screen:
            pl.screenshot(title + ".png")
            try:
                pl.remove_actor(ac1)
            except Exception:
                pass
            self.pl = pl
        else:
            pl.show(
                full_screen=False, auto_close=False, interactive_update=True
            )

    def plot3DonSTL(
        self,
        field="E",
        component="z",
        clim=None,
        cmap="jet",
        log_scale=False,
        stl_with_field=None,
        field_opacity=1.0,
        tolerance=None,
        stl_transparent=None,
        stl_opacity=0.1,
        stl_colors="white",
        clip_plane=False,
        clip_interactive=False,
        clip_normal="-x",
        clip_origin=[0, 0, 0],
        clip_box=False,
        clip_bounds=None,
        title=None,
        off_screen=False,
        n=None,
        zoom=0.5,
        camera_position=None,
        **kwargs,
    ):
        """Built-in 3D plotting sampling fields onto STL surfaces.

        Samples a cell-centered field onto STL surfaces and visualizes the
        sampled field. Supports clipping with a plane or a static box and
        rendering the STL surfaces with or without sampled field data.

        Parameters
        ----------
        field : str, optional
            Field to sample and plot: 'E', 'H', or 'J'. Two-letter values like
            'Ex' are accepted and override ``component``. Default 'E'.
        component : str, optional
            Component to plot ('x', 'y', 'z', 'Abs'). Default 'z'.
        clim : sequence, optional
            Color limits for the plotted field.
        cmap : str, Colormap, optional
            Colormap for field rendering (default 'jet').
        log_scale : bool, optional
            Use logarithmic color scaling when True (default False).
        stl_with_field : str or list, optional
            STL key or list of keys whose surfaces will be sampled with the
            selected field.
        field_opacity : float, optional
            Opacity for the sampled field rendering on STL surfaces.
        tolerance : float, optional
            Sampling tolerance passed to ``pyvista.PolyData.sample``.
        stl_transparent : str or list, optional
            STL solids to add to the scene with transparent rendering.
        stl_opacity : float, optional
            Opacity for non-sampled STL solids (default 0.1).
        stl_colors : str or list, optional
            Colors used for STL solids (default 'white').
        clip_interactive : bool, optional
            Enable an interactive clipping plane for sampled surfaces.
        clip_plane : bool, optional
            Clip the STL surface with a plane and show the field on the plane.
        clip_normal : str, optional
            Normal direction for clipping operations (default '-x').
        clip_origin : sequence, optional
            Origin for planar clipping when ``clip_plane`` is True.
        clip_box : bool, optional
            Clip sampled surfaces using a static box (default False).
        clip_bounds : sequence, optional
            Bounds for the static clip box as
            ``[xmin,xmax,ymin,ymax,zmin,zmax]``.
        title : str, optional
            Basename used to save screenshots when ``off_screen=True``.
        off_screen : bool, optional
            Render off-screen and save screenshots when True.
        n : int, optional
            Timestep index appended to saved filenames.
        zoom : float, optional
            Camera zoom factor (default 0.5).
        camera_position : str, sequence, or None, optional
            Camera position preset or coordinates (default None).
        **kwargs : dict
            Extra keyword arguments forwarded to ``pyvista.Plotter.add_mesh``.

        Returns
        -------
        None

        Notes
        -----
        - When running with MPI (`use_mpi=True`), plotting is not supported.
        - The method modifies the PyVista plotter state and may save screenshots if
          `off_screen=True`.
        """
        if self.use_mpi:
            print("[!] plot3D is not supported when `use_mpi=True`")
            return

        if len(field) == 2:  # support for e.g. field='Ex'
            component = field[1]
            field = field[0]

        if title is None:
            title = field + component + "3d"

        if self.plotter_active and not off_screen:
            self.plotter_active = False

        if not self.plotter_active:
            pl = pv.Plotter(off_screen=off_screen, lighting="none")
            light = pv.Light(light_type="headlight")
            pl.add_light(light)

            # Plot stl surface(s)
            if stl_transparent is not None:
                if type(stl_transparent) is str:
                    key = stl_transparent
                    surf = self.grid.read_stl(key)
                    pl.add_mesh(
                        surf,
                        color=stl_colors,
                        opacity=stl_opacity,
                        smooth_shading=True,
                    )

                elif type(stl_transparent) is list:
                    for i, key in enumerate(stl_transparent):
                        surf = self.grid.read_stl(key)
                        if type(stl_colors) is list:
                            pl.add_mesh(
                                surf,
                                color=stl_colors[i],
                                opacity=stl_opacity,
                                smooth_shading=True,
                            )
                        else:
                            pl.add_mesh(
                                surf,
                                color=stl_colors,
                                opacity=stl_opacity,
                                smooth_shading=True,
                            )

            if off_screen:
                self.plotter_active = True
        else:
            pl = self.pl

        # Plot field
        if field == "E":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.E[:, :, :, component], self.N
            )

        elif field == "H":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.H[:, :, :, component], self.N
            )

        elif field == "J":
            self.grid.grid.cell_data[field + component] = np.reshape(
                self.J[:, :, :, component], self.N
            )
        else:
            print("`field` value not valid")

        points = self.grid.grid.cell_data_to_point_data()  # interpolate

        # Interpolate fields on stl
        if stl_with_field is not None:
            if type(stl_with_field) is str:
                key = stl_with_field
                surf = self.grid.read_stl(key)
                if clip_plane:
                    try:
                        surf = surf.clip_closed_surface(
                            normal=clip_normal, origin=clip_origin
                        ).subdivide_adaptive(max_edge_len=3 * np.min(self.dz))
                    except Exception as e:
                        surf = surf.clip(
                            normal=clip_normal,
                            origin=clip_origin,
                            invert=False,
                        ).subdivide_adaptive(max_edge_len=3 * np.min(self.dz))
                        if self.verbose > 1:
                            print(f"[!] '{key}' surface non-manifold: {e}")

                fieldonsurf = surf.sample(points, tolerance=tolerance)

                if clip_interactive:  # interactive plotting with a plane
                    ac1 = pl.add_mesh_clip_plane(
                        fieldonsurf,
                        normal=clip_normal,
                        normal_rotation=False,
                        scalars=field + component,
                        opacity=field_opacity,
                        cmap=cmap,
                        clim=clim,
                        log_scale=log_scale,
                        **kwargs,
                    )

                elif clip_box:  # Clip a rectangle of the domain
                    if clip_bounds is None:
                        Lx, Ly = (
                            (self.grid.xmax - self.grid.xmin),
                            (self.grid.ymax - self.grid.ymin),
                        )
                        clip_bounds = [
                            self.grid.xmax - Lx / 2,
                            self.grid.xmax,
                            self.grid.ymax - Ly / 2,
                            self.grid.ymax,
                            self.grid.zmin,
                            self.grid.zmax,
                        ]

                    ac1 = pl.add_mesh(
                        fieldonsurf.clip_box(bounds=clip_bounds),
                        cmap=cmap,
                        clim=clim,
                        scalars=field + component,
                        opacity=field_opacity,
                        log_scale=log_scale,
                        **kwargs,
                    )

                else:
                    ac1 = pl.add_mesh(
                        fieldonsurf,
                        cmap=cmap,
                        clim=clim,
                        scalars=field + component,
                        opacity=field_opacity,
                        log_scale=log_scale,
                        **kwargs,
                    )

            elif type(stl_with_field) is list:
                for i, key in enumerate(stl_with_field):
                    surf = self.grid.read_stl(key)
                    if clip_plane:
                        try:
                            surf = surf.clip_closed_surface(
                                normal=clip_normal, origin=clip_origin
                            ).subdivide_adaptive(
                                max_edge_len=3 * np.min(self.dz)
                            )
                        except Exception as e:
                            surf = surf.clip(
                                normal=clip_normal,
                                origin=clip_origin,
                                invert=False,
                            ).subdivide_adaptive(
                                max_edge_len=3 * np.min(self.dz)
                            )
                            if self.verbose > 1:
                                print(f"[!] '{key}' surface non-manifold: {e}")

                    fieldonsurf = surf.sample(points, tolerance=tolerance)

                    if clip_interactive:  # interactive plotting with a plane
                        ac1 = pl.add_mesh_clip_plane(
                            fieldonsurf,
                            normal=clip_normal,
                            normal_rotation=False,
                            scalars=field + component,
                            opacity=field_opacity,
                            cmap=cmap,
                            clim=clim,
                            log_scale=log_scale,
                            **kwargs,
                        )
                    elif clip_box:  # Clip a rectangle of the domain
                        if clip_bounds is None:
                            Lx, Ly = (
                                (self.grid.xmax - self.grid.xmin),
                                (self.grid.ymax - self.grid.ymin),
                            )
                            clip_bounds = [
                                self.grid.xmax - Lx / 2,
                                self.grid.xmax,
                                self.grid.ymax - Ly / 2,
                                self.grid.ymax,
                                self.grid.zmin,
                                self.grid.zmax,
                            ]
                        ac1 = pl.add_mesh(
                            fieldonsurf.clip_box(bounds=clip_bounds),
                            cmap=cmap,
                            clim=clim,
                            scalars=field + component,
                            opacity=field_opacity,
                            log_scale=log_scale,
                            **kwargs,
                        )
                    else:
                        ac1 = pl.add_mesh(
                            fieldonsurf,
                            cmap=cmap,
                            clim=clim,
                            scalars=field + component,
                            opacity=field_opacity,
                            log_scale=log_scale,
                            **kwargs,
                        )
        if camera_position is None:
            pl.camera_position = "zx"
            pl.camera.azimuth += 30
            pl.camera.elevation += 30
        else:
            pl.camera_position = camera_position

        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.camera.zoom(zoom)
        pl.add_axes()
        pl.enable_anti_aliasing()
        # pl.enable_3_lights()

        if n is not None:
            pl.add_title(
                field + component + f" field, timestep={n}",
                font="times",
                font_size=12,
            )
            title += "_" + str(n).zfill(6)

        # Save
        if off_screen:
            pl.screenshot(title + ".png")
            pl.remove_actor(ac1)
            self.pl = pl
        else:
            pl.show(
                full_screen=False, auto_close=False, interactive_update=True
            )

    def plot2D(
        self,
        field="E",
        component="z",
        plane="ZY",
        pos=0.5,
        norm=None,
        vmin=None,
        vmax=None,
        figsize=[8, 4],
        cmap="jet",
        patch_alpha=0.1,
        patch_reverse=False,
        add_patch=False,
        title=None,
        off_screen=False,
        n=None,
        interpolation="antialiased",
        dpi=100,
        return_handles=False,
    ):
        """Built-in 2D plotting of a field slice using matplotlib.

        Renders a 2D cut through the structured grid and displays the selected
        field component using ``matplotlib.pyplot.imshow``. Optionally adds a
        patch showing STL masks as overlays.

        Parameters
        ----------
        field : str, optional
            Field to plot: 'E', 'H', or 'J'. Two-letter values (e.g. 'Ex') are
            accepted and override ``component``. Default 'E'.
        component : str, optional
            Component to plot ('x','y','z','Abs'). Default 'z'.
        plane : str or sequence, optional
            Cut plane specified as 'XY', 'ZY', 'ZX' or a sequence of slices
            and an integer index ``[x,y,z]``. Default 'XZ'.
        pos : float, optional
            Relative position in the normal direction for the cut (0-1).
            Default 0.5 (center).
        norm : str or None, optional
            Color normalization for ``imshow`` ('linear','log','symlog').
        vmin, vmax : scalar, optional
            Color limits for the plot.
        figsize : sequence, optional
            Figure size for the matplotlib figure (default [8,4]).
        cmap : str, optional
            Colormap for the field plot (default 'jet').
        patch_alpha : float, optional
            Alpha value for STL mask overlays (default 0.1).
        patch_reverse : bool, optional
            If True, reverse the mask for the patch overlay.
        add_patch : str or list, optional
            STL key or list of keys to overlay as a mask patch on the plot.
        title : str, optional
            Basename used to save screenshots when ``off_screen=True``.
        off_screen : bool, optional
            If True save the figure to disk instead of showing it interactively.
        n : int, optional
            Timestep index appended to saved filenames.
        interpolation : str, optional
            Interpolation method for ``imshow`` (default 'antialiased').
        dpi : int, optional
            Dots per inch for the figure (default 100).
        return_handles : bool, optional
            If True return the ``(fig, ax)`` handles instead of showing.

        Returns
        -------
        None or tuple
            Returns (fig, ax) if ``return_handles=True``, otherwise None.

        Notes
        -----
        - When running with MPI (`use_mpi=True`), only rank 0 will display or save the
          figure.
        - STL patch overlays are not supported when running under MPI.
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        xmin, xmax = self.grid.xmin, self.grid.xmax
        ymin, ymax = self.grid.ymin, self.grid.ymax
        zmin, zmax = self.grid.zmin, self.grid.zmax
        _z = self.z

        if self.use_mpi:
            zmin, zmax = self.grid.ZMIN, self.grid.ZMAX
            Nz = self.grid.NZ
            _z = self.grid.Z

        if type(field) is str:
            if len(field) == 2:  # support for e.g. field='Ex'
                component = field[1]
                field = field[0]
            elif len(field) == 4:  # support for e.g. field='EAbs'
                component = field[1:]
                field = field[0]

        if title is None:
            title = field + component + "2d"

        if type(plane) is not str and len(plane) == 3:
            x, y, z = plane[0], plane[1], plane[2]

            if type(plane[2]) is int:
                cut = f"(x,y,a) a={round(self.z[z], 3)}"
                xax, yax = "y", "x"
                extent = [
                    self.y[y].min(),
                    self.y[y].max(),
                    self.x[x].min(),
                    self.x[x].max(),
                ]

            if type(plane[0]) is int:
                cut = f"(a,y,z) a={round(self.x[x], 3)}"
                xax, yax = "z", "y"
                extent = [
                    _z[z].min(),
                    _z[z].max(),
                    self.y[y].min(),
                    self.y[y].max(),
                ]

            if type(plane[1]) is int:
                cut = f"(x,a,z) a={round(self.y[y], 3)}"
                xax, yax = "z", "x"
                extent = [
                    _z[z].min(),
                    _z[z].max(),
                    self.x[x].min(),
                    self.x[x].max(),
                ]

        elif plane == "XY":
            x, y, z = slice(0, Nx), slice(0, Ny), int(Nz * pos)  # plane XY
            cut = f"(x,y,a) a={round(pos * (zmax - zmin) + zmin, 3)}"
            xax, yax = "y", "x"
            extent = [ymin, ymax, xmin, xmax]

        elif plane == "ZY" or plane == "YZ":
            x, y, z = int(Nx * pos), slice(0, Ny), slice(0, Nz)  # plane ZY
            cut = f"(a,y,z) a={round(pos * (xmax - xmin) + xmin, 3)}"
            xax, yax = "z", "y"
            extent = [zmin, zmax, ymin, ymax]

        elif plane == "ZX" or plane == "XZ":
            x, y, z = slice(0, Nx), int(Ny * pos), slice(0, Nz)  # plane XZ
            cut = f"(x,a,z) a={round(pos * (ymax - ymin) + ymin, 3)}"
            xax, yax = "z", "x"
            extent = [zmin, zmax, xmin, xmax]

        else:
            print(
                "Plane needs to be an array of slices [x,y,z] or a str 'XY', 'ZY', 'ZX'"
            )

        if self.use_mpi:  # only in rank=0
            _field = self.mpi_gather(field, x=x, y=y, z=z, component=component)

            if self.rank == 0:
                fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
                im = ax.imshow(
                    _field,
                    cmap=cmap,
                    norm=norm,
                    extent=extent,
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation=interpolation,
                )

                fig.colorbar(
                    im,
                    cax=make_axes_locatable(ax).append_axes(
                        "right", size="5%", pad=0.05
                    ),
                )
                ax.set_title(f"Wakis {field}{component}{cut}")
                ax.set_xlabel(xax)
                ax.set_ylabel(yax)

                if n is not None:
                    fig.suptitle(
                        "$"
                        + str(field)
                        + "_{"
                        + str(component)
                        + "}$ field, timestep="
                        + str(n)
                    )
                    title += "_" + str(n).zfill(6)

                fig.tight_layout()

                if off_screen:
                    fig.savefig(title + ".png")
                    plt.clf()
                    plt.close(fig)
                elif return_handles:
                    return fig, ax
                else:
                    plt.show(block=False)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            if field == "E":
                _field = self.E[x, y, z, component]
            if field == "H":
                _field = self.H[x, y, z, component]
            if field == "J":
                _field = self.J[x, y, z, component]

            im = ax.imshow(
                _field,
                cmap=cmap,
                norm=norm,
                extent=extent,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                interpolation=interpolation,
            )

            fig.colorbar(
                im,
                cax=make_axes_locatable(ax).append_axes(
                    "right", size="5%", pad=0.05
                ),
            )
            ax.set_title(f"Wakis {field}{component}{cut}")
            ax.set_xlabel(xax)
            ax.set_ylabel(yax)

            # Patch stl - not supported when running MPI
            if add_patch is not None:
                if type(add_patch) is str:
                    mask = np.reshape(self.grid.grid[add_patch], (Nx, Ny, Nz))
                    patch = np.ones((Nx, Ny, Nz))
                    if patch_reverse:
                        patch[mask] = np.nan
                    else:
                        patch[np.logical_not(mask)] = np.nan
                    ax.imshow(
                        patch[x, y, z],
                        cmap="Greys",
                        extent=extent,
                        origin="lower",
                        alpha=patch_alpha,
                    )

                elif type(add_patch) is list:
                    for solid in add_patch:
                        mask = np.reshape(self.grid.grid[solid], (Nx, Ny, Nz))
                        patch = np.ones((Nx, Ny, Nz))
                        if patch_reverse:
                            patch[mask] = np.nan
                        else:
                            patch[np.logical_not(mask)] = np.nan
                        ax.imshow(
                            patch[x, y, z],
                            cmap="Greys",
                            extent=extent,
                            origin="lower",
                            alpha=patch_alpha,
                        )

            if n is not None:
                fig.suptitle(
                    "$"
                    + str(field)
                    + "_{"
                    + str(component)
                    + "}$ field, timestep="
                    + str(n)
                )
                title += "_" + str(n).zfill(6)

            fig.tight_layout()

            if off_screen:
                fig.savefig(title + ".png")
                plt.clf()
                plt.close(fig)
            elif return_handles:
                return fig, ax
            else:
                plt.show(block=False)

    def plot1D(
        self,
        field="E",
        component="z",
        line="z",
        pos=[0.5],
        xscale="linear",
        yscale="linear",
        xlim=None,
        ylim=None,
        figsize=[8, 4],
        title=None,
        off_screen=False,
        n=None,
        colors=None,
        **kwargs,
    ):
        """Built-in 1D plotting of a field line using matplotlib.

        Plots 1D cuts along a grid line (x, y or z) for the selected field
        component. Supports plotting multiple positions and MPI gathering when
        running in parallel.

        Parameters
        ----------
        field : str, optional
            Field to plot: 'E', 'H', or 'J'. Two-letter values (e.g. 'Ex') are
            accepted and override ``component``. Default 'E'.
        component : str, optional
            Component to plot ('x','y','z','Abs'). Default 'z'.
        line : str or sequence, optional
            Line specification as 'x','y','z' or a sequence of three indices
            or slices (e.g. [0, slice(10,Ny-10), 0]). Default 'z'.
        pos : float or sequence, optional
            Relative position(s) (0-1) used when ``line`` is a single-axis
            specifier. Default 0.5.
        xscale, yscale : str, optional
            Axis scaling passed to matplotlib (default 'linear').
        xlim, ylim : sequence, optional
            Axis limits for the plot.
        figsize : sequence, optional
            Figure size for the matplotlib figure (default [8,4]).
        title : str, optional
            Basename used to save screenshots when ``off_screen=True``.
        off_screen : bool, optional
            If True save the figure to disk instead of showing it interactively.
        n : int, optional
            Timestep index appended to saved filenames.
        colors : sequence, optional
            Colors used for each plotted line when multiple positions are
            requested.
        **kwargs : dict
            Extra keyword arguments forwarded to ``matplotlib.axes.Axes.plot``.

        Returns
        -------
        None

        Notes
        -----
        - When running with MPI (`use_mpi=True`), only rank 0 will display or save the
          figure.
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        xmin, xmax = self.grid.xmin, self.grid.xmax
        ymin, ymax = self.grid.ymin, self.grid.ymax
        zmin, zmax = self.grid.zmin, self.grid.zmax
        _z = self.z

        if self.use_mpi:
            zmin, zmax = self.grid.ZMIN, self.grid.ZMAX
            Nz = self.grid.NZ
            _z = self.grid.Z
            if self.rank == 0:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plotkw = {"lw": 1.2, "ls": "-"}

        if colors is None:
            colors = [
                "k",
                "#5ccfe6",
                "#bae67e",
                "#ffae57",
                "#fdb6d0",
                "#ffd580",
                "#a2aabc",
            ]
        plotkw.update(kwargs)

        if len(field) == 2:  # support for e.g. field='Ex'
            component = field[1]
            field = field[0]

        if title is None:
            title = field + component + "1d"

        if type(pos) is not list:  # support for a list of cut positions
            pos_arr = [pos]
        else:
            pos_arr = pos

        for i, pos in enumerate(pos_arr):
            if type(line) is not str and len(line) == 3:
                x, y, z = line[0], line[1], line[2]

                # z-axis
                if type(line[2]) is slice:
                    cut = f"(a,b,z) a={round(self.x[x], 3)}, b={round(self.y[y], 3)}"
                    xax = "z"
                    xx = _z[z]
                    xlims = (_z[z].min(), _z[z].max())

                # x-axis
                elif type(line[0]) is slice:
                    cut = (
                        f"(x,a,b) a={round(self.y[y], 3)}, b={round(_z[z], 3)}"
                    )
                    xax = "x"
                    xx = self.x[x]
                    xlims = (self.x[x].min(), self.x[x].max())

                # y-axis
                elif type(line[1]) is slice:
                    cut = (
                        f"(a,y,b) a={round(self.x[x], 3)}, b={round(_z[z], 3)}"
                    )
                    xax = "y"
                    xx = self.y[y]
                    xlims = (self.y[y].min(), self.y[y].max())

            elif line.lower() == "x":
                x, y, z = slice(0, Nx), int(Ny * pos), int(Nz * pos)  # x-axis
                cut = f"(x,a,b) a={round(self.y[y], 3)}, b={round(_z[z], 3)}"
                xax = "x"
                xx = self.x[x]
                xlims = (xmin, xmax)

            elif line.lower() == "y":
                x, y, z = int(Nx * pos), slice(0, Ny), int(Nz * pos)  # y-axis
                cut = f"(a,y,b) a={round(self.x[x], 3)}, b={round(_z[z], 3)}"
                xax = "y"
                xx = self.y[y]
                xlims = (ymin, ymax)

            elif line.lower() == "z":
                x, y, z = int(Nx * pos), int(Ny * pos), slice(0, Nz)  # z-axis
                cut = (
                    f"(a,b,z) a={round(self.x[x], 3)}, b={round(self.y[y], 3)}"
                )
                xax = "z"
                xx = _z[z]
                xlims = (zmin, zmax)

            else:
                print(
                    "line needs to be an array of slices [x,y,z] or a str 'x', 'y', 'z'"
                )

            if i == 0:  # first one on top
                zorder = 10
            else:
                zorder = i

            if self.use_mpi:  # only in rank=0
                _field = self.mpi_gather(
                    field, x=x, y=y, z=z, component=component
                )

                if self.rank == 0:
                    ax.plot(
                        xx,
                        _field,
                        color=colors[i],
                        zorder=zorder,
                        label=f"{field}{component}{cut}",
                        **plotkw,
                    )
            else:
                if field == "E":
                    _field = self.E[x, y, z, component]
                if field == "H":
                    _field = self.H[x, y, z, component]
                if field == "J":
                    _field = self.J[x, y, z, component]

                ax.plot(
                    xx,
                    _field,
                    color=colors[i],
                    zorder=zorder,
                    label=f"{field}{component}{cut}",
                    **plotkw,
                )

        if not self.use_mpi:
            yax = f"{field}{component} amplitude"

            ax.set_title(
                f"Wakis {field}{component}" + (len(pos_arr) == 1) * f"{cut}"
            )
            ax.set_xlabel(xax)
            ax.set_ylabel(yax, color=colors[0])
            ax.set_xlim(xlims)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            if len(pos_arr) > 1:
                ax.legend(loc=1)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            if n is not None:
                fig.suptitle(
                    "$"
                    + field
                    + "_{"
                    + component
                    + "}$ field, timestep="
                    + str(n)
                )
                title += "_" + str(n).zfill(6)

            fig.tight_layout()

            if off_screen:
                fig.savefig(title + ".png")
                plt.clf()
                plt.close(fig)

            else:
                plt.show()

        elif self.use_mpi and self.rank == 0:
            yax = f"{field}{component} amplitude"

            ax.set_title(
                f"Wakis {field}{component}" + (len(pos_arr) == 1) * f"{cut}"
            )
            ax.set_xlabel(xax)
            ax.set_ylabel(yax, color=colors[0])
            ax.set_xlim(xlims)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            if len(pos_arr) > 1:
                ax.legend(loc=1)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            if n is not None:
                fig.suptitle(
                    "$"
                    + field
                    + "_{"
                    + component
                    + "}$ field, timestep="
                    + str(n)
                )
                title += "_" + str(n).zfill(6)

            fig.tight_layout()

            if off_screen:
                fig.savefig(title + ".png")
                plt.clf()
                plt.close(fig)

            else:
                plt.show()

    def inspect(
        self,
        wake=None,
        window_size=None,
        off_screen=False,
        opacity=1.0,
        inactive_opacity=0.1,
        add_silhouette=False,
        specular=0.5,
        smooth_shading=False,
    ):
        """Interactive inspection of STL solids and optional wake geometry.

        Provides a PyVista-based UI to toggle visibility and highlighting for
        the package STL solids. When a ``wake`` object is provided the method
        also creates a beam and an integration-path actor and exposes check
        boxes to toggle them. A master checkbox toggles all solids at once.

        Parameters
        ----------
        wake : object or None, optional
            Optional wake-like object (for example a ``WakeSolver`` instance)
            that provides ``xsource``, ``ysource``, ``xtest`` and ``ytest``
            attributes used to place a beam and an integration path. When
            provided the object is stored as ``self.wake``. Default ``None``.
        window_size : sequence or None, optional
            Tuple ``(width, height)`` passed to ``pyvista.Plotter`` to set
            the window size. Default ``None`` (use PyVista defaults).
        off_screen : bool, optional
            If True the plotter is not shown interactively and the method
            returns the created ``pyvista.Plotter`` instance for programmatic
            control and screenshot capture. Default ``False``.
        opacity : float, optional
            Opacity applied to solids when they are active/highlighted
            (default 1.0).
        inactive_opacity : float, optional
            Opacity applied to solids when inactive (default 0.1).
        add_silhouette : bool, optional
            When True add a silhouette actor behind each solid for emphasis.
        specular : float, optional
            Specular lighting value forwarded to ``pyvista.Plotter.add_mesh``
            for STL actors (default 0.5).
        smooth_shading : bool, optional
            Enable smooth shading for STL meshes (default False).

        Returns
        -------
        pyvista.Plotter or None
            Returns the created ``pyvista.Plotter`` when ``off_screen`` is
            True so the caller can save screenshots or control rendering
            programmatically. Otherwise shows the interactive window and
            returns ``None``.

        Notes
        -----
        - The method expects ``self.stl_solids`` and ``self.stl_colors`` to
          be populated and uses ``self.grid.read_stl(key)`` to load surfaces.
        - When running under MPI the method prints a warning and returns
          immediately (interactive plotting is not supported under MPI).
        """
        if self.use_mpi:
            print("[!] plot3D is not supported when `use_mpi=True`")
            return
        if wake is not None:
            self.wake = wake

        # Initialize plotter
        pl = pv.Plotter(window_size=window_size)
        solid_state = {}
        for key, path in self.stl_solids.items():
            surf = self.grid.read_stl(key)
            color = self.stl_colors.get(key, "lightgray")
            actor = pl.add_mesh(
                surf,
                color=color,
                name=key,
                opacity=inactive_opacity,
                silhouette=False,
                smooth_shading=smooth_shading,
                specular=specular,
            )
            sil = None
            if add_silhouette:
                sil = pl.add_silhouette(surf, color="black", line_width=3.0)
                sil.SetVisibility(False)

            solid_state[key] = {
                "actor": actor,
                "silhouette": sil,
                "active_opacity": opacity,
                "inactive_opacity": inactive_opacity,
                "checked": False,
                "highlight": False,
                "button": None,
            }

        # UI scale and solid checkboxes positioning
        w, h = pl.window_size
        ui = h / 800.0
        box = max(16, int(20 * ui))
        font = max(8, int(12 * ui))
        pad = int(10 * ui)
        dy = box + pad
        cx = int(10 * ui)
        cy = h // 2

        # checkboxes callbacks for solids and master (all On/Off)
        color_on = "green"
        color_off = "white"

        def apply(state):
            if state["highlight"]:
                state["actor"].GetProperty().SetOpacity(
                    state["active_opacity"]
                )
                if add_silhouette:
                    state["silhouette"].SetVisibility(True)
            else:
                state["actor"].GetProperty().SetOpacity(
                    state["inactive_opacity"]
                )
                if add_silhouette:
                    state["silhouette"].SetVisibility(False)

        def make_cb(name):
            def _cb(v):
                s = solid_state[name]
                s["checked"] = bool(v)
                s["highlight"] = s["checked"]
                apply(s)

            return _cb

        master_on = True

        def master_cb(v):
            nonlocal master_on
            master_on = bool(v)
            for s in solid_state.values():
                s["checked"] = master_on
                s["highlight"] = master_on
                btn = s["button"]
                if btn:
                    rep = btn.GetRepresentation()
                    if hasattr(rep, "SetState"):
                        rep.SetState(1 if master_on else 0)
                apply(s)

        pl.add_checkbox_button_widget(
            master_cb,
            value=False,
            color_on=color_on,
            color_off=color_off,
            position=(cx, cy),
            size=box,
        )
        pl.add_text("All on", position=(cx + box + pad, cy), font_size=font)

        for i, name in enumerate(solid_state):
            y = cy - (i + 2) * dy
            _color_on = self.stl_colors.get(name, color_on)
            if str(_color_on) == "white" or list(_color_on) == [1.0, 1.0, 1.0]:
                _color_on = "gray"
            btn = pl.add_checkbox_button_widget(
                make_cb(name),
                value=False,
                color_on=_color_on,
                color_off=color_off,
                position=(cx, y),
                size=box,
            )
            solid_state[name]["button"] = btn
            pl.add_text(name, position=(cx + box + pad, y), font_size=font)

        # Add beam & integration path checkboxes if wake object is passed
        if self.wake is not None:
            z_center = 0.5 * (self.grid.zmin + self.grid.zmax)
            z_height = self.grid.zmax - self.grid.zmin
            radius = 0.005 * max(
                self.grid.xmax - self.grid.xmin,
                self.grid.ymax - self.grid.ymin,
            )

            beam = pv.Cylinder(
                center=(self.wake.xsource, self.wake.ysource, z_center),
                direction=(0, 0, 1),
                height=z_height,
                radius=radius * 1.1,
            )
            path = pv.Cylinder(
                center=(self.wake.xtest, self.wake.ytest, z_center),
                direction=(0, 0, 1),
                height=z_height,
                radius=radius,
            )

            beam_actor = pl.add_mesh(
                beam, color="orange", name="beam", opacity=1.0
            )
            beam_actor.SetVisibility(True)
            path_actor = pl.add_mesh(
                path, color="blue", name="integration_path", opacity=1.0
            )
            path_actor.SetVisibility(True)

            bx = int(w - box - 200 * ui)
            to = box + pad

            def beam_cb(v):
                beam_actor.SetVisibility(bool(v))

            def path_cb(v):
                path_actor.SetVisibility(bool(v))

            pl.add_checkbox_button_widget(
                path_cb,
                value=True,
                color_off=color_off,
                color_on="blue",
                position=(bx, cy + dy),
                size=box,
            )
            pl.add_text(
                "Integration path", position=(bx + to, cy + dy), font_size=font
            )

            pl.add_checkbox_button_widget(
                beam_cb,
                value=True,
                color_off=color_off,
                color_on="orange",
                position=(bx, cy),
                size=box,
            )
            pl.add_text("Beam", position=(bx + to, cy), font_size=font)

        pl.set_background("mistyrose", top="white")
        self._add_logo_widget(pl)
        pl.add_axes()

        # Save
        if off_screen:
            pl.off_screen = True
            return pl
        else:
            pl.show(auto_close=False, interactive_update=True)
            return None

    def _add_logo_widget(self, pl):
        """Add packaged logo via importlib.resources (Python 3.9+)."""
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
