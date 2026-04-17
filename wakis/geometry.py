# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import re


def extract_colors_from_stp(stp_file):
    """
    Extracts a mapping from solid names to RGB color values from a STEP (.stp) file.

    Args:
        stp_file (str): Path to the STEP file.

    Returns:
        dict[str, list[float]]: A dictionary mapping solid names to [R, G, B] colors.
    """
    solids, _ = extract_names_from_stp(stp_file)

    colors = []
    stl_colors = {}

    color_pattern = re.compile(r"#\d+=COLOUR_RGB\('?',([\d.]+),([\d.]+),([\d.]+)\);")

    # Extract colors
    with open(stp_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            color_match = color_pattern.search(line)
            if color_match:
                r = float(color_match.group(1))
                g = float(color_match.group(2))
                b = float(color_match.group(3))
                colors.append([r, g, b])

    # Map solids to colors by order of appearance (colors >=solids)
    for i in range(len(list(solids.keys()))):
        solid = solids[list(solids.keys())[i]]
        solid_re = re.sub(r"[^a-zA-Z0-9_-]", "-", solid)
        stl_colors[f"{str(i).zfill(3)}_{solid_re}"] = colors[i]

    return stl_colors


def extract_materials_from_stp(stp_file):
    """
    Extracts a mapping from solid names to materials from a STEP (.stp) file.

    Args:
        stp_file (str): Path to the STEP file.

    Returns:
        dict[str, str]: A dictionary mapping solid names to material names.
    """

    solids, materials = extract_names_from_stp(stp_file)
    stl_materials = {}
    for i in range(len(list(solids.keys()))):
        solid = solids[list(solids.keys())[i]]
        try:
            mat = materials[list(solids.keys())[i]].lower()
        except KeyError:
            print(f"Solid #{list(solids.keys())[i]} has no assigned material")
            mat = "None"

        # Remove problematic characters
        solid_re = re.sub(r"[^a-zA-Z0-9_-]", "-", solid)
        mat_re = re.sub(r"[^a-zA-Z0-9_-]", "-", mat)
        stl_materials[f"{str(i).zfill(3)}_{solid_re}"] = mat_re

    return stl_materials


def extract_solids_from_stp(stp_file, path=None):
    """
    Extracts a mapping from solid names to STL file names from a STEP (.stp) file.
    Args:
        stp_file (str): Path to the STEP file.
        path (str) (optional): default: None, path to save the STL (.stl) files
    Returns:
        dict[str, str]: A dictionary mapping solid names to STL file names.
    """
    if path is not None and not path.endswith("/"):
        path += "/"
    solids, materials = extract_names_from_stp(stp_file)
    stl_solids = {}
    for i in range(len(list(solids.keys()))):
        solid = solids[list(solids.keys())[i]]
        try:
            mat = materials[list(solids.keys())[i]]
        except KeyError:
            print(f"Solid #{list(solids.keys())[i]} has no assigned material")
            mat = "None"

        # Remove problematic characters
        solid_re = re.sub(r"[^a-zA-Z0-9_-]", "-", solid)
        mat_re = re.sub(r"[^a-zA-Z0-9_-]", "-", mat)
        name = f"{str(i).zfill(3)}_{solid_re}_{mat_re}"
        if path is not None:
            stl_solids[f"{str(i).zfill(3)}_{solid_re}"] = path + name + ".stl"
        else:
            stl_solids[f"{str(i).zfill(3)}_{solid_re}"] = name + ".stl"

    return stl_solids


def extract_names_from_stp(stp_file):
    """
    Extracts solid names and their corresponding materials from a STEP (.stp) file.

    This function parses a given STEP file to identify solid objects and their assigned materials.
    The solid names are extracted from `MANIFOLD_SOLID_BREP` statements, while the materials are
    linked via `PRESENTATION_LAYER_ASSIGNMENT` statements.

    Args:
        stp_file (str): Path to the STEP (.stp) file.

    Returns:
        tuple[dict[int, str], dict[int, str]]:
            - A dictionary mapping solid IDs to their names.
            - A dictionary mapping solid IDs to their corresponding material names.

    Example:
        >>> solids, materials = extract_names_from_stp("example.stp")
        >>> print(solids)
        {37: "Vacuum|Half_cell_dx", 39: "Be window left"}
        >>> print(materials)
        {37: "Vacuum", 39: "Berillium"}
    """
    solid_dict = {}
    material_dict = {}

    # Compile regex patterns
    # solid_pattern = re.compile(r"#(\d+)=MANIFOLD_SOLID_BREP\('([^']+)'.*;")
    solid_pattern = re.compile(
        r"#(\d+)=ADVANCED_BREP_SHAPE_REPRESENTATION\('([^']*)',\(([^)]*)\),#\d+\);"
    )
    material_pattern = re.compile(
        r"#(\d+)=PRESENTATION_LAYER_ASSIGNMENT\('([^']+)','[^']+',\(#([\d#,]+)\)\);"
    )

    # First pass: extract solids
    with open(stp_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            solid_match = solid_pattern.search(line)
            if solid_match:
                # solid_number = int(solid_match.group(1)) #if MANIFOLD
                solid_number = int(
                    solid_match.group(3).split(",")[0].strip().lstrip("#")
                )
                solid_name = solid_match.group(2)
                solid_dict[solid_number] = solid_name

    # Second pass: extract materials
    with open(stp_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            material_match = material_pattern.search(line)
            if material_match:
                material_name = material_match.group(2)
                solid_numbers = [
                    int(num.strip("#")) for num in material_match.group(3).split(",")
                ]
                for solid_number in solid_numbers:
                    if solid_number in solid_dict:
                        material_dict[solid_number] = material_name

    return solid_dict, material_dict


def get_stp_unit_scale(stp_file):
    """
    Reads the unit definition from a STEP (.stp or .step) file and determines the
    scale factor required to convert the geometry to meters.

    This function:
    - Opens and scans the header section of the STEP file.
    - Detects the SI base unit definition (e.g., millimeter, centimeter, meter).
    - Returns a scale factor to convert the geometry to meters.
    - Handles missing or unreadable unit information gracefully.

    Args:
        stp_file (str): Path to the STEP (.stp or .step) file.

    Returns:
        float: Scale factor to convert STEP geometry to meters.
               Defaults to 1.0 if no valid unit information is found.
    """

    unit_map = {
        ".MILLI.": 1e-3,
        ".CENTI.": 1e-2,
        ".DECI.": 1e-1,
        ".KILO.": 1e3,
        "$": 1.0,  # '$' indicates no prefix, i.e. plain meters
    }

    try:
        with open(stp_file, "r", encoding="utf-8", errors="ignore") as f:
            header = f.read(10000)  # read only the beginning of the file

        match = re.search(
            r"SI_UNIT\s*\(\s*(\.\w+\.)?\s*,\s*\.METRE\.\s*\)",
            header,
            re.IGNORECASE,
        )

        if match:
            prefix = match.group(1).upper() if match.group(1) else "$"
            scale = unit_map.get(prefix, 1.0)
            print(f"Detected STEP unit: {prefix} → scale to meters: {scale}")
            return scale
        else:
            print("No unit found, files remain in original unit.")
            return 1.0

    except Exception as exc:
        print(f"Error reading unit from STEP file: {exc}")
        print("Files remain in original unit.")

        return 1.0


def generate_stl_solids_from_stp(stp_file, results_path=None):
    """
    Extracts solid objects from a STEP (.stp) file and exports them as STL files.

    This function:
    - Imports the STEP file using `cadquery`.
    - Extracts solid names and their materials using `extract_names_from_stp()`.
    - Sanitizes solid and material names by replacing problematic characters.
    - Scales the solid to meter using `get_stp_unit_scale()`.
    - Saves each solid as an STL file in the current folder (default) or the given path.

    Args:
        stp_file (str): Path to the STEP (.stp) file.
        results_path (str) (optional): default: '', path to save the STL (.stl) files

    Raises:
        Exception: If `cadquery` is not installed, it prompts the user to install it.

    Example:
        >>> extract_stl_solids_from_stp("example.stp")
        000_Vacuum-Half_cell_dx_Vacuum.stl
        001_Be_window_left_Berillium.stl
    """

    try:
        import cadquery as cq
    except ImportError:
        raise Exception("""This function needs the open-source package `cadquery`
                        To install it in a conda environment do:

                        `pip install cadquery`

                        [!] We recommend having a dedicated conda environment to avoid version issues
                        """)

    stp = cq.importers.importStep(stp_file)

    scale_factor = get_stp_unit_scale(stp_file)
    if scale_factor != 1.0:
        print(f"Scaling geometry to meters (factor={scale_factor}).")
        scaled_solids = [solid.scale(scale_factor) for solid in stp.objects[0]]
        stp.objects = [scaled_solids]

    solid_dict = extract_solids_from_stp(stp_file, results_path)

    print(f"Generating stl from file: {stp_file}... ")
    for i, obj in enumerate(stp.objects[0]):
        name = solid_dict[list(solid_dict.keys())[i]]
        print(name)
        obj.exportStl(name)

    return solid_dict


def measure_stl_slice(
    stl_file,
    plane="ZX",
    position=None,
    stl_opacity=0.1,
    stl_color="blue",
    smooth_shading=False,
    off_screen=False,
):
    """
    Interactive 2D slice visualization of an STL file with distance measurement.

    Display a 2D cross-section of an STL solid with an interactive slider
    to adjust the slice position. Includes point picking with snapping to mesh
    vertices and manual 2-click distance measurement. Measurements are cleared
    when the slice position changes.

    Parameters
    ----------
    stl_file : str
        Path to the STL file to load and visualize.
    plane : {'XY', 'ZY', 'ZX'}, optional
        Plane of the slice. Default 'ZX'.

        - 'XY' → normal along Z, slider controls Z position.
        - 'ZY' → normal along X, slider controls X position.
        - 'ZX' → normal along Y, slider controls Y position.
    position : float or None, optional
        Initial position of the slice along the normal axis. If None, uses
        the center of the domain along that axis.
    stl_opacity : float, optional
        Opacity of the STL (0 = fully transparent, 1 = fully opaque).
        Default 0.1.
    stl_color : str, optional
        Color for the STL solid slice (default 'blue').
    smooth_shading : bool, optional
        Enable smooth shading for the STL surface. Default False.
    off_screen : bool, optional
        If True, render off-screen and return the plotter object.

    Returns
    -------
    pyvista.Plotter or None
        Returns the created plotter if ``off_screen=True``, otherwise None.

    Notes
    -----
    - The slice is updated interactively via the slider widget.
    - Point picking is enabled via a checkbox toggle labeled "Measure".
    - Clicking two points on the slice creates a distance measurement.
    - Measurements are automatically cleared when the slice position changes or
      when the "Clear Measurement" button is clicked.
    - Picked points snap to mesh vertices on the current slice only;
      free-space picking is disabled.

    Example
    -------
    >>> import wakis.geometry as geom
    >>> pl = geom.measure_stl_slice("path/to/geometry.stl", plane="ZX", off_screen=True)
    >>> pl.screenshot("stl_slice.png")
    """
    import numpy as np
    import pyvista as pv

    # Load the STL file
    surf = pv.read(stl_file)

    # Compute domain bounds from the loaded surface
    xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds

    plane = plane.upper()
    plane_to_normal = {"XY": "z", "ZY": "x", "ZX": "y"}
    if plane not in plane_to_normal:
        raise ValueError(f"plane must be one of 'XY', 'ZY', 'ZX', got '{plane}'")

    normal = plane_to_normal[plane]
    if normal == "x":
        axis_min, axis_max = xmin, xmax
        slider_title = "X Position"
        center = ((ymin + ymax) / 2, (zmin + zmax) / 2)

        def origin_fn(v):
            return (v, center[0], center[1])

    elif normal == "y":
        axis_min, axis_max = ymin, ymax
        slider_title = "Y Position"
        center = ((xmin + xmax) / 2, (zmin + zmax) / 2)

        def origin_fn(v):
            return (center[0], v, center[1])

    else:  # z
        axis_min, axis_max = zmin, zmax
        slider_title = "Z Position"
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2)

        def origin_fn(v):
            return (center[0], center[1], v)

    if position is None:
        position = (axis_min + axis_max) / 2

    pv.global_theme.allow_empty_mesh = True
    pl = pv.Plotter(off_screen=off_screen)

    # --- 3D STL context mesh (non-pickable) ---
    pl.add_mesh(
        surf,
        color=stl_color,
        opacity=stl_opacity,
        smooth_shading=smooth_shading,
        pickable=False,
        name="stl_3d",
    )

    # --- Initial slice ---
    initial_slice = surf.slice(normal=normal, origin=origin_fn(position))
    slice_actor = pl.add_mesh(
        initial_slice,
        color=stl_color,
        line_width=3,
        smooth_shading=smooth_shading,
        pickable=True,
        name="stl_slice",
    )

    # --- Measurement state ---
    measurement_state = {
        "measurement_enabled": False,
        "picked_points": [],
        "point_actors": [],
        "line_actor": None,
        "distance_text_actor": None,
        "current_slice": initial_slice,
    }

    # --- Clear measurement function ---
    def clear_measurement():
        """Remove all measurement graphics from the scene."""
        if measurement_state["line_actor"] is not None:
            try:
                pl.remove_actor(measurement_state["line_actor"])
            except Exception:
                pass
            measurement_state["line_actor"] = None

        for actor in measurement_state["point_actors"]:
            try:
                pl.remove_actor(actor)
            except Exception:
                pass
        measurement_state["point_actors"] = []

        if measurement_state["distance_text_actor"] is not None:
            try:
                pl.remove_actor(measurement_state["distance_text_actor"])
            except Exception:
                pass
            measurement_state["distance_text_actor"] = None

        measurement_state["picked_points"] = []

    # --- Update function for slider ---
    def update_slice(val):
        """Update the slice when slider moves and clear measurements."""
        new_slice = surf.slice(normal=normal, origin=origin_fn(val))
        slice_actor.mapper.SetInputData(new_slice)
        measurement_state["current_slice"] = new_slice
        clear_measurement()
        pl.render()

    # --- Point pick callback ---
    def on_point_pick(point):
        """Handle point picking for measurement."""
        if not measurement_state["measurement_enabled"]:
            return

        # Snap to nearest point on the current slice
        if measurement_state["current_slice"].n_points == 0:
            return

        distances = np.linalg.norm(
            measurement_state["current_slice"].points - point, axis=1
        )
        nearest_idx = np.argmin(distances)
        snapped_point = measurement_state["current_slice"].points[nearest_idx]

        measurement_state["picked_points"].append(snapped_point)

        # Add a point actor
        point_cloud = pv.PolyData(snapped_point)
        pt_actor = pl.add_mesh(
            point_cloud,
            color="red",
            point_size=10,
            render_points_as_spheres=True,
            name=f"picked_point_{len(measurement_state['point_actors'])}",
        )
        measurement_state["point_actors"].append(pt_actor)

        # If we have two points, draw the measurement
        if len(measurement_state["picked_points"]) == 2:
            p1, p2 = measurement_state["picked_points"]
            distance = np.linalg.norm(p2 - p1)

            # Draw line segment
            line = pv.Line(p1, p2)
            line_actor = pl.add_mesh(
                line,
                color="red",
                line_width=2,
                name="measurement_line",
            )
            measurement_state["line_actor"] = line_actor

            # Add distance label
            label_text = f"Distance: {distance:.6f} m"
            text_actor = pl.add_text(
                label_text,
                position="lower_left",
                font_size=10,
                color="red",
                name="distance_label",
            )
            measurement_state["distance_text_actor"] = text_actor

            # Reset for next measurement
            measurement_state["picked_points"] = []

        pl.render()

    # --- Slider ---
    pl.add_slider_widget(
        update_slice,
        [axis_min, axis_max],
        value=position,
        title=slider_title,
        pointa=(0.8, 0.6),
        pointb=(0.95, 0.6),
        style="modern",
    )

    # --- Measurement toggle checkbox ---
    def toggle_measurement(state):
        measurement_state["measurement_enabled"] = bool(state)

    ui_scale = pl.window_size[1] / 800.0
    box_size = max(16, int(20 * ui_scale))
    font_size = max(8, int(12 * ui_scale))
    pad = int(10 * ui_scale)
    cx = int(10 * ui_scale)

    pl.add_checkbox_button_widget(
        toggle_measurement,
        value=False,
        color_on="red",
        color_off="white",
        position=(cx, int(pl.window_size[1] / 2)),
        size=box_size,
    )
    pl.add_text(
        "Enable picking",
        position=(cx + box_size + pad, int(pl.window_size[1] / 2)),
        font_size=font_size,
    )

    # --- Clear measurement button ---
    def clear_measurement_widget(_):
        clear_measurement()
        pl.render()

    pl.add_checkbox_button_widget(
        clear_measurement_widget,
        value=False,
        color_on="yellow",
        color_off="white",
        position=(cx, int(pl.window_size[1] / 2) - (box_size + pad)),
        size=box_size,
    )
    pl.add_text(
        "Clear picks",
        position=(cx + box_size + pad, int(pl.window_size[1] / 2) - (box_size + pad)),
        font_size=font_size,
    )

    # --- Enable point picking ---
    pl.enable_point_picking(
        callback=on_point_pick,
        tolerance=0.025,
        left_clicking=True,
        picker="cell",
        pickable_window=False,
        show_point=False,
        show_message="Left-click on the slice to pick measurement points",
        font_size=12,
    )

    # --- Camera orientation ---
    pl.set_background("mistyrose", top="white")
    pl.add_axes()
    pl.camera_position = plane.lower()

    # --- STL bounds info ---
    bounds_text = (
        f"STL bounds:\n"
        f"  X: [{xmin:.6f}, {xmax:.6f}]\n"
        f"  Y: [{ymin:.6f}, {ymax:.6f}]\n"
        f"  Z: [{zmin:.6f}, {zmax:.6f}]"
    )
    pl.add_text(
        bounds_text,
        position="lower_right",
        font_size=9,
        color="black",
        name="bounds_label",
    )

    if off_screen:
        return pl
    else:
        pl.show()
        return None
