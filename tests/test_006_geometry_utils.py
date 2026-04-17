import sys

sys.path.append("../wakis")

from wakis import geometry


class TestGeometryUtils:
    # Compare data
    STP_FILE = "tests/stl/006_muonCavity.stp"

    COLORS = {
        "000_Vacuum-Half_cell_dx": [0.5, 0.800000011920929, 1.0],
        "001_Be-windows-Be-window-left": [
            0.752941012382507,
            0.752941012382507,
            0.752941012382507,
        ],
        "002_Be-windows-Be-window-right": [
            0.752941012382507,
            0.752941012382507,
            0.752941012382507,
        ],
        "003_Walls-Cavity-walls": [1.0, 0.615685999393463, 0.235293999314308],
        "004_Vacuum-Half_cell_sx": [0.5, 0.800000011920929, 1.0],
    }

    MATERIALS = {
        "000_Vacuum-Half_cell_dx": "vacuum",
        "001_Be-windows-Be-window-left": "berillium",
        "002_Be-windows-Be-window-right": "berillium",
        "003_Walls-Cavity-walls": "copper--annealed-",
        "004_Vacuum-Half_cell_sx": "vacuum",
    }

    SOLIDS = {
        "000_Vacuum-Half_cell_dx": "000_Vacuum-Half_cell_dx_Vacuum.stl",
        "001_Be-windows-Be-window-left": "001_Be-windows-Be-window-left_Berillium.stl",
        "002_Be-windows-Be-window-right": "002_Be-windows-Be-window-right_Berillium.stl",
        "003_Walls-Cavity-walls": "003_Walls-Cavity-walls_Copper--annealed-.stl",
        "004_Vacuum-Half_cell_sx": "004_Vacuum-Half_cell_sx_Vacuum.stl",
    }

    UNITS = 0.001

    def test_colors(self):
        colors = geometry.extract_colors_from_stp(self.STP_FILE)
        assert colors == self.COLORS

    def test_materials(self):
        materials = geometry.extract_materials_from_stp(self.STP_FILE)
        assert materials == self.MATERIALS

    def test_solids(self):
        solids = geometry.extract_solids_from_stp(self.STP_FILE)
        assert solids == self.SOLIDS

    def test_units(self):
        units = geometry.get_stp_unit_scale(self.STP_FILE)
        assert units == self.UNITS

    def test_measure_stl_slice(self):
        stl_file = "tests/stl/007_vacuum_cavity.stl"
        pl = geometry.measure_stl_slice(stl_file, plane="ZX", off_screen=True)
        assert pl is not None
