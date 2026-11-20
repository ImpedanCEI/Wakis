import os  # noqa
import sys  # noqa

import pytest  # noqa


def test_dependency_imports():
    import h5py  # noqa
    import numpy  # noqa
    import pyvista  # noqa
    import scipy  # noqa
    from tqdm import tqdm  # noqa


def test_module_imports():
    sys.path.append("../wakis")

    from wakis import Field  # noqa
    from wakis import GridFIT3D  # noqa
    from wakis import SolverFIT3D  # noqa
    from wakis import WakeSolver  # noqa
    from wakis.materials import material_lib  # noqa
    from wakis.sources import Beam  # noqa
    from wakis.sources import PlaneWave  # noqa
    from wakis.sources import Pulse  # noqa
    from wakis.sources import WavePacket  # noqa


def test_source_package_exports():
    import wakis.sources as sources
    from wakis.sources.beam import Beam
    from wakis.sources.dipole import Dipole
    from wakis.sources.gaussian_packet import GaussianPacket
    from wakis.sources.mode_packet import ModePacket
    from wakis.sources.plane_wave import PlaneWave
    from wakis.sources.pulse import Pulse
    from wakis.sources.source import Source, WaveformSource
    from wakis.sources.wave_packet import WavePacket

    source_classes = {
        "Beam": Beam,
        "Dipole": Dipole,
        "GaussianPacket": GaussianPacket,
        "ModePacket": ModePacket,
        "PlaneWave": PlaneWave,
        "Pulse": Pulse,
        "Source": Source,
        "WaveformSource": WaveformSource,
        "WavePacket": WavePacket,
    }

    for name, source_class in source_classes.items():
        assert getattr(sources, name) is source_class
