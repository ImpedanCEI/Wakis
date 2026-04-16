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
