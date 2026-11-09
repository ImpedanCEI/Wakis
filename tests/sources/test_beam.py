"""Test beam-current deposition, profile stability, and plotting."""

import matplotlib.pyplot as plt
import numpy as np

from wakis.sources import Beam


def test_source_updates_component(unit_solver):
    beam = Beam(sigmaz=0.01, ti=0.0)

    beam.update(unit_solver, 0.0)

    assert np.any(unit_solver.J[:, :, :, "z"])
    assert not np.any(unit_solver.J[:, :, :, "x"])
    assert not np.any(unit_solver.J[:, :, :, "y"])
    assert beam.Jold.shape == unit_solver.z.shape


def test_beam_plot_uses_longitudinal_current_profile(monkeypatch):
    beam = Beam(q=2e-9, sigmaz=0.01, ti=0.0)
    times = np.array([0.0, beam.sigmaz / beam.v])
    monkeypatch.setattr(plt, "show", lambda: None)

    beam.plot(times)

    current = plt.gcf().axes[0].lines[0].get_ydata()
    np.testing.assert_allclose(current[1] / current[0], np.exp(-0.5))
    plt.close()


def test_beam_edge_locations():
    """Deposit the Gaussian at Jz edge midpoints on a nonuniform grid."""
    from wakis import GridFIT3D, SolverFIT3D

    grid = GridFIT3D(
        x=np.array([-0.03, -0.01, 0.0, 0.02, 0.05]),
        y=np.array([-0.04, -0.02, 0.01, 0.03, 0.06]),
        z=np.array([-0.1, -0.06, -0.01, 0.05, 0.13, 0.2]),
        verbose=0,
    )
    solver = SolverFIT3D(grid, source_type="direct", verbose=0)
    beam = Beam(q=2e-9, sigmaz=0.03, ti=0.0, xsource=0.003, ysource=0.008)
    t = 0.08 / beam.v - solver.dt / 2
    beam.update(solver, t)

    # The nearest transverse nodes are (0, 0.01); the bunch center is z=-0.02.
    z = (grid.z[:-1] + grid.z[1:]) / 2
    area = (grid.x[3] - grid.x[1]) / 2 * (grid.y[3] - grid.y[1]) / 2
    expected = beam.q * beam.v / area / (np.sqrt(2 * np.pi) * beam.sigmaz)
    expected *= np.exp(-0.5 * ((z + 0.02) / beam.sigmaz) ** 2)
    np.testing.assert_allclose(solver.J[2, 2, :-1, "z"], expected)
    assert not np.any(solver.J[2, 2, -1, "z"])
    assert not np.any(solver.J[1, 2, :, "z"])
