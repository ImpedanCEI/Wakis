"""The z-cell widths the wake-potential integrals multiply Ez by.

`WakeSolver.dz` reaches the integration in three shapes: a scalar on a uniform
grid, one primal-edge length per field sample, or `np.diff(z)`, which the MPI
writer stores and which has one entry fewer than there are samples. Each shape
must integrate the same field to the same wake potential; before #58 a scalar
raised `TypeError` at `dz[k]` and a `np.diff` array ran one cell short.
"""

import numpy as np
import pytest

from wakis import WakeSolver

# A uniform grid expressed each way it reaches the solver.
UNIFORM = [
    pytest.param(2.0, id="scalar"),
    pytest.param(np.array([2.0, 2.0, 2.0, 2.0]), id="per-cell"),
    pytest.param(np.array([2.0, 2.0, 2.0]), id="np-diff"),
]


def _longitudinal_solver(dz, *, counter_moving=False, skip_cells=0, z=None):
    """A solver with one transverse cell and a field that grows with z."""
    solver = WakeSolver.__new__(WakeSolver)  # no grid, no files
    coordinates = np.array([0.0, 1.0, 3.0, 6.0])
    solver.z = coordinates if z is None else z
    solver.zf = coordinates
    solver.dz = dz
    solver.t = np.linspace(0.0, 8e-9, 9)
    solver.ti = 0.0
    solver.v = 1e9
    solver.q = 1.0
    solver.wakelength = 4e-9
    solver.counter_moving = counter_moving
    solver.n_transverse_cells = 0
    solver.skip_cells = skip_cells
    solver.verbose = 0
    solver.save = False
    solver.folder = ""
    solver.log = lambda *args, **kwargs: None
    field = np.array([0.0, 1.0, 2.0, 3.0]).reshape(1, 1, 4)

    class _H5(dict):
        def close(self):
            pass

    solver.Ez_hf = _H5({f"#{n}": field for n in range(len(solver.t))})
    return solver


@pytest.mark.parametrize("dz", UNIFORM)
def test_every_dz_shape_integrates_the_same_uniform_grid(dz):
    """A scalar, per-cell and np.diff spelling of one grid agree."""
    solver = _longitudinal_solver(dz)
    solver.calc_long_WP_3d()
    # (0 + 1 + 2 + 3) * 2.0 / q
    assert np.isclose(abs(solver.WP[0]), 12.0 / 1e12, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("dz", UNIFORM)
def test_skipped_cells_keep_the_widths_aligned(dz):
    """`skip_cells` trims the coordinates and the widths together."""
    solver = _longitudinal_solver(dz, skip_cells=1)
    solver.calc_long_WP_3d()
    # Two cells remain, so only the middle of the field contributes.
    assert np.isclose(abs(solver.WP[0]), 6.0 / 1e12, rtol=1e-9, atol=0.0)


def test_the_outer_edges_are_recovered_when_z_is_not_given():
    """Without `z`, the domain edges come from `zf` and the edge widths."""
    solver = _longitudinal_solver(2.0, z=None)
    solver.calc_long_WP_3d()
    assert np.isclose(abs(solver.WP[0]), 12.0 / 1e12, rtol=1e-9, atol=0.0)


def test_counter_moving_uses_the_width_of_the_cell_it_reads():
    """Both directions must integrate the same cells with the same widths.

    The counter-moving loop reads `Ezt[-k-1]`, so it has to multiply by
    `dz[-k-1]`. Pairing it with `dz[k]` halves the result for these widths
    (20 -> 10), which is what this pins. `atol=0` on purpose: the values are
    ~1e-11 and numpy's default atol of 1e-8 would call a factor of two close.
    """
    widths = np.array([1.0, 2.0, 3.0, 4.0])
    forward = _longitudinal_solver(widths, counter_moving=False)
    backward = _longitudinal_solver(widths, counter_moving=True)
    forward.calc_long_WP_3d()
    backward.calc_long_WP_3d()
    assert np.isclose(abs(backward.WP[0]), abs(forward.WP[0]), rtol=1e-9, atol=0.0)
    assert np.isclose(abs(forward.WP[0]), 20.0 / 1e12, rtol=1e-9, atol=0.0)
