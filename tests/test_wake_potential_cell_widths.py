"""The z-cell widths the wake-potential integrals multiply Ez by.

The same code serves a uniform grid (`dz` is a scalar), a grid that stores one
width per cell, and the MPI writer, which stores `np.diff(z)` and so has one
entry fewer than there are cells. Each case has to end up with one width per
selected cell: a scalar left as a scalar raised `TypeError` at `dz[k]`, and a
`np.diff` array ran one cell short (issue #58).
"""

import numpy as np
import pytest

from wakis import WakeSolver


def _solver(zf, dz):
    """A solver with just the z-axis state the width helper reads."""
    solver = WakeSolver.__new__(WakeSolver)  # no grid, no files
    solver.zf = np.asarray(zf, dtype=float)
    solver.z = solver.zf
    solver.dz = dz
    return solver


def test_a_scalar_dz_is_broadcast_to_one_width_per_cell():
    solver = _solver([0.0, 1.0, 2.0, 3.0], 0.5)
    assert np.allclose(solver._cell_widths(np.s_[:], 4), [0.5] * 4)


def test_a_np_diff_dz_keeps_one_width_per_cell():
    zf = np.array([0.0, 0.2, 0.5, 0.6, 1.0])
    solver = _solver(zf, np.diff(zf))
    widths = solver._cell_widths(np.s_[:], len(zf))
    assert widths.shape == (len(zf),)
    assert np.allclose(widths, [0.2, 0.3, 0.1, 0.4, 0.4])


def test_skipped_cells_slice_the_widths_like_the_coordinates():
    zf = np.array([0.0, 0.2, 0.5, 0.6, 1.0])
    solver = _solver(zf, np.array([0.2, 0.3, 0.1, 0.4, 0.5]))
    assert np.allclose(solver._cell_widths(slice(1, -1), 3), [0.3, 0.1, 0.4])


def test_a_dz_that_matches_no_known_shape_is_refused():
    solver = _solver([0.0, 1.0, 2.0, 3.0], np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="dz has 2 entries"):
        solver._cell_widths(np.s_[:], 4)


def _longitudinal_solver(dz, counter_moving):
    """A solver with one transverse cell and a field that grows with z."""
    solver = WakeSolver.__new__(WakeSolver)
    z = np.array([0.0, 1.0, 3.0, 6.0])
    solver.z = z
    solver.zf = z
    solver.dz = dz
    solver.t = np.linspace(0.0, 8e-9, 9)
    solver.ti = 0.0
    solver.v = 1e9
    solver.q = 1.0
    solver.wakelength = 4e-9
    solver.counter_moving = counter_moving
    solver.n_transverse_cells = 0
    solver.skip_cells = 0
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


def test_counter_moving_uses_the_width_of_the_cell_it_reads():
    """Both directions must integrate the same cells with the same widths.

    With one width per cell the counter-moving loop reads `Ezt[-k-1]`, so it
    has to multiply by `dz[-k-1]`. Pairing it with `dz[k]` instead halves the
    result for these widths (20 -> 10), which is what this pins.
    """
    widths = np.array([1.0, 2.0, 3.0, 4.0])
    forward = _longitudinal_solver(widths, counter_moving=False)
    backward = _longitudinal_solver(widths, counter_moving=True)
    forward.calc_long_WP_3d()
    backward.calc_long_WP_3d()
    # atol=0 on purpose: the values are ~1e-11 and numpy's default atol of 1e-8
    # would call a factor-of-two difference close.
    assert np.isclose(abs(backward.WP[0]), abs(forward.WP[0]), rtol=1e-9, atol=0.0)
    assert np.isclose(abs(forward.WP[0]), 20.0 / 1e12, rtol=1e-9, atol=0.0)
