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
