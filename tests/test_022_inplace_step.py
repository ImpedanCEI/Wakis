"""The in-place E/H update reproduces the previous out-of-place expressions."""

import numpy as np
import pytest

from wakis import GridFIT3D, SolverFIT3D


def _old_update_H(solver):
    """H update as written before the in-place change."""
    solver.H.fromarray(
        solver.H.toarray() - solver.dt * solver.tDsiDmuiDaC * solver.E.toarray()
    )


def _old_update_E(solver):
    """E update as written before the in-place change."""
    solver.E.fromarray(
        solver.E.toarray()
        + solver.dt
        * (
            solver.itDaiDepsDstC * solver.H.toarray()
            - solver.ieps.toarray() * solver.J.toarray()
        )
    )


def _seeded_solver(bc, use_gpu=False):
    x = np.cumsum(np.r_[0.0, np.linspace(0.8, 1.3, 9)]) * 1e-2  # stretched
    y = np.linspace(0.0, 0.08, 9)
    z = np.linspace(0.0, 0.11, 12)
    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver = SolverFIT3D(
        grid, bc_low=[bc] * 3, bc_high=[bc] * 3, use_gpu=use_gpu, verbose=0
    )
    rng = np.random.default_rng(7)
    xp = solver.E.xp
    for field in (solver.E, solver.H, solver.J):
        field.fromarray(xp.asarray(rng.standard_normal(field.array.size)))
    return solver


@pytest.mark.parametrize("bc", ["pec", "periodic"])
def test_inplace_step_matches_previous_expressions(bc, use_gpu):
    """30 steps agree with the out-of-place expressions to round-off.

    J is seeded nonzero, so the ieps * J term of the E update is exercised.
    The arrays of E and H are updated in place (same objects).
    """
    new = _seeded_solver(bc, use_gpu)
    ref = _seeded_solver(bc, use_gpu)
    ref._update_H = lambda: _old_update_H(ref)
    ref._update_E = lambda: _old_update_E(ref)

    e_obj, h_obj = new.E.array, new.H.array
    for _ in range(30):
        new.one_step()
        ref.one_step()
    assert new.E.array is e_obj and new.H.array is h_obj

    for a, b in ((new.E, ref.E), (new.H, ref.H)):
        va, vb = a.toarray(), b.toarray()
        if use_gpu:
            va, vb = va.get(), vb.get()
        scale = np.max(np.abs(vb))
        assert scale > 0
        # Only the order of the dt multiplication differs: a few ulp per step
        np.testing.assert_allclose(va, vb, rtol=0, atol=1e-12 * scale)
