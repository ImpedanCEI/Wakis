"""Material changes through ``SolverFIT3D.update_tensors``.

Modifying the material Fields and calling ``update_tensors`` before time
stepping must give the same simulation as passing that material at
construction.
"""

import numpy as np
import pytest
from scipy.constants import epsilon_0 as eps_0

from wakis import GridFIT3D, SolverFIT3D


def _lossy_box(eps_r, sigma, dt):
    grid = GridFIT3D(0, 1, 0, 1, 0, 1, 5, 5, 5, verbose=0)
    return SolverFIT3D(
        grid,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        bg=[eps_r, 1.0, sigma],
        dt=dt,
        verbose=0,
    )


def test_update_tensors_matches_constructed_material():
    """A dielectric set via update_tensors behaves as one set at construction.

    Both solvers are conductive (so the conduction current J_old is used),
    share the time step and start from the same random fields, including the
    ghost samples that the first step clears.
    """
    eps_r, sigma, dt = 4.0, 1e-3, 1e-10
    reference = _lossy_box(eps_r, sigma, dt)
    updated = _lossy_box(1.0, sigma, dt)
    updated.ieps *= 1 / eps_r
    updated.update_tensors("ieps")
    np.testing.assert_allclose(updated.ieps.toarray(), 1 / (eps_r * eps_0))

    rng = np.random.default_rng(3)
    E0 = rng.normal(size=reference.E.array.size)
    H0 = rng.normal(size=reference.H.array.size)
    for solver in (reference, updated):
        solver.E.fromarray(E0)
        solver.H.fromarray(H0)
        for _ in range(20):
            solver.one_step()

    np.testing.assert_allclose(updated.E.toarray(), reference.E.toarray())
    np.testing.assert_allclose(updated.H.toarray(), reference.H.toarray())


def test_update_tensors_after_first_step_raises():
    solver = _lossy_box(1.0, 0.0, None)
    solver.one_step()
    with pytest.raises(RuntimeError, match="before the first time step"):
        solver.update_tensors()
