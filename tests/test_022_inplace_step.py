"""Public-API verification of single-step source response and cavity oscillation."""

import numpy as np
import pytest
from scipy.constants import c

from wakis import GridFIT3D, SolverFIT3D


@pytest.mark.parametrize("bc", ["pec", "periodic"])
def test_first_step_current_source(bc, use_gpu):
    """Advance one step from zero fields with a constant current density J [A/m^2].

    Because E = 0 initially, curl E = 0 and H [A/m] remains zero after the magnetic
    half-step. The electric update then reduces to
    E = 0 + dt * (curl H - ieps * J) = -dt * ieps * J [V/m].
    """
    grid = GridFIT3D(0.0, 0.09, 0.0, 0.08, 0.0, 0.11, 6, 7, 8, verbose=0)
    solver = SolverFIT3D(
        grid, bc_low=[bc] * 3, bc_high=[bc] * 3, use_gpu=use_gpu, verbose=0
    )

    currents = {"x": 1.5, "y": -2.25, "z": 3.75}
    for comp, value in currents.items():
        solver.J[:, :, :, comp] = value

    solver.one_step()

    for axis, comp in enumerate("xyz"):
        h = solver.H.to_matrix(comp)
        e = solver.E.to_matrix(comp)
        ieps = solver.ieps.to_matrix(comp)
        j = solver.J.to_matrix(comp)
        if use_gpu:
            h, e, ieps, j = h.get(), e.get(), ieps.get(), j.get()

        np.testing.assert_array_equal(h, 0.0)

        expected = -solver.dt * ieps * j
        nonzero = ieps != 0.0
        if bc == "periodic":
            physical = tuple(slice(0, len(ax)) for ax in grid.primal_coords[comp])
            assert np.all(nonzero[physical])
            # Periodic wrap zeroes the inactive zero-length terminal ghost plane.
            ghost = [slice(None)] * 3
            ghost[axis] = -1
            nonzero[tuple(ghost)] = False

        assert np.any(nonzero)
        np.testing.assert_allclose(e[nonzero], expected[nonzero], rtol=1e-12, atol=0)
        np.testing.assert_array_equal(e[~nonzero], 0.0)


def test_pec_cavity_mode_frequency(use_gpu):
    """Seed the TM110 mode in a closed PEC box and check its oscillation frequency [Hz].

    The discrete Yee dispersion for TM110 (kx = pi / a, ky = pi / b, kz = 0) satisfies
    sin(omega * dt / 2)^2 / (c * dt)^2 = sin(kx * dx / 2)^2 / dx^2 + sin(ky * dy / 2)^2 / dy^2.
    """
    a, b, d = 0.12, 0.10, 0.04
    cells_x, cells_y, cells_z = 12, 10, 4
    nx, ny, nz = cells_x + 1, cells_y + 1, cells_z + 1

    grid = GridFIT3D(0.0, a, 0.0, b, 0.0, d, nx, ny, nz, verbose=0)
    solver = SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, use_gpu=use_gpu, verbose=0
    )

    x_ez, y_ez, z_ez = grid.primal_coords["z"]
    ez_mode = np.sin(np.pi * x_ez[:, None] / a) * np.sin(np.pi * y_ez[None, :] / b)
    solver.E[:, :, : len(z_ez), "z"] = ez_mode[:, :, None]

    dx = a / cells_x
    dy = b / cells_y
    kx = np.pi / a
    ky = np.pi / b
    rhs = (np.sin(kx * dx / 2.0) / dx) ** 2 + (np.sin(ky * dy / 2.0) / dy) ** 2
    omega_exact = 2.0 / solver.dt * np.arcsin(c * solver.dt * np.sqrt(rhs))
    f_exact = omega_exact / (2.0 * np.pi)
    period_exact = 1.0 / f_exact

    target_periods = 6
    n_steps = int(np.ceil((target_periods + 0.5) * period_exact / solver.dt))
    ix, iy = nx // 2, ny // 2

    centre_column = np.empty((n_steps + 1, len(z_ez)))
    centre_column[0] = solver.E[ix, iy, : len(z_ez), "z"]
    for step in range(1, n_steps + 1):
        solver.one_step()
        centre_column[step] = solver.E[ix, iy, : len(z_ez), "z"]

    # TM110 is z-invariant across all physical z layers of the centre column.
    assert np.ptp(centre_column, axis=1).max() < 1e-12
    signal = centre_column[:, 0]

    crossings = np.flatnonzero(signal[:-1] * signal[1:] <= 0.0)
    t_cross = (
        crossings + signal[crossings] / (signal[crossings] - signal[crossings + 1])
    ) * solver.dt
    m_periods = 0.5 * (len(t_cross) - 1)
    assert m_periods >= 4.0

    # Tolerance derivation: each crossing time is resolved to within O(dt), so the
    # period estimate across M periods of length T has relative error <~ dt / (M * T);
    # accounting for both end crossings gives rtol = 2 * dt / (M * T).
    rtol = 2.0 * solver.dt / (m_periods * period_exact)

    half_periods = np.diff(t_cross)
    np.testing.assert_allclose(half_periods, 0.5 * period_exact, rtol=rtol, atol=0)

    f_est = m_periods / (t_cross[-1] - t_cross[0])
    np.testing.assert_allclose(f_est, f_exact, rtol=rtol, atol=0)
