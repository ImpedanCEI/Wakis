"""Tests for second-order linear time interpolation of Ez in WakeSolver."""

import h5py
import numpy as np
import pytest
from scipy.constants import c, epsilon_0

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


def _box_modes(a, b, d, x0, y0, sigma, n_sigma=9.0):
    """Compute modal angular frequencies and loss factors of a closed PEC box.

    For a cavity ``[0, a] x [0, b] x [0, d]`` crossed at ``(x0, y0)`` by an
    ultrarelativistic bunch (``v = c``), only modes with longitudinal electric
    field ``E_z = sin(kx x) sin(ky y) cos(kz z)`` couple to the beam.

    Parameters
    ----------
    a, b, d : float
        Cavity dimensions along x, y, z [m].
    x0, y0 : float
        Transverse beam position in cavity coordinates ``[0, a] x [0, b]`` [m].
    sigma : float
        Bunch rms longitudinal length [m].
    n_sigma : float, optional
        Mode wavenumber cutoff ``k * sigma <= n_sigma``. Default is 9.0.

    Returns
    -------
    omega : ndarray
        Modal angular frequencies [rad/s].
    kn : ndarray
        Modal loss factors [V/C].
    """
    kmax = n_sigma / sigma
    m = np.arange(1, int(kmax * a / np.pi) + 1)
    n = np.arange(1, int(kmax * b / np.pi) + 1)
    p = np.arange(0, int(kmax * d / np.pi) + 1)
    sx = np.sin(m * np.pi * x0 / a)
    sy = np.sin(n * np.pi * y0 / b)
    m, sx = m[np.abs(sx) > 1e-12], sx[np.abs(sx) > 1e-12]
    n, sy = n[np.abs(sy) > 1e-12], sy[np.abs(sy) > 1e-12]
    m_grid, n_grid, p_grid = np.meshgrid(m, n, p, indexing="ij")
    sx_grid, sy_grid = np.meshgrid(sx, sy, indexing="ij")
    sx_grid, sy_grid = sx_grid[..., None], sy_grid[..., None]
    kx = m_grid * np.pi / a
    ky = n_grid * np.pi / b
    kz = p_grid * np.pi / d
    kt2 = kx**2 + ky**2
    k = np.sqrt(kt2 + kz**2)
    keep = k <= kmax
    k = k[keep]
    kz = kz[keep]
    kt2 = kt2[keep]
    sx_grid = np.broadcast_to(sx_grid, keep.shape)[keep]
    sy_grid = np.broadcast_to(sy_grid, keep.shape)[keep]
    p_grid = p_grid[keep]

    integral = 0.5 * (
        (np.exp(1j * (k + kz) * d) - 1.0) / (1j * (k + kz))
        + (np.exp(1j * (k - kz) * d) - 1.0) / (1j * (k - kz))
    )
    v2 = (sx_grid * sy_grid) ** 2 * np.abs(integral) ** 2
    energy = np.where(p_grid == 0, a * b * d / 4.0, a * b * d / 8.0 * k**2 / kt2)
    kn = v2 / (2.0 * epsilon_0 * energy)
    return k * c, kn


def _analytic_loss_factor(omega, kn, sigma):
    """Compute the Gaussian-bunch analytic loss factor from modal parameters.

    Parameters
    ----------
    omega : ndarray
        Modal angular frequencies [rad/s].
    kn : ndarray
        Modal loss factors [V/C].
    sigma : float
        Bunch rms longitudinal length [m].

    Returns
    -------
    float
        Total loss factor ``sum k_n exp(-(omega * sigma / c)^2)`` [V/C].
    """
    return float(np.sum(kn * np.exp(-((omega * sigma / c) ** 2))))


def _run_synthetic_wake(tmp_path, dt, mode):
    """Write a synthetic Ez.h5 with an analytic traveling pulse and compute WP.

    The field is ``Ez(x, y, z, t) = P(x, y) * g(t - ti - distance(z) / v)``,
    where ``g(tau) = e0 * exp(-tau^2 / (2 * sigma_t^2)) * cos(omega * tau)``
    and ``distance(z)`` is ``z - zmin`` (co-moving) or ``zmax - z``
    (counter-moving). Along the test-charge trajectory
    ``t_k(s) = ti + (distance(z_k) + s) / v``, the argument is ``tau = s / v``
    at every midpoint ``z_k``, so the exact spatial integral over ``[zmin, zmax]``
    is ``+/- P(x, y) * L * g(s / v) / (q * 1e12)`` with zero spatial
    quadrature error.
    """
    v = c
    q = 1e-12
    length = 0.08
    nz = 32
    nodes = np.linspace(0.0, length, nz + 1)
    z = 0.5 * (nodes[:-1] + nodes[1:])
    dz = np.diff(nodes)
    xf = np.array([-1e-3, 0.0, 1e-3])
    yf = np.array([-2e-3, 0.0, 2e-3])
    dx = np.diff(xf)
    dy = np.diff(yf)
    x_grid, y_grid = np.meshgrid(xf, yf, indexing="ij")
    profile_xy = 1.0 + 150.0 * x_grid - 80.0 * y_grid

    sigma_z = 0.012
    sigma_t = sigma_z / v
    omega = 2.0 * np.pi * v / 0.024
    e0 = 100.0
    ti = 0.03 / v
    wakelength = 0.03

    counter = mode == "3d_counter"
    distance = (nodes[-1] - z) if counter else (z - nodes[0])

    tmax = ti + (length + wakelength) / v + 4.0 * dt
    nt = int(np.ceil(tmax / dt)) + 1
    t = np.arange(nt, dtype=np.float64) * dt

    ez_path = tmp_path / f"Ez_{mode}_{nt}.h5"
    with h5py.File(ez_path, "w") as hf:
        hf["x"], hf["y"], hf["z"] = xf, yf, z
        hf["dx"], hf["dy"], hf["dz"] = dx, dy, dz
        hf["t"] = t
        for n, tn in enumerate(t):
            tau = tn - ti - distance / v
            g_val = e0 * np.exp(-0.5 * (tau / sigma_t) ** 2) * np.cos(omega * tau)
            hf[f"#{n:05d}"] = profile_xy[:, :, None] * g_val[None, None, :]

    wake = WakeSolver(
        wakelength=wakelength,
        q=q,
        sigmaz=sigma_z,
        beta=1.0,
        ti=ti,
        counter_moving=counter,
        Ez_file=str(ez_path),
        results_folder=str(tmp_path),
        save=False,
        verbose=0,
    )
    if mode == "longitudinal":
        wake.calc_long_WP()
    else:
        wake.calc_long_WP_3d()

    tau_s = wake.s / v
    g_exact = e0 * np.exp(-0.5 * (tau_s / sigma_t) ** 2) * np.cos(omega * tau_s)
    sign = -1.0 if counter else 1.0
    wp_exact = sign * length * g_exact / (q * 1e12)

    # For g(tau) = e0 * exp(-tau^2 / (2 * sigma_t^2)) * cos(omega * tau),
    # |g''(tau)| attains its maximum at tau = 0: |g''(0)| = e0 * (1/sigma_t^2 + omega^2).
    # Linear interpolation on [t_m, t_{m+1}] has pointwise error
    # 0.5 * w * (1 - w) * dt^2 * |g''(xi)| <= (dt^2 / 8) * max|d2Ez/dt2|.
    # Integrating over length L and dividing by (q * 1e12) gives the bound:
    max_d2_ez_dt2 = e0 * (1.0 / sigma_t**2 + omega**2)
    bound = (dt**2 / 8.0) * max_d2_ez_dt2 * length / (q * 1e12)
    return wake, wp_exact, profile_xy, bound


@pytest.mark.parametrize("mode", ["longitudinal", "3d_co", "3d_counter"])
def test_wake_time_sampling_second_order(tmp_path, mode):
    """Verify second-order accuracy and error bound of Ez time interpolation."""
    dz = 0.08 / 32
    # Non-integer c*dt/dz = 0.55 and 0.275 ensure non-trivial fractional weights
    # w = ts/dt - floor(ts/dt) across the longitudinal cells.
    dt_coarse = 0.55 * dz / c
    dt_fine = 0.5 * dt_coarse

    wake_coarse, wp_exact_coarse, _, _ = _run_synthetic_wake(tmp_path, dt_coarse, mode)
    wake_fine, wp_exact_fine, profile_xy, bound_fine = _run_synthetic_wake(
        tmp_path, dt_fine, mode
    )

    err_coarse = float(np.max(np.abs(wake_coarse.WP - wp_exact_coarse)))
    err_fine = float(np.max(np.abs(wake_fine.WP - wp_exact_fine)))

    assert err_fine > 0.0
    assert err_coarse / err_fine >= 3.0
    assert err_fine < bound_fine

    if mode != "longitudinal":
        err_3d_fine = float(
            np.max(
                np.abs(
                    wake_fine.WP_3d
                    - profile_xy[:, :, None] * wp_exact_fine[None, None, :]
                )
            )
        )
        assert err_3d_fine < np.max(np.abs(profile_xy)) * bound_fine


@pytest.mark.parametrize("mode", ["longitudinal", "3d_co", "3d_counter"])
def test_wake_time_sampling_boundaries(tmp_path, mode):
    """Verify out-of-range arrival times do not contribute and it == nt - 1 is kept."""
    v = c
    q = 1e-12
    z = np.array([0.01])
    dz = np.array([0.02])
    dt = 0.01 / v
    ti = 0.02 / v
    # Three saved steps starting at t[0] = 1.5 * dt: t = [1.5, 2.5, 3.5] * dt (nt = 3).
    # With z = 0.01 m (1.0 * v * dt from zmin = 0 and zmax = 0.02 m) and
    # s = np.arange(-0.02, 0.03, 0.01) = [-0.02, -0.01, 0.0, 0.01, 0.02]:
    # ts / dt = (z - zmin + s + ti * v) / (v * dt) - t[0] / dt
    #         = [-0.5, 0.5, 1.5, 2.5, 3.5].
    # - x = -0.5: floor(x) = -1 < 0 -> outside, 0.0 (int(-0.5) == 0 would sample step 0)
    # - x =  0.5: floor(x) =  0     -> 0.5 * 10 + 0.5 * 20 = 15.0
    # - x =  1.5: floor(x) =  1     -> 0.5 * 20 + 0.5 * 40 = 30.0
    # - x =  2.5: floor(x) =  2 == nt - 1 -> last stored step, 40.0
    # - x =  3.5: floor(x) =  3 >= nt     -> outside, 0.0
    t = (1.5 + np.arange(3, dtype=np.float64)) * dt
    xf = np.array([-1e-3, 0.0, 1e-3])
    yf = np.array([-1e-3, 0.0, 1e-3])
    vals = np.array([10.0, 20.0, 40.0])

    ez_path = tmp_path / f"Ez_bounds_{mode}.h5"
    with h5py.File(ez_path, "w") as hf:
        hf["x"], hf["y"], hf["z"] = xf, yf, z
        hf["dx"], hf["dy"], hf["dz"] = np.diff(xf), np.diff(yf), dz
        hf["t"] = t
        for n, val in enumerate(vals):
            hf[f"#{n:05d}"] = np.full((3, 3, 1), val, dtype=np.float64)

    wake = WakeSolver(
        wakelength=0.03,
        q=q,
        sigmaz=0.01,
        beta=1.0,
        ti=ti,
        counter_moving=(mode == "3d_counter"),
        Ez_file=str(ez_path),
        results_folder=str(tmp_path),
        save=False,
        verbose=0,
    )
    if mode == "longitudinal":
        wake.calc_long_WP()
    else:
        wake.calc_long_WP_3d()

    sign = -1.0 if mode == "3d_counter" else 1.0
    expected = sign * dz[0] * np.array([0.0, 15.0, 30.0, 40.0, 0.0]) / (q * 1e12)
    np.testing.assert_allclose(wake.WP, expected, rtol=0, atol=1e-12)


def _closed_box_loss_factor(tmp_path, dz):
    """Run a closed PEC box wake simulation at longitudinal step dz [m]."""
    a = 0.032
    b = 0.032
    d = 0.096
    sigmaz = 3.2e-3
    dx = 2.0 * dz
    dy = 2.0 * dz
    wakelength = 0.3

    nx_cells = round(a / dx)
    ny_cells = round(b / dy)
    nz_cells = round(d / dz)

    run_dir = tmp_path / f"box_{nz_cells}"
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = GridFIT3D(
        -a / 2.0,
        a / 2.0,
        -b / 2.0,
        b / 2.0,
        0.0,
        d,
        nx_cells + 1,
        ny_cells + 1,
        nz_cells + 1,
        verbose=0,
    )
    wake = WakeSolver(
        wakelength=wakelength,
        q=1e-9,
        sigmaz=sigmaz,
        beta=1.0,
        xsource=0.0,
        ysource=0.0,
        xtest=0.0,
        ytest=0.0,
        skip_cells=0,
        compute_plane="longitudinal",
        results_folder=str(run_dir),
        Ez_file=str(run_dir / "Ez.h5"),
        save=False,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        wake,
        cfln=0.5,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        bg=[1.0, 1.0, 0.0],
        dtype=np.float64,
        use_gpu=False,
        verbose=0,
    )
    solver.wakesolve(wakelength=wakelength, compute_plane="longitudinal")
    return -float(wake.calc_loss_factor(save=False))


def test_closed_pec_box_loss_factor(tmp_path):
    """Compare WakeSolver loss factor on a closed PEC box with the analytic modal sum."""
    a = 0.032
    b = 0.032
    d = 0.096
    sigmaz = 3.2e-3
    omega, kn = _box_modes(a, b, d, a / 2.0, b / 2.0, sigmaz)
    k_analytic = _analytic_loss_factor(omega, kn, sigmaz) * 1e-12

    dz_coarse = 1.6e-3
    dz_fine = 0.8e-3
    k_coarse = _closed_box_loss_factor(tmp_path, dz_coarse)
    k_fine = _closed_box_loss_factor(tmp_path, dz_fine)

    err_coarse = abs(k_coarse - k_analytic) / k_analytic
    err_fine = abs(k_fine - k_analytic) / k_analytic

    # Halving the mesh spacing in all directions reduces the error at second order;
    # at the characteristic bunch wavenumber kc = 1 / sigmaz, the leading second-order
    # grid-dispersion relative error scale is ((kc*dx)^2 + (kc*dy)^2 + (kc*dz)^2) / 12.
    dx_fine = 2.0 * dz_fine
    dy_fine = 2.0 * dz_fine
    dispersion_tol = (
        (dx_fine / sigmaz) ** 2 + (dy_fine / sigmaz) ** 2 + (dz_fine / sigmaz) ** 2
    ) / 12.0
    assert err_coarse / err_fine >= 2.5
    assert err_fine < dispersion_tol
