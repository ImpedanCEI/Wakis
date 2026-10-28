"""Wake potential on the transverse stencil (calc_long_WP_3d + calc_trans_WP).

Uses synthetic Ez(x, y, z, t) data, so no field simulation is needed.
"""

import numpy as np
import pytest

from wakis import WakeSolver

c = 299792458.0


class _FakeH5(dict):
    """Dict standing in for the h5py file written by wakesolve."""

    def close(self):
        pass


def _wake_from_field(q, profile_xy, tmp_path):
    """Run the 3d wake integration on Ez = profile_xy(x, y) * f(z, t)."""
    nz, nt = 40, 120
    z = np.linspace(0.0, 0.1, nz)
    dz = np.full(nz, z[1] - z[0])
    dt = 0.5 * (z[1] - z[0]) / c
    t = np.arange(nt) * dt
    xf = np.array([-1e-3, 0.0, 1e-3])
    yf = np.array([-2e-3, 0.0, 2e-3])
    X, Y = np.meshgrid(xf, yf, indexing="ij")
    shape = profile_xy(X, Y)[:, :, None]

    hf = _FakeH5()
    for n in range(nt):
        hf[f"#{n:05d}"] = shape * 1e3 * np.sin(2 * np.pi * n / 30) * np.ones(nz)

    wake = WakeSolver(q=q, results_folder=str(tmp_path), save=False, verbose=0)
    wake.Ez_hf = hf
    wake.t, wake.zf, wake.z, wake.dz = t, z, z, dz
    wake.xf, wake.yf = xf, yf
    wake.ti = 0.0
    wake.calc_long_WP_3d()
    wake.calc_trans_WP()
    return wake


@pytest.mark.parametrize("q", [1e-9, 1e-12])
def test_uniform_field_gives_no_transverse_wake(q, tmp_path):
    # Ez independent of (x, y): every stencil point sees the same longitudinal
    # wake, so the Panofsky-Wenzel gradient must vanish exactly
    wake = _wake_from_field(q, lambda x, y: np.ones_like(x), tmp_path)

    for i in range(3):
        for j in range(3):
            np.testing.assert_array_equal(wake.WP_3d[i, j], wake.WP)
    assert np.max(np.abs(wake.WP)) > 0
    np.testing.assert_array_equal(wake.WPx, 0.0)
    np.testing.assert_array_equal(wake.WPy, 0.0)


def test_wake_potential_scales_with_inverse_charge(tmp_path):
    # The same field divided by q must give WP proportional to 1/q
    wake_nC = _wake_from_field(1e-9, lambda x, y: np.ones_like(x), tmp_path)
    wake_pC = _wake_from_field(1e-12, lambda x, y: np.ones_like(x), tmp_path)
    np.testing.assert_allclose(wake_pC.WP, 1e3 * wake_nC.WP, rtol=1e-12)


def test_linear_field_gives_constant_gradient(tmp_path):
    # Ez = (1 + a x + b y) f(z, t): WP_3d is linear in (x, y) and the central
    # difference is exact, WPx = -a * int WP0 ds and WPy = -b * int WP0 ds
    a, b = 50.0, -20.0
    wake = _wake_from_field(1e-9, lambda x, y: 1 + a * x + b * y, tmp_path)

    ds = wake.s[2] - wake.s[1]
    int_WP0 = np.concatenate(([0.0], np.cumsum(wake.WP)[:-1])) * ds
    scale = np.max(np.abs(int_WP0))
    np.testing.assert_allclose(wake.WPx, -a * int_WP0, rtol=0, atol=1e-10 * a * scale)
    np.testing.assert_allclose(
        wake.WPy, -b * int_WP0, rtol=0, atol=1e-10 * abs(b) * scale
    )
