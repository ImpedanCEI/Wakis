"""Transverse wake potential from WP_3d through the Panofsky-Wenzel theorem."""

import numpy as np
import pytest

from wakis import WakeSolver


def _wake(xf, yf, profile_xy, tmp_path, ns=400):
    """WakeSolver with WP_3d = profile_xy(x, y) * g(s) on the 3x3 stencil."""
    wake = WakeSolver(results_folder=str(tmp_path), save=False, verbose=0)
    wake.s = np.linspace(-0.01, 0.2, ns)
    g = np.exp(-(((wake.s - 0.02) / 0.005) ** 2)) * np.cos(80 * wake.s)
    X, Y = np.meshgrid(xf, yf, indexing="ij")
    wake.WP_3d = profile_xy(X, Y)[:, :, None] * g
    wake.WP = wake.WP_3d[1, 1]
    wake.xf, wake.yf = np.asarray(xf), np.asarray(yf)
    ds = wake.s[1] - wake.s[0]
    int_g = np.concatenate(([0.0], np.cumsum(g)[:-1])) * ds
    return wake, int_g


def _reference_loop(wake):
    """Previous implementation: per-sample sums and (f+ - f-) / (h- + h+)."""
    ds = wake.s[2] - wake.s[1]
    int_WP = np.zeros_like(wake.WP_3d)
    for n in range(len(wake.s)):
        int_WP[:, :, n] = np.sum(wake.WP_3d[:, :, 0:n], axis=-1) * ds
    dxf, dyf = np.diff(wake.xf), np.diff(wake.yf)
    WPx = -(int_WP[2, 1] - int_WP[0, 1]) / (dxf[0] + dxf[1])
    WPy = -(int_WP[1, 2] - int_WP[1, 0]) / (dyf[0] + dyf[1])
    return WPx, WPy


def test_uniform_grid_matches_previous_implementation(tmp_path):
    xf = np.array([-1e-3, 0.0, 1e-3])
    yf = np.array([-2e-3, 0.0, 2e-3])
    wake, _ = _wake(xf, yf, lambda x, y: 1 + 30 * x - 7 * y + 900 * x * y, tmp_path)
    WPx_ref, WPy_ref = _reference_loop(wake)

    wake.calc_trans_WP()

    scale = np.max(np.abs(WPx_ref))
    np.testing.assert_allclose(wake.WPx, WPx_ref, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(wake.WPy, WPy_ref, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("ratio", [1.0, 1.5, 3.0])
def test_quadratic_profile_exact_on_nonuniform_grid(ratio, tmp_path):
    # W(x, y) = 1 + a x + b x^2 + c y + d y^2: dW/dx = a + 2 b x0 at the centre
    # node x0, which the 3-point non-uniform stencil reproduces exactly
    a, b, c, d = 40.0, 2e4, -15.0, 5e3
    x0, y0 = 2e-4, -1e-4
    h = 1e-3
    xf = np.array([x0 - h, x0, x0 + ratio * h])
    yf = np.array([y0 - ratio * h, y0, y0 + h])
    profile = lambda x, y: 1 + a * x + b * x**2 + c * y + d * y**2  # noqa: E731
    wake, int_g = _wake(xf, yf, profile, tmp_path)

    wake.calc_trans_WP()

    scale = np.max(np.abs(int_g))
    np.testing.assert_allclose(
        wake.WPx, -(a + 2 * b * x0) * int_g, rtol=0, atol=1e-9 * abs(a) * scale
    )
    np.testing.assert_allclose(
        wake.WPy, -(c + 2 * d * y0) * int_g, rtol=0, atol=1e-9 * abs(c) * scale
    )


def test_previous_stencil_is_first_order_on_nonuniform_grid(tmp_path):
    # The symmetric formula picks up b * (h2 - h1) on unequal spacings, which
    # is the error the non-uniform weights remove
    a, b = 40.0, 2e4
    h1, h2 = 1e-3, 2e-3
    xf = np.array([-h1, 0.0, h2])
    yf = np.array([-1e-3, 0.0, 1e-3])
    wake, int_g = _wake(xf, yf, lambda x, y: 1 + a * x + b * x**2, tmp_path)
    WPx_ref, _ = _reference_loop(wake)

    wake.calc_trans_WP()

    i = np.argmax(np.abs(int_g))
    assert WPx_ref[i] / -int_g[i] == pytest.approx(a + b * (h2 - h1), rel=1e-9)
    assert wake.WPx[i] / -int_g[i] == pytest.approx(a, rel=1e-9)
