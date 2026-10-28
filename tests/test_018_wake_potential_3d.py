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


@pytest.mark.parametrize("compute_plane", ["longitudinal", "both", "counter"])
@pytest.mark.parametrize("skip_cells", [0, 1])
@pytest.mark.parametrize("use_domain_axis", [False, True])
def test_edge_wake_integration(tmp_path, compute_plane, skip_cells, use_domain_axis):
    """Integrate a traveling ramp at edge midpoints with nonuniform dz.

    The analytic wake is proportional to s and the integral of 1 + 2*z.
    Left-sample time interpolation has an error bounded by one timestep.
    """
    import h5py

    nodes = np.array([-0.1, -0.08, -0.04, 0.02, 0.1])
    dz = np.diff(nodes)
    z = (nodes[:-1] + nodes[1:]) / 2
    dt = 0.0005 / c
    ti = 0.01 / c
    t = np.arange(700) * dt
    xy = np.array([-0.002, 0.0, 0.003])
    shape = 1 + 100 * xy[:, None] + 200 * xy[None, :]
    counter = compute_plane == "counter"
    distance = nodes[-1] - z if counter else z - nodes[0]
    path = tmp_path / "Ez.h5"
    with h5py.File(path, "w") as hf:
        hf["x"], hf["y"], hf["z"] = xy, xy, z
        hf["dz"], hf["t"] = dz, t
        for n, time in enumerate(t):
            # A ramp in distance behind the moving bunch center.
            hf[f"#{n:05d}"] = (
                shape[:, :, None] * (1 + 2 * z) * (c * (time - ti) - distance)
            )

    wake = WakeSolver(
        q=1e-12,
        ti=ti,
        wakelength=0.025,
        skip_cells=skip_cells,
        counter_moving=counter,
        Ez_file=str(path),
        results_folder=str(tmp_path),
        save=False,
        verbose=0,
    )
    if use_domain_axis:
        wake.z = nodes
    if compute_plane == "longitudinal":
        wake.calc_long_WP()
    else:
        wake.calc_long_WP_3d()

    left, right = nodes[skip_cells], nodes[-skip_cells - 1]
    integral = (right - left) + (right**2 - left**2)
    # The saved central x/y node has shape=1. Allow only time-quantization error.
    expected = (-1 if counter else 1) * integral * wake.s
    np.testing.assert_allclose(
        wake.WP, expected, rtol=0, atol=integral * c * dt * 1.001
    )
    if compute_plane != "longitudinal":
        np.testing.assert_allclose(
            wake.WP_3d,
            shape[:, :, None] * expected,
            rtol=0,
            atol=shape.max() * integral * c * dt * 1.001,
        )


@pytest.mark.parametrize("compute_plane", ["longitudinal", "both"])
def test_wakesolve_edge_output(tmp_path, monkeypatch, compute_plane):
    """Save Ez on physical edges and retain the domain origin for integration."""
    import h5py

    from wakis import GridFIT3D, SolverFIT3D

    monkeypatch.chdir(tmp_path)
    x = np.array([-0.04, -0.02, 0.0, 0.01, 0.03])
    y = np.array([-0.03, -0.01, 0.02, 0.04, 0.06])
    nodes = np.array([-0.08, -0.04, 0.01, 0.07, 0.14])
    z = (nodes[:-1] + nodes[1:]) / 2
    path = tmp_path / "Ez.h5"
    wake = WakeSolver(
        q=1e-12,
        sigmaz=0.02,
        ti=0.01 / c,
        xsource=0.009,
        ysource=0.019,
        xtest=0.009,
        ytest=0.019,
        Ez_file=str(path),
        results_folder=str(tmp_path),
        save=False,
        verbose=0,
    )
    grid = GridFIT3D(x=x, y=y, z=nodes, verbose=0)
    solver = SolverFIT3D(grid, wake, source_type="direct", verbose=0)
    field = 1 + x[:, None, None] + 2 * y[None, :, None] + 3 * z[None, None, :]
    solver.E[:, :, :-1, "z"] = field
    solver.wakesolve(wakelength=0.03, compute_plane=compute_plane, save_J=True)

    xx = 3 if compute_plane == "longitudinal" else slice(2, 5)
    yy = 2 if compute_plane == "longitudinal" else slice(1, 4)
    with h5py.File(path, "r") as hf:
        np.testing.assert_array_equal(hf["x"][()], x[xx])
        np.testing.assert_array_equal(hf["y"][()], y[yy])
        np.testing.assert_array_equal(hf["z"][()], z)
        np.testing.assert_array_equal(hf["dz"][()], np.diff(nodes))
        np.testing.assert_allclose(hf["#00000"][()], field[xx, yy])
    np.testing.assert_array_equal(wake.zf, z)
    assert np.isfinite(wake.WP).all()
    with h5py.File(tmp_path / "Jz.h5", "r") as hf:
        np.testing.assert_allclose(hf["z"][()], z)
        np.testing.assert_allclose(hf["t"][0], solver.dt / 2)
        bunch_center = nodes[0] + wake.v * (solver.dt / 2 - wake.ti)
        area = (x[4] - x[2]) / 2 * (y[3] - y[1]) / 2
        current = wake.q * wake.v / area / (np.sqrt(2 * np.pi) * wake.sigmaz)
        current *= np.exp(-0.5 * ((z - bunch_center) / wake.sigmaz) ** 2)
        actual = hf["#00000"][()]
        if compute_plane == "both":
            actual = actual[1, 1]
        np.testing.assert_allclose(actual, current)
