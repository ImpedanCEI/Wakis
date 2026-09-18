"""Small serial wake example with a PyVista cylindrical shell."""

import numpy as np
import pytest
import pyvista as pv

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


def test_loss_factor_calculates_missing_wake_and_profile(tmp_path):
    wake = WakeSolver(results_folder=str(tmp_path), save=False, verbose=0)

    def calculate_wake():
        wake.s = np.linspace(-0.01, 0.01, 101)
        wake.WP = np.full_like(wake.s, 2.0)

    wake.calc_long_WP = calculate_wake
    assert wake.calc_loss_factor() == pytest.approx(2.0)
    assert wake.lambdas is not None


@pytest.mark.slow
def test_016_wake_factors(tmp_path):
    height = 0.093
    cavity = pv.Cylinder(
        center=(0, 0, height / 2),
        direction=(0, 0, 1),
        radius=0.0085,
        height=height,
        resolution=32,
    ).triangulate()
    shell = (
        pv.Disc(inner=0.0085, outer=0.01, r_res=1, c_res=32)
        .extrude((0, 0, height), capping=True)
        .triangulate()
    )
    solids = {"cavity": tmp_path / "cavity.stl", "shell": tmp_path / "shell.stl"}
    cavity.save(solids["cavity"])
    shell.save(solids["shell"])
    bounds = shell.bounds
    grid = GridFIT3D(
        *bounds,
        12,
        10,
        30,
        stl_solids={name: str(path) for name, path in solids.items()},
        stl_materials={"cavity": "vacuum", "shell": [1.0, 1.0, 30.0]},
        stl_method="implicit_distance",
        verbose=0,
    )
    wake = WakeSolver(
        q=1e-9,
        sigmaz=4.5e-3,
        ysource=1e-3,
        skip_cells=4,
        results_folder=str(tmp_path),
        save=True,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        wake,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        use_stl=True,
        use_gpu=False,
        dtype=np.float64,
        verbose=0,
    )
    solver.wakesolve(wakelength=0.03, plot=False)

    loss = wake.calc_loss_factor()
    kick_x, kick_y = wake.calc_kick_factors()
    weighted = np.trapz(wake.lambdas, wake.s)

    assert np.isfinite(loss)
    assert loss == pytest.approx(np.trapz(wake.WP * wake.lambdas, wake.s) / weighted)
    assert kick_x is None  # The source is centered in x.
    assert np.isfinite(kick_y)
    assert kick_y == pytest.approx(
        np.trapz(wake.WPy * wake.lambdas, wake.s) / weighted / wake.ysource
    )
    assert (tmp_path / "loss_factor.txt").is_file()
    assert (tmp_path / "kick_factors.txt").is_file()
