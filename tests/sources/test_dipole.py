"""Test dipole injection, plotting, and radiated-field propagation."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wakis.sources import Dipole


def test_source_updates_component(unit_solver):
    source = Dipole(field="Hy", f=1e9, amplitude=3.0, phase=np.pi / 2)

    source.update(unit_solver, 0.0)

    center = (unit_solver.Nx // 2, unit_solver.Ny // 2, unit_solver.Nz // 2)
    assert unit_solver.H[center[0], center[1], center[2], "y"] == 3.0
    assert not np.any(unit_solver.E.toarray())
    assert not np.any(unit_solver.J.toarray())


def test_dipole_plot_shows_injected_component(monkeypatch):
    source = Dipole(field="Hy", f=1e9, amplitude=3.0, phase=np.pi / 2)
    monkeypatch.setattr(plt, "show", lambda: None)

    source.plot(np.array([0.0]))

    line = plt.gcf().axes[0].lines[0]
    assert line.get_label() == "Hy"
    assert line.get_ydata()[0] == pytest.approx(3.0)
    plt.close()


def test_dipole_radiates_an_oscillating_localized_field(
    propagation_solver, simulate_source
):
    source = Dipole(
        field="Ex",
        xs=propagation_solver.Nx // 2,
        ys=propagation_solver.Ny // 2,
        zs=8,
        f=8e9,
    )

    result = simulate_source(
        propagation_solver,
        source,
        ("E", "x"),
        ("E", "x"),
        ("H", "y"),
        steps=80,
    )

    assert result["observed"].min() < 0.0 < result["observed"].max()
    assert np.max(np.abs(result["companion"])) > 1e-5
    peak = np.argmax(np.abs(result["observed"]))
    assert abs(result["profile_x"][peak, propagation_solver.Nx // 2]) > 2 * abs(
        result["profile_x"][peak, 0]
    )
