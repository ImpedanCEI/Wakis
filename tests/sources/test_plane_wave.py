import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c, mu_0

from wakis.sources import PlaneWave


def test_source_updates_component(source_solver):
    source = PlaneWave(zs=2, nodes=1, f=1e9, amplitude=2.0)

    source.update(source_solver, 0.0)

    np.testing.assert_allclose(source_solver.H[:, :, 2, "y"], 2.0)
    assert np.all(source_solver.E[:, :, 2, "x"] > 0.0)
    assert not np.any(source_solver.H[:, :, 1, "y"])

    source.update(source_solver, source.tmax + source_solver.dt)

    assert not np.any(source_solver.H[:, :, 2, "y"])
    assert not np.any(source_solver.E[:, :, 2, "x"])


def test_plane_wave_follows_harmonic_phase_and_positive_z_direction(source_solver):
    source = PlaneWave(zs=2, f=1e9, amplitude=2.0)
    period = 1.0 / source.f

    source.update(source_solver, 0.0)
    assert np.all(source_solver.E[:, :, 2, "x"] * source_solver.H[:, :, 2, "y"] > 0)
    source.update(source_solver, period / 4)
    np.testing.assert_allclose(source_solver.H[:, :, 2, "y"], 0.0, atol=1e-14)
    source.update(source_solver, period / 2)
    np.testing.assert_allclose(source_solver.H[:, :, 2, "y"], -2.0)


def test_plane_wave_uses_medium_wave_impedance(source_solver):
    source = PlaneWave(zs=2, f=1e9, amplitude=2.0, beta=0.8)

    source.update(source_solver, 0.0)

    electric = source_solver.E[source_solver.Nx // 2, source_solver.Ny // 2, 2, "x"]
    magnetic = source_solver.H[source_solver.Nx // 2, source_solver.Ny // 2, 2, "y"]
    np.testing.assert_allclose(electric / magnetic, mu_0 * source.vp)


def test_plane_wave_uses_vacuum_impedance_at_beta_one(source_solver):
    source = PlaneWave(zs=2, f=1e9, amplitude=2.0, beta=1.0)

    source.update(source_solver, 0.0)

    electric = source_solver.E[source_solver.Nx // 2, source_solver.Ny // 2, 2, "x"]
    magnetic = source_solver.H[source_solver.Nx // 2, source_solver.Ny // 2, 2, "y"]
    np.testing.assert_allclose(electric / magnetic, mu_0 * c)


def test_plane_wave_requires_positive_frequency():
    with pytest.raises(ValueError, match="positive frequency"):
        PlaneWave()


def test_plane_wave_plot_matches_update_sign(monkeypatch):
    source = PlaneWave(f=1e9)
    monkeypatch.setattr(plt, "show", lambda: None)

    source.plot(np.array([0.0]))

    magnetic_axis, electric_axis = plt.gcf().axes
    assert [line.get_label() for line in magnetic_axis.lines] == ["Hy"]
    assert [line.get_label() for line in electric_axis.lines] == ["Ex"]
    magnetic = magnetic_axis.lines[0].get_ydata()[0]
    electric = electric_axis.lines[0].get_ydata()[0]
    assert magnetic > 0.0
    assert electric > 0.0
    plt.close()
