import matplotlib.pyplot as plt
import numpy as np
import pytest

from wakis.sources import Dipole


def test_source_updates_component(source_solver):
    source = Dipole(field="Hy", f=1e9, amplitude=3.0, phase=np.pi / 2)

    source.update(source_solver, 0.0)

    center = (source_solver.Nx // 2, source_solver.Ny // 2, source_solver.Nz // 2)
    assert source_solver.H[center[0], center[1], center[2], "y"] == 3.0
    assert not np.any(source_solver.E.toarray())
    assert not np.any(source_solver.J.toarray())


def test_dipole_plot_shows_injected_component(monkeypatch):
    source = Dipole(field="Hy", f=1e9, amplitude=3.0, phase=np.pi / 2)
    monkeypatch.setattr(plt, "show", lambda: None)

    source.plot(np.array([0.0]))

    line = plt.gcf().axes[0].lines[0]
    assert line.get_label() == "Hy"
    assert line.get_ydata()[0] == pytest.approx(3.0)
    plt.close()
