import matplotlib.pyplot as plt
import numpy as np
import pytest

from wakis.sources import ModePacket


def test_source_updates_component(source_solver):
    source = ModePacket(zs=2, f=1e9, amplitude=2.0, sigma_t=1e-9, tinj=0.0)

    source.update(source_solver, 0.0)

    electric = source_solver.E[:, :, 2, "x"]
    assert np.allclose(electric[:, 0], 0.0)
    assert np.allclose(electric[:, -1], 0.0, atol=1e-15)
    assert electric[:, source_solver.Ny // 2].max() == pytest.approx(2.0)
    assert not np.any(source_solver.H.toarray())


def test_mode_packet_plot_shows_modal_drive(monkeypatch):
    source = ModePacket(f=1e9, amplitude=2.0, sigma_t=1e-9, tinj=0.0)
    monkeypatch.setattr(plt, "show", lambda: None)

    source.plot(np.array([0.0]))

    line = plt.gcf().axes[0].lines[0]
    assert line.get_label() == "Ex"
    assert line.get_ydata()[0] == pytest.approx(2.0)
    plt.close()
