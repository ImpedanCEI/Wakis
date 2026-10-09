import pytest

from wakis import GridFIT3D, SolverFIT3D


@pytest.fixture
def source_solver():
    grid = GridFIT3D(-0.05, 0.05, -0.05, 0.05, 0.0, 0.1, 5, 5, 8, verbose=0)
    return SolverFIT3D(grid, source_type="direct", verbose=0)


@pytest.fixture
def plot_source_simulation(request):
    """Plot normalized end-to-end source traces when requested."""
    if not request.config.getoption("--debug-plots"):
        return lambda time, traces, title: None

    import matplotlib.pyplot as plt
    import numpy as np

    def plot(time, traces, title):
        fig, ax = plt.subplots()
        for label, values in traces.items():
            values = np.asarray(values)
            scale = np.max(np.abs(values))
            normalized = values if scale == 0 else values / scale
            ax.plot(time * 1e9, normalized, label=label)
        ax.set(
            title=f"{request.node.name}: {title}",
            xlabel="Time [ns]",
            ylabel="Normalized field",
        )
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    return plot
