import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--gpu", action="store_true", help="Enable GPU support for SolverFIT3D"
    )
    parser.addoption(
        "--interactive",
        action="store_true",
        help="Enable interactive (onscreen) PyVista rendering",
    )
    parser.addoption(
        "--debug-plots",
        action="store_true",
        help="Show simulation versus static reference comparison plots",
    )


@pytest.fixture(scope="session")
def use_gpu(request):
    return request.config.getoption("--gpu")


@pytest.fixture(scope="session")
def flag_offscreen(request):
    # If --interactive is set, offscreen should be False
    return not request.config.getoption("--interactive")


@pytest.fixture
def plot_comparison(request):
    """Show the values used by a regression assertion when requested."""
    if not request.config.getoption("--debug-plots"):
        return lambda actual, expected, title: None

    import matplotlib.pyplot as plt
    import numpy as np

    def plot(actual, expected, title):
        actual = np.atleast_1d(actual)
        expected = np.atleast_1d(expected)
        fig, ax = plt.subplots()
        ax.plot(np.arange(actual.size), actual, label="Simulation")
        ax.plot(
            np.arange(expected.size),
            expected,
            "o",
            label="Static reference",
            markersize=3,
        )
        ax.set(title=f"{request.node.name}: {title}", xlabel="Sample index")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    return plot
