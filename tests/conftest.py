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


@pytest.fixture(scope="session")
def use_gpu(request):
    return request.config.getoption("--gpu")


@pytest.fixture(scope="session")
def flag_offscreen(request):
    # If --interactive is set, offscreen should be False
    return not request.config.getoption("--interactive")
