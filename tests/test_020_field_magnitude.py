"""Field magnitude at the cell centres vs. index-wise on FIT grids (issue #116)."""

import numpy as np
import pytest

from wakis import Field, GridFIT3D

GHOST = 1e30  # written into terminal ghost storage; must never reach the result


def _make_grid(Nx=11, Ny=13, Nz=15, nonuniform=False):
    """Build a uniform or smoothly stretched non-uniform GridFIT3D."""
    xmin, xmax = -0.4, 0.8
    ymin, ymax = -0.5, 0.7
    zmin, zmax = -0.3, 0.9
    if not nonuniform:
        return GridFIT3D(
            xmin, xmax, ymin, ymax, zmin, zmax, Nx=Nx, Ny=Ny, Nz=Nz, verbose=0
        )

    xi = np.linspace(0.0, 1.0, Nx)
    eta = np.linspace(0.0, 1.0, Ny)
    zeta = np.linspace(0.0, 1.0, Nz)
    x = xmin + (xmax - xmin) * (xi + 0.15 * np.sin(np.pi * xi))
    y = ymin + (ymax - ymin) * (eta + 0.12 * np.sin(2.0 * np.pi * eta))
    z = zmin + (zmax - zmin) * (zeta + 0.18 * np.sin(np.pi * zeta))
    return GridFIT3D(x=x, y=y, z=z, verbose=0)


def _coords(grid, staggering):
    return grid.dual_coords if staggering == "H" else grid.primal_coords


def _sample(grid, staggering, fns, dtype=float):
    """Sample analytic components at their true positions, as the solver stores them.

    Storage outside each component's coordinate axes (terminal ghosts) is filled
    with GHOST so that any leak of ghost storage into the result is detected.
    """
    coords = _coords(grid, staggering)
    field = Field(grid.Nx, grid.Ny, grid.Nz, dtype=dtype, coordinates=coords)
    for comp, fn in zip("xyz", fns):
        X, Y, Z = np.meshgrid(*coords[comp], indexing="ij")
        mat = np.full((grid.Nx, grid.Ny, grid.Nz), GHOST, dtype=dtype)
        mat[: X.shape[0], : X.shape[1], : X.shape[2]] = np.broadcast_to(
            fn(X, Y, Z), X.shape
        )
        field.from_matrix(mat, comp)
    return field


def _indexwise_abs(field):
    """Index-wise norm of the stored components (the pre-#116 behaviour)."""
    return np.sqrt(sum(np.abs(field.to_matrix(c)) ** 2 for c in "xyz"))


def _exact_centre_abs(grid, fns):
    """Exact |F| at the physical cell centres, shape (Nx-1, Ny-1, Nz-1)."""
    X, Y, Z = np.meshgrid(grid.tx[1:-1], grid.ty[1:-1], grid.tz[1:-1], indexing="ij")
    return np.sqrt(
        sum(np.abs(np.broadcast_to(fn(X, Y, Z), X.shape)) ** 2 for fn in fns)
    )


LINEAR = (
    lambda x, y, z: 2.3 * x - 1.1 * y + 0.7 * z + 0.4,
    lambda x, y, z: 0.9 * x - 1.7 * y + 1.3 * z - 0.2,
    lambda x, y, z: -1.4 * x + 0.8 * y + 3.1 * z + 0.5,
)


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("staggering", ["E", "H"])
@pytest.mark.parametrize("as_matrix", [True, False])
def test_constant_field_magnitude(nonuniform, staggering, as_matrix):
    """A constant vector field has the exact magnitude at every cell centre."""
    grid = _make_grid(9, 10, 11, nonuniform=nonuniform)
    c = (3.2, -1.5, 4.7)
    field = _sample(grid, staggering, [lambda x, y, z, v=v: v for v in c])
    mag = field.get_abs(as_matrix=as_matrix)
    n_centres = (grid.Nx - 1, grid.Ny - 1, grid.Nz - 1)
    assert mag.shape == (n_centres if as_matrix else (int(np.prod(n_centres)),))
    np.testing.assert_allclose(mag, np.linalg.norm(c), rtol=1e-15, atol=0)


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("staggering", ["E", "H"])
def test_linear_field_magnitude_exact(nonuniform, staggering):
    """Linear interpolation is exact for F(r) = A r + d, also on stretched grids.

    The index-wise norm combines components from displaced points and has an
    O(h) error, so the test also fails if the interpolation is dropped.
    """
    grid = _make_grid(10, 11, 12, nonuniform=nonuniform)
    exact = _exact_centre_abs(grid, LINEAR)
    field = _sample(grid, staggering, LINEAR)

    mag = field.get_abs()
    np.testing.assert_allclose(mag, exact, rtol=1e-13, atol=0)

    indexwise = _indexwise_abs(field)[:-1, :-1, :-1]
    err_indexwise = np.max(np.abs(indexwise - exact))
    print(f"{staggering} nonuniform={nonuniform}: index-wise error {err_indexwise:.3e}")
    assert err_indexwise > 1e-2


@pytest.mark.parametrize("staggering", ["E", "H"])
def test_smooth_field_convergence_nonuniform(staggering):
    """Smooth field on refined stretched grids: collocated error is 2nd order."""
    fns = (
        lambda x, y, z: np.sin(1.7 * x + 0.9 * y - 0.6 * z + 0.3),
        lambda x, y, z: np.cos(0.8 * x - 1.5 * y + 1.1 * z - 0.2),
        lambda x, y, z: np.sin(1.2 * x + 0.7 * y + 1.6 * z + 0.4),
    )
    ncells = [8, 16, 32]
    errs_col, errs_idx = [], []
    for nc in ncells:
        grid = _make_grid(nc + 1, nc + 1, nc + 1, nonuniform=True)
        exact = _exact_centre_abs(grid, fns)
        field = _sample(grid, staggering, fns)
        errs_col.append(np.max(np.abs(field.get_abs() - exact)))
        idx = _indexwise_abs(field)[:-1, :-1, :-1]
        errs_idx.append(np.max(np.abs(idx - exact)))

    h = 1.0 / np.array(ncells)
    order_col = np.polyfit(np.log(h), np.log(errs_col), 1)[0]
    order_idx = np.polyfit(np.log(h), np.log(errs_idx), 1)[0]
    print(f"{staggering}: collocated order {order_col:.3f}, index-wise {order_idx:.3f}")
    assert 1.8 < order_col < 2.2
    assert 0.8 < order_idx < 1.2


def test_complex_field_magnitude():
    """Complex (phasor) fields use |F|^2, at the cell centres and index-wise."""
    grid = _make_grid(7, 8, 9, nonuniform=True)
    c = (1.0 + 2.0j, -0.5j, 3.0 - 1.0j)
    field = _sample(grid, "E", [lambda x, y, z, v=v: v for v in c], dtype=complex)
    expected = np.sqrt(sum(abs(v) ** 2 for v in c))
    np.testing.assert_allclose(field.get_abs(), expected, rtol=1e-15)
    plain = Field(grid.Nx, grid.Ny, grid.Nz, dtype=complex)
    plain.array = field.array
    np.testing.assert_allclose(plain.get_abs()[:-1, :-1, :-1], expected, rtol=1e-15)


def test_index_wise_without_coordinates():
    """Fields without coordinates keep the index-wise norm on the full storage."""
    grid = _make_grid(6, 7, 8)
    field = Field(grid.Nx, grid.Ny, grid.Nz)
    field.array = _sample(grid, "E", LINEAR).array
    np.testing.assert_array_equal(field.get_abs(), _indexwise_abs(field))
    assert field.get_abs(as_matrix=False).shape == (field.N,)


def test_getitem_abs_indexes_storage():
    """field[i, j, k, 'abs'] indexes the storage layout, as before."""
    grid = _make_grid(6, 7, 8)
    field = _sample(grid, "E", LINEAR)
    np.testing.assert_array_equal(field[:, :, 3, "abs"], _indexwise_abs(field)[:, :, 3])


def test_solver_fields_carry_coordinates():
    """The solver attaches primal coordinates to E and J and dual ones to H."""
    from wakis import SolverFIT3D

    grid = _make_grid(6, 7, 8, nonuniform=True)
    solver = SolverFIT3D(grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0)
    assert solver.E.coordinates is grid.primal_coords
    assert solver.J.coordinates is grid.primal_coords
    assert solver.H.coordinates is grid.dual_coords
    assert solver.E.get_abs().shape == (5, 6, 7)
    assert solver.H.get_abs().shape == (5, 6, 7)


def test_nonstandard_layout_raises():
    """A coordinate layout other than nodes or cell centres fails loudly."""
    grid = _make_grid(6, 7, 8)
    coords = dict(grid.primal_coords)
    x, y, z = coords["y"]
    coords["y"] = (x[:-2], y, z)  # two samples short along x
    field = Field(grid.Nx, grid.Ny, grid.Nz, coordinates=coords)
    with pytest.raises(ValueError, match="samples along axis 0"):
        field.get_abs()


@pytest.mark.parametrize("staggering", ["E", "H"])
def test_abs_plot_uses_cell_centres(staggering):
    """The magnitude is plotted on the cell-centre axes, which match its shape."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = _make_grid(6, 7, 8, nonuniform=True)
    field = _sample(grid, staggering, LINEAR)
    centres = field._cell_centre_coords()
    for axis, t in zip(centres, (grid.tx, grid.ty, grid.tz)):
        np.testing.assert_array_equal(axis, t[1:-1])
    assert field.get_abs().shape == tuple(len(a) for a in centres)
    field.inspect3D(field="abs", backend="matplotlib", off_screen=True)
    plt.close("all")


@pytest.mark.parametrize("staggering", ["E", "H"])
def test_gpu_matches_cpu(use_gpu, staggering):
    """On the GPU the averaging stays on the device and matches the CPU result."""
    if not use_gpu:
        pytest.skip("needs --gpu")
    grid = _make_grid(9, 10, 11, nonuniform=True)
    cpu = _sample(grid, staggering, LINEAR, dtype=complex)
    gpu = Field(
        grid.Nx,
        grid.Ny,
        grid.Nz,
        dtype=complex,
        use_gpu=True,
        coordinates=_coords(grid, staggering),
    )
    for c in "xyz":
        gpu.from_matrix(cpu.to_matrix(c), c)
    for as_matrix in (True, False):
        mag = gpu.get_abs(as_matrix=as_matrix)
        assert isinstance(mag, np.ndarray)  # returned on the host
        np.testing.assert_allclose(
            mag, cpu.get_abs(as_matrix=as_matrix), rtol=1e-14, atol=0
        )
