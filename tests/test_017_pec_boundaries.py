import numpy as np
import pytest

from wakis.gridFIT3D import GridFIT3D
from wakis.solverFIT3D import SolverFIT3D


class TestBoundaryConditions:
    @pytest.mark.parametrize(
        "axis, tangential_components",
        [
            (0, ("y", "z")),  # x boundary
            (1, ("x", "z")),  # y boundary
            (2, ("x", "y")),  # z boundary
        ],
    )
    def test_pec_low_and_high_boundary_mask(
        self,
        axis,
        tangential_components,
    ):
        """
        Regression test for PEC boundary masking.

        WAKIS stores only Nx*Ny*Nz DOFs per field component.
        Therefore, tangential E-field DOFs exist explicitly on the
        low-side boundaries, but not on the high-side boundaries.

        Low-side PEC boundaries must mask the corresponding stored
        tangential E-field DOFs.

        High-side PEC boundaries must not mask the last stored
        interior field layer.
        """

        grid = GridFIT3D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            Nx=3,
            Ny=4,
            Nz=5,
            verbose=0,
        )

        # Use periodic boundaries in the other two directions so that
        # only the PEC boundary in the tested direction affects BC.
        bc_low = ["periodic", "periodic", "periodic"]
        bc_high = ["periodic", "periodic", "periodic"]

        bc_low[axis] = "pec"
        bc_high[axis] = "pec"

        solver = SolverFIT3D(
            grid=grid,
            bc_low=bc_low,
            bc_high=bc_high,
            verbose=0,
        )

        for component in tangential_components:
            if axis == 0:
                # Exclude edges shared with the y/z boundaries. Their current
                # treatment is independent of the high-side PEC regression tested here.
                low_mask = solver.BC[0, 1:-1, 1:-1, component]
                high_mask = solver.BC[-1, 1:-1, 1:-1, component]

            elif axis == 1:
                low_mask = solver.BC[1:-1, 0, 1:-1, component]
                high_mask = solver.BC[1:-1, -1, 1:-1, component]

            else:
                low_mask = solver.BC[1:-1, 1:-1, 0, component]
                high_mask = solver.BC[1:-1, 1:-1, -1, component]

            # The explicit low-side tangential DOFs belong to the PEC
            # boundary and therefore have to be removed.
            assert np.all(low_mask == 0)

            # The high-side boundary DOFs are not stored explicitly.
            # The final stored layer is an interior layer and must remain.
            assert np.all(high_mask == 1)