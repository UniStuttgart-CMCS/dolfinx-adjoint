"""Unit tests for the helpers of the Taylor tests."""

import numpy as np
import pytest
from dolfinx import fem, mesh

from dolfinx_graph_ad.verification import _perturbed, _rates


@pytest.mark.parametrize("steps", [[1e-2, 5e-3, 2.5e-3], [1e-1, 1e-2, 1e-3]])
def test_rates_are_the_log_slopes_of_a_power_law(steps: list) -> None:
    """Catch rates computed as log2 of the ratio, right only for halving steps.

    The residuals 3 ε² converge at rate 2 for any sequence of steps ε.
    """
    residuals = [3.0 * step**2 for step in steps]
    np.testing.assert_allclose(_rates(residuals, steps), 2.0)


@pytest.mark.parametrize("shape", [(), (2,)], ids=["scalar", "vector"])
def test_perturbed_function_moves_every_local_entry(
    unit_square_mesh_per_comm: mesh.Mesh, shape: tuple
) -> None:
    """Catch a copy moved without its ghost entries, or without all components of a block."""

    V = fem.functionspace(unit_square_mesh_per_comm, ("Lagrange", 1, shape))
    control = fem.Function(V)
    control.interpolate(lambda x: 1.0 + x[: V.dofmap.bs])
    direction = fem.Function(V)
    direction.interpolate(lambda x: x[: V.dofmap.bs] ** 2)

    perturbed = _perturbed(control, direction, 0.5)

    np.testing.assert_allclose(
        perturbed.x.array, control.x.array + 0.5 * direction.x.array
    )
