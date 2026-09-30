"""
Taylor tests for the plane elasticity adjoint gradients.

The function space of this problem is a blocked vector space, so this module
covers the case in which a dof of the boundary condition addresses a block of
components rather than a single entry of the gradient.
"""

import numpy as np
import pytest
from dolfinx import fem, mesh

from dolfinx_adjoint import Graph


def _convergence_rates(errors, steps):
    errors = np.asarray(errors)
    steps = np.asarray(steps)
    return np.log(errors[1:] / errors[:-1]) / np.log(steps[1:] / steps[:-1])


@pytest.mark.parametrize(
    "unit_square_mesh", [mesh.CellType.triangle], indirect=True, ids=["triangle"]
)
def test_plane_elasticity_taylor_bc(plane_elasticity_problem):
    """Taylor test for J with respect to the controlled boundary condition."""
    evaluation = plane_elasticity_problem.evaluate(graph=Graph())
    (gradient,) = evaluation.graph.backprop(evaluation.J, evaluation.u_D)
    direction = fem.Function(evaluation.u_D.function_space)
    direction.interpolate(lambda x: np.stack((np.sin(np.pi * x[0]), 1.0 + 0.5 * x[1])))
    derivative = gradient.dot(direction.x.petsc_vec)

    steps = 1e-2 * 0.5 ** np.arange(4)
    errors0 = []
    errors = []
    for step in steps:
        u_D = evaluation.u_D.copy()
        u_D.x.petsc_vec.axpy(step, direction.x.petsc_vec)
        value = plane_elasticity_problem.evaluate(u_D=u_D).value
        errors0.append(abs(value - evaluation.value))
        errors.append(abs(value - evaluation.value - step * derivative))

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.05)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.05)
