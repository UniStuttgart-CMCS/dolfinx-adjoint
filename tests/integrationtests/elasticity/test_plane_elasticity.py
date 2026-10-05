"""
Taylor tests for the plane elasticity adjoint gradients.
"""

import numpy as np
from dolfinx import fem

from dolfinx_graph_ad import Graph
from dolfinx_graph_ad.verification import _perturbed, _rates, _remainders


def test_plane_elasticity_taylor_bc(plane_elasticity_problem):
    """Taylor test for J with respect to the controlled boundary condition."""
    evaluation = plane_elasticity_problem.evaluate(graph=Graph())
    (gradient,) = evaluation.graph.backprop(evaluation.J, evaluation.u_D)
    direction = fem.Function(evaluation.u_D.function_space)
    direction.interpolate(lambda x: np.stack((np.sin(np.pi * x[0]), 1.0 + 0.5 * x[1])))
    steps = 1e-2 * 0.5 ** np.arange(4)

    R0, R1 = _remainders(
        lambda step: plane_elasticity_problem.evaluate(
            u_D=_perturbed(evaluation.u_D, direction, step)
        ).value,
        evaluation.value,
        gradient.dot(direction.x.petsc_vec),
        steps,
    )

    np.testing.assert_allclose(_rates(R0, steps), np.ones(len(steps) - 1), atol=0.05)
    np.testing.assert_allclose(
        _rates(R1, steps), 2 * np.ones(len(steps) - 1), atol=0.05
    )
