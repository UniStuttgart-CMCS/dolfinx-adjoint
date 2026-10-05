"""
Taylor tests for the Stokes equation adjoint gradients.

This test module verifies the gradients computed by the automatic
differentiation implementation by perturbing an input of the forward problem and
checking that the first-order Taylor remainder converges with rate two.
"""

import numpy as np
from dolfinx import fem

from dolfinx_graph_ad.verification import _perturbed, _rates, _remainders


def test_Stokes_taylor_nu(stokes_problem, stokes_evaluation):
    """Taylor test for J with respect to the viscosity nu."""
    evaluation = stokes_evaluation
    (gradient,) = evaluation.graph.backprop(evaluation.J, evaluation.nu)
    nu = float(evaluation.nu.value)
    steps = 1e-2 * 0.5 ** np.arange(4)

    R0, R1 = _remainders(
        lambda step: stokes_problem.evaluate(nu=nu + step).value,
        evaluation.value,
        gradient,
        steps,
    )

    np.testing.assert_allclose(_rates(R0, steps), np.ones(len(steps) - 1), atol=0.05)
    np.testing.assert_allclose(
        _rates(R1, steps), 2 * np.ones(len(steps) - 1), atol=0.05
    )


def test_Stokes_taylor_g(stokes_problem, stokes_evaluation):
    """Taylor test for J with respect to the obstacle boundary condition g.

    The direction spans the whole collapsed velocity space, so a gradient entry off
    the obstacle, where J does not depend on g, is not masked by the direction.
    """
    evaluation = stokes_evaluation
    (gradient,) = evaluation.graph.backprop(evaluation.J, evaluation.g)
    direction = fem.Function(evaluation.g.function_space)
    direction.interpolate(
        lambda x: np.stack((1.0 + np.sin(np.pi * x[1]), np.cos(np.pi * x[0])))
    )
    steps = 1e-2 * 0.5 ** np.arange(4)

    R0, R1 = _remainders(
        lambda step: stokes_problem.evaluate(
            g=_perturbed(evaluation.g, direction, step)
        ).value,
        evaluation.value,
        gradient.dot(direction.x.petsc_vec),
        steps,
    )

    np.testing.assert_allclose(_rates(R0, steps), np.ones(len(steps) - 1), atol=0.05)
    np.testing.assert_allclose(
        _rates(R1, steps), 2 * np.ones(len(steps) - 1), atol=0.05
    )
