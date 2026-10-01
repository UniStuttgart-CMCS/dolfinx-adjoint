"""
Taylor tests for the Heat Equation adjoint gradients.

This test module verifies the gradients computed by the automatic
differentiation implementation for a time-dependent problem by perturbing the
initial condition and checking that the first-order Taylor remainder converges
with rate two.
"""

import numpy as np
from dolfinx import fem

from dolfinx_adjoint.verification import _perturbed, _rates, _remainders


def test_Heat_taylor_initial(heat_equation_problem, heat_equation_evaluation):
    """Taylor test for J with respect to the initial condition.

    The direction is a low mode, which the time steps damp only mildly, so that the
    remainder stays well above round-off.
    """
    evaluation = heat_equation_evaluation
    (gradient,) = evaluation.graph.backprop(evaluation.J, evaluation.initial_guess)
    direction = fem.Function(evaluation.initial_guess.function_space)
    direction.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + x[0])
    steps = 1e-2 * 0.5 ** np.arange(4)

    R0, R1 = _remainders(
        lambda step: heat_equation_problem.evaluate(
            initial_guess=_perturbed(evaluation.initial_guess, direction, step)
        ).value,
        evaluation.value,
        gradient.dot(direction.x.petsc_vec),
        steps,
    )

    np.testing.assert_allclose(_rates(R0, steps), np.ones(len(steps) - 1), atol=0.05)
    np.testing.assert_allclose(
        _rates(R1, steps), 2 * np.ones(len(steps) - 1), atol=0.05
    )
