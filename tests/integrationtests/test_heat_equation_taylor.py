"""
Taylor tests for the Heat Equation adjoint gradients.

This test module verifies the gradients computed by the automatic
differentiation implementation for a time-dependent problem by perturbing the
initial condition and checking that the first-order Taylor remainder converges
with rate two.
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
def test_Heat_taylor_initial(heat_equation_problem):
    """Taylor test for J with respect to the initial condition.

    The direction is a low mode, which the time steps damp only mildly, so that the
    remainder stays well above round-off.
    """
    evaluation = heat_equation_problem.evaluate(graph=Graph())
    gradient = evaluation.graph.backprop(evaluation.J, evaluation.initial_guess)
    direction = fem.Function(evaluation.initial_guess.function_space)
    direction.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + x[0])
    derivative = gradient.dot(direction.x.petsc_vec)

    steps = 1e-2 * 0.5 ** np.arange(4)
    errors0 = []
    errors = []
    for step in steps:
        initial_guess = evaluation.initial_guess.copy()
        initial_guess.x.petsc_vec.axpy(step, direction.x.petsc_vec)
        value = heat_equation_problem.evaluate(initial_guess=initial_guess).value
        errors0.append(abs(value - evaluation.value))
        errors.append(abs(value - evaluation.value - step * derivative))

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.05)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.05)
