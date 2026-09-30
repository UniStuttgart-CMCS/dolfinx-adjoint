"""
Taylor tests for the Stokes equation adjoint gradients.

This test module verifies the gradients computed by the automatic
differentiation implementation by perturbing an input of the forward problem and
checking that the first-order Taylor remainder converges with rate two.
"""

import numpy as np
from dolfinx import fem

from dolfinx_adjoint import Graph


def _convergence_rates(errors, steps):
    errors = np.asarray(errors)
    steps = np.asarray(steps)
    return np.log(errors[1:] / errors[:-1]) / np.log(steps[1:] / steps[:-1])


def test_Stokes_taylor_nu(stokes_problem):
    """Taylor test for J with respect to the viscosity nu."""
    evaluation = stokes_problem.evaluate(graph=Graph())
    (gradient,) = evaluation.graph.backprop(evaluation.J, evaluation.nu)
    direction = 1.0
    derivative = gradient * direction
    nu = float(evaluation.nu.value)

    steps = 1e-2 * 0.5 ** np.arange(4)
    errors0 = []
    errors = []
    for step in steps:
        value = stokes_problem.evaluate(nu=nu + step * direction).value
        errors0.append(abs(value - evaluation.value))
        errors.append(abs(value - evaluation.value - step * derivative))

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.05)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.05)


def test_Stokes_taylor_g(stokes_problem):
    """Taylor test for J with respect to the obstacle boundary condition g.

    The direction spans the whole collapsed velocity space, so a gradient entry off
    the obstacle, where J does not depend on g, is not masked by the direction.
    """
    evaluation = stokes_problem.evaluate(graph=Graph())
    (gradient,) = evaluation.graph.backprop(evaluation.J, evaluation.g)
    direction = fem.Function(evaluation.g.function_space)
    direction.interpolate(
        lambda x: np.stack((1.0 + np.sin(np.pi * x[1]), np.cos(np.pi * x[0])))
    )
    derivative = gradient.dot(direction.x.petsc_vec)

    steps = 1e-2 * 0.5 ** np.arange(4)
    errors0 = []
    errors = []
    for step in steps:
        g = evaluation.g.copy()
        g.x.petsc_vec.axpy(step, direction.x.petsc_vec)
        value = stokes_problem.evaluate(g=g).value
        errors0.append(abs(value - evaluation.value))
        errors.append(abs(value - evaluation.value - step * derivative))

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.05)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.05)
