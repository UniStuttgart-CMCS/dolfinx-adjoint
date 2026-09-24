"""
Taylor tests for the Linear Elasticity adjoint gradients.

This test module verifies the gradients computed by the automatic
differentiation implementation by perturbing an input of the forward problem and
checking that the first-order Taylor remainder converges with rate two.
"""

import numpy as np

from dolfinx_adjoint import Graph


def _convergence_rates(errors, steps):
    errors = np.asarray(errors)
    steps = np.asarray(steps)
    return np.log(errors[1:] / errors[:-1]) / np.log(steps[1:] / steps[:-1])


def test_material_param_lambda_taylor(linear_elasticity_problem):
    """Taylor test for J with respect to the material parameter λ."""
    evaluation = linear_elasticity_problem.evaluate(graph=Graph())
    gradient = evaluation.graph.backprop(id(evaluation.J), id(evaluation.lambda_))
    direction = 1.0
    derivative = gradient * direction
    lambda_ = float(evaluation.lambda_.value)

    steps = 1e-2 * 0.5 ** np.arange(4)
    errors0 = []
    errors = []
    for step in steps:
        value = linear_elasticity_problem.evaluate(
            lambda_=lambda_ + step * direction
        ).value
        errors0.append(abs(value - evaluation.value))
        errors.append(abs(value - evaluation.value - step * derivative))

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.05)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.05)


def test_material_param_mu_taylor(linear_elasticity_problem):
    """Taylor test for J with respect to the material parameter μ."""
    evaluation = linear_elasticity_problem.evaluate(graph=Graph())
    gradient = evaluation.graph.backprop(id(evaluation.J), id(evaluation.mu))
    direction = 1.0
    derivative = gradient * direction
    mu = float(evaluation.mu.value)

    steps = 1e-2 * 0.5 ** np.arange(4)
    errors0 = []
    errors = []
    for step in steps:
        value = linear_elasticity_problem.evaluate(mu=mu + step * direction).value
        errors0.append(abs(value - evaluation.value))
        errors.append(abs(value - evaluation.value - step * derivative))

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.05)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.05)
