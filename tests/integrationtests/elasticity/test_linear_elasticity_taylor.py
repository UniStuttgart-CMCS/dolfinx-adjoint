"""
Taylor tests for the Linear Elasticity adjoint gradients.

This test module verifies the gradients computed by the automatic
differentiation implementation by perturbing an input of the forward problem and
checking that the first-order Taylor remainder converges with rate two.
"""

import numpy as np
import pytest

from dolfinx_adjoint.verification import _rates, _remainders


@pytest.mark.parametrize("parameter", ["lambda_", "mu"])
def test_material_param_taylor(
    linear_elasticity_problem,
    linear_elasticity_evaluation,
    parameter,
):
    """Taylor test for J with respect to the material parameters λ and μ."""
    evaluation = linear_elasticity_evaluation
    control = getattr(evaluation, parameter)
    (gradient,) = evaluation.graph.backprop(evaluation.J, control)
    value = float(control.value)
    steps = 1e-2 * 0.5 ** np.arange(4)

    R0, R1 = _remainders(
        lambda step: linear_elasticity_problem.evaluate(
            **{parameter: value + step}
        ).value,
        evaluation.value,
        gradient,
        steps,
    )

    np.testing.assert_allclose(_rates(R0, steps), np.ones(len(steps) - 1), atol=0.05)
    np.testing.assert_allclose(
        _rates(R1, steps), 2 * np.ones(len(steps) - 1), atol=0.05
    )
