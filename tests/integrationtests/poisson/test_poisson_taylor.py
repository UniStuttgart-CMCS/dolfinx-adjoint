"""
Taylor tests for the Poisson equation adjoint gradients.

This test module verifies the gradients computed by the automatic
differentiation implementation by perturbing an input of the forward problem and
checking that the first-order Taylor remainder converges with rate two.
"""

import numpy as np
from dolfinx import fem

from dolfinx_adjoint.verification import _perturbed, _rates, _remainders


def test_Poisson_taylor_f(poisson_problem, poisson_evaluation):
    """Taylor test for J with respect to the forcing term f (graph backprop)."""
    evaluation = poisson_evaluation
    (gradient,) = evaluation.graph.backprop(evaluation.J, evaluation.f)
    direction = fem.Function(evaluation.f.function_space)
    direction.interpolate(lambda x: x[0] + np.sin(x[1]))
    steps = 1e-2 * 0.5 ** np.arange(4)

    R0, R1 = _remainders(
        lambda step: poisson_problem.evaluate(
            f=_perturbed(evaluation.f, direction, step)
        ).value,
        evaluation.value,
        gradient.dot(direction.x.petsc_vec),
        steps,
    )

    np.testing.assert_allclose(_rates(R0, steps), np.ones(len(steps) - 1), atol=0.05)
    np.testing.assert_allclose(
        _rates(R1, steps), 2 * np.ones(len(steps) - 1), atol=0.05
    )


def test_Poisson_taylor_nu(poisson_problem, poisson_evaluation):
    """Taylor test for J with respect to the diffusion coefficient nu."""
    evaluation = poisson_evaluation
    (gradient,) = evaluation.graph.backprop(evaluation.J, evaluation.nu)
    nu = float(evaluation.nu.value)
    steps = 1e-2 * 0.5 ** np.arange(4)

    R0, R1 = _remainders(
        lambda step: poisson_problem.evaluate(nu=nu + step).value,
        evaluation.value,
        gradient,
        steps,
    )

    np.testing.assert_allclose(_rates(R0, steps), np.ones(len(steps) - 1), atol=0.05)
    np.testing.assert_allclose(
        _rates(R1, steps), 2 * np.ones(len(steps) - 1), atol=0.05
    )


def test_Poisson_taylor_bc(poisson_problem, poisson_evaluation):
    """Taylor test for J with respect to the controlled boundary condition."""

    evaluation = poisson_evaluation
    (gradient,) = evaluation.graph.backprop(evaluation.J, evaluation.u_D)
    direction = fem.Function(evaluation.u_D.function_space)
    direction.interpolate(lambda x: 1.0 + np.sin(np.pi * x[1]) + x[0])
    steps = 1e-2 * 0.5 ** np.arange(4)

    R0, R1 = _remainders(
        lambda step: poisson_problem.evaluate(
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
