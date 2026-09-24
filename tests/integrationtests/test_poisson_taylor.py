"""
Taylor tests for the Poisson equation adjoint gradients.

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


def test_Poisson_taylor_f(poisson_problem):
    """Taylor test for J with respect to the forcing term f (graph backprop)."""
    evaluation = poisson_problem.evaluate(graph=Graph())
    gradient = evaluation.graph.backprop(id(evaluation.J), id(evaluation.f))
    direction = fem.Function(evaluation.f.function_space)
    direction.interpolate(lambda x: x[0] + np.sin(x[1]))
    derivative = gradient.dot(direction.x.petsc_vec)

    steps = 1e-2 * 0.5 ** np.arange(4)
    errors0 = []
    errors = []
    for step in steps:
        f = evaluation.f.copy()
        f.x.petsc_vec.axpy(step, direction.x.petsc_vec)
        value = poisson_problem.evaluate(f=f).value
        errors0.append(abs(value - evaluation.value))
        errors.append(abs(value - evaluation.value - step * derivative))

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)


def test_Poisson_taylor_nu(poisson_problem):
    """Taylor test for J with respect to the diffusion coefficient nu."""
    evaluation = poisson_problem.evaluate(graph=Graph())
    gradient = evaluation.graph.backprop(id(evaluation.J), id(evaluation.nu))
    direction = 1.0
    derivative = gradient * direction
    nu = float(evaluation.nu.value)

    steps = 1e-2 * 0.5 ** np.arange(4)
    errors0 = []
    errors = []
    for step in steps:
        value = poisson_problem.evaluate(nu=nu + step * direction).value
        errors0.append(abs(value - evaluation.value))
        errors.append(abs(value - evaluation.value - step * derivative))

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)


def test_Poisson_taylor_bc(poisson_problem):
    """Taylor test for J with respect to the controlled boundary condition."""
    evaluation = poisson_problem.evaluate(graph=Graph())
    gradient = evaluation.graph.backprop(id(evaluation.J), id(evaluation.u_D))
    smooth = fem.Function(evaluation.u_D.function_space)
    smooth.interpolate(lambda x: 1.0 + np.sin(np.pi * x[1]) + x[0])
    direction = fem.Function(evaluation.u_D.function_space)
    dofs = poisson_problem.control_dofs
    direction.x.array[dofs] = smooth.x.array[dofs]
    derivative = gradient.dot(direction.x.petsc_vec)

    steps = 1e-2 * 0.5 ** np.arange(4)
    errors0 = []
    errors = []
    for step in steps:
        u_D = evaluation.u_D.copy()
        u_D.x.petsc_vec.axpy(step, direction.x.petsc_vec)
        value = poisson_problem.evaluate(u_D=u_D).value
        errors0.append(abs(value - evaluation.value))
        errors.append(abs(value - evaluation.value - step * derivative))

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)
