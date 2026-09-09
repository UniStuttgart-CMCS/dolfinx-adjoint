"""
Taylor tests for the Poisson equation adjoint gradients.

This test module verifies the gradients computed by the automatic
differentiation implementation by perturbing an input of the forward problem and
checking that the first-order Taylor remainder converges with rate two.
"""

import numpy as np
from dolfinx import fem
from mpi4py import MPI


def _convergence_rates(errors: np.ndarray, steps: np.ndarray) -> list:
    rates = []
    for i in range(1, len(steps)):
        rates.append(
            np.log(errors[i] / errors[i - 1]) / np.log(steps[i] / steps[i - 1])
        )
    return rates


def test_Poisson_taylor_f(poisson_problem):
    """Taylor test for J with respect to the forcing term f (graph backprop)."""
    graph_ = poisson_problem["graph_"]
    f = poisson_problem["f"]
    J = poisson_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = f.function_space.mesh.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    df = fem.Function(f.function_space)
    df.interpolate(lambda x: x[0] + np.sin(x[1]))

    grad = graph_.backprop(id(J), id(f))
    grad_array = grad.array if hasattr(grad, "array") else np.array(grad)
    dJ = comm.allreduce(np.dot(grad_array, df.x.array), op=MPI.SUM)

    f_org = f.x.array.copy()
    step_length = 1e-2
    steps = [step_length * (0.5**i) for i in range(4)]
    errors0 = []
    errors = []
    try:
        for h in steps:
            f.x.array[:] = f_org + h * df.x.array
            graph_.recalculate()
            Jh = comm.allreduce(J_node.object, op=MPI.SUM)
            errors0.append(abs(Jh - J0))
            errors.append(abs(Jh - J0 - h * dJ))
    finally:
        f.x.array[:] = f_org
        graph_.recalculate()

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)


def test_Poisson_taylor_nu(poisson_problem):
    """Taylor test for J with respect to the diffusion coefficient nu."""
    graph_ = poisson_problem["graph_"]
    nu = poisson_problem["nu"]
    J = poisson_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = nu.domain.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    direction = 1.0
    grad = graph_.backprop(id(J), id(nu))
    dJ = comm.allreduce(float(grad) * direction, op=MPI.SUM)

    nu_org = float(nu.value)
    step_length = 1e-2
    steps = [step_length * (0.5**i) for i in range(4)]
    errors0 = []
    errors = []
    try:
        for h in steps:
            nu.value = float(nu_org + h * direction)
            graph_.recalculate()
            Jh = comm.allreduce(J_node.object, op=MPI.SUM)
            errors0.append(abs(Jh - J0))
            errors.append(abs(Jh - J0 - h * dJ))
    finally:
        nu.value = nu_org
        graph_.recalculate()

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)


def test_Poisson_taylor_bc(poisson_problem):
    """Taylor test for J with respect to the controlled boundary condition."""
    graph_ = poisson_problem["graph_"]
    uD_control = poisson_problem["uD_control"]
    control_dofs = poisson_problem["control_dofs"]
    J = poisson_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = uD_control.function_space.mesh.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    grad = graph_.backprop(id(J), id(uD_control))
    grad_array = grad.array if hasattr(grad, "array") else np.array(grad)

    # The perturbation direction is a fixed smooth field restricted to the
    # controlled dofs. It is deliberately independent of the computed gradient,
    # so that error components orthogonal to the gradient are also detected.
    smooth = fem.Function(uD_control.function_space)
    smooth.interpolate(lambda x: 1.0 + np.sin(np.pi * x[1]) + x[0])
    direction = fem.Function(uD_control.function_space)
    direction.x.array[:] = 0.0
    direction.x.array[control_dofs] = smooth.x.array[control_dofs]

    dJ = comm.allreduce(np.dot(grad_array, direction.x.array), op=MPI.SUM)

    u_org = uD_control.x.array.copy()
    step_length = 1e-2
    steps = [step_length * (0.5**i) for i in range(4)]
    errors0 = []
    errors = []
    try:
        for h in steps:
            uD_control.x.array[:] = u_org + h * direction.x.array
            graph_.recalculate()
            Jh = comm.allreduce(J_node.object, op=MPI.SUM)
            errors0.append(abs(Jh - J0))
            errors.append(abs(Jh - J0 - h * dJ))
    finally:
        uD_control.x.array[:] = u_org
        graph_.recalculate()

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)
