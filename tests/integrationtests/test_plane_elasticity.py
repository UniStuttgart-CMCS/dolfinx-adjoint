"""
Taylor tests for the plane elasticity adjoint gradients.

The function space of this problem is a blocked vector space, so this module
covers the case in which a dof of the boundary condition addresses a block of
components rather than a single entry of the gradient.
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


def test_plane_elasticity_taylor_bc(plane_elasticity_problem):
    """Taylor test for J with respect to the controlled boundary condition."""
    graph_ = plane_elasticity_problem["graph_"]
    uD_control = plane_elasticity_problem["uD_control"]
    J = plane_elasticity_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = uD_control.function_space.mesh.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    grad = graph_.backprop(id(J), id(uD_control))
    grad_array = grad.array if hasattr(grad, "array") else np.array(grad)

    direction = fem.Function(uD_control.function_space)
    direction.interpolate(lambda x: np.stack((np.sin(np.pi * x[0]), 1.0 + 0.5 * x[1])))

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
