"""
Integration tests for the Heat Equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method for time-dependent problems.
"""

import numpy as np
import ufl
from dolfinx import fem, la
from dolfinx.fem.petsc import LinearProblem


def test_Heat_initial(heat_equation_problem):
    """
    Test gradient of J with respect to the initial condition.

    We compute the derivative of J with respect to the initial guess $u_0$:

        $$\\frac{dJ}{du_0} = \\frac{\\partial J}{\\partial u_N} \\frac{du_N}{du_{N-1}} \\cdots \\frac{du_1}{du_0} + \\frac{\\partial J}{\\partial u_0}$$

    where $u_i$ is the solution at time step $i$.

    The first and last terms $\\frac{\\partial J}{\\partial u_N}$ and $\\frac{\\partial J}{\\partial u_0}$
    can be easily obtained using UFL.

    The term $\\frac{du_N}{du_{N-1}}$ is obtained by solving the adjoint equation. Taking the derivative
    of the residual equation $F(u_N, u_{N-1}) = 0$ with respect to $u_{N-1}$:

        $$\\frac{dF}{du_{N-1}} = \\frac{\\partial F}{\\partial u_N} \\frac{du_N}{du_{N-1}} + \\frac{\\partial F}{\\partial u_{N-1}} = 0$$

        $$\\Rightarrow \\frac{du_N}{du_{N-1}} = - \\left(\\frac{\\partial F}{\\partial u_N}\\right)^{-1} \\frac{\\partial F}{\\partial u_{N-1}}$$

    This leads to the first adjoint equation:

        $$\\left(\\frac{\\partial F^T}{\\partial u_N}\\right) \\lambda_N = - \\frac{\\partial J^T}{\\partial u_N}$$

    and subsequent adjoint equations:

        $$\\left(\\frac{\\partial F^T}{\\partial u_{i-1}}\\right) \\lambda_{i-1} = - \\lambda_i \\frac{\\partial F^T}{\\partial u_{i-1}}$$

    Finally:

        $$\\frac{dJ}{du_0} = \\lambda_1^T \\frac{\\partial F}{\\partial u_0} + \\frac{\\partial J}{\\partial u_0}$$
    """
    graph_ = heat_equation_problem["graph_"]
    J_form = heat_equation_problem["J_form"]
    J = heat_equation_problem["J"]
    initial_guess = heat_equation_problem["initial_guess"]
    u_next = heat_equation_problem["u_next"]
    u_prev = heat_equation_problem["u_prev"]
    F = heat_equation_problem["F"]
    u_iterations = heat_equation_problem["u_iterations"]

    dJdu = ufl.derivative(J_form, u_next)
    dJdu_0 = ufl.derivative(J_form, initial_guess)

    rhs = dJdu

    for i in range(len(u_iterations) - 1, 0, -1):
        F_i = ufl.replace(F, {u_next: u_iterations[i], u_prev: u_iterations[i - 1]})
        dF_idu_i = ufl.derivative(F_i, u_iterations[i])

        lambda_i = LinearProblem(
            ufl.adjoint(dF_idu_i),
            -rhs,
            petsc_options_prefix="adjoint_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_error_if_not_converged": True,
            },
        ).solve()

        dF_idu_i_1 = ufl.derivative(F_i, u_iterations[i - 1])
        rhs = ufl.action(ufl.adjoint(dF_idu_i_1), lambda_i)

    gradient = rhs + dJdu_0
    gradient = fem.assemble_vector(fem.form(gradient))
    gradient.scatter_reverse(la.InsertMode.add)
    gradient.scatter_forward()

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(graph_.backprop(id(J), id(initial_guess)), gradient.array)
