"""
Integration tests for the Heat Equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method for time-dependent problems.
"""

import numpy as np
import ufl
from dolfinx import fem


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

    dJdu_0 = fem.assemble_vector(fem.form(dJdu_0)).array
    dJdu = fem.assemble_vector(fem.form(dJdu)).array

    rhs = dJdu

    for i in range(len(u_iterations) - 1, 0, -1):
        F_i = ufl.replace(F, {u_next: u_iterations[i], u_prev: u_iterations[i - 1]})
        dF_idu_i = ufl.derivative(F_i, u_iterations[i])
        dF_idu_i = fem.assemble_matrix(fem.form(dF_idu_i)).to_dense()

        lambda_i = np.linalg.solve(dF_idu_i.transpose(), -rhs.transpose())

        dF_idu_i_1 = ufl.derivative(F_i, u_iterations[i - 1])
        dF_idu_i_1 = fem.assemble_matrix(fem.form(dF_idu_i_1)).to_dense()
        rhs = lambda_i.transpose() @ dF_idu_i_1

    gradient = rhs + dJdu_0

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(graph_.backprop(id(J), id(initial_guess)), gradient)
