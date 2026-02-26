"""
Integration tests for the Stokes equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

import numpy as np
import ufl
from dolfinx import fem
from petsc4py.PETSc import ScalarType


def test_Stokes_dJdnu(stokes_problem):
    """
    Test gradient of J with respect to viscosity ν.

    We compute the derivative of J with respect to ν:

        dJ/dν = ∂J/∂u * du/dν + ∂J/∂ν                               (2.1)

    The first term ∂J/∂u is easy to compute. The second term du/dν is handled
    using the adjoint problem:

        dF/dν = ∂F/∂u * du/dν + ∂F/∂ν = 0
        => du/dν = -(∂F/∂u)^{-1} * ∂F/∂ν                              (2.2)

    Inserting (2.2) into (2.1) yields:

        dJ/dν = - ∂J/∂u * (∂F/∂u)^{-1} * ∂F/∂ν + ∂J/∂ν                (2.3)

    This leads to the adjoint equation:

        (∂F^T/∂u) * θ = -∂J^T/∂u

    And with the adjoint solution θ:

        dJ/dν = θ^T * ∂F/∂ν + ∂J/∂ν                                  (2.5)
    """
    mesh = stokes_problem["mesh"]
    up = stokes_problem["up"]
    nu = stokes_problem["nu"]
    F = stokes_problem["F"]
    J_form = stokes_problem["J_form"]
    J = stokes_problem["J"]
    bcs = stokes_problem["bcs"]
    bc_dofs_total = stokes_problem["bc_dofs_total"]
    graph_ = stokes_problem["graph_"]

    DG0 = fem.functionspace(mesh, ("DG", 0))
    nu_function = fem.Function(DG0, name="nu")
    nu_function.x.array[:] = ScalarType(1.0)

    J_form_replaced = ufl.replace(J_form, {nu: nu_function})
    F_replaced = ufl.replace(F, {nu: nu_function})

    dJdu = ufl.derivative(J_form, up)
    dJdnu = ufl.derivative(J_form_replaced, nu_function)
    dFdnu = ufl.derivative(F_replaced, nu_function)
    dFdu = ufl.derivative(F, up)

    dJdu = fem.assemble_vector(fem.form(dJdu)).array
    dJdnu = fem.assemble_scalar(fem.form(dJdnu))
    dFdnu = fem.assemble_vector(fem.form(dFdnu)).array
    dFdu = fem.assemble_matrix(fem.form(dFdu), bcs=bcs).to_dense()

    # Apply the boundary conditions to the rhs of the adjoint problem
    for bc_dof in bc_dofs_total:
        dJdu[int(bc_dof)] = 0

    adjoint_solution = np.linalg.solve(dFdu.transpose(), -dJdu.transpose())
    gradient = adjoint_solution.transpose() @ dFdnu + dJdnu

    assert np.allclose(graph_.backprop(id(J), id(nu)), gradient)


def test_Stokes_dJdg(stokes_problem):
    """
    Test gradient of J with respect to boundary condition g.

    We calculate the derivative of the objective function with respect to the
    Dirichlet boundary condition g:

        dJ(u,p,g)/dg = ∂J/∂up * ∂up/∂g + ∂J/∂g

    Since u is defined on a mixed element space, we need to map the derivatives
    appropriately between function spaces.

    From the residual equation 0 = F(u, p):

        0 = ∂F/∂up * ∂up/∂g + ∂F/∂g

    This leads to the adjoint equation:

        (∂F/∂up)^T * θ = -(∂J/∂up)^T

    And the gradient computation:

        dJ/dg = θ^T * ∂F/∂g + ∂J/∂g

    We extract only the boundary values by multiplying with an identity matrix
    nonzero on the boundary.
    """
    V = stokes_problem["V"]
    V_u_map = stokes_problem["V_u_map"]
    up = stokes_problem["up"]
    g = stokes_problem["g"]
    F = stokes_problem["F"]
    J_form = stokes_problem["J_form"]
    J = stokes_problem["J"]
    bcs = stokes_problem["bcs"]
    bc_dofs_total = stokes_problem["bc_dofs_total"]
    dofs_obstacle = stokes_problem["dofs_obstacle"]
    graph_ = stokes_problem["graph_"]

    argument = ufl.TrialFunction(V)
    dJdu = ufl.derivative(J_form, up, argument)

    dJdg = ufl.derivative(J_form, g)

    argument = ufl.TrialFunction(V)
    dFdu = ufl.derivative(F, up, argument)

    dJdu = fem.assemble_vector(fem.form(dJdu)).array
    dJdg = fem.assemble_vector(fem.form(dJdg)).array

    dFdg = fem.assemble_matrix(
        fem.form(ufl.derivative(F, up, ufl.TrialFunction(up.function_space)))
    ).to_dense()
    dFdu = fem.assemble_matrix(fem.form(dFdu), bcs=bcs).to_dense()

    # Apply the boundary conditions to the rhs of the adjoint problem
    for bc_dof in bc_dofs_total:
        dJdu[int(bc_dof)] = 0

    adjoint_solution = np.linalg.solve(dFdu.transpose(), -dJdu.transpose())

    gradient = adjoint_solution.transpose() @ dFdg

    matrix = np.zeros((len(gradient), len(gradient)))
    for index in dofs_obstacle[0]:
        matrix[index, index] = 1.0

    gradient = (matrix @ gradient)[V_u_map] + dJdg
    assert np.allclose(gradient, graph_.backprop(id(J), id(g)))
