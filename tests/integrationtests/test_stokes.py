"""
Integration tests for the Stokes equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

import numpy as np
import ufl
from dolfinx import fem, la
from dolfinx.fem.petsc import LinearProblem


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
    bcs_dofs = stokes_problem["bcs_dofs"]
    graph_ = stokes_problem["graph_"]

    DG0 = fem.functionspace(mesh, ("DG", 0))
    nu_function = fem.Function(DG0, name="nu")
    nu_function.x.array[:] = nu.value

    J_form_replaced = ufl.replace(J_form, {nu: nu_function})
    F_replaced = ufl.replace(F, {nu: nu_function})

    dJdu = ufl.derivative(J_form, up)
    dJdnu = ufl.derivative(J_form_replaced, nu_function)
    dFdnu = ufl.derivative(F_replaced, nu_function)
    dFdu = ufl.derivative(F, up)

    # Homogeneous conditions on all constrained velocity and pressure DOFs
    zero = fem.Function(up.function_space)
    bcs_adjoint = fem.dirichletbc(zero, bcs_dofs)

    adjoint_solution = LinearProblem(
        ufl.adjoint(dFdu),
        -dJdu,
        bcs=[bcs_adjoint],
        petsc_options_prefix="adjoint_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        },
    ).solve()
    gradient = ufl.action(ufl.adjoint(dFdnu), adjoint_solution) + dJdnu
    gradient = fem.assemble_scalar(fem.form(gradient))

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
    bcs_dofs = stokes_problem["bcs_dofs"]
    dofs_obstacle = stokes_problem["dofs_obstacle"]
    graph_ = stokes_problem["graph_"]

    argument = ufl.TrialFunction(V)
    dJdu = ufl.derivative(J_form, up, argument)

    dJdg = ufl.derivative(J_form, g)

    dFdu = ufl.derivative(F, up, argument)
    dFdg = ufl.derivative(F, up, argument)

    # Homogeneous conditions on all constrained velocity and pressure DOFs
    zero = fem.Function(up.function_space)
    bcs_adjoint = fem.dirichletbc(zero, bcs_dofs)

    adjoint_solution = LinearProblem(
        ufl.adjoint(dFdu),
        -dJdu,
        bcs=[bcs_adjoint],
        petsc_options_prefix="adjoint_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        },
    ).solve()
    gradient = ufl.action(ufl.adjoint(dFdg), adjoint_solution)
    gradient = fem.assemble_vector(fem.form(gradient))
    gradient.scatter_reverse(la.InsertMode.add)
    gradient.scatter_forward()

    dJdg_vec = fem.assemble_vector(fem.form(dJdg))
    dJdg_vec.scatter_reverse(la.InsertMode.add)
    dJdg_vec.scatter_forward()

    # Extract obstacle boundary values and map to the collapsed velocity space.
    boundary_gradient = np.zeros_like(gradient.array)
    boundary_gradient[dofs_obstacle[0]] = gradient.array[dofs_obstacle[0]]
    gradient = boundary_gradient[V_u_map] + dJdg_vec.array

    assert np.allclose(gradient, graph_.backprop(id(J), id(g)))
