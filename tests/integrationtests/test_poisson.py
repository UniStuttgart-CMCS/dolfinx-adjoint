"""
Integration tests for the Poisson equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

import numpy as np
import ufl
from dolfinx import fem, la
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType


def test_Poisson_dJdf(poisson_problem):
    """
    Test gradient of J with respect to forcing term f.

    We turn to the explicit adjoint approach to calculate the gradient of J(u) with respect to f.
    Therefore, we first define a new reduced functional R(f) = J(u(f),f) and take the derivative with respect to f:
        dJ/df = dR/df = ∂J/∂u * du/df + ∂J/∂f                       (1.1)

    The first term ∂J/∂u and the last term ∂J/∂f are easy to compute.
    The second term du/df will be handled using the adjoint problem.

    By taking the derivative of F(u) = 0 with respect to f, we obtain a representation of du/df:
        dF/df = ∂F/∂u * du/df + ∂F/∂f = 0
        => du/df = -(∂F/∂u)^-1 * ∂F/∂f                              (1.2)

    Inserting (1.2) into (1.1) yields

        dJ/df = - ∂J/∂u * (∂F/∂u)^-1 * ∂F/∂f + ∂J/∂f                (1.3)

    Our adjoint approach thus is to solve the adjoint problem
        ∂Fᵀ/∂u * λ =  - ∂Jᵀ/∂u                                      (1.4)
    and then compute the gradient of J(u) with respect to f using (1.3).
        dJ/df = λᵀ * ∂F/∂f + ∂J/∂f                                  (1.5)
    """
    F = poisson_problem["F"]
    uh = poisson_problem["uh"]
    f = poisson_problem["f"]
    J_form = poisson_problem["J_form"]
    bcs_dofs = poisson_problem["bcs_dofs"]
    graph_ = poisson_problem["graph_"]
    J = poisson_problem["J"]

    dFdu = ufl.derivative(F, uh)
    dFdf = ufl.derivative(F, f)
    dJdu = ufl.derivative(J_form, uh)
    dJdf = ufl.derivative(J_form, f)

    # Boundary conditions of adjoint must be set to zero since there cannot be any contribution from the boundary to the gradient of J with respect to variables except for the boundary condition itself.
    bcs_adjoint = fem.dirichletbc(
        ScalarType(0.0),
        bcs_dofs,
        uh.function_space,
    )

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
    gradient = ufl.action(ufl.adjoint(dFdf), adjoint_solution) + dJdf

    gradient_df = fem.assemble_vector(fem.form(gradient))
    gradient_df.scatter_reverse(la.InsertMode.add)
    gradient_df.scatter_forward()

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(graph_.backprop(id(J), id(f)), gradient_df.array[:])


def test_Poisson_dJdnu(poisson_problem):
    """
    Test gradient of J with respect to diffusion coefficient ν.

    In this test we explicitly calculate the adjoint of J with respect to ν.
    Again we come up with
        dJ/dν = ∂J/∂u * du/dν + ∂J/∂ν                               (2.1)

    The first term ∂J/∂u is easy to compute, and to calculate the term ∂J/∂ν using a fem.function does the trick
    in order to use the automatic differentiation in ufl. Here we have to keep in mind that we still need
    a scalar quantity as the value of ∂J/∂ν.

    The second term du/dν will be handled using the adjoint problem similar to (1.2)
        dF/dν = ∂F/∂u * du/dν + ∂F/∂ν = 0
        => du/dν = -(∂F/∂u)^-1 * ∂F/∂ν                              (2.2)

    Inserting (2.2) into (2.1) yields
        dJ/dν = - ∂J/∂u * (∂F/∂u)^-1 * ∂F/∂ν + ∂J/∂ν                (2.3)

    Our adjoint approach thus is to solve the adjoint problem
        ∂Fᵀ/∂u * λ =  - ∂Jᵀ/∂u                                      (2.4)
    and then compute the gradient of J(u) with respect to ν using (2.3).
        dJ/dν = λᵀ * ∂F/∂ν + ∂J/∂ν                                  (2.5)
    """
    domain = poisson_problem["domain"]
    F = poisson_problem["F"]
    uh = poisson_problem["uh"]
    nu = poisson_problem["nu"]
    J_form = poisson_problem["J_form"]
    bcs_dofs = poisson_problem["bcs_dofs"]
    graph_ = poisson_problem["graph_"]
    J = poisson_problem["J"]

    DG0 = fem.functionspace(domain, ("DG", 0))
    nu_function = fem.Function(DG0, name="nu")
    nu_function.x.array[:] = nu.value

    J_form_replaced = ufl.replace(J_form, {nu: nu_function})
    F_replaced = ufl.replace(F, {nu: nu_function})

    dJdu = ufl.derivative(J_form, uh)
    dJdnu = ufl.derivative(J_form_replaced, nu_function)
    dFdnu = ufl.derivative(F_replaced, nu_function)
    dFdu = ufl.derivative(F, uh)

    # Boundary conditions of adjoint must be set to zero since there cannot be any contribution from the boundary to the gradient of J with respect to variables except for the boundary condition itself.
    bcs_adjoint = fem.dirichletbc(
        ScalarType(0.0),
        bcs_dofs,
        uh.function_space,
    )

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

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(graph_.backprop(id(J), id(nu)), gradient)


def test_Poisson_dJdbc(poisson_problem):
    """
    Test gradient of J with respect to boundary condition u_D.

    We turn to the explicit adjoint approach to calculate the gradient of J(u) with respect to u_D on the left boundary.
    Therefore, we first define a new reduced functional R(u_D) = J(u(u_D),u_D) and take the derivative with respect to u_D:
        dJ/du_D = ∂J/∂u * du/du_D + ∂J/∂u_D                         (3.1)

    The first term ∂J/∂u is easy to compute and for the second term we use the adjoint problem similar to (1.2)
        dF/du_D = ∂F/∂u * du/du_D + ∂F/∂u_D = 0
        => du/du_D = -(∂F/∂u)^-1 * ∂F/∂u_D                          (3.2)

    Inserting (3.2) into (3.1) yields
        dJ/du_D = - ∂J/∂u * (∂F/∂u)^-1 * ∂F/∂u_D + ∂J/∂u_D          (3.3)

    We first need to look closer to the term ∂F/∂u_D.

    Since the boundary condition is not directly in the ufl representation of the form F,
    we need to look at how the boundary conditions are applied when constructing a nonlinear problem.
    In the case of the Poisson problem, the boundary conditions are applied to the linearized problem,
    by lifting the linear term. With the form F = a - L = 0, we get the assembled form F = A*u - b = 0.
    The boundary conditions are now applied by lifting the b vector:
        b = b - α A (u_D - x₀)
    where α is a scaling factor (in this case α=-1) and x_0 is some weird thing that is not important here.
    The term ∂F/∂u_D is now given by
        ∂F/∂u_D = - α A

    Our adjoint approach thus is to solve the adjoint problem
        ∂Fᵀ/∂u * λ =  - ∂Jᵀ/∂u                                      (3.4)
    and then compute the gradient of J(u) with respect to u_D using (3.3).
        dJ/du_D = λᵀ * ∂F/∂u_D + ∂J/∂u_D                            (3.5)

    Here ∂J/∂u_D is not defined. Since the function on the boundary u_D is defined for Ω and not for ∂Ω,
    especially not the part of the boundary where the boundary condition is applied. We need to extract these values
    and set everything else to zero.
    """
    F = poisson_problem["F"]
    uh = poisson_problem["uh"]
    uD_L = poisson_problem["uD_L"]
    J_form = poisson_problem["J_form"]
    boundary_dofs_L = poisson_problem["boundary_dofs_L"]
    bcs_dofs = poisson_problem["bcs_dofs"]
    graph_ = poisson_problem["graph_"]
    J = poisson_problem["J"]

    dFdu = ufl.derivative(F, uh)
    dJdu = ufl.derivative(J_form, uh)
    dFdbc = ufl.derivative(F, uh, ufl.TrialFunction(uh.function_space))

    # Boundary conditions of adjoint must be set to zero since there cannot be any contribution from the boundary to the gradient of J with respect to variables except for the boundary condition itself.
    bcs_adjoint = fem.dirichletbc(
        ScalarType(0.0),
        bcs_dofs,
        uh.function_space,
    )

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
    gradient = ufl.action(ufl.adjoint(dFdbc), adjoint_solution)

    gradient = fem.assemble_vector(fem.form(gradient))
    gradient.scatter_reverse(la.InsertMode.add)
    gradient.scatter_forward()

    # Extract gradient values only at the boundary
    matrix = np.zeros((len(gradient.array), len(gradient.array)))
    for index in boundary_dofs_L:
        matrix[index, index] = 1.0

    gradient = matrix @ gradient.array

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(graph_.backprop(id(J), id(uD_L)), gradient)
