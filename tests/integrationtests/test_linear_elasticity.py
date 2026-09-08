"""
Integration tests for the Linear Elasticity adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType


def test_material_param_lambda(linear_elasticity_problem):
    """
    Test gradient of J with respect to the material parameter λ.

    We compute the derivative of J with respect to the material parameter λ:

        dJ/dλ = ∂J/∂u * du/dλ                                             (1.1)

    We can obtain an expression for du/dλ by using the residual equation 0 = F(u(λ); λ):

        0 = dF/dλ = dF/du * du/dλ + ∂F/∂λ
        => du/dλ = -dF/du^{-1} * ∂F/∂λ                                    (1.2)

    Inserting (1.2) into (1.1) yields:

        dJ/dλ = ∂J/∂u * -dF/du^{-1} * ∂F/∂λ

    This leads to the adjoint equation:

        (∂F^T/∂u) * θ = -∂J^T/∂u

    And with the adjoint solution θ, we can compute the derivative of J with respect to λ:

        dJ/dλ = θ^T * ∂F/∂λ
    """
    domain = linear_elasticity_problem["domain"]
    u = linear_elasticity_problem["u"]
    lambda_ = linear_elasticity_problem["lambda_"]
    F = linear_elasticity_problem["F"]
    J_form = linear_elasticity_problem["J_form"]
    J = linear_elasticity_problem["J"]
    bcs_dofs = linear_elasticity_problem["bcs_dofs"]
    graph_ = linear_elasticity_problem["graph_"]

    dJdu = ufl.derivative(J_form, u)
    dFdu = ufl.derivative(F, u)

    bcs_adjoint = fem.dirichletbc(
        np.array([0.0, 0.0, 0.0], dtype=ScalarType),
        bcs_dofs,
        u.function_space,
    )

    # Define a new form to use ufl.derivative for the derivative of F with respect to λ
    DG0 = fem.functionspace(domain, ("DG", 0))
    lambda_func = fem.Function(DG0, name="lambda")
    lambda_func.x.array[:] = lambda_.value
    F_replace = ufl.replace(F, {lambda_: lambda_func})
    dFdlambda = ufl.derivative(F_replace, lambda_func)

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
    dJdlambda = fem.assemble_scalar(
        fem.form(ufl.action(ufl.adjoint(dFdlambda), adjoint_solution))
    )

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(dJdlambda, graph_.backprop(id(J), id(lambda_)))


def test_material_param_mu(linear_elasticity_problem):
    """
    Test gradient of J with respect to the material parameter μ.

    We compute the derivative of J with respect to the material parameter μ:

        dJ/dμ = ∂J/∂u * du/dμ                                             (1.1)

    We can obtain an expression for du/dμ by using the residual equation 0 = F(u(μ); μ):

        0 = dF/dμ = dF/du * du/dμ + ∂F/∂μ
        => du/dμ = -dF/du^{-1} * ∂F/∂μ                                    (1.2)

    Inserting (1.2) into (1.1) yields:

        dJ/dμ = ∂J/∂u * -dF/du^{-1} * ∂F/∂μ

    This leads to the adjoint equation:

        (∂F^T/∂u) * θ = -∂J^T/∂u

    And with the adjoint solution θ, we can compute the derivative of J with respect to μ:

        dJ/dμ = θ^T * ∂F/∂μ
    """
    domain = linear_elasticity_problem["domain"]
    u = linear_elasticity_problem["u"]
    mu = linear_elasticity_problem["mu"]
    F = linear_elasticity_problem["F"]
    J_form = linear_elasticity_problem["J_form"]
    J = linear_elasticity_problem["J"]
    bcs_dofs = linear_elasticity_problem["bcs_dofs"]
    graph_ = linear_elasticity_problem["graph_"]

    dJdu = ufl.derivative(J_form, u)
    dFdu = ufl.derivative(F, u)

    bcs_adjoint = fem.dirichletbc(
        np.array([0.0, 0.0, 0.0], dtype=ScalarType),
        bcs_dofs,
        u.function_space,
    )

    # Define a new form to use ufl.derivative for the derivative of F with respect to μ
    DG0 = fem.functionspace(domain, ("DG", 0))
    mu_func = fem.Function(DG0, name="mu")
    mu_func.x.array[:] = mu.value
    F_replace = ufl.replace(F, {mu: mu_func})
    dFdmu = ufl.derivative(F_replace, mu_func)

    adjoint_solution = LinearProblem(
        ufl.adjoint(dFdu),
        -dJdu,
        bcs=[bcs_adjoint],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        },
        petsc_options_prefix="adjoint_",
    ).solve()
    dJdmu = fem.assemble_scalar(
        fem.form(ufl.action(ufl.adjoint(dFdmu), adjoint_solution))
    )

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(dJdmu, graph_.backprop(id(J), id(mu)))
