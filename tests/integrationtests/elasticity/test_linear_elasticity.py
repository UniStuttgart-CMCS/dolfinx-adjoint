"""
Integration tests for the Linear Elasticity adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType


@pytest.mark.parametrize("parameter", ["lambda_", "mu"])
def test_material_param(linear_elasticity_evaluation, parameter):
    """
    Test gradient of J with respect to the material parameter m, which is λ or μ.

    We compute the derivative of J with respect to the material parameter m:

        dJ/dm = ∂J/∂u * du/dm                                             (1.1)

    We can obtain an expression for du/dm by using the residual equation 0 = F(u(m); m):

        0 = dF/dm = dF/du * du/dm + ∂F/∂m
        => du/dm = -dF/du^{-1} * ∂F/∂m                                    (1.2)

    Inserting (1.2) into (1.1) yields:

        dJ/dm = ∂J/∂u * -dF/du^{-1} * ∂F/∂m

    This leads to the adjoint equation:

        (∂F^T/∂u) * θ = -∂J^T/∂u

    And with the adjoint solution θ, we can compute the derivative of J with respect to m:

        dJ/dm = θ^T * ∂F/∂m
    """
    problem = linear_elasticity_evaluation.problem
    domain = problem.domain
    u = linear_elasticity_evaluation.u
    control = getattr(linear_elasticity_evaluation, parameter)
    F = linear_elasticity_evaluation.F
    J_form = linear_elasticity_evaluation.J_form
    J = linear_elasticity_evaluation.J
    bcs_dofs = problem.bcs_dofs
    graph_ = linear_elasticity_evaluation.graph

    dJdu = ufl.derivative(J_form, u)
    dFdu = ufl.derivative(F, u)

    bcs_adjoint = fem.dirichletbc(
        np.array([0.0, 0.0, 0.0], dtype=ScalarType),
        bcs_dofs,
        u.function_space,
    )

    # Define a new form to use ufl.derivative for the derivative of F with respect to the material control
    DG0 = fem.functionspace(domain, ("DG", 0))
    control_function = fem.Function(DG0, name=parameter)
    control_function.x.array[:] = control.value
    F_replace = ufl.replace(F, {control: control_function})
    dFdm = ufl.derivative(F_replace, control_function)

    adjoint_solution = LinearProblem(
        ufl.adjoint(dFdu),
        -dJdu,
        bcs=[bcs_adjoint],
        petsc_options_prefix="adjoint_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
    ).solve()

    gradient = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.action(ufl.adjoint(dFdm), adjoint_solution))),
        op=MPI.SUM,
    )

    # Compare automatic differentiation result with explicit adjoint calculation
    assert np.allclose(gradient, graph_.backprop(J, control)[0])
