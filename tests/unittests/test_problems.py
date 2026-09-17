"""Unit tests for graph-tracked problems and their derivatives."""

import numpy as np
import ufl
from dolfinx import mesh
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph, fem


def test_problem_constant_gradient_is_taken_at_the_current_value(
    unit_square_mesh: mesh.Mesh,
) -> None:
    """The problem edge differentiates at the value of c, not its constructor argument."""
    domain = unit_square_mesh
    graph_ = Graph()
    c = fem.Constant(domain, ScalarType(1.0), graph=graph_)
    c.value = 2.0

    V = fem.functionspace(domain, ("Lagrange", 1))
    uh = fem.Function(V, name="uh", graph=graph_)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    direct_solver = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = fem.petsc.LinearProblem(
        c**2 * ufl.inner(u, v) * ufl.dx,
        ufl.conj(v) * ufl.dx,
        u=uh,
        petsc_options_prefix="test_problem_constant_gradient_",
        petsc_options=direct_solver,
        adjoint_petsc_options=direct_solver,
        graph=graph_,
    )
    problem.solve(graph=graph_)

    J_form = ufl.inner(uh, uh) * ufl.dx
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    gradient = graph_.backprop(id(J), id(c))

    # P1 contains the exact solution u = c⁻², so J = c⁻⁴ on the unit square and
    # dJ/dc = -4 c⁻⁵ is -0.125. Evaluating ∂F/∂c = 2c u v at the constructor
    # argument c = 1 halves it to -0.0625.
    assert np.isclose(gradient, -0.125)
