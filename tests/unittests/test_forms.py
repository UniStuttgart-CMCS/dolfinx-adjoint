"""Unit tests for graph-tracked forms and their derivatives."""

import numpy as np
import ufl
from dolfinx import mesh
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph, fem


def test_form_constant_edge_gradient_without_coefficient(
    unit_square_mesh: mesh.Mesh,
) -> None:
    """A constant-only form must not reference an undefined coefficient.

    The exact gradient also catches invalid mesh access and scalar assembly
    of a derivative with a test argument. The seed catches ignored input values.
    """
    domain = unit_square_mesh
    graph_ = Graph()
    c = fem.Constant(domain, ScalarType(3.0), graph=graph_)

    J_form = 0.5 * c**2 * ufl.dx(domain=domain)
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    seed = -2.5
    gradient = graph_.backprop(id(J), id(c), seed=seed)

    # The unit square has area 1, so dJ/dc at c = 3 is 3.
    assert np.isclose(gradient, seed * 3.0)


def test_form_constant_edge_gradient_with_coefficient(
    unit_square_mesh: mesh.Mesh,
) -> None:
    """The constant edge must capture c rather than the form's coefficient."""
    domain = unit_square_mesh
    graph_ = Graph()
    c = fem.Constant(domain, ScalarType(2.0), graph=graph_)

    V = fem.functionspace(domain, ("Lagrange", 1))
    weight = fem.Function(V, graph=graph_)
    weight.interpolate(lambda x: 1.0 + x[0])

    J_form = 0.5 * c**2 * weight * ufl.dx(domain=domain)
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    seed = -2.5
    gradient = graph_.backprop(id(J), id(c), seed=seed)

    # Integral of 1 + x over the unit square is 1.5, so dJ/dc = 2 * 1.5 = 3.
    # Differentiating w.r.t. weight in direction 1 would give 0.5 * 2**2 = 2.
    assert np.isclose(gradient, seed * 3.0)
