"""Regressions for exact Function assignment versions."""

import pytest
import ufl
from dolfinx import fem

from dolfinx_adjoint import Graph
from dolfinx_adjoint import fem as fem_ad


@pytest.mark.parametrize("versions", [(None, None), (3, None), (3, 7)])
def test_function_self_assignment_preserves_the_original_controls_derivative(
    unit_square_mesh_per_comm, versions
):
    """Catch assignments losing the original control or skipping nonadjacent versions."""
    graph_ = Graph()
    V = fem.functionspace(unit_square_mesh_per_comm, ("Lagrange", 1))
    u = fem_ad.Function(V, name="u")
    u.x.array[:] = 2.0
    graph_.track(u)
    direction = fem_ad.Function(V)
    direction.x.array[:] = 1.0

    try:
        original = graph_.get_node(u)
        for version in versions:
            u.assign(u, graph=graph_, version=version)

        J = fem_ad.assemble_scalar(
            fem_ad.form(u**2 * ufl.dx, graph=graph_), graph=graph_
        )
        (gradient,) = graph_.backprop(J, original)

        assert gradient.dot(direction.x.petsc_vec) == pytest.approx(4.0)
    finally:
        graph_.release()
