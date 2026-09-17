"""
Unit tests for the restriction performed by
:py:class:`dolfinx_adjoint.fem.bcs.DirichletBC_Edge`."""

import numpy as np
import pytest
from basix.ufl import element, mixed_element
from dolfinx import default_scalar_type, la, mesh
from dolfinx.common import IndexMap
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx_adjoint import Graph, fem
from dolfinx_adjoint.fem.bcs import DirichletBC_Edge
from dolfinx_adjoint.node import AbstractNode


def _restrict(input_value: PETSc.Vec, dofs) -> np.ndarray:
    """Run the restriction of a DirichletBC_Edge on the given input."""
    # The value lives in the space of the input, so its dofs are the same.
    size = input_value.getLocalSize()
    layout = la.vector(IndexMap(MPI.COMM_SELF, size), dtype=default_scalar_type)
    edge = DirichletBC_Edge(
        AbstractNode(object(), name="value"),
        AbstractNode(object(), name="bc"),
        ctx=[dofs, dofs, layout.petsc_vec],
    )
    edge.input_value = input_value
    return edge.calculate_adjoint().array


def test_restriction_keeps_only_the_controlled_dofs():
    """The gradient is masked onto the controlled dofs."""
    gradient = la.vector(IndexMap(MPI.COMM_SELF, 5), dtype=default_scalar_type)
    gradient.array[:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    dofs = np.array([0, 2, 4], dtype=np.int32)

    output = _restrict(gradient.petsc_vec, dofs)

    np.testing.assert_allclose(output, [1.0, 0.0, 3.0, 0.0, 5.0])


def test_restriction_with_two_controlled_dofs():
    """A boundary holding exactly two dofs is restricted like any other."""

    gradient = la.vector(IndexMap(MPI.COMM_SELF, 5), dtype=default_scalar_type)
    gradient.array[:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    dofs = np.array([1, 3], dtype=np.int32)

    output = _restrict(gradient.petsc_vec, dofs)

    np.testing.assert_allclose(output, [0.0, 2.0, 0.0, 4.0, 0.0])


@pytest.fixture(params=[0, 1], ids=["vector sub space", "scalar sub space"])
def collapsed_condition(request, unit_square_mesh: mesh.Mesh):
    """A tracked Dirichlet condition on a sub space of a mixed space, with its value on
    the collapsed sub space.

    DOLFINx collapses the vector sub space by rebuilding its dofmap and the scalar sub
    space by extracting it, so both parameters cover a different ownership of the dofs.
    """
    graph_ = Graph()
    domain = unit_square_mesh
    u_elem = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    p_elem = element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, mixed_element([u_elem, p_elem]))
    V_sub, _ = V.sub(request.param).collapse()

    def on_boundary(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        )

    g = fem.Function(V_sub, name="g", graph=graph_)
    dofs = fem.locate_dofs_geometrical((V.sub(request.param), V_sub), on_boundary)
    bc = fem.dirichletbc(g, dofs, V.sub(request.param), graph=graph_)

    return graph_, V, g, bc, dofs


def test_restriction_into_a_collapsed_space(collapsed_condition):
    """The restricted gradient is moved into the collapsed space along the dof pairs of
    the condition, which DOLFINx uses to set the value, u[dofs[0]] = g[dofs[1]]."""
    graph_, V, g, bc, dofs = collapsed_condition

    gradient = la.vector(
        V.dofmap.index_map, V.dofmap.index_map_bs, dtype=default_scalar_type
    )
    gradient.array[:] = np.random.default_rng(V.mesh.comm.rank).random(
        gradient.array.size
    )

    edge = graph_.get_edge(graph_.get_node(id(g)), graph_.get_node(id(bc)))
    edge.input_value = gradient.petsc_vec
    output = edge.calculate_adjoint()

    # The pairs are sorted by the dofs in V, so the first num_owned pairs are owned.
    _, num_owned = bc.dof_indices()
    expected = np.zeros(output.getLocalSize(), dtype=default_scalar_type)
    expected[dofs[1][:num_owned]] = gradient.array[dofs[0][:num_owned]]

    assert V.mesh.comm.allreduce(np.allclose(output.array, expected), op=MPI.LAND)
