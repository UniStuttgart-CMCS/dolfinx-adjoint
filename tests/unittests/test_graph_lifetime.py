"""Unit tests for the lifetime of the computational graph."""

import pytest

from dolfinx_adjoint import Edge, Graph, Node, fem


def test_assigning_a_function_from_itself_records_no_self_loop(unit_square_mesh):
    """An assignment of a function to itself records no edge that closes on its own node."""
    graph_ = Graph()
    V = fem.functionspace(unit_square_mesh, ("Lagrange", 1))
    u = fem.Function(V, name="u", graph=graph_)

    u.assign(u, graph=graph_)

    self_loops = [edge for edge in graph_.edges if edge.predecessor is edge.successor]
    assert not self_loops


def test_backprop_rejects_a_graph_with_a_cycle():
    """Backpropagation refuses a graph in which an edge leads back to its predecessor."""
    graph_ = Graph()
    first = Node(object(), name="first")
    second = Node(object(), name="second")
    for node in (first, second):
        graph_.add_node(node)
    graph_.add_edge(Edge(first, second))
    graph_.add_edge(Edge(second, first))

    with pytest.raises(RuntimeError, match="cycle"):
        graph_.backprop(id(second.object))
