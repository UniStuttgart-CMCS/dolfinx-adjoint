"""Unit tests for the lifetime of the computational graph."""

import weakref

import pytest

from dolfinx_adjoint import Edge, Graph, Node, fem


class _Value:
    """A stand-in value like ``object()``, which unlike it can be referenced weakly."""


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


def test_release_frees_what_an_edge_saved_while_a_node_is_still_held():
    """Releasing frees the context of an edge, although the caller still holds its node."""
    graph_ = Graph()
    control = Node(object(), name="control")
    output = Node(object(), name="output")
    for node in (control, output):
        graph_.add_node(node)
    saved = _Value()
    edge = Edge(control, output, ctx=[saved])
    output.append_gradFuncs(edge)
    graph_.add_edge(edge)
    saved = weakref.ref(saved)
    del edge

    graph_.release()

    assert saved() is None


def test_release_frees_the_object_of_a_node_while_the_node_is_still_held():
    """Releasing frees the object of a node, although the caller still holds the node."""
    graph_ = Graph()
    node = Node(_Value(), name="node")
    graph_.add_node(node)
    held = weakref.ref(node.object)

    graph_.release()

    assert held() is None
