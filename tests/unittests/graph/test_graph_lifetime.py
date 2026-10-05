"""Unit tests for the lifetime of the computational graph."""

import weakref

import pytest
from dolfinx import fem

from dolfinx_graph_ad import Edge, Graph, Node
from dolfinx_graph_ad import fem as fem_ad


class _Value:
    """A stand-in value like ``object()``, which unlike it can be referenced weakly."""


@pytest.mark.parametrize(
    "versions, expected",
    [((None, None), [0, 1, 2]), ((3, None), [0, 3, 4]), ((3, 7), [0, 3, 7])],
)
def test_assigning_a_function_from_itself_records_no_self_loop(
    unit_square_mesh, versions, expected
):
    """Catch reused default versions, ignored explicit versions, or self-loop edges."""
    graph_ = Graph()
    V = fem.functionspace(unit_square_mesh, ("Lagrange", 1))
    u = fem_ad.Function(V, name="u", graph=graph_)

    nodes = [graph_.get_node(u)]
    for version in versions:
        options = {} if version is None else {"version": version}
        u.assign(u, graph=graph_, **options)
        nodes.append(graph_.get_node(u))

    assert (
        [node.version for node in nodes],
        [(edge.predecessor, edge.successor) for edge in graph_.edges],
    ) == (expected, list(zip(nodes, nodes[1:])))


def test_release_frees_what_an_edge_saved_while_a_node_is_still_held():
    """Releasing frees the context of an edge, although the caller still holds its node."""
    graph_ = Graph()
    control = Node(object(), name="control")
    output = Node(object(), name="output")
    for node in (control, output):
        graph_.add_node(node)
    saved = _Value()
    edge = Edge(control, output, ctx=[saved])
    output.grad_fns.append(edge)
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
