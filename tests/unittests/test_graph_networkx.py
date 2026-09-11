"""Unit tests for the networkx representation of the computational graph."""

import pytest

import dolfinx_adjoint.graph as graph
from dolfinx_adjoint.edge import Edge
from dolfinx_adjoint.node import AbstractNode, Node


def _chain():
    """Build the graph ``predecessor -> successor`` with a single edge."""
    _graph = graph.Graph()

    predecessor = AbstractNode(object(), name="predecessor")
    successor = AbstractNode(object(), name="successor")
    for node in (predecessor, successor):
        _graph.add_node(node)

    edge = Edge(predecessor, successor)
    _graph.add_edge(edge)

    return _graph, predecessor, successor, edge


@pytest.mark.parametrize("representation_built", [False, True])
def test_add_edge_rejects_parallel_edges(representation_built):
    """Reject equal endpoints without changing the graph or its representation."""
    _graph, predecessor, successor, edge = _chain()
    parallel_edge = Edge(predecessor, successor)

    if representation_built:
        _graph.to_networkx()

    with pytest.raises(ValueError):
        _graph.add_edge(parallel_edge)

    assert _graph.edges == [edge]
    assert list(_graph.to_networkx().edges(data="edge")) == [
        (id(predecessor), id(successor), edge)
    ]


def test_to_networkx_preserves_the_data_flow():
    """The networkx edges point from the predecessor to the successor.

    The nodes and edges themselves are kept in the representation, so that they can be
    accessed from it.
    """
    _graph, predecessor, successor, edge = _chain()

    nx_graph = _graph.to_networkx()

    assert nx_graph.has_edge(id(predecessor), id(successor))
    assert not nx_graph.has_edge(id(successor), id(predecessor))

    assert nx_graph.nodes[id(predecessor)]["node"] is predecessor
    assert nx_graph.nodes[id(successor)]["node"] is successor
    assert nx_graph[id(predecessor)][id(successor)]["edge"] is edge


def test_to_networkx_contains_nodes_added_after_the_first_call():
    """A node added after the representation was built is contained in it."""
    _graph, _, _, _ = _chain()

    _graph.to_networkx()

    added = AbstractNode(object(), name="added")
    _graph.add_node(added)

    assert id(added) in _graph.to_networkx()


def test_to_networkx_contains_edges_added_after_the_first_call():
    """An edge added after the representation was built is contained in it.

    Only the edge is added afterwards, independently of the node it connects to.
    """
    _graph, _, successor, _ = _chain()

    added = AbstractNode(object(), name="added")
    _graph.add_node(added)

    _graph.to_networkx()

    _graph.add_edge(Edge(successor, added))

    assert _graph.to_networkx().has_edge(id(successor), id(added))


def test_to_networkx_colours_operation_nodes_like_abstract_nodes():
    """A node without a numerical value keeps its colour when it is a derived class."""

    class OperationNode(AbstractNode):
        """Minimal test subclass standing in for the operation nodes of the package."""

    _graph = graph.Graph()

    abstract = AbstractNode(object(), name="abstract")
    operation = OperationNode(object(), name="operation")
    value = Node(object(), name="value")
    for node in (abstract, operation, value):
        _graph.add_node(node)

    nx_graph = _graph.to_networkx()

    assert (
        nx_graph.nodes[id(operation)]["color"] == nx_graph.nodes[id(abstract)]["color"]
    )
    # The colours have to stay distinguishable, so that colouring every node alike
    # does not satisfy the assertion above.
    assert nx_graph.nodes[id(value)]["color"] != nx_graph.nodes[id(abstract)]["color"]
