"""Object lookup must preserve versions, node ownership and mesh identity."""

import pytest

from dolfinx_adjoint import Graph
from dolfinx_adjoint.node import Node


@pytest.fixture
def recorded_versions():
    graph = Graph()
    value = object()
    nodes = [Node(value), Node(object()), Node(value, version=3)]
    for node in nodes:
        graph.add_node(node)
    return graph, value, nodes


@pytest.mark.parametrize("version", [None, 0, 3, 4])
def test_object_lookup_selects_version(recorded_versions, version):
    """Catch lost versions or regrouping interleaved nodes by object identity."""
    graph, value, nodes = recorded_versions
    expected = {None: nodes[2], 0: nodes[0], 3: nodes[2], 4: None}[version]

    assert (graph.get_node(value, version=version), list(graph.nodes)) == (
        expected,
        nodes,
    )


@pytest.mark.parametrize("options", [{}, {"version": None}])
def test_exact_node_lookup_preserves_identity(recorded_versions, options):
    """Catch an exact node resolving to a different version of the same object."""
    graph, _, nodes = recorded_versions

    assert graph.get_node(nodes[0], **options) is nodes[0]


@pytest.mark.parametrize("version", [0, 3, 4])
def test_exact_node_lookup_rejects_version(recorded_versions, version):
    """Catch accepting a version with an exact node, even when it matches."""
    graph, _, nodes = recorded_versions

    with pytest.raises(
        TypeError, match="version cannot be supplied with an exact node"
    ):
        graph.get_node(nodes[0], version=version)


def test_foreign_node_lookup_is_rejected(recorded_versions):
    """Catch accepting a foreign node merely because its ID and version match."""
    graph, value, _ = recorded_versions
    other = Graph()
    foreign = Node(value)
    other.add_node(foreign)

    assert graph.get_node(foreign) is None
