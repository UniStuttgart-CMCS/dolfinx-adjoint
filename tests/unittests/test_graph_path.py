"""Unit tests for the path marking of the computational graph."""

import pytest

import dolfinx_adjoint.graph as graph
from dolfinx_adjoint.edge import Edge
from dolfinx_adjoint.node import AbstractNode


def _build(edges: dict):
    """Build a graph from ``{name: (predecessor, successor)}``, given in flow direction."""

    names = []
    for predecessor, successor in edges.values():
        for name in (predecessor, successor):
            if name not in names:
                names.append(name)

    _graph = graph.Graph()
    nodes = {}
    for name in names:
        nodes[name] = AbstractNode(object(), name=name)
        _graph.add_node(nodes[name])

    built_edges = {}
    for key, (predecessor, successor) in edges.items():
        built_edges[key] = Edge(nodes[predecessor], nodes[successor])
        _graph.add_edge(built_edges[key])

    return _graph, nodes, built_edges

def _assert_marked(edges: dict, expected: set):
    """Assert that exactly the edges in ``expected`` are marked."""
    marked = {key for key, edge in edges.items() if getattr(edge, "marked", False)}
    assert marked == expected


def test_get_path_does_not_mark_unrelated_branches():
    """Only the edges connecting the variable to the objective are marked."""
    # variable -> mid -> objective, with a second input to mid, a dead end
    # branching off mid, an operation feeding the variable and an operation
    # performed on the objective.
    _graph, nodes, edges = _build(
        {
            "e1": ("variable", "mid"),
            "e2": ("mid", "objective"),
            "other": ("other_input", "mid"),
            "dead_end": ("mid", "dead_end"),
            "upstream": ("upstream", "variable"),
            "post_processing": ("objective", "post_processing"),
        }
    )

    _graph.get_path(id(nodes["variable"]), id(nodes["objective"]))

    _assert_marked(edges, {"e1", "e2"})


def test_get_path_marks_all_parallel_paths():
    """Every path from the variable to the objective is marked."""
    # variable -> {upper, lower} -> objective and variable -> objective, so that
    # paths of different lengths connect the variable and the objective
    _graph, nodes, edges = _build(
        {
            "e_upper_in": ("variable", "upper"),
            "e_upper_out": ("upper", "objective"),
            "e_lower_in": ("variable", "lower"),
            "e_lower_out": ("lower", "objective"),
            "e_direct": ("variable", "objective"),
        }
    )

    _graph.get_path(id(nodes["variable"]), id(nodes["objective"]))

    _assert_marked(
        edges,
        {"e_upper_in", "e_upper_out", "e_lower_in", "e_lower_out", "e_direct"},
    )


def test_get_path_with_no_connection_marks_none():
    """A variable that the objective does not depend on marks no edge.

    The disconnected query is performed before and after a successful query, so that
    it also covers the complete clearing of the marks of the successful query.
    """

    _graph, nodes, edges = _build(
        {
            "e1": ("variable", "mid"),
            "e2": ("mid", "objective"),
            "e3": ("other_input", "other_output"),
        }
    )

    _graph.get_path(id(nodes["other_input"]), id(nodes["objective"]))
    _assert_marked(edges, set())

    _graph.get_path(id(nodes["variable"]), id(nodes["objective"]))
    _assert_marked(edges, {"e1", "e2"})

    _graph.get_path(id(nodes["other_input"]), id(nodes["objective"]))
    _assert_marked(edges, set())


def test_get_path_resets_previous_marks():
    """A second call to get_path does not keep the marks of the first call."""
    _graph, nodes, edges = _build(
        {
            "e1": ("variable", "mid"),
            "e2": ("mid", "objective"),
            "e3": ("other_input", "mid"),
        }
    )

    _graph.get_path(id(nodes["variable"]), id(nodes["objective"]))
    _assert_marked(edges, {"e1", "e2"})

    _graph.get_path(id(nodes["other_input"]), id(nodes["objective"]))
    _assert_marked(edges, {"e3", "e2"})


def test_get_path_in_reverse_marks_nothing():
    """The path is directed: querying it against the data flow marks no edge."""
    _graph, nodes, edges = _build(
        {
            "e1": ("variable", "mid"),
            "e2": ("mid", "objective"),
        }
    )

    # Variable and objective are exchanged, so no forward path exists
    _graph.get_path(id(nodes["objective"]), id(nodes["variable"]))

    _assert_marked(edges, set())


def test_get_path_with_isolated_control_marks_nothing():
    """A control that is registered in the graph but unused marks no edge."""
    _graph, nodes, edges = _build(
        {
            "e1": ("variable", "mid"),
            "e2": ("mid", "objective"),
        }
    )

    # A control that has been added to the graph but is not used in any operation
    isolated = AbstractNode(object(), name="isolated")
    _graph.add_node(isolated)

    _graph.get_path(id(isolated), id(nodes["objective"]))

    _assert_marked(edges, set())


def test_get_path_uses_nodes_added_after_the_first_call():
    """The path marking sees the nodes and edges added after the first call."""
    _graph, nodes, edges = _build({"e1": ("predecessor", "successor")})

    _graph.get_path(id(nodes["predecessor"]), id(nodes["successor"]))

    # Extend the graph after the networkx representation has been built
    added = AbstractNode(object(), name="added")
    _graph.add_node(added)
    added_edge = Edge(nodes["successor"], added)
    _graph.add_edge(added_edge)

    _graph.get_path(id(nodes["predecessor"]), id(added))

    assert edges["e1"].marked is True
    assert added_edge.marked is True


def test_get_path_with_an_unknown_node_keeps_the_previous_marks():
    """A query that raises does not change the marking of the previous query."""
    _graph, nodes, edges = _build(
        {
            "e1": ("variable", "mid"),
            "e2": ("mid", "objective"),
        }
    )

    _graph.get_path(id(nodes["variable"]), id(nodes["objective"]))

    unregistered = AbstractNode(object(), name="unregistered")
    with pytest.raises(ValueError):
        _graph.get_path(id(unregistered), id(nodes["objective"]))

    _assert_marked(edges, {"e1", "e2"})


def test_get_dependencies_marks_only_dependencies_and_clears_previous_marks():
    """Dependency marking excludes downstream edges and clears unrelated marks."""
    _graph, nodes, edges = _build(
        {
            "e1": ("variable", "form"),
            "other": ("other_input", "form"),
            "e_form": ("form", "objective"),
            "post": ("objective", "post_processing"),
            "unrelated": ("unrelated_input", "unrelated_output"),
        }
    )

    _graph.get_path(id(nodes["unrelated_input"]), id(nodes["unrelated_output"]))
    _assert_marked(edges, {"unrelated"})

    _graph.get_dependencies(id(nodes["objective"]))

    _assert_marked(edges, {"e1", "other", "e_form"})
