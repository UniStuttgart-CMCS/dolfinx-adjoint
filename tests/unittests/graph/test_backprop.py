"""Unit tests for the backpropagation through the computational graph."""

import numpy as np
import pytest
from petsc4py import PETSc

import dolfinx_graph_ad.graph as graph
from dolfinx_graph_ad.graph import Edge, Node


class ScaledEdge(Edge):
    """Test double with a constant derivative."""

    def __init__(self, predecessor: Node, successor: Node, factor: float = 1.0):
        super().__init__(predecessor, successor)
        self.factor = factor

    def calculate_adjoint(self, value):
        return value * self.factor


def _build(edges: list):
    """Build a graph from ``(name, predecessor, successor, factor)`` tuples.

    The edges are given in the order in which the forward operations are performed.
    Returns the graph and the objects, nodes and edges, the last three as dictionaries
    of names.
    """
    names = []
    for _, predecessor, successor, _ in edges:
        for name in (predecessor, successor):
            if name not in names:
                names.append(name)

    _graph = graph.Graph()
    objects = {}
    nodes = {}
    for name in names:
        objects[name] = object()
        nodes[name] = Node(objects[name], name=name)
        _graph.add_node(nodes[name])

    built_edges = {}
    for name, predecessor, successor, factor in edges:
        edge = ScaledEdge(nodes[predecessor], nodes[successor], factor=factor)
        nodes[successor].grad_fns.append(edge)
        _graph.add_edge(edge)
        built_edges[name] = edge

    return _graph, objects, nodes, built_edges


@pytest.fixture
def single_edge_graph():
    """A single operation with derivative two."""
    return _build([("e1", "variable", "objective", 2.0)])


@pytest.fixture
def chain_graph():
    """Two operations with derivatives two and three."""
    return _build(
        [
            ("e1", "variable", "mid", 2.0),
            ("e2", "mid", "objective", 3.0),
        ]
    )


@pytest.fixture
def two_input_graph():
    """Two controls, with downstream and disconnected operations to exclude."""
    return _build(
        [
            ("e1", "variable", "form", 2.0),
            ("other", "other_input", "form", 3.0),
            ("e_form", "form", "objective", 1.0),
            ("post", "objective", "post_processing", 5.0),
            ("unrelated", "unrelated_input", "unrelated_output", 7.0),
        ]
    )


@pytest.fixture(params=[1.0, 5.0], ids=["unit_seed", "scaled_seed"])
def seed(request):
    """A non-unit seed exposes ignored seeds and hardcoded self-derivatives."""
    return request.param


# Gradient values and storage


def test_backprop_repeats_the_chain_derivative_without_accumulating(chain_graph, seed):
    """Each call scales the chain derivative by its seed, without stale gradients."""
    _graph, objects, _, _ = chain_graph

    gradients = [
        _graph.backprop(objects["objective"], objects["variable"], grad_outputs=seed)[0]
        for _ in range(2)
    ]

    assert gradients == pytest.approx([6.0 * seed, 6.0 * seed])


# Selection of controls


def test_backprop_updates_gradients_when_switching_controls_and_modes(two_input_graph):
    """Catch stale gradients or branches lost when a previous call skipped them."""
    _graph, objects, nodes, _ = two_input_graph

    # Switch controls directly, then expand to all dependencies and restrict again.
    gradients = []
    for control in ("variable", "other_input", None, "variable"):
        variable = None if control is None else objects[control]
        _graph.backprop(objects["objective"], variable)
        gradients.append((nodes["variable"].grad, nodes["other_input"].grad))

    assert gradients == [(2.0, None), (None, 3.0), (2.0, 3.0), (2.0, None)]


# Several controls in one propagation


@pytest.mark.parametrize(
    "controls", [("outer", "inner"), ("inner", "outer"), ("outer", "latest")]
)
def test_backprop_returns_the_total_derivative_of_nested_controls(controls):
    """Catch conflating versions or losing a path into an intermediate control."""
    # Propagating the union of the paths without capturing the inner control leaves it
    # without a gradient, stopping at the inner control drops the outer control's path
    # through it, and following only the first control's path loses the outer control
    # in the second order.
    _graph = graph.Graph()
    value = object()
    nodes = {
        "outer": Node(value),
        "inner": Node(value, version=1),
        "objective": Node(object()),
    }
    for node in nodes.values():
        _graph.add_node(node)
    for start, end, factor in (
        ("outer", "inner", 2.0),
        ("inner", "objective", 3.0),
        ("outer", "objective", 5.0),
    ):
        edge = ScaledEdge(nodes[start], nodes[end], factor=factor)
        nodes[end].grad_fns.append(edge)
        _graph.add_edge(edge)
    # The object selects the latest version; an exact node selects its own version.
    inputs = {"outer": nodes["outer"], "inner": nodes["inner"], "latest": value}
    expected = {"outer": 2.0 * 3.0 + 5.0, "inner": 3.0, "latest": 3.0}

    gradients = _graph.backprop(nodes["objective"], [inputs[name] for name in controls])

    assert gradients == pytest.approx(tuple(expected[name] for name in controls))


@pytest.mark.parametrize(
    "controls", [("objective",), ("variable", "objective")], ids=["alone", "with_input"]
)
def test_backprop_returns_the_seed_as_the_derivative_of_the_function_itself(
    single_edge_graph, seed, controls
):
    """Catch a function that loses or hardcodes its self-derivative.

    Alone, the path is empty, so the seed is only stored if the function is captured
    without an edge being executed. With its input, the function's own input edge is on
    the path, so the seed is only stored if the function is captured as a control rather
    than as the end of a path.
    """
    _graph, objects, _, _ = single_edge_graph
    expected = {"variable": 2.0 * seed, "objective": seed}

    gradients = _graph.backprop(
        objects["objective"], [objects[name] for name in controls], grad_outputs=seed
    )

    assert gradients == pytest.approx(tuple(expected[name] for name in controls))


def test_backprop_returns_a_gradient_that_does_not_share_the_seed(single_edge_graph):
    """Catch a gradient stored as the vector it arrived in.

    The derivative of a function with respect to itself arrives as the seed, which the
    caller still holds, so scaling the returned gradient, e.g. for an optimisation step,
    would change the seed of the next call.
    """
    _graph, objects, _, _ = single_edge_graph
    seed = PETSc.Vec().createSeq(1, comm=PETSc.COMM_SELF)
    seed.set(1.0)

    (gradient,) = _graph.backprop(
        objects["objective"], objects["objective"], grad_outputs=seed
    )
    gradient.scale(3.0)

    np.testing.assert_allclose(seed.array_r, [1.0])


@pytest.mark.parametrize(
    "controls", [("variable", "other_input"), ("other_input", "variable")]
)
def test_backprop_returns_each_controls_derivative_in_requested_order(
    two_input_graph, controls
):
    """Catch dropping a control or returning gradients in recording order."""
    _graph, objects, _, _ = two_input_graph
    expected = {"variable": 2.0, "other_input": 3.0}

    gradients = _graph.backprop(
        objects["objective"], [objects[name] for name in controls]
    )

    assert gradients == pytest.approx(tuple(expected[name] for name in controls))


@pytest.mark.parametrize("next_control", ["variable", None])
def test_backprop_clears_the_gradients_of_the_previous_controls(
    chain_graph, next_control
):
    """A former control keeps no gradient once the next call selects other controls.

    Catch a gradient that is not reset between calls, and a control of an earlier call
    that keeps storing gradients in the node although it is only passed through.
    """
    _graph, objects, nodes, _ = chain_graph
    _graph.backprop(objects["objective"], [objects["variable"], objects["mid"]])

    variable = None if next_control is None else objects[next_control]
    _graph.backprop(objects["objective"], variable)

    assert nodes["mid"].grad is None


# Shared branches


def test_backprop_sums_both_paths_through_a_shared_input():
    """Catch losing or double-counting a branch at a shared intermediate value."""
    _graph, objects, _, _ = _build(
        [
            ("e_in", "variable", "mid", 2.0),
            ("e_upper", "mid", "upper", 3.0),
            ("e_lower", "mid", "lower", 5.0),
            ("e_upper_out", "upper", "objective", 7.0),
            ("e_lower_out", "lower", "objective", 11.0),
        ]
    )
    (gradient,) = _graph.backprop(objects["objective"], objects["variable"])

    assert gradient == pytest.approx(2.0 * (3.0 * 7.0 + 5.0 * 11.0))


def test_backprop_includes_dependencies_added_after_a_previous_call(chain_graph):
    """Catch stale topology silently omitting a newly recorded derivative path."""
    _graph, objects, nodes, _ = chain_graph
    (before,) = _graph.backprop(objects["objective"], objects["variable"])

    added = Node(object())
    _graph.add_node(added)
    for predecessor, successor, factor in (
        (nodes["objective"], added, 7.0),
        (nodes["mid"], added, 5.0),
    ):
        edge = ScaledEdge(predecessor, successor, factor=factor)
        successor.grad_fns.append(edge)
        _graph.add_edge(edge)

    (after,) = _graph.backprop(added, objects["variable"])

    assert (before, after) == pytest.approx((6.0, 2.0 * (3.0 * 7.0 + 5.0)))


# Several functions in one propagation


@pytest.fixture
def two_output_graph():
    """Two functions of one control, sharing the control's input operation."""
    return _build(
        [
            ("e_shared", "variable", "form", 2.0),
            ("e_first", "form", "first", 3.0),
            ("e_second", "form", "second", 5.0),
        ]
    )


def test_backprop_weights_several_functions_by_their_seeds(two_output_graph):
    """The gradient of several functions is the sum of theirs, weighted by the seeds."""
    # Seeding only the first function, or pairing the seeds with the wrong functions,
    # returns a gradient without error.
    _graph, objects, _, _ = two_output_graph

    gradients = _graph.backprop(
        [objects["first"], objects["second"]],
        objects["variable"],
        grad_outputs=[7.0, 11.0],
    )

    assert gradients == pytest.approx((2.0 * (3.0 * 7.0 + 5.0 * 11.0),))


def test_topological_order_follows_the_edges_and_breaks_ties_by_recording():
    """The order of a propagation is the unique topological order of the nodes the
    function depends on, with ties broken by recording order."""
    # p is recorded before its input a, like a problem recorded by its constructor.
    # Taking the recording order puts p before a, so backprop, which reverses the order,
    # passes on the adjoint of a before p has added its contribution to it, without
    # error. Taking networkx's topological_sort puts b before p, an order that holds on
    # every rank only by networkx's iteration order. Not restricting the order to the
    # dependencies of f also visits x.
    _graph, _, nodes, _ = _build(
        [
            ("e1", "p", "c", 1.0),
            ("e2", "a", "p", 1.0),
            ("e3", "b", "f", 1.0),
            ("e4", "c", "f", 1.0),
            ("e5", "a", "x", 1.0),
        ]
    )

    order = _graph._topological_order([nodes["f"]])

    assert order == [nodes[name] for name in ("a", "p", "c", "b", "f")]
