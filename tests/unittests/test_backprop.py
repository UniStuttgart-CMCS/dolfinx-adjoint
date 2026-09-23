"""Unit tests for the backpropagation through the computational graph."""

import pytest

import dolfinx_adjoint.graph as graph
from dolfinx_adjoint.edge import Edge
from dolfinx_adjoint.node import AbstractNode, Node


class LinearEdge(Edge):
    """Test double with a constant derivative and a flag indicating execution."""

    def __init__(self, predecessor: Node, successor: Node, factor: float = 1.0):
        super().__init__(predecessor, successor)
        self.factor = factor
        self.called = False

    def calculate_adjoint(self):
        self.called = True
        return self.input_value * self.factor


def _build(edges: list, node_types: dict = None):
    """Build a graph from ``(name, predecessor, successor, factor)`` tuples.

    The edges are given in the order in which the forward operations are performed and
    ``node_types`` overrides the node class of a node. Returns the graph and the objects,
    nodes and edges, the last three as dictionaries of names.
    """
    node_types = node_types or {}

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
        nodes[name] = node_types.get(name, Node)(objects[name], name=name)
        _graph.add_node(nodes[name])

    built_edges = {}
    for name, predecessor, successor, factor in edges:
        edge = LinearEdge(nodes[predecessor], nodes[successor], factor=factor)
        nodes[successor].append_gradFuncs(edge)
        edge.set_next_functions(nodes[predecessor].get_gradFuncs())
        _graph.add_edge(edge)
        built_edges[name] = edge

    return _graph, objects, nodes, built_edges


def _executed_edges(edges: dict):
    """Get the names of the edges that have been evaluated."""
    return {name for name, edge in edges.items() if edge.called}


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


def test_backprop_repeats_the_chain_derivative_without_accumulating(chain_graph):
    """Each call returns the product of the derivatives, without stale gradients."""
    _graph, objects, _, _ = chain_graph

    gradients = [
        _graph.backprop(id(objects["objective"]), id(objects["variable"]))
        for _ in range(2)
    ]

    assert gradients == pytest.approx([6.0, 6.0])


def test_backprop_sums_parallel_paths():
    """Gradients of parallel paths are accumulated in the variable."""
    _graph, objects, _, _ = _build(
        [
            ("e_upper_in", "variable", "upper", 2.0),
            ("e_upper_out", "upper", "form", 3.0),
            ("e_lower_in", "variable", "lower", 5.0),
            ("e_lower_out", "lower", "form", 7.0),
            ("e_form", "form", "objective", 1.0),
        ]
    )

    gradient = _graph.backprop(id(objects["objective"]), id(objects["variable"]))

    assert gradient == pytest.approx(2.0 * 3.0 + 5.0 * 7.0)


def test_backprop_stores_the_gradient_in_an_intermediate_control(chain_graph):
    """A selected intermediate node receives the gradient; its input does not."""
    _graph, objects, nodes, _ = chain_graph

    gradient = _graph.backprop(id(objects["objective"]), id(objects["mid"]))

    assert gradient == nodes["mid"].get_grad() == pytest.approx(3.0)
    assert nodes["variable"].get_grad() is None


def test_backprop_accumulates_in_specialised_nodes():
    """Nodes of a derived node class receive their gradient as well."""

    class DerivedNode(Node):
        """Minimal test subclass for checking inherited gradient accumulation."""

    _graph, objects, _, _ = _build(
        [
            ("e1", "variable", "objective", 2.0),
        ],
        node_types={"variable": DerivedNode},
    )

    gradient = _graph.backprop(id(objects["objective"]), id(objects["variable"]))

    assert gradient == pytest.approx(2.0)


def test_backprop_seeds_all_gradient_functions_of_the_function():
    """A function that is reached by several edges is seeded on all of them."""
    # The paths merge in the function itself, so the function has two gradient
    # functions, in contrast to the paths merging in an intermediate node.
    _graph, objects, _, _ = _build(
        [
            ("e_upper_in", "variable", "upper", 2.0),
            ("e_upper_out", "upper", "objective", 3.0),
            ("e_lower_in", "variable", "lower", 5.0),
            ("e_lower_out", "lower", "objective", 7.0),
        ]
    )

    gradient = _graph.backprop(id(objects["objective"]), id(objects["variable"]))

    assert gradient == pytest.approx(2.0 * 3.0 + 5.0 * 7.0)


def test_backprop_without_a_variable_stores_gradients_only_in_dependency_leaves(
    two_input_graph,
):
    """Unrestricted propagation stores gradients in the objective's dependency leaves."""
    _graph, objects, nodes, _ = two_input_graph

    assert _graph.backprop(id(objects["objective"])) is None

    assert nodes["variable"].get_grad() == pytest.approx(2.0)
    assert nodes["other_input"].get_grad() == pytest.approx(3.0)
    assert nodes["form"].get_grad() is None
    assert nodes["unrelated_input"].get_grad() is None
    assert nodes["post_processing"].get_grad() is None


def test_backprop_clears_gradients_when_switching_controls(two_input_graph):
    """Selecting another control clears the previously stored gradient."""
    _graph, objects, nodes, _ = two_input_graph

    _graph.backprop(id(objects["objective"]), id(objects["variable"]))
    assert nodes["variable"].get_grad() == pytest.approx(2.0)

    _graph.backprop(id(objects["objective"]), id(objects["other_input"]))
    assert nodes["variable"].get_grad() is None


def test_backprop_scales_the_derivative_with_the_seed(chain_graph, seed):
    """The seed is the adjoint value of the function and scales the derivative."""
    _graph, objects, _, _ = chain_graph

    gradient = _graph.backprop(
        id(objects["objective"]), id(objects["variable"]), seed=seed
    )

    assert gradient == pytest.approx(seed * 6.0)


def test_backprop_of_the_function_with_respect_to_itself(single_edge_graph, seed):
    """The derivative of the function with respect to itself is the seed."""
    _graph, objects, nodes, _ = single_edge_graph

    gradient = _graph.backprop(
        id(objects["objective"]), id(objects["objective"]), seed=seed
    )

    assert gradient == nodes["objective"].get_grad() == pytest.approx(seed)


# Execution of marked edges


def test_backprop_skips_unmarked_objective_inputs():
    """Only the selected input's edge is executed when seeding the objective."""
    # Select the second input to catch propagation that always seeds the first.
    _graph, objects, _, edges = _build(
        [
            ("other", "other_input", "objective", 3.0),
            ("e_variable", "variable", "objective", 2.0),
        ]
    )

    _graph.backprop(id(objects["objective"]), id(objects["variable"]))

    assert _executed_edges(edges) == {"e_variable"}


@pytest.mark.parametrize("control, expected", [("mid", {"e2"}), ("objective", set())])
def test_backprop_stops_execution_at_the_selected_control(
    chain_graph, control, expected
):
    """The selected control's own input operations must not execute."""
    _graph, objects, _, edges = chain_graph

    _graph.backprop(id(objects["objective"]), id(objects[control]))

    assert _executed_edges(edges) == expected


def test_backprop_updates_execution_when_switching_controls_and_modes(two_input_graph):
    """Each query executes its own paths without losing previously skipped branches."""
    _graph, objects, _, edges = two_input_graph

    # Switch controls directly, then expand to all dependencies and restrict again.
    for control, expected in (
        ("variable", {"e1", "e_form"}),
        ("other_input", {"other", "e_form"}),
        (None, {"e1", "other", "e_form"}),
        ("variable", {"e1", "e_form"}),
    ):
        for edge in edges.values():
            edge.called = False

        variable_id = None if control is None else id(objects[control])
        _graph.backprop(id(objects["objective"]), variable_id)

        assert _executed_edges(edges) == expected, f"control={control!r}"


# Validation of the arguments


def test_backprop_rejects_an_unknown_function(single_edge_graph):
    """An unregistered function is reported with the id that has been passed."""
    _graph, objects, _, _ = single_edge_graph

    # Report the supplied id, rather than id(None) from a failed node lookup.
    unregistered = object()
    with pytest.raises(ValueError, match=str(id(unregistered))):
        _graph.backprop(id(unregistered), id(objects["variable"]))


def test_backprop_rejects_an_unknown_variable(single_edge_graph):
    """An unregistered variable is reported with the id that has been passed."""
    _graph, objects, _, _ = single_edge_graph

    unregistered = object()
    with pytest.raises(ValueError, match=str(id(unregistered))):
        _graph.backprop(id(objects["objective"]), id(unregistered))


def test_backprop_with_an_unknown_argument_keeps_the_previous_gradients(
    single_edge_graph,
):
    """A call that is rejected does not discard the gradients of the previous call."""
    _graph, objects, nodes, _ = single_edge_graph

    _graph.backprop(id(objects["objective"]), id(objects["variable"]))
    assert nodes["variable"].get_grad() == pytest.approx(2.0)

    unregistered = object()
    with pytest.raises(ValueError):
        _graph.backprop(id(objects["objective"]), id(unregistered))

    assert nodes["variable"].get_grad() == pytest.approx(2.0)


def test_backprop_rejects_a_variable_that_cannot_store_a_gradient():
    """A variable without a numerical value is rejected before the propagation."""
    _graph, objects, _, edges = _build(
        [
            ("e1", "variable", "objective", 2.0),
        ],
        node_types={"variable": AbstractNode},
    )

    with pytest.raises(TypeError):
        _graph.backprop(id(objects["objective"]), id(objects["variable"]))

    assert _executed_edges(edges) == set()


def test_backprop_rejects_a_variable_the_function_does_not_depend_on(chain_graph):
    """A variable the function does not depend on is rejected instead of yielding None."""
    _graph, objects, nodes, _ = chain_graph

    with pytest.raises(ValueError):
        _graph.backprop(id(objects["mid"]), id(objects["objective"]))

    assert nodes["mid"].get_grad() is None
