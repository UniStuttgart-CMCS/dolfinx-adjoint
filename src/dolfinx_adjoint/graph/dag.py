import gc
import os
from collections.abc import Sequence, ValuesView
from typing import Any, Callable

import networkx as nx
from dolfinx.mesh import Mesh
from networkx import DiGraph

from .edge import Edge
from .node import AbstractNode, Node
from .values import add


class Graph:
    """Class for the computational graph

    The computational graph is a directed acyclic graph (DAG) that represents the
    operations, objects and dependencies in a forward simulation in DOLFINx.

    Attributes:
        nodes (ValuesView): Nodes in recording order, added through :py:meth:`add_node`.
        edges (ValuesView): Edges in recording order, added through :py:meth:`add_edge`.

    Example:
        The graph object can be initialised and the default nodes and edges can be added as follows:
        >>> graph = Graph()
        >>> node1 = Node(object_1, name = "Node 1")
        >>> node2 = Node(object_2, name = "Node 2")
        >>> edge = Edge(node1, node2)
        >>> graph.add_node(node1)
        >>> graph.add_node(node2)
        >>> graph.add_edge(edge)

        The derivatives can then be caculated using the backpropagation method:
        >>> graph.backprop(object_2, object_1)

    """

    def __init__(self):
        """Constructor for the Graph class

        The constructor initialises the lists to store the nodes and edges of the graph.

        """
        # Exact-version lookup and iteration share the recording order.
        self._nodes = {}
        self._latest = {}
        # Keyed by their endpoints so recording a time loop needs no list scans.
        self._edges = {}

        # Cached networkx graph for fast descendants/ancestors queries
        self._nx_graph = None

    @property
    def nodes(self) -> ValuesView[AbstractNode]:
        """The recorded nodes, in insertion order.

        Returns:
            ValuesView: A live view of all versions. Add nodes with :py:meth:`add_node`.
        """
        return self._nodes.values()

    @property
    def edges(self) -> ValuesView[Edge]:
        """The recorded edges, in insertion order.

        Returns:
            ValuesView: A live view of all edges. Add edges with :py:meth:`add_edge`.
        """
        return self._edges.values()

    def add_node(self, node: Node):
        """Add a node to the graph

        Args:
            node (Node): The node to be added to the graph

        """
        latest = self._latest.get(node.id)
        if latest is not None and node.version <= latest.version:
            raise ValueError(
                f"Version {node.version} of {node} must follow its latest recorded "
                f"version {latest.version}."
            )
        self._nodes[node.id, node.version] = node
        self._latest[node.id] = node
        if self._nx_graph is not None:
            self._add_node_to_networkx(self._nx_graph, node)

    def add_edge(self, edge: Edge):
        """Add an edge to the graph

        Args:
            edge (Edge): The edge to be added to the graph

        Raises:
            ValueError: If an edge already connects the predecessor to the successor

        Note:
            Both nodes of the edge have to be part of the graph, since the graph is what
            keeps them alive: an edge references its successor weakly, so that the graph
            stays free of reference cycles.

        """
        key = (edge.predecessor, edge.successor)
        if key in self._edges:
            raise ValueError(
                f"An edge from {edge.predecessor} to {edge.successor} is already part of the graph."
            )
        self._edges[key] = edge
        if self._nx_graph is not None:
            self._add_edge_to_networkx(self._nx_graph, edge)

    def get_node(self, value, version=None) -> AbstractNode | None:
        """Get the node of a recorded object or an exact node owned by this graph.

        Args:
            value: The recorded object, mesh or exact node.
            version (int, optional): The recorded version of an object. Defaults to
                its latest version. Must be None when value is an exact node, which
                already identifies its version.

        Returns:
            AbstractNode or None: The matching node, or None if it is not recorded
            in this graph. An exact node must match by identity.

        Raises:
            TypeError: If a non-None version is supplied with an exact node.
        """

        if isinstance(value, AbstractNode):
            if version is not None:
                raise TypeError("Version cannot be supplied with an exact node.")
            node = self._nodes.get((value.id, value.version))
            return node if node is value else None
        if isinstance(value, Mesh):
            # Geometry nodes use the C++ mesh identity shared with UFL cargo.
            value = value._cpp_object
        if version is None:
            return self._latest.get(id(value))
        return self._nodes.get((id(value), version))

    def get_edge(self, predecessor: Node, successor: Node):
        """Get an edge from the graph

        Args:
            predecessor (Node): The predecessor node of the edge
            successor (Node): The successor node of the edge

        Returns:
            Edge: The edge with the given predecessor and successor

        """
        return self._edges.get((predecessor, successor))

    def print(self, detailed=False) -> None:
        """
        Print the graph structure

        Args:
            detailed (bool, optional): Whether to print the graph in detailed mode. Defaults to False.

        """
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            # Fallback for environments without a terminal (e.g., Jupyter notebooks)
            terminal_width = 80

        print("#" * terminal_width)

        # Prints the graph object define in the __str__ method
        print(self)

        print("Nodes:")
        for node in self.nodes:
            print(f"\t{node}")

        print("Edges:")
        for edge in self.edges:
            print(f"\t{edge}")

        if not detailed:
            print("#" * terminal_width)
        else:
            print("Gradient functions:")
            for node in self.nodes:
                if node.grad_fns == []:
                    continue
                print(f"\t{node}")
                for grad_fn in node.grad_fns:
                    print(f"\t\t{grad_fn}")
            print("Next functions:")
            for edge in self.edges:
                if edge.next_functions == []:
                    continue
                print(f"\t{edge}")
                for next_function in edge.next_functions:
                    print(f"\t\t{next_function}")
            print("#" * terminal_width)

    def __str__(self) -> str:
        """String representation of the graph

        Returns:
            str: String representation of the graph

        """
        return f"Graph object with {len(self.nodes)} nodes and {len(self.edges)} edges."

    @staticmethod
    def _add_edge_to_networkx(nx_graph: DiGraph, edge: Edge) -> None:
        """Add an edge of the graph to its networkx representation"""
        tag = "" if edge.__class__.__name__ == "Edge" else edge.__class__.__name__
        nx_graph.add_edge(edge.predecessor, edge.successor, tag=tag, edge=edge)

    @staticmethod
    def _add_node_to_networkx(nx_graph: DiGraph, node: AbstractNode) -> None:
        """Add a node of the graph to its networkx representation"""
        color = "lightblue" if isinstance(node, Node) else "pink"
        nx_graph.add_node(node, name=node.name, node=node, color=color)

    @property
    def nx_graph(self) -> DiGraph:
        """The cached NetworkX representation of the graph.

        Nodes themselves are the keys. The representation is cached and cleared when
        the graph is deleted, so it is empty if kept past the graph.

        Returns:
            nx_graph (nx.DiGraph): The networkx graph representation of the graph
        """

        if self._nx_graph is None:
            nx_graph = nx.DiGraph()
            for node in self.nodes:
                self._add_node_to_networkx(nx_graph, node)
            for edge in self.edges:
                self._add_edge_to_networkx(nx_graph, edge)
            self._nx_graph = nx_graph
        return self._nx_graph

    def visualise(
        self,
        ax=None,
        style: str | Callable = "planar",
        print_edge_labels=True,
        path=None,
    ):
        """Visualise the graph

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to draw the graph in. If None, a new figure and axes are created. Defaults to None.
            style (str, optional): The style of the visualisation. Defaults to "planar".
            print_edge_labels (bool, optional): Whether to print the edge labels. Defaults to True.
            path (set, optional): The set of edges to be highlighted in the visualisation. Defaults to None.

        Returns:
            matplotlib.axes.Axes: The axes containing the visualisation of the graph.
        """

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        labels = nx.get_node_attributes(self.nx_graph, "name")
        edge_labels = nx.get_edge_attributes(self.nx_graph, "tag")
        path = set() if path is None else path
        edge_colors = {
            key: "black" if edge in path else "grey"
            for key, edge in nx.get_edge_attributes(self.nx_graph, "edge").items()
        }
        node_colors = nx.get_node_attributes(self.nx_graph, "color")
        layouts = {
            "planar": nx.planar_layout,
            "shell": nx.shell_layout,
            "random": nx.random_layout,
            "spring": nx.spring_layout,
        }
        # One layout places both the graph and its edge labels.
        if not callable(style):
            edge_pos = layouts.get(style, nx.spring_layout)(self.nx_graph)
        else:
            edge_pos = style(self.nx_graph)
        nx.draw(
            self.nx_graph,
            pos=edge_pos,
            ax=ax,
            labels=labels,
            node_color=node_colors.values(),
            edge_color=edge_colors.values(),
            with_labels=True,
        )
        if print_edge_labels:
            nx.draw_networkx_edge_labels(
                self.nx_graph, pos=edge_pos, edge_labels=edge_labels, ax=ax
            )
        return ax

    def backprop(
        self,
        outputs: Any | Sequence[Any],
        inputs: Any | Sequence[Any] | None = None,
        grad_outputs: Any | Sequence[Any] = 1.0,
    ) -> tuple[Any, ...] | None:
        """
        Perform backpropagation in the graph

        The gradients of all previous calls are reset before the propagation is started.

        Multiple outputs produce one gradient of their weighted sum per control.
        For scalar objectives J_j and seeds s_j, the gradient for control m_i is
        sum_j s_j * dJ_j/dm_i. To obtain separate gradients for each objective,
        call backprop once per objective with all requested controls.

        If controls are given, only the edges on the paths from the controls to the
        outputs are executed, in a single propagation. The propagation therefore ends at
        the controls, whose gradients are stored and returned, even if a control is an
        intermediate node of the graph. A control that depends on another control stores
        its gradient and also passes it on, so that each control receives its total
        derivative.

        If no control is given, all edges the outputs depend on are executed and the
        propagation continues until it reaches the leaves of this dependency subgraph.
        Gradients are then stored only in these leaves; intermediate nodes are passed
        through without storing their gradients. The gradients can be read afterwards
        from :py:attr:`Node.grad` of the respective nodes.

        Args:
            outputs: The recorded object to differentiate, or its exact Node, or a
                sequence of them.
                An object selects its latest version; a Node selects that version.
            inputs: A control object or Node, or a sequence of control objects
                and Nodes in the desired return order.
                Defaults to None. If None, the propagation is carried out down to the dependency
                leaves of the outputs.
            grad_outputs (float or PETSc.Vec, or a sequence of them, optional): The seed,
                i.e. the adjoint value the propagation is started with, or one per output.

        Returns:
            tuple or None: One gradient of the weighted output sum per requested
            control, in control order. An entry is None for a control no output
            depends on. Without controls, returns None and stores leaf gradients.
            A PETSc.Vec gradient has the control's vector layout, including ghost
            entries, but only its entries owned by the calling rank are valid.

        Raises:
            TypeError: If a control does not represent a numerical value and can
                therefore not store a gradient.

        Example:
            Given two recorded scalar objectives and three recorded controls:

            >>> controls = (m1, m2, m3)
            >>> summed = graph_.backprop((J1, J2), controls)
            >>> weighted = graph_.backprop((J1, J2), controls, grad_outputs=(1.0, 0.5))

            summed and weighted each contain three gradients.
        """

        if not isinstance(outputs, Sequence):
            outputs = (outputs,)
        output_nodes = [self.get_node(value) for value in outputs]
        if not isinstance(grad_outputs, Sequence):
            grad_outputs = (grad_outputs,) * len(output_nodes)
        order = self._topological_order(output_nodes)
        if inputs is not None and not isinstance(inputs, Sequence):
            inputs = (inputs,)
        control_nodes = (
            [] if inputs is None else [self.get_node(value) for value in inputs]
        )
        for node in control_nodes:
            if not isinstance(node, Node):
                raise TypeError(
                    f"{node} does not represent a numerical value and can therefore not store a gradient."
                )

        self.reset_grads()

        # Select the edges to be executed, either along a path or all the dependencies if no control is specificied.
        if control_nodes:
            marked = self.get_path(control_nodes, output_nodes)
            targets = control_nodes
        else:
            marked = self.get_dependencies(output_nodes)
            targets = [
                node
                for node in order
                if not any(edge in marked for edge in node.grad_fns)
            ]

        # Propagate the adjoint values in reverse topological order, from the outputs to the controls or leaves.
        # The rule to propagate values through an edge is to call the edge's calculate_adjoint method with the adjoint value of the successor node.
        captured = self._propagate_reverse(
            order,
            marked,
            output_nodes,
            grad_outputs,
            targets,
            lambda edge, value: edge.calculate_adjoint(value),
        )
        for node, adjoint in captured.items():
            if isinstance(node, Node):
                node.accumulate_grad(adjoint)

        if control_nodes:
            return tuple(node.grad for node in control_nodes)

    def get_path(
        self,
        start_nodes: Sequence[AbstractNode],
        end_nodes: Sequence[AbstractNode],
    ) -> set[Edge]:
        """
        Get the edges on the paths from the start nodes to the end nodes

        Args:
            start_nodes (Sequence[AbstractNode]): The start nodes, whose
                paths are collected together.
            end_nodes (Sequence[AbstractNode]): The end nodes, whose
                paths are collected together.

        Returns:
            set[Edge]: The edges along the paths.

        """

        descendants_of_start = set(start_nodes).union(
            *(nx.descendants(self.nx_graph, node) for node in start_nodes)
        )
        ancestors_of_end = set(end_nodes).union(
            *(nx.ancestors(self.nx_graph, node) for node in end_nodes)
        )

        return {
            edge
            for edge in self.edges
            if edge.predecessor in descendants_of_start
            and edge.successor in ancestors_of_end
        }

    def get_dependencies(self, end_nodes: Sequence[AbstractNode]) -> set[Edge]:
        """
        Get the edges of all operations the end nodes are the result of

        Args:
            end_nodes (Sequence[AbstractNode]): The end nodes, whose
                dependencies are collected together.

        Returns:
            set[Edge]: The edges into the end nodes and into the nodes they depend on,
            see :py:meth:`get_path`.


        """

        upstream_of_end = set(end_nodes).union(
            *(nx.ancestors(self.nx_graph, node) for node in end_nodes)
        )

        return {edge for edge in self.edges if edge.successor in upstream_of_end}

    def _topological_order(
        self, output_nodes: Sequence[AbstractNode]
    ) -> list[AbstractNode]:
        """The nodes the outputs depend on, in topological order."""

        upstream = set().union(
            *(nx.ancestors(self.nx_graph, node) | {node} for node in output_nodes)
        )
        position = {node: index for index, node in enumerate(self.nodes)}
        return [
            node
            for node in nx.lexicographical_topological_sort(
                self.nx_graph, key=position.__getitem__
            )
            if node in upstream
        ]

    def _propagate_reverse(
        self,
        order: list[AbstractNode],
        marked: set[Edge],
        output_nodes: Sequence[AbstractNode],
        seeds: Sequence[Any],
        control_nodes: Sequence[AbstractNode],
        rule: Callable[[Edge, Any], Any],
    ) -> dict[AbstractNode, Any]:
        """Reverse: propagate values in reverse topological order through the graph

        Instead of sending values directly downstream through the edges, we collect
        the values at each node and send them downstream once per node. This allows
        for a deterministic order of the edge contributions and increases efficiency,
        since every edge is only called once per rank even if it has multiple successors.

        This is technically achieved by holding intermediate values inside a pending buffer,
        which is then flushed to the next nodes once the current node is processed.
        All endpoints of propagation are collected in a captured buffer, which is returned to the caller.

        The actual propagation is done using the provided rule, which is a edge specific callable
        specified by the current edge and the value to be propagated. The rule is responsible for
        calculating the contribution of the current edge to the next node.

        Args:
            order (list[AbstractNode]): The topological order of the nodes to propagate through.
            marked (set[Edge]): The relevant edges to propagate through, e.g. those on the paths
                from the controls to the outputs.
            output_nodes (Sequence[AbstractNode]): The output nodes to start the propagation from.
            seeds (Sequence[Any]): The initial values to seed the propagation.
            control_nodes (Sequence[AbstractNode]): The nodes at which the propagation
                ends and whose values are returned.
            rule: The contribution ``rule(edge, value)`` describing how the value is propagated
                through the edge.

        Returns:
            dict: The values of the reached control nodes, keyed by node.
        """

        controls = set(control_nodes)

        # Buffers for final values and intermediate values
        captured: dict[AbstractNode, Any] = {}
        pending: dict[AbstractNode, Any] = {}

        # Asign the intial values that should be propagated to the output nodes
        for node, value in zip(output_nodes, seeds):
            pending[node] = add(pending.get(node), value)

        # Propagate the values in reverse topological order, from the outputs to the controls.
        for node in reversed(order):

            # Get the accumulated value of the node, if it has been reached by the propagation. If not, skip it.
            value = pending.pop(node, None)
            if value is None:
                continue

            # Capture the value of the node if it is a control (intermediate or final)
            if node in controls:
                captured[node] = value

            # Propagate the value through the relevant edges
            edges = [edge for edge in node.grad_fns if edge in marked]
            for edge in edges:
                predecessor = edge.predecessor
                pending[predecessor] = add(pending.get(predecessor), rule(edge, value))
        return captured

    def reset_grads(self):
        """
        Reset the gradients in the graph

        """

        for node in self.nodes:
            # Abstract nodes do not represent a numerical value and have no gradient.
            if isinstance(node, Node):
                node.reset_grad()

    def release(self):
        """
        Release the values saved in the graph

        """
        # Collect the UFL cycles first, for the reason and TODO given in __del__
        gc.collect()
        for edge in self.edges:
            edge.release()
        for node in self.nodes:
            node.release()

    def recalculate(self):
        """
        Recalculate the graph

        """
        for node in self.nodes:
            node()

    def __del__(self):
        """
        Destructor for the graph

        """
        if self._nx_graph is not None:
            self._nx_graph.clear()

        # TODO: Remove gc.collect() once UFL's MultiFunction no longer keeps handlers bound to itself
        # (ufl/corealg/multifunction.py, `self._handlers`): each ufl.replace, action and
        # adjoint leaves a Replacer cycle that pins the Functions in its mapping.
        gc.collect()
        for edge in self.edges:
            del edge
        for node in self.nodes:
            del node
        del self
