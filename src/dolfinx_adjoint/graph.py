import os

import networkx as nx
from networkx import DiGraph

from .edge import Edge
from .node import AbstractNode, Node


class Graph:
    """Class for the computational graph

    The computational graph is a directed acyclic graph (DAG) that represents the
    operations, objects and dependencies in a forward simulation in DOLFINx.

    Attributes:
        nodes (list): List of nodes in the graph
        edes (list): List of edges in the graph

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
        self.nodes = []
        self.edges = []

        # Cached networkx graph for fast descendants/ancestors queries
        self._nx_graph = None

    def add_node(self, node: Node):
        """Add a node to the graph

        Args:
            node (Node): The node to be added to the graph

        """
        self.nodes.append(node)

    def add_edge(self, edge: Edge):
        """Add an edge to the graph

        Args:
            edge (Edge): The edge to be added to the graph

        """
        self.edges.append(edge)

    def get_node(self, id: int, version=None):
        """Get a node from the graph

        Args:
            id (int): The python-id of the node to be retrieved
            version (int, optional): The version of the node to be retrieved. Defaults to None.

        Returns:
            Node: The node with the given id and version

        """
        # Get all nodes with the given id
        node_versions = {}
        for node in self.nodes:
            if node.id == id:
                node_versions[node.version] = node
        if node_versions == {}:
            return None
        # If no version is given, return the latest version
        if version is None:
            latest_version = max(node_versions.keys())
            return node_versions[latest_version]
        # If a version is given, return the node with the given version
        elif version in node_versions.keys():
            return node_versions[version]
        else:
            return None

    def get_edge(self, predecessor: Node, successor: Node):
        """Get an edge from the graph

        Args:
            predecessor (Node): The predecessor node of the edge
            successor (Node): The successor node of the edge

        Returns:
            Edge: The edge with the given predecessor and successor

        """
        for edge in self.edges:
            if edge.predecessor == predecessor and edge.successor == successor:
                return edge
        return None

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
                if node.get_gradFuncs() == []:
                    continue
                print(f"\t{node}")
                for gradFunc in node.get_gradFuncs():
                    print(f"\t\t{gradFunc}")
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
    def _edge_color(edge: Edge) -> str:
        """Get the color of an edge, black if it is part of the marked path"""
        return "black" if hasattr(edge, "marked") else "grey"

    @classmethod
    def _add_edge_to_networkx(cls, nx_graph: DiGraph, edge: Edge) -> None:
        """Add an edge of the graph to its networkx representation"""
        tag = "" if edge.__class__.__name__ == "Edge" else edge.__class__.__name__
        nx_graph.add_edge(
            id(edge.predecessor),
            id(edge.successor),
            tag=tag,
            color=cls._edge_color(edge),
            edge=edge,
        )

    def to_networkx(self) -> DiGraph:
        """Convert the graph to a networkx graph

        Returns:
            nx_graph (nx.DiGraph): The networkx graph representation of the graph
        """
        if self._nx_graph is not None:
            # Update dynamic edge colors based on marking
            for edge in self.edges:
                u = id(edge.predecessor)
                v = id(edge.successor)
                if self._nx_graph.has_edge(u, v):
                    self._nx_graph[u][v]["color"] = self._edge_color(edge)
            return self._nx_graph

        nx_graph = nx.DiGraph()
        for node in self.nodes:
            nx_graph.add_node(id(node), name=node.name, node=node)
            if type(node) == AbstractNode:
                nx_graph.nodes[id(node)]["color"] = "pink"
            else:
                nx_graph.nodes[id(node)]["color"] = "lightblue"

        for edge in self.edges:
            self._add_edge_to_networkx(nx_graph, edge)

        self._nx_graph = nx_graph
        return nx_graph

    def _get_networkx_graph(self) -> DiGraph:
        """Get the networkx graph representation of the graph.

        Returns:
            DiGraph: The networkx graph representation of the graph.
        """
        if self._nx_graph is None:
            return self.to_networkx()
        else:
            return self._nx_graph

    def visualise(self, filename="graph.pdf", style="planar", print_edge_labels=True):
        import matplotlib.pyplot as plt

        """Visualise the graph

        Args:
            filename (str, optional): The filename of the visualisation. Defaults to "graph.pdf".
            style (str, optional): The style of the visualisation. Defaults to "planar".
            print_edge_labels (bool, optional): Whether to print the edge labels. Defaults to True.

        """
        plt.figure(figsize=(10, 8))
        nx_graph = self._get_networkx_graph()
        labels = nx.get_node_attributes(nx_graph, "name")
        edge_labels = nx.get_edge_attributes(nx_graph, "tag")
        edge_colors = nx.get_edge_attributes(nx_graph, "color")
        node_colors = nx.get_node_attributes(nx_graph, "color")
        if style == "planar":
            nx.draw_planar(
                nx_graph,
                labels=labels,
                node_color=node_colors.values(),
                edge_color=edge_colors.values(),
                with_labels=True,
            )
            edge_pos = nx.planar_layout(nx_graph)
        elif style == "shell":
            nx.draw_shell(
                nx_graph,
                labels=labels,
                node_color=node_colors.values(),
                edge_color=edge_colors.values(),
                with_labels=True,
            )
            edge_pos = nx.shell_layout(nx_graph)
        elif style == "random":
            nx.draw_random(
                nx_graph,
                labels=labels,
                node_color=node_colors.values(),
                edge_color=edge_colors.values(),
                with_labels=True,
            )
            edge_pos = nx.random_layout(nx_graph)
        else:
            if style != "spring":
                print("Given style is not implemented. Using spring layout")
            nx.draw(
                nx_graph,
                labels=labels,
                node_color=node_colors.values(),
                edge_color=edge_colors.values(),
                with_labels=True,
            )
            edge_pos = nx.spring_layout(nx_graph)
        if print_edge_labels:
            nx.draw_networkx_edge_labels(
                nx_graph, pos=edge_pos, edge_labels=edge_labels
            )
        plt.savefig(filename)

    def backprop(self, function_id: int, variable_id=None):
        """
        Perform backpropagation in the graph

        Args:
            function_id (int): The id of the function to be differentiated
            variable_id (int, optional): The id of the variable with respect to which the differentiation is performed. Defaults to None.
                If None, the differentiation is performed with respect to all variables in the graph.

        """
        self.reset_grads()
        function_node = self.get_node(function_id)
        if variable_id is not None:
            variable_node = self.get_node(variable_id)
            self.get_path(id(variable_node), id(function_node))
        else:
            for edge in self.edges:
                edge.marked = True
        grad_func = function_node.get_gradFuncs()[0]
        grad_func(1.0)
        if variable_id is not None:
            return self.get_node(variable_id).get_grad()

    def get_path(self, start_id: int, end_id: int):
        """
        Get the path from the start node to the end node by marking the edges

        Args:
            start_id (int): The id of the start node
            end_id (int): The id of the end node
        """
        for edge in self.edges:
            edge.marked = False

        nx_graph = self._get_networkx_graph()

        descendants_of_start = nx.descendants(nx_graph, start_id)
        descendants_of_start.add(start_id)
        ancestors_of_end = nx.ancestors(nx_graph, end_id)
        ancestors_of_end.add(end_id)

        for edge in self.edges:
            succ_id = id(edge.successor)
            pred_id = id(edge.predecessor)
            if succ_id in descendants_of_start and pred_id in ancestors_of_end:
                edge.marked = True

    def reset_grads(self):
        """
        Reset the gradients in the graph

        """
        for node in self.nodes:
            # Since abstract nodes do not have gradients, we skip them
            try:
                node.reset_grad()
            except:
                pass

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
        for edge in self.edges:
            del edge
        for node in self.nodes:
            del node
        del self
