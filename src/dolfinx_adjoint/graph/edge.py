import weakref
from typing import Any

from dolfinx_adjoint.graph.node import Node


class Edge:
    """
    The edge class provides the functionality to represent the edges in the graph.

    Edges are used to keep track of the dependencies betweeen the nodes in the graph and
    are important for the backpropagation of the gradients. To this end, the edges store
    the adjoint equations to calculate the derivative of the successor node with respect
    to the predecessor node.

    Attributes:
        predecessor (Node): The predecessor node of the edge
        ctx (Any): The context variable of the edge

    """

    def __init__(self, predecessor: Node, successor: Node, ctx=None):
        """
        The constructor for the Edge class.

        Args:
            predecessor (Node): The predecessor node of the edge
            successor (Node): The successor node of the edge. None marks an edge that is
                not part of the graph, e.g. the edge that seeds the backpropagation.
            ctx (Any, optional): The context variable of the edge

        Raises:
            ValueError: If the predecessor node is None, i.e. the operation was recorded
                with an input that is not part of the graph. Such an operation does not
                depend on a tracked value and therefore has no edge to record.

        """

        if predecessor is None:
            raise ValueError(
                f"The edge into {successor} has no predecessor node, since the operation was recorded with an input that is not part of the graph."
            )
        self.predecessor = predecessor
        self.successor = successor
        self.ctx = ctx

    @property
    def successor(self):
        """The successor node of the edge, referenced weakly, or None for an edge that ends nowhere."""
        return None if self._successor is None else self._successor()

    @successor.setter
    def successor(self, node):
        """Reference the successor node weakly."""
        self._successor = None if node is None else weakref.ref(node)

    def release(self):
        """
        Releases the values saved in the edge.

        """
        self.ctx = None

    @property
    def next_functions(self) -> list:
        """The edges into the predecessor node, which the propagation continues with."""
        return self.predecessor.grad_fns

    def calculate_adjoint(self, value: Any) -> Any:
        """
        This method calculates the default adjoint equation for the edge, which
        corresponds to the derivative:

            d(successor)/d(predecessor) = 1.0

        This operator is stored in the edge and applied to the input. The adjoint value of the predecessor
        node is generally calculated as follows:

            adjoint(predecessor) = adjoint(successor) * d(successor)/d(predecessor)

        The computed value only corresponds to the contribution of the current edge to the predecessor node. The total adjoint value of the predecessor node is the sum of all contributions from all edges into it.

        Args:
            value (Any): The adjoint value of the successor node, or shared
                operation data prepared by a downstream edge.

        Returns:
            Any: The adjoint contribution to the predecessor node. Edges between
            operations may return shared adjoint data; edges into numerical nodes
            return a scalar or PETSc.Vec gradient, of which only the entries owned by
            the calling rank have to be valid. None if it vanishes.

        """

        return value

    def __str__(self):
        """
        Returns the string representation of the edge.

        Returns:
            str: The string representation of the edge

        """
        return f"{str(self.predecessor)} -> {str(self.successor)}"
