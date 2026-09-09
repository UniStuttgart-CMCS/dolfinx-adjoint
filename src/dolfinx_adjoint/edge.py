from typing import Any

import petsc4py.PETSc as PETSc

from .node import Node


class Edge:
    """
    The edge class provides the functionality to represent the edges in the graph.

    Edges are used to keep track of the dependencies betweeen the nodes in the graph and
    are important for the backpropagation of the gradients. To this end, the edges store
    the adjoint equations to calculate the derivative of the successor node with respect
    to the predecessor node.

    Calling an edge will perform the backpropagation of the graph, by calculating the adjoint
    value and passing it to the edges connected to the predecessor node.

    Attributes:
        predecessor (Node): The predecessor node of the edge
        successor (Node): The successor node of the edge
        next_functions (list): The list of the gradient functions that are connected to the edge
        ctx (Any): The context variable of the edge
        input_value (float or PETSc.Vec): The input value of the edge

    """

    def __init__(self, predecessor: Node, successor: Node, ctx=None, input_value=None):
        """
        The constructor for the Edge class.

        Args:
            predecessor (Node): The predecessor node of the edge
            successor (Node): The successor node of the edge
            ctx (Any, optional): The context variable of the edge
            input_value (float or PETSc.Vec, optional): The input value of the edge

        """
        self.predecessor = predecessor
        self.successor = successor
        self.next_functions = []
        self.ctx = ctx
        self.input_value = input_value

    def set_next_functions(self, funcList: list):
        """
        This method sets the next functions in the path of the edge.

        Typically the next functions are the edges that are connected to the predecessor node.

        Args:
            funcList (list): A list of the gradient functions that are connected to the edge

        """
        self.next_functions = funcList

    def calculate_adjoint(self):
        """
        This method calculates the default adjoint equation for the edge, which
        corresponds to the derivative:

            d(successor)/d(predecessor) = 1.0

        This operator is stored in the edge and applied to the input. The adjoint value of the predeccessor
        node is generally calculated as follows:

            adjoint(predecessor) += adjoint(successor) * d(successor)/d(predecessor)

        Returns:
            float or PETSc.Vec: The adjoint value of the predecessor node
        """
        return self.input_value

    def __call__(self, value: float or PETSc.Vec):
        """
        This method is used to perform the backpropagation of the edge.

        By calling an edge, the adjoint value is calculated and passed to edges connected to the predecessor node.
        These edges are automatically called with the adjoint value, performing the backpropagation through the graph.

        Args:
            value (float or PETSc.Vec): The adjoint value of the successor node

        Note:
            This method is called automatically when the backpropagation of the whole graph is performed.
            It should not be modified. The user should implement the calculate_adjoint method to define
            the adjoint equation.

        """

        # Compute adjoint value
        self.input_value = value
        grad_value = self.calculate_adjoint()

        # Extract all marked next functions
        next_functions = [
            function
            for function in self.next_functions
            if getattr(function, "marked", True)
        ]
        for function in next_functions:
            function(grad_value)

        # Accumulate gradient if end of path
        if not next_functions and isinstance(self.predecessor, Node):
            self.predecessor.accumulate_grad(grad_value)

    def __str__(self):
        """
        Returns the string representation of the edge.

        Returns:
            str: The string representation of the edge

        """
        return f"{str(self.predecessor)} -> {str(self.successor)}"

    def __del__(self):
        """
        Destructor for the Edge class.

        """
        del self.ctx
        del self.input_value
        del self
