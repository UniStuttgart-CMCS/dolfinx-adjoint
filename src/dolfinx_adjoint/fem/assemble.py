from typing import Any

from dolfinx import fem

import dolfinx_adjoint.graph as graph


def assemble_scalar(*args, **kwargs):
    """OVERLOADS: :py:func:`dolfinx.fem.assemble_scalar`.
    Assemble functional. The returned value is local and not accumulated across processes.

    The overloaded function adds the functionality to keep track of the dependencies
    in the computational graph. The original functionality is kept.

    Args:
        args: Arguments to :py:func:`dolfinx.fem.assemble_scalar`.
        kwargs: Keyword arguments to :py:func:`dolfinx.fem.assemble_scalar`.
        graph (graph, optional): An additional keyword argument to specifier whether the assemble
            operation should be added to the graph. If not present, the original functionality
            of dolfinx is used without any additional functionalities.

    Returns:
        float: The computed scalar on the calling rank

    Note:
        When a form is assembled into a scalar value, the information about its dependencies is lost,
        and the resulting scalar does not support automatic differentiation. To this end
        the graph is used to keep track of the dependencies between the resulting scalar value and the
        used form to obtain the scalar value.

    """
    _graph = kwargs.pop("graph", None)
    output = fem.assemble_scalar(*args, **kwargs)
    if _graph is None:
        return output

    # Creating and adding node to graph
    assemble_node = AssembleScalarNode(output, args[0])
    _graph.add_node(assemble_node)

    # Create edge between form and assemble
    form_node = _graph.get_node(id(args[0]))

    # The default edge is sufficient, since assembling a scalar does not require any additional operations
    # for the gradients
    assemble_edge = graph.Edge(form_node, assemble_node)
    assemble_node.set_gradFuncs([assemble_edge])

    # Create connectivity to previous edges
    assemble_edge.set_next_functions(form_node.get_gradFuncs())
    _graph.add_edge(assemble_edge)

    return output


class AssembleScalarNode(graph.Node):
    """
    Node for the operation :py:func:`dolfinx.fem.assemble_scalar`.

    In order to assemble the scalar value from the form in the forward pass,
    the form needs to be saved in the node.

    Attributes:
        object (Any): The object that is being represented by the node
        M (dolfinx.fem.Form): The form that is being assembled

    """

    def __init__(self, object: Any, M: fem.Form):
        """
        Constructor for the AssembleScalarNode

        Args:
            object (Any): The object that is being represented by the node
            M (dolfinx.fem.Form): The form that is being assembled

        """
        super().__init__(object, name="AssembleScalar")
        self.M = M

    def __call__(self):
        """
        The call method to perform the assemble operation.

        Returns:
            float: The computed scalar on the local rank
        """
        output = fem.assemble_scalar(self.M)
        self.object = output
        return output
