from dolfinx import fem

import dolfinx_adjoint.graph as graph
from dolfinx_adjoint.fem._recording import bind_arguments


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
    if _graph is None:
        return fem.assemble_scalar(*args, **kwargs)

    M = bind_arguments(fem.assemble_scalar, *args, **kwargs)["M"]

    # Get the node corresponding to the form
    form_node = _graph.get_node(M).bind(_graph)
    # The versions the form is assembled with, which its edges differentiate at.
    ufl_form = form_node.ufl_form
    form_node.values = {
        value: _graph.capture(value)
        for value in (*ufl_form.coefficients(), *ufl_form.constants())
    }
    output = fem.assemble_scalar(*args, **kwargs)

    # Creating and adding node to graph
    assemble_node = graph.Node(output, name="AssembleScalar")
    _graph.add_node(assemble_node)

    # Create edge between form and assemble

    # The default edge is sufficient, since assembling a scalar does not require any additional operations
    # for the gradients
    assemble_edge = graph.Edge(form_node, assemble_node)
    assemble_node.grad_fns = [assemble_edge]

    _graph.add_edge(assemble_edge)

    return output
