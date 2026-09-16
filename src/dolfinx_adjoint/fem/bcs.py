import numpy as np
from dolfinx import fem

import dolfinx_adjoint.graph as graph


def dirichletbc(*args, **kwargs):
    """OVERLOADS: :py:func:`dolfinx.fem.dirichletbc`.
    Creates a representation of a Dirichlet boundary condition in

    The overloaded function adds the functionality to keep track of the dependencies
    in the computational graph. The original functionality is kept.

    Args:
        args: Arguments to :py:func:`dolfinx.fem.dirichletbc`.
        kwargs: Keyword arguments to :py:func:`dolfinx.fem.dirichletbc`.
        graph: An additional keyword argument to specifier wheter the assemble
            operation should be added to the graph. If not present, the original functionality
            of dolfinx is used without any additional functionalities.

    """
    _graph = kwargs.pop("graph", None)
    output = fem.dirichletbc(*args, **kwargs)
    if _graph is None:
        return output

    # Creating and adding node to graph
    dirichletbc_node = graph.AbstractNode(output)
    _graph.add_node(dirichletbc_node)

    # The boundary condition provides its dofs unrolled, so that they address the
    # entries of the gradient also in a blocked vector space, where the dofs given
    # to the boundary condition address blocks of components. They are a single
    # array also when the boundary condition couples a sub space with its
    # collapsed space, in which case the indices are the ones of the sub space.
    dofs, num_owned = output.dof_indices()

    dofs_arg = args[1] if len(args) > 1 else kwargs["dofs"]
    value_dofs = dofs_arg[1] if np.ndim(dofs_arg) == 2 else dofs

    value = args[0]
    template = value.x.petsc_vec if isinstance(value, fem.Function) else value.value
    ctx = [dofs[:num_owned], value_dofs[:num_owned], template]

    # Creating the edge between the DirichletBC and the function defining the value of the BC
    value_node = _graph.get_node(id(args[0]))
    dirichletbc_edge = DirichletBC_Edge(value_node, dirichletbc_node, ctx=ctx)
    dirichletbc_edge.set_next_functions(value_node.get_gradFuncs())
    dirichletbc_node.set_gradFuncs([dirichletbc_edge])
    _graph.add_edge(dirichletbc_edge)

    return output


class DirichletBC_Edge(graph.Edge):
    """
    Edge providing the adjoint equation for the derivative of the DirichletBC to the function defining the value of the BC.

    """

    def calculate_adjoint(self):
        """
        The method provides the adjoint equation for the derivative of the DirichletBC to the function defining the value of the BC.

        Since the boundary condition is only applied to a part of the domain, the derivative of the boundary condition
        applies the accumulated gradient to the function only defined on the relevant boundary.

        Returns:
            (PETSc.Vec): The accumulated gradient up to this point in the computational graph.
            It has the layout of the vector of the function defining the value of the boundary
            condition, but only its entries owned by the calling rank are valid.

        """
        # Extract variables from contextvariable ctx
        dofs, value_dofs, template = self.ctx

        values = self.input_value.array_r
        gradient = template.duplicate()
        gradient.zeroEntries()
        gradient.array_w[value_dofs] = values[dofs]

        return gradient
