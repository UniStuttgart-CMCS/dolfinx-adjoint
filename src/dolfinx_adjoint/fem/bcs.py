import numpy as np
from dolfinx import fem, la

import dolfinx_adjoint.graph as graph
from dolfinx_adjoint.utils import bind_arguments


def dirichletbc(*args, **kwargs):
    """OVERLOADS: :py:func:`dolfinx.fem.dirichletbc`.
    Creates a representation of a Dirichlet boundary condition in

    The overloaded function adds the functionality to keep track of the dependencies
    in the computational graph. The original functionality is kept.

    The boundary condition carries the gradient of the value defining it, so a value
    that is not part of the graph leaves nothing to record and the boundary condition
    behaves like the one of DOLFINx. A value DOLFINx creates itself, from an array or a
    scalar, is never part of the graph, since the caller does not hold it.

    Args:
        args: Arguments to :py:func:`dolfinx.fem.dirichletbc`.
        kwargs: Keyword arguments to :py:func:`dolfinx.fem.dirichletbc`.
        graph: An additional keyword argument to specifier wheter the assemble
            operation should be added to the graph. If not present, the original functionality
            of dolfinx is used without any additional functionalities.

    Raises:
        NotImplementedError: If the value is a constant holding more than one value.

    """

    _graph = kwargs.pop("graph", None)
    output = fem.dirichletbc(*args, **kwargs)
    if _graph is None:
        return output

    arguments = bind_arguments(fem.dirichletbc, *args, **kwargs)
    value = arguments["value"]

    value_node = _graph.get_node(id(value))
    if value_node is None:
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

    if isinstance(value, fem.Function):
        value_dofs = arguments["dofs"][1] if np.ndim(arguments["dofs"]) == 2 else dofs
        ctx = [dofs[:num_owned], value_dofs[:num_owned], value.x.petsc_vec]
        dirichletbc_edge = DirichletBC_Function_Edge(
            value_node, dirichletbc_node, ctx=ctx
        )
    else:
        if np.size(value.value) > 1:
            raise NotImplementedError(
                "The gradient of a boundary condition whose value is a constant of more "
                "than one value requires a sum for each of its components."
            )
        # The derivative of the boundary condition broadcasts the single value of the
        # constant onto the dofs it constrains, so the adjoint equation only has to
        # pick those dofs out of the accumulated gradient.
        space = output.function_space
        indicator = la.vector(
            space.dofmap.index_map, space.dofmap.index_map_bs, dtype=value.dtype
        )
        indicator.array[dofs[:num_owned]] = 1.0

        dirichletbc_edge = DirichletBC_Constant_Edge(
            value_node, dirichletbc_node, ctx=[indicator]
        )

    dirichletbc_edge.set_next_functions(value_node.get_gradFuncs())
    dirichletbc_node.set_gradFuncs([dirichletbc_edge])
    _graph.add_edge(dirichletbc_edge)

    return output


class DirichletBC_Function_Edge(graph.Edge):
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


class DirichletBC_Constant_Edge(graph.Edge):
    """
    Edge providing the adjoint equation for the derivative of the DirichletBC to the constant defining the value of the BC.

    """

    def calculate_adjoint(self):
        """
        The method provides the adjoint equation for the derivative of the DirichletBC to the constant defining the value of the BC.

        Since the constant holds a single value on every dof the boundary condition
        constrains, the derivative of the boundary condition broadcasts it onto those
        dofs, and the adjoint equation sums the accumulated gradient over them. The sum
        is the inner product with a vector marking those dofs, which PETSc takes over the
        entries every rank owns and reduces over the communicator of the vectors.

        Returns:
            (float): The accumulated gradient up to this point in the computational
            graph, which the contributions of the same constant from the forms it
            appears in are accumulated with.

        """
        # Extract variables from contextvariable ctx
        (indicator,) = self.ctx

        return self.input_value.dot(indicator.petsc_vec)
