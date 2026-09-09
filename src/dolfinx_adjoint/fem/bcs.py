import numpy as np
import scipy.sparse as sps
from dolfinx import fem
from petsc4py import PETSc

import dolfinx_adjoint.graph as graph


def dirichletbc(*args, map=None, **kwargs):
    """OVERLOADS: :py:func:`dolfinx.fem.dirichletbc`.
    Creates a representation of a Dirichlet boundary condition in

    The overloaded function adds the functionality to keep track of the dependencies
    in the computational graph. The original functionality is kept.

    Args:
        args: Arguments to :py:func:`dolfinx.fem.dirichletbc`.
        kwargs: Keyword arguments to :py:func:`dolfinx.fem.dirichletbc`.
        map : Defines the map between the function space of the boundary
            condition and the function space of the problem.
        graph: An additional keyword argument to specifier wheter the assemble
            operation should be added to the graph. If not present, the original functionality
            of dolfinx is used without any additional functionalities.

    Note:
        The map is used to define the map between the function space of the boundary
        condition and the function space of the problem. This is useful when the function
        space of the boundary condition and the function space of the problem are different.
        The map is stored as an array where the index is equivalent to the index in the correct
        space and the value is the index in the wrong space.

    """
    if not "graph" in kwargs:
        output = fem.dirichletbc(*args, **kwargs)
    else:
        _graph = kwargs["graph"]
        del kwargs["graph"]
        output = fem.dirichletbc(*args, **kwargs)

        # Creating and adding node to graph
        dirichletbc_node = graph.AbstractNode(output)
        _graph.add_node(dirichletbc_node)

        # The boundary condition provides its dofs unrolled, so that they address the
        # entries of the gradient also in a blocked vector space, where the dofs given
        # to the boundary condition address blocks of components. They are a single
        # array also when the boundary condition couples a sub space with its
        # collapsed space, in which case the indices are the ones of the sub space.
        dofs = output.dof_indices()[0]
        ctx = [dofs, map]

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

        """
        # Extract variables from contextvariable ctx
        dofs, map = self.ctx

        size = np.shape(self.input_value)[0]
        matrix = sps.csr_matrix((np.ones(np.size(dofs)), (dofs, dofs)), (size, size))

        # The map is used to define the map between the function space of the boundary
        # condition and the function space of the problem. This is useful when the function
        # space of the boundary condition and the function space of the problem are different.
        # The map is stored as an array where the index is equivalent to the index in the correct
        # space and the value is the index in the wrong space.
        if map is None:
            output = self.input_value @ matrix
        else:
            output = (self.input_value @ matrix)[map]

        # Convert to petsc vector
        return PETSc.Vec().createWithArray(output)
