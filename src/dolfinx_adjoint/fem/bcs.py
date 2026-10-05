import numpy as np
from dolfinx import fem
from petsc4py import PETSc

import dolfinx_adjoint.graph as graph
from dolfinx_adjoint.fem._calculus import real_space
from dolfinx_adjoint.fem._recording import bind_arguments
from dolfinx_adjoint.graph.dolfinx_helpers import scalar, zeros


def dirichletbc(*args, **kwargs):
    """OVERLOADS: :py:func:`dolfinx.fem.dirichletbc`.
    Creates a representation of a Dirichlet boundary condition in

    The overloaded function adds the functionality to keep track of the dependencies
    in the computational graph. The original functionality is kept.

    The boundary condition carries the gradient of the value defining it, so a value
    that is not part of the graph leaves nothing to record and the boundary condition
    behaves like the one of DOLFINx.

    Args:
        args: Arguments to :py:func:`dolfinx.fem.dirichletbc`.
        kwargs: Keyword arguments to :py:func:`dolfinx.fem.dirichletbc`.
        graph: An additional keyword argument to specify whether the boundary
            condition should be added to the graph. If not present, the original functionality
            of dolfinx is used without any additional functionalities.

    """

    _graph = kwargs.pop("graph", None)
    output = fem.dirichletbc(*args, **kwargs)
    if _graph is None:
        return output
    arguments = bind_arguments(fem.dirichletbc, *args, **kwargs)
    arguments["value"] = output.g

    if _graph.get_node(arguments["value"]) is None:
        return output

    dirichletbc_node = DirichletBCNode(output, arguments)
    _graph.add_node(dirichletbc_node)
    dirichletbc_node._record_dirichletbc(_graph)
    return output


class DirichletBCNode(graph.AbstractNode):
    """Node of a boundary condition, with the version of the value it applies.

    A boundary condition is created once and applies whatever its value holds when a
    problem is solved. Every solve therefore depends on the version of the value that is
    the latest at the solve, which a node created with the boundary condition cannot
    know.

    Attributes:
        arguments (dict): The arguments of the :py:func:`dolfinx.fem.dirichletbc` call.
    """

    def __init__(self, object, arguments, version=0):
        super().__init__(object, version=version)
        self.arguments = arguments

    def bind(self, _graph) -> "DirichletBCNode":
        """Record a fresh boundary-condition version with the value this solve reads.

        Args:
            _graph (Graph): The graph the solve is recorded in.

        Returns:
            DirichletBCNode: A new version with an edge from the latest recorded
            version of the boundary value.
        """
        dirichletbc_node = DirichletBCNode(
            self.object, self.arguments, version=self.version + 1
        )
        _graph.add_node(dirichletbc_node)
        dirichletbc_node._record_dirichletbc(_graph)
        return dirichletbc_node

    def _record_dirichletbc(self, _graph):
        """Record the edge into this version from the latest version of its value."""
        output, arguments = self.object, self.arguments
        value = arguments["value"]
        value_node = _graph.get_node(value)

        # The boundary condition provides its dofs unrolled, so that they address the
        # entries of the gradient also in a blocked vector space, where the dofs given
        # to the boundary condition address blocks of components. They are a single
        # array also when the boundary condition couples a sub space with its
        # collapsed space, in which case the indices are the ones of the sub space.
        dofs, num_owned = output.dof_indices()

        if isinstance(value, fem.Function):
            value_dofs = (
                arguments["dofs"][1] if np.ndim(arguments["dofs"]) == 2 else dofs
            )
            ctx = [dofs[:num_owned], value_dofs[:num_owned], value.x.petsc_vec]
            dirichletbc_edge = DirichletBC_Function_Edge(value_node, self, ctx=ctx)
        else:
            ctx = [dofs[:num_owned], value, arguments["V"]]
            dirichletbc_edge = DirichletBC_Constant_Edge(value_node, self, ctx=ctx)

        self.grad_fns = [dirichletbc_edge]
        _graph.add_edge(dirichletbc_edge)

    def release(self):
        """Releases the boundary condition and the arguments it was created with."""
        super().release()
        self.arguments = None


class DirichletBC_Function_Edge(graph.Edge):
    """
    Edge providing the adjoint equation for the derivative of the DirichletBC to the function defining the value of the BC.

    """

    def calculate_adjoint(self, value: PETSc.Vec):
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

        # Sum into the local dofs, including ghosts, and at shared dofs across ranks.
        gradient = zeros(template)
        with gradient.localForm() as local:
            np.add.at(local.array, value_dofs, value.array_r[dofs])
        gradient.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return gradient


class DirichletBC_Constant_Edge(graph.Edge):
    """
    Edge providing the adjoint equation for the derivative of the DirichletBC to the constant defining the value of the BC.

    Attributes:
        ctx (list): The owned dofs of the boundary condition, the constant and the space
            the boundary condition constrains.
    """

    _interpolation = None

    def release(self):
        """Releases the values saved in the edge and its interpolation matrix."""
        super().release()
        self._interpolation = None

    def calculate_adjoint(self, value: PETSc.Vec):
        """
        The method provides the adjoint equation for the derivative of the DirichletBC to the constant defining the value of the BC.

        The cached DOLFINx interpolation matrix supplies the component mapping.
        Its PETSc transpose product accumulates contributions across mesh ranks; masking only owned entries avoids counting ghost sensitivities twice.

        Returns:
            float or complex or PETSc.Vec: The accumulated gradient up to this point in the computational
            graph, which the contributions of the same constant from the forms it appears in are accumulated with.
        """

        owned_boundary_dofs, constant, V = self.ctx
        if self._interpolation is None:
            # The interpolation matrix and a vector of its rows, created collectively
            # at the first derivative.
            interpolation = fem.petsc.interpolation_matrix(real_space(constant), V)
            interpolation.assemble()
            self._interpolation = (interpolation, interpolation.createVecLeft())
        interpolation, input_cache = self._interpolation
        input_cache.zeroEntries()
        input_cache.array_w[owned_boundary_dofs] = value.array_r[owned_boundary_dofs]
        gradient = interpolation.createVecRight()
        interpolation.multTranspose(input_cache, gradient)
        return scalar(constant, gradient)
