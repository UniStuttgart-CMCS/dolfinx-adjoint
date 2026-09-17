from typing import Any

import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx_adjoint.graph as graph
from dolfinx_adjoint.utils import bind_arguments


def form(*args, **kwargs):
    """OVERLOADS: :py:func:`dolfinx.fem.form`.
    Create a Form or an array of Forms.

    The overloaded function adds the functionality to keep track of the dependencies
    in the computational graph and the adjoint equations. The original functionality is kept.

    Args:
        args: Arguments to :py:func:`dolfinx.fem.assemble_scalar`.
        kwargs: Keyword arguments to :py:func:`dolfinx.fem.assemble_scalar`.
        graph (graph, optional): An additional keyword argument to specifier whether the assemble
            operation should be added to the graph. If not present, the original functionality
            of dolfinx is used without any additional functionalities.

    Returns:
        Compiled finite element Form.

    Note:
        When a form is compiled from a UFL form, the dependencies in the symbolic equation is lost.
        The resulting compiled form does not support symbolic differentiation. The graph and the custom edges
        are used to keep track of the dependencies and the adjoint equations.

    """
    _graph = kwargs.pop("graph", None)
    output = fem.form(*args, **kwargs)
    if _graph is None:
        return output

    ufl_form = bind_arguments(fem.form, *args, **kwargs)["form"]

    # Creating and adding node to graph
    form_node = FormNode(output, ufl_form)
    _graph.add_node(form_node)

    # Creating and adding edges to the graph if the coefficients are in the graph
    for coefficient in ufl_form.coefficients():
        coefficient_node = _graph.get_node(id(coefficient))
        if not coefficient_node == None:
            ctx = [ufl_form, coefficient]
            coefficient_edge = Form_Coefficient_Edge(
                coefficient_node, form_node, ctx=ctx
            )
            form_node.append_gradFuncs(coefficient_edge)
            coefficient_edge.set_next_functions(coefficient_node.get_gradFuncs())
            _graph.add_edge(coefficient_edge)

    # Creating and adding edges to the graph if the constants are in the graph
    for constant in ufl_form.constants():
        constant_node = _graph.get_node(id(constant))
        if not constant_node == None:
            ctx = [ufl_form, constant]
            constant_edge = Form_Constant_Edge(constant_node, form_node, ctx=ctx)
            form_node.append_gradFuncs(constant_edge)
            constant_edge.set_next_functions(constant_node.get_gradFuncs())
            _graph.add_edge(constant_edge)

    return output


class FormNode(graph.AbstractNode):
    """
    Node for the operation :py:func:`dolfinx.fem.form`.

    In order to compile the form from the ufl form in the forward pass,
    the ufl form needs to be saved in the node.

    Attributes:
        object (Any): The object that is being compiled.
        ufl_form (ufl.form.Form): The ufl form that is being compiled.

    """

    def __init__(self, object: Any, ufl_form: ufl.form.Form):
        """
        Constructor for the FormNode.

        Args:
            object (Any): The object that is being compiled.
            ufl_form (ufl.form.Form): The ufl form that is being compiled.

        """
        super().__init__(object, name="Form")
        self.ufl_form = ufl_form

    def __call__(self):
        """
        The call method to perform the compile form operation.

        Returns:
            Compiled finite element Form.

        """
        output = fem.form(self.ufl_form)
        self.object = output
        return output


class Form_Coefficient_Edge(graph.Edge):
    """
    Edge providing the adjoint equation for the derivative of the form with respect to a coefficient.

    """

    def calculate_adjoint(self):
        """
        The method provides the adjoint equation for the derivative of the form with respect to a coefficient.

        Since the symbolic equations are available in the ufl form, the derivative can be calculated using the symbolic
        differentiation provided by UFL.

        Returns:
            (PETSc.Vec): The accumulated gradient up to this point in the computational graph.
            Only its entries owned by the calling rank are valid.

        """
        ufl_form, coefficient = self.ctx
        derivative = ufl.derivative(ufl_form, coefficient)

        output = fem.petsc.assemble_vector(fem.form(derivative))

        output.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        output.scale(self.input_value)

        return output


class Form_Constant_Edge(graph.Edge):
    """
    Edge providing the adjoint equation for the derivative of the form with respect to a scalar constant.
    """

    def calculate_adjoint(self):
        """
        The method provides the adjoint equation for the derivative of the form with respect to a scalar constant.

        Since the symbolic equations are available in the ufl form, the derivative can be calculated using the symbolic
        differentiation provided by UFL. However, the constant needs to be replaced by a function in order to use the
        UFL functionality. A uniform unit direction varies the scalar parameter everywhere and leaves no test argument, so the derivative can be assembled as a scalar.

        Returns:
            float or complex: The accumulated gradient up to this point in the computational graph.

        """

        ufl_form, constant = self.ctx

        # Create a function based on the constant in order to use ufl.derivative
        domain = constant.domain
        DG0 = fem.functionspace(domain, ("DG", 0))
        function = fem.Function(DG0)
        function.x.array[:] = constant.value

        replaced_form = ufl.replace(ufl_form, {constant: function})

        derivative = ufl.derivative(replaced_form, function, ufl.as_ufl(1.0))

        return self.input_value * domain.comm.allreduce(
            fem.assemble_scalar(fem.form(derivative)), op=MPI.SUM
        )
