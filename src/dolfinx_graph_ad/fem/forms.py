from typing import Any

import ufl
from dolfinx import fem

import dolfinx_graph_ad.graph as graph
from dolfinx_graph_ad.fem._calculus import (
    assemble_vector,
    real_function,
    ufl_derivative_constant,
)
from dolfinx_graph_ad.fem._recording import bind_arguments
from dolfinx_graph_ad.graph.dolfinx_helpers import resolve_capture, scalar, zeros


def form(*args, **kwargs):
    """OVERLOADS: :py:func:`dolfinx.fem.form`.
    Create a Form or an array of Forms.

    The overloaded function adds the functionality to keep track of the dependencies
    in the computational graph and the adjoint equations. The original functionality is kept.

    Args:
        args: Arguments to :py:func:`dolfinx.fem.form`.
        kwargs: Keyword arguments to :py:func:`dolfinx.fem.form`.
        graph (graph, optional): An additional keyword argument to specifier whether the assemble
            operation should be added to the graph. If not present, the original functionality
            of dolfinx is used without any additional functionalities.

    Returns:
        Compiled finite element Form.

    Note:
        When a form is compiled from a UFL form, the dependencies in the symbolic equation is lost.
        The resulting compiled form does not support symbolic differentiation. The graph and the custom edges are used to keep track of the dependencies and the adjoint equations.

    Note:
        The arguments the form is compiled with are recorded together with it, so that the
        compilations the node and its edges perform later are the compilation the caller
        asked for.

    """

    _graph = kwargs.pop("graph", None)
    output = fem.form(*args, **kwargs)
    if _graph is None:
        return output

    arguments = bind_arguments(fem.form, *args, **kwargs)
    ufl_form = arguments.pop("form")

    # The first version of the form; later versions are recorded by FormNode.bind.
    form_node = FormNode(output, ufl_form, **arguments)
    _graph.add_node(form_node)
    form_node._record_form(_graph)
    return output


class FormNode(graph.AbstractNode):
    """
    Node for the operation :py:func:`dolfinx.fem.form`.

    In order to compile the form from the ufl form in the forward pass,
    the ufl form needs to be saved in the node, together with the arguments it was
    compiled with.

    Attributes:
        object (Any): The object that is being compiled.
        ufl_form (ufl.form.Form): The ufl form that is being compiled.
        kwargs (dict): The arguments of :py:func:`dolfinx.fem.form` the form was
            compiled with, keyed by parameter name and without the form itself.
        values (dict): Copies of the coefficients and constants of the form, keyed by
            the originals, taken when the form was assembled. None before.

    """

    def __init__(self, object: Any, ufl_form: ufl.form.Form, version=0, **kwargs):
        """
        Constructor for the FormNode.

        Args:
            object (Any): The object that is being compiled.
            ufl_form (ufl.form.Form): The ufl form that is being compiled.
            version (int, optional): The version of the compiled form.
            kwargs: The remaining arguments of the recorded :py:func:`dolfinx.fem.form`
                call, keyed by parameter name.

        """

        super().__init__(object, version=version, name="Form")
        self.ufl_form = ufl_form
        self.kwargs = kwargs
        self.values = None

    def bind(self, _graph) -> "FormNode":
        """Record a fresh form version with the inputs this assembly reads.

        Args:
            _graph (Graph): The graph the assembly is recorded in.

        Returns:
            FormNode: The node of the form for this assembly.
        """
        form_node = FormNode(
            self.object, self.ufl_form, version=self.version + 1, **self.kwargs
        )
        _graph.add_node(form_node)
        form_node._record_form(_graph)
        return form_node

    def _record_form(self, _graph):
        """Record the edges into this form version from the latest versions of the
        recorded values it depends on.
        """
        # Creating and adding edges to the graph if the coefficients are in the graph
        for coefficient in self.ufl_form.coefficients():
            coefficient_node = _graph.get_node(coefficient)
            if coefficient_node is not None:
                coefficient_edge = Form_Coefficient_Edge(
                    coefficient_node, self, ctx=coefficient
                )
                self.grad_fns.append(coefficient_edge)
                _graph.add_edge(coefficient_edge)

        # Creating and adding edges to the graph if the constants are in the graph
        for constant in self.ufl_form.constants():
            constant_node = _graph.get_node(constant)
            if constant_node is not None:
                constant_edge = Form_Constant_Edge(constant_node, self, ctx=constant)
                self.grad_fns.append(constant_edge)
                _graph.add_edge(constant_edge)

    def recorded(self):
        """The form at the values it was assembled with, and these values.

        Returns:
            tuple: The form and the values it is evaluated at, keyed by the originals.

        Raises:
            RuntimeError: If the form has not been assembled, so that no values were
                copied.
        """
        if self.values is None:
            raise RuntimeError(
                f"{self} has not been assembled and has no values to be differentiated at."
            )
        values = {
            coefficient: resolve_capture(captured)
            for coefficient, captured in self.values.items()
        }
        return ufl.replace(self.ufl_form, values), values

    def release(self):
        """
        Releases the compiled form, the ufl form, the arguments it was compiled from and
        the values it was assembled with.

        """

        super().release()
        self.ufl_form = None
        self.kwargs = None
        self.values = None


class Form_Coefficient_Edge(graph.Edge):
    """The scalar form's reverse derivative at its recorded input values.

    Attributes:
        ctx: The original coefficient differentiated by this edge.
    """

    def calculate_adjoint(self, value):
        """
        The method provides the adjoint equation for the derivative of the form with respect to a coefficient.

        Returns:
            (PETSc.Vec): The accumulated gradient up to this point in the computational graph.
            Only its entries owned by the calling rank are valid.

        """
        coefficient = self.ctx
        ufl_form, values = self.successor.recorded()
        form_derivative = ufl.derivative(ufl_form, values[coefficient])
        gradient = assemble_vector(form_derivative, self.successor.kwargs, scale=value)

        # A form that does not depend on the coefficient has a zero gradient.
        return zeros(coefficient.x.petsc_vec) if gradient is None else gradient


class Form_Constant_Edge(graph.Edge):
    """
    Edge providing the adjoint equation for the derivative of the form with respect to a constant.

    Attributes:
        ctx: The original Constant differentiated by this edge.
    """

    def calculate_adjoint(self, value):
        """
        The method provides the adjoint equation for the derivative of the form with respect to a constant.

        Returns:
            float or complex or PETSc.Vec: The accumulated gradient up to this point in the
            computational graph, a number for a scalar constant.

        """

        constant = self.ctx
        ufl_form, values = self.successor.recorded()
        function = real_function(constant)
        form_derivative = ufl_derivative_constant(
            ufl_form,
            values[constant],
            stand_in_function=function,
        )
        gradient = assemble_vector(form_derivative, self.successor.kwargs, scale=value)

        # A form that does not depend on the constant has a zero gradient.
        if gradient is None:
            if not constant.ufl_shape:
                return 0.0
            gradient = zeros(function.x.petsc_vec)
        return scalar(constant, gradient)
