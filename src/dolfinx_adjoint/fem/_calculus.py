"""First-order UFL derivatives of the recorded forms."""

import ufl
from basix.ufl import real_element
from dolfinx import fem
from petsc4py import PETSc
from ufl.algorithms import expand_derivatives

from dolfinx_adjoint.graph.dolfinx_helpers import constant_mesh


def real_space(constant: ufl.Constant) -> fem.FunctionSpace:
    """The space of a real element of the shape of a Constant, on the mesh of the Constant."""
    mesh = constant_mesh(constant)
    return fem.functionspace(
        mesh, real_element(mesh.basix_cell(), value_shape=constant.ufl_shape)
    )


def real_function(constant: ufl.Constant) -> fem.Function:
    """A Function that stands in for a Constant where UFL differentiates."""
    return fem.Function(real_space(constant), dtype=constant.dtype)


def ufl_derivative_constant(
    expression, terminal, argument=None, stand_in_function=None
):
    """The derivative of a form or expression with respect to a Function or Constant.

    Args:
        expression: A form or expression.
        terminal: The Function or Constant.
        argument: The argument or direction of the derivative. Defaults to a new
            argument in the space of the Function, or of the stand-in of the Constant.
        stand_in_function: For a Constant, a Function on a real element with the shape
            of the Constant, which supplies the space of the new argument.

    Returns:
        The derivative, with the Constant in place again.
    """
    if not isinstance(terminal, ufl.Constant):
        return ufl.derivative(expression, terminal, argument)
    replaced = ufl.replace(expression, {terminal: stand_in_function})
    result = ufl.derivative(replaced, stand_in_function, argument)
    return ufl.replace(result, {stand_in_function: terminal})


def assemble_vector(form: ufl.Form, kwargs: dict, scale=None) -> PETSc.Vec | None:
    """Assemble a vector, weight its rank-local integrals and sum them at shared dofs.

    Args:
        form: A one-form whose derivatives are expanded before assembly.
        kwargs: The arguments the form is compiled with.
        scale: A rank-local weight of the integrals, applied before contributions of
            other ranks are added, or None.

    Returns:
        The vector, of which only the owned entries are valid, or None.
    """
    form = expand_derivatives(form)
    # An expanded form vanishes when no integral is left.
    if form.empty():
        return None
    vector = fem.petsc.assemble_vector(fem.form(form, **kwargs))
    if scale is not None:
        with vector.localForm() as local:
            local.scale(scale)
    vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vector
