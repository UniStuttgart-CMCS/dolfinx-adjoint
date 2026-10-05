"""DOLFINx mesh lookup and recorded derivative value helpers."""

import weakref
from typing import Any

from dolfinx import mesh as dolfinx_mesh
from petsc4py import PETSc


def constant_mesh(constant) -> dolfinx_mesh.Mesh:
    """The DOLFINx mesh of a Constant.

    An overloaded Constant keeps the mesh it was created on. A Constant of DOLFINx, e.g.
    the one :py:func:`dolfinx.fem.dirichletbc` creates for an array, reaches it through
    its UFL domain only, whose cargo is the C++ mesh, which is wrapped without copying
    or communicating.

    Args:
        constant (dolfinx.fem.Constant): The Constant.

    Returns:
        dolfinx.mesh.Mesh: The mesh.
    """
    domain = getattr(constant, "domain", None)
    if isinstance(domain, dolfinx_mesh.Mesh):
        return domain
    domain = constant.ufl_domain()
    return dolfinx_mesh.Mesh(domain.ufl_cargo(), domain)


def resolve_capture(entry: Any) -> Any:
    """Resolve a captured node reference to its snapshot, or return an untracked value.

    Args:
        entry: A weak reference returned by ``Graph.capture``, or an untracked value.

    Returns:
        The recorded version's saved value, or the untracked value unchanged.

    Raises:
        ReferenceError: If the node owning a captured value no longer exists.
        RuntimeError: If the captured version has no saved value.
    """
    if not isinstance(entry, weakref.ReferenceType):
        return entry
    node = entry()
    if node is None:
        raise ReferenceError("The recorded version is no longer alive.")
    if node.snapshot is None:
        raise RuntimeError(f"The values of {node} have not been saved.")
    return node.snapshot


def add(total: Any, value: Any) -> Any:
    """Sum the adjoint values a node receives, without modifying either of them.

    This helper is necessary because the adjoint values can be scalars, or distributed PETSc.Vec
    or nothing at all, and a simple summation would corrupt the data.

    Args:
        total (Any): The values summed so far, a scalar or PETSc.Vec, or None if
            there are none yet.
        value (Any): The next value, of the same kind as total, or None if it vanishes.

    Returns:
        Any: The sum, a new vector if both values are vectors, or the other value if
        one of them is None.

    Raises:
        TypeError: If the values cannot be summed.
    """
    if total is None:
        return value
    if value is None:
        return total
    if isinstance(total, PETSc.Vec):
        result = total.copy()
        result.axpy(1.0, value)
        return result
    return total + value


def zeros(like: PETSc.Vec) -> PETSc.Vec:
    """A zero vector in the layout of another."""
    vector = like.duplicate()
    vector.zeroEntries()
    return vector


def scalar(constant, gradient: PETSc.Vec):
    """The gradient of a Constant from the vector of its components.

    The components are copied to a vector on the communicator of the mesh of the
    Constant, as the vector they are assembled into lives on the communicator of a
    temporary real space, which DOLFINx 0.11 frees with that space.
    """
    try:
        if not constant.ufl_shape:
            return gradient.sum()
        components = PETSc.Vec().createMPI(
            gradient.getSizes(), comm=constant_mesh(constant).comm
        )
        components.array_w[:] = gradient.array_r[: components.getLocalSize()]
        return components
    finally:
        gradient.destroy()
