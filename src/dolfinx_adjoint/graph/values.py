"""The values a propagation passes along the edges of a graph, and how they are summed."""

from typing import Any

from petsc4py import PETSc


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
