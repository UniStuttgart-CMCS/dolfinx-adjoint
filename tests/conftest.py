"""Numerical checks shared by the unit and the integration tests."""

import numpy as np
import pytest
from petsc4py import PETSc


@pytest.fixture
def mode() -> str:
    """The reverse mode retained by these regression tests."""
    return "reverse"


def _pair(derivative, direction):
    """A derivative applied to a direction, summed over all ranks.

    Args:
        derivative: A PETSc.Vec, whose owned entries are paired, or a number.
        direction: A Function or PETSc.Vec for a vector, an array of the shape of a
            Constant for the vector of its components, a number for a number.
    """
    if isinstance(derivative, PETSc.Vec):
        if isinstance(direction, PETSc.Vec):
            vector = direction
        elif isinstance(direction, np.ndarray):
            # The components of a Constant, each owned by one rank, as in its gradient.
            vector = derivative.duplicate()
            start, end = derivative.getOwnershipRange()
            vector.array[:] = direction.ravel()[start:end]
        else:
            vector = direction.x.petsc_vec
        return derivative.dot(vector)
    return derivative * direction


def _directional_derivative(graph, output, control, direction, mode, comm, seed=1.0):
    """Apply the recorded reverse derivative to a control direction.

    Args:
        graph: The graph of this forward evaluation.
        output: The recorded scalar or Function output.
        control: The recorded control or exact node.
        direction: The direction paired with the gradient.
        mode: The reverse mode of the retained regressions.
        comm: The forward evaluation's communicator.
        seed: The output adjoint seed.
    """
    (gradient,) = graph.backprop(output, control, grad_outputs=seed)
    return _pair(gradient, direction)


@pytest.fixture(scope="session")
def directional_derivative():
    """The derivative of a recorded output in a direction, in one mode of the graph.

    Returns:
        A function ``(graph, output, control, direction, mode, comm, seed=1.0)``, see
        :py:func:`_directional_derivative`.
    """
    return _directional_derivative
