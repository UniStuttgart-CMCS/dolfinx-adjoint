"""Unit tests for gradient accumulation in computational graph nodes."""

import numpy as np
from dolfinx import default_scalar_type, la
from dolfinx.common import IndexMap
from mpi4py import MPI

from dolfinx_adjoint.node import Node


def test_accumulate_grad_sums_shared_vector():
    """Three contributions sharing a vector each contribute once to the sum."""
    node = Node(object())
    contribution = la.vector(IndexMap(MPI.COMM_SELF, 1), dtype=default_scalar_type)
    contribution.array[:] = [1.0]

    # Identity paths can pass the same adjoint vector to the accumulator.
    for _ in range(3):
        node.accumulate_grad(contribution.petsc_vec)

    np.testing.assert_allclose(node.get_grad().array, [3.0])


def test_accumulate_grad_preserves_contribution():
    """Accumulation preserves the input values needed by other branches."""
    node = Node(object())
    contribution = la.vector(IndexMap(MPI.COMM_SELF, 1), dtype=default_scalar_type)
    contribution.array[:] = [1.0]

    for _ in range(2):
        node.accumulate_grad(contribution.petsc_vec)
        np.testing.assert_allclose(contribution.array, [1.0])
