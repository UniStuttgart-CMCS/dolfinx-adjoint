"""
Unit tests for the restriction performed by
:py:class:`dolfinx_adjoint.fem.bcs.DirichletBC_Edge`."""

import numpy as np
from dolfinx import default_scalar_type, la
from dolfinx.common import IndexMap
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx_adjoint.fem.bcs import DirichletBC_Edge
from dolfinx_adjoint.node import AbstractNode


def _restrict(input_value: PETSc.Vec, dofs, map=None) -> np.ndarray:
    """Run the restriction of a DirichletBC_Edge on the given input."""
    edge = DirichletBC_Edge(
        AbstractNode(object(), name="value"),
        AbstractNode(object(), name="bc"),
        ctx=[dofs, map],
    )
    edge.input_value = input_value
    return edge.calculate_adjoint().array


def test_restriction_keeps_only_the_controlled_dofs():
    """The gradient is masked onto the controlled dofs."""
    gradient = la.vector(IndexMap(MPI.COMM_SELF, 5), dtype=default_scalar_type)
    gradient.array[:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    dofs = np.array([0, 2, 4], dtype=np.int32)

    output = _restrict(gradient.petsc_vec, dofs)

    np.testing.assert_allclose(output, [1.0, 0.0, 3.0, 0.0, 5.0])


def test_restriction_with_two_controlled_dofs():
    """A boundary holding exactly two dofs is restricted like any other."""

    gradient = la.vector(IndexMap(MPI.COMM_SELF, 5), dtype=default_scalar_type)
    gradient.array[:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    dofs = np.array([1, 3], dtype=np.int32)

    output = _restrict(gradient.petsc_vec, dofs)

    np.testing.assert_allclose(output, [0.0, 2.0, 0.0, 4.0, 0.0])


def test_restriction_into_a_collapsed_space():
    """The map moves the restricted gradient into the collapsed space."""
    gradient = la.vector(IndexMap(MPI.COMM_SELF, 5), dtype=default_scalar_type)
    gradient.array[:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    dofs = np.array([1, 3], dtype=np.int32)
    # Entry i of the collapsed space corresponds to entry map[i] of the sub space.
    map = np.array([1, 3, 4], dtype=np.int32)

    output = _restrict(gradient.petsc_vec, dofs, map=map)

    np.testing.assert_allclose(output, [2.0, 4.0, 0.0])
