"""Shared fixtures for unit tests."""

import pytest
from dolfinx import mesh
from mpi4py import MPI


@pytest.fixture(scope="module")
def unit_square_mesh():
    """Create one unit-square mesh on COMM_SELF per test module."""
    return mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
