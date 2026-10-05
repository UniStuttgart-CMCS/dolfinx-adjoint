"""Meshes and parameters shared by the integration tests.

Every PDE problem is set up in the conftest.py of the folder whose tests use it.
"""

from __future__ import annotations

import pytest
from dolfinx import mesh
from mpi4py import MPI


@pytest.fixture(scope="module")
def unit_square_mesh() -> mesh.Mesh:
    """Create a 64-by-64 triangle mesh of the unit square per test module."""
    return mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)


@pytest.fixture(
    scope="module",
    params=["nonlinear", "linear"],
)
def solver(request):
    return request.param


@pytest.fixture(scope="module")
def unit_interval_mesh() -> mesh.Mesh:
    """Create a unit-interval mesh with 12 cells per test module."""
    return mesh.create_unit_interval(MPI.COMM_WORLD, 12)
