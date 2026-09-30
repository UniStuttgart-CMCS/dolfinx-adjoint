"""Meshes and parameters shared by the integration tests.

Every PDE problem is set up in the conftest.py of the folder whose tests use it.
"""

from __future__ import annotations

import pytest
from dolfinx import mesh
from mpi4py import MPI


@pytest.fixture(
    scope="module",
    params=[mesh.CellType.triangle, mesh.CellType.quadrilateral],
    ids=["triangle", "quadrilateral"],
)
def unit_square_mesh(request) -> mesh.Mesh:
    """Create a 64-by-64 unit-square mesh per cell type and test module."""
    return mesh.create_unit_square(MPI.COMM_WORLD, 64, 64, request.param)


@pytest.fixture(
    scope="module",
    params=["nonlinear", "linear"],
)
def solver(request):
    return request.param
