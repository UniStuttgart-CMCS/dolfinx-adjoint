"""Shared fixtures for unit tests."""

import pytest
from dolfinx import mesh
from mpi4py import MPI


@pytest.fixture(scope="module")
def unit_square_mesh() -> mesh.Mesh:
    """Create one unit-square mesh on COMM_SELF per test module."""
    return mesh.create_unit_square(MPI.COMM_SELF, 4, 4)


@pytest.fixture(
    scope="module",
    params=[MPI.COMM_SELF, MPI.COMM_WORLD],
    ids=["COMM_SELF", "COMM_WORLD"],
)
def unit_square_mesh_per_comm(request: pytest.FixtureRequest) -> mesh.Mesh:
    """Create one unit-square mesh per communicator and test module."""
    return mesh.create_unit_square(request.param, 4, 4)


@pytest.fixture(scope="module")
def left_half_unit_sqaure_mesh(
    unit_square_mesh: mesh.Mesh,
) -> tuple[mesh.Mesh, mesh.EntityMap]:
    """The left half of the unit square, with the map back to the cells it was cut from."""
    tdim = unit_square_mesh.topology.dim
    cells = mesh.locate_entities(unit_square_mesh, tdim, lambda x: x[0] <= 0.5)
    submesh, cell_map, _, _ = mesh.create_submesh(unit_square_mesh, tdim, cells)
    return submesh, cell_map
