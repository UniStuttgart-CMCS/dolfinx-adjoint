"""
Integration tests for the setup of the Poisson problem.

This test module verifies the premises that the gradient tests in
``test_poisson.py`` and ``test_poisson_taylor.py`` build on: the controlled part
of the boundary is the one the fixture claims it is, and the computational graph
replays the forward problem when an input is modified.
"""

import numpy as np
import pytest
from mpi4py import MPI


@pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason="Boundary dof ownership is checked locally in this test",
)
@pytest.mark.parametrize("solver", ["linear"], indirect=True)
@pytest.mark.parametrize("boundary_condition", ["inflow"], indirect=True)
def test_Poisson_controlled_boundary_dofs(poisson_problem):
    """Check the open controlled boundary in serial."""
    uD_control = poisson_problem["uD_control"]
    controlled_dofs = poisson_problem["control_dofs"]
    assert controlled_dofs.size > 0
    dof_coordinates = uD_control.function_space.tabulate_dof_coordinates()
    controlled_coordinates = dof_coordinates[controlled_dofs]
    invalid = np.count_nonzero(
        ~np.isclose(controlled_coordinates[:, 0], 0.0)
        | np.isclose(controlled_coordinates[:, 1], 0.0)
        | np.isclose(controlled_coordinates[:, 1], 1.0)
    )
    assert invalid == 0


def test_Poisson_graph_recalculate(poisson_problem):
    """Ensure graph.recalculate updates J after modifying f."""
    graph_ = poisson_problem["graph_"]
    f = poisson_problem["f"]
    J = poisson_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = f.function_space.mesh.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    f_org = f.x.array.copy()

    f.x.array[:] = f_org + 0.5
    graph_.recalculate()
    J1 = comm.allreduce(J_node.object, op=MPI.SUM)
    assert not np.isclose(J1, J0)

    f.x.array[:] = f_org
    graph_.recalculate()
    J2 = comm.allreduce(J_node.object, op=MPI.SUM)
    np.testing.assert_allclose(J2, J0)
