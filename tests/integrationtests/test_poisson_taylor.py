import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI


def _convergence_rates(errors: np.ndarray, steps: np.ndarray) -> list:
    rates = []
    for i in range(1, len(steps)):
        rates.append(
            np.log(errors[i] / errors[i - 1]) / np.log(steps[i] / steps[i - 1])
        )
    return rates


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


def test_Poisson_taylor_f(poisson_problem):
    """Taylor test for J with respect to the forcing term f (graph backprop)."""
    graph_ = poisson_problem["graph_"]
    f = poisson_problem["f"]
    J = poisson_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = f.function_space.mesh.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    df = fem.Function(f.function_space)
    df.interpolate(lambda x: x[0] + np.sin(x[1]))

    grad = graph_.backprop(id(J), id(f))
    grad_array = grad.array if hasattr(grad, "array") else np.array(grad)
    dJ = comm.allreduce(np.dot(grad_array, df.x.array), op=MPI.SUM)

    f_org = f.x.array.copy()
    step_length = 1e-2
    steps = [step_length * (0.5**i) for i in range(4)]
    errors0 = []
    errors = []
    try:
        for h in steps:
            f.x.array[:] = f_org + h * df.x.array
            graph_.recalculate()
            Jh = comm.allreduce(J_node.object, op=MPI.SUM)
            errors0.append(abs(Jh - J0))
            errors.append(abs(Jh - J0 - h * dJ))
    finally:
        f.x.array[:] = f_org
        graph_.recalculate()

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)


def test_Poisson_taylor_nu(poisson_problem):
    """Taylor test for J with respect to the diffusion coefficient nu."""
    graph_ = poisson_problem["graph_"]
    nu = poisson_problem["nu"]
    J = poisson_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = nu.domain.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    direction = 1.0
    grad = graph_.backprop(id(J), id(nu))
    dJ = comm.allreduce(float(grad) * direction, op=MPI.SUM)

    nu_org = float(nu.value)
    step_length = 1e-2
    steps = [step_length * (0.5**i) for i in range(4)]
    errors0 = []
    errors = []
    try:
        for h in steps:
            nu.value = float(nu_org + h * direction)
            graph_.recalculate()
            Jh = comm.allreduce(J_node.object, op=MPI.SUM)
            errors0.append(abs(Jh - J0))
            errors.append(abs(Jh - J0 - h * dJ))
    finally:
        nu.value = nu_org
        graph_.recalculate()

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)


@pytest.mark.xfail(
    reason=(
        "Dirichlet BC sensitivity in graph path does not yet match lifted system. "
        "See ISSUES.md#bc-taylor"
    ),
    strict=False,
)
def test_Poisson_taylor_bc(poisson_problem):
    """Taylor test for J with respect to the left boundary condition uD_L."""
    graph_ = poisson_problem["graph_"]
    uD_L = poisson_problem["uD_L"]
    boundary_dofs_L = poisson_problem["boundary_dofs_L"]
    J = poisson_problem["J"]

    J_node = graph_.get_node(id(J))
    assert J_node is not None

    comm = uD_L.function_space.mesh.comm
    J0 = comm.allreduce(J_node.object, op=MPI.SUM)

    grad = graph_.backprop(id(J), id(uD_L))
    grad_array = grad.array if hasattr(grad, "array") else np.array(grad)

    direction = fem.Function(uD_L.function_space)
    direction.x.array[:] = 0.0
    direction.x.array[boundary_dofs_L] = grad_array[boundary_dofs_L]

    local_norm = np.dot(direction.x.array, direction.x.array)
    norm = comm.allreduce(local_norm, op=MPI.SUM)
    if norm == 0.0:
        pytest.skip("Boundary gradient is zero; Taylor test is not informative.")
    direction.x.array[:] /= np.sqrt(norm)

    dJ = comm.allreduce(np.dot(grad_array, direction.x.array), op=MPI.SUM)

    u_org = uD_L.x.array.copy()
    step_length = 1e-2
    steps = [step_length * (0.5**i) for i in range(4)]
    errors0 = []
    errors = []
    try:
        for h in steps:
            uD_L.x.array[:] = u_org + h * direction.x.array
            graph_.recalculate()
            Jh = comm.allreduce(J_node.object, op=MPI.SUM)
            errors0.append(abs(Jh - J0))
            errors.append(abs(Jh - J0 - h * dJ))
    finally:
        uD_L.x.array[:] = u_org
        graph_.recalculate()

    rates0 = _convergence_rates(errors0, steps)
    np.testing.assert_allclose(rates0, 1.0, atol=0.2)
    rates = _convergence_rates(errors, steps)
    np.testing.assert_allclose(rates, 2.0, atol=0.2)
