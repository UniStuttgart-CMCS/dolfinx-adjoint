"""Unit tests for graph-tracked problems and their derivatives."""

import numpy as np
import pytest
import ufl
from dolfinx import mesh
from dolfinx.fem.petsc import assemble_vector
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph, fem


def test_problem_constant_gradient_is_taken_at_the_current_value(
    unit_square_mesh: mesh.Mesh,
) -> None:
    """The problem edge differentiates at the value of c, not its constructor argument."""
    domain = unit_square_mesh
    graph_ = Graph()
    c = fem.Constant(domain, ScalarType(1.0), graph=graph_)
    c.value = 2.0

    V = fem.functionspace(domain, ("Lagrange", 1))
    uh = fem.Function(V, name="uh", graph=graph_)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    direct_solver = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = fem.petsc.LinearProblem(
        c**2 * ufl.inner(u, v) * ufl.dx,
        ufl.conj(v) * ufl.dx,
        u=uh,
        petsc_options_prefix="test_problem_constant_gradient_",
        petsc_options=direct_solver,
        adjoint_petsc_options=direct_solver,
        graph=graph_,
    )
    problem.solve(graph=graph_)

    J_form = ufl.inner(uh, uh) * ufl.dx
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    gradient = graph_.backprop(id(J), id(c))

    # P1 contains the exact solution u = c⁻², so J = c⁻⁴ on the unit square and
    # dJ/dc = -4 c⁻⁵ is -0.125. Evaluating ∂F/∂c = 2c u v at the constructor
    # argument c = 1 halves it to -0.0625.
    assert np.isclose(gradient, -0.125)


def test_nonlinear_problem_constant_edge_gradient(
    unit_square_mesh_per_comm: mesh.Mesh,
) -> None:
    """The scalar gradient includes the seed and contributions from every rank."""
    domain = unit_square_mesh_per_comm
    graph_ = Graph()
    c = fem.Constant(domain, ScalarType(8.0), graph=graph_)

    V = fem.functionspace(domain, ("Lagrange", 1))
    uh = fem.Function(V, graph=graph_)
    # Supply the exact state, including ghosts, to test the edge without a forward solve.
    uh.x.array[:] = 2.0
    v = ufl.TestFunction(V)
    F = (uh**3 - c) * ufl.conj(v) * ufl.dx
    problem = fem.petsc.NonlinearProblem(
        F,
        uh,
        petsc_options_prefix="test_constant_direction_",
        adjoint_petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-14,
            "ksp_error_if_not_converged": True,
        },
        graph=graph_,
    )
    edge = graph_.get_edge(graph_.get_node(id(c)), graph_.get_node(id(problem)))

    x = ufl.SpatialCoordinate(domain)
    seed = assemble_vector(fem.form((1 + x[0]) * ufl.conj(v) * ufl.dx))
    try:
        seed.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        edge.input_value = seed
        gradient = edge.calculate_adjoint()
    finally:
        seed.destroy()

    # u = c^(1/3), J = integral((1 + x) u) = 3u/2 on the unit square.
    # Thus dJ/dc = (3/2)/(3u^2) = 1/8 at c = 8, on every rank.
    assert np.isclose(gradient, 0.125)


@pytest.mark.parametrize("measure_name", ["dx", "ds", "dS"])
@pytest.mark.parametrize(
    "value",
    [3.0, (3.0,), (3.0, 2.0), (3.0, 2.0, 1.0)],
    ids=["scalar", "vector1", "vector2", "vector3"],
)
def test_problem_constant_gradient_on_measures(
    unit_square_mesh_per_comm: mesh.Mesh, measure_name: str, value: float | tuple
) -> None:
    """Catch incorrect Constant components."""
    domain = unit_square_mesh_per_comm
    graph_ = Graph()
    value = np.asarray(value, dtype=ScalarType)
    c = fem.Constant(domain, value, graph=graph_)
    measure = ufl.Measure(measure_name, domain=domain)
    measure_size = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(1.0 * measure)), op=MPI.SUM
    )

    V = fem.functionspace(domain, ("Lagrange", 1))
    uh = fem.Function(V, graph=graph_)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    restricted_c = c("+") if measure_name == "dS" else c
    load_test = ufl.avg(v) if measure_name == "dS" else v
    L = ufl.inner(restricted_c, restricted_c) * ufl.conj(load_test) * measure

    # Both partial derivatives are independent of uh, so no forward solve is needed.
    problem = fem.petsc.LinearProblem(
        a,
        L,
        u=uh,
        petsc_options_prefix="test_constant_measures_",
        adjoint_petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-14,
            "ksp_error_if_not_converged": True,
        },
        graph=graph_,
    )
    edge = graph_.get_edge(graph_.get_node(id(c)), graph_.get_node(id(problem)))

    seed = -2.5
    adjoint_input = assemble_vector(fem.form(seed * ufl.conj(v) * ufl.dx))
    try:
        adjoint_input.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        edge.input_value = adjoint_input
        gradient = edge.calculate_adjoint()
    finally:
        adjoint_input.destroy()

    expected = seed * 2.0 * value * measure_size
    if value.ndim == 0:
        assert np.isscalar(gradient) and np.isclose(gradient, expected)
    else:
        assert gradient.getSize() == expected.size
        if gradient.getLocalSize() > 0:
            np.testing.assert_allclose(gradient.array_r, expected.ravel())
