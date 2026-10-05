"""Unit tests for graph-tracked problems and their derivatives."""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph
from dolfinx_adjoint import fem as fem_ad


@pytest.mark.parametrize("measure_name", ["dx", "ds", "dS"])
@pytest.mark.parametrize(
    "value",
    [3.0, (3.0,), (3.0, 2.0, 1.0)],
    ids=["scalar", "vector1", "vector3"],
)
def test_problem_constant_gradient_on_measures(
    unit_square_mesh_per_comm: mesh.Mesh, measure_name: str, value: float | tuple
) -> None:
    """Catch incorrect Constant components, lost restrictions on dS, or a scalar summed
    on one rank only.

    J = s ∫ u for the projection u of the load |c|² v, so J = s |c|² |measure|, whose
    derivative 2 s c |measure| does not depend on u.
    """

    domain = unit_square_mesh_per_comm
    graph_ = Graph()
    value = np.asarray(value, dtype=ScalarType)
    c = fem_ad.Constant(domain, value, graph=graph_)
    measure = ufl.Measure(measure_name, domain=domain)
    measure_size = domain.comm.allreduce(
        fem_ad.assemble_scalar(fem_ad.form(1.0 * measure)), op=MPI.SUM
    )

    V = fem.functionspace(domain, ("Lagrange", 1))
    uh = fem_ad.Function(V, graph=graph_)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    restricted_c = c("+") if measure_name == "dS" else c
    load_test = ufl.avg(v) if measure_name == "dS" else v
    L = ufl.inner(restricted_c, restricted_c) * ufl.conj(load_test) * measure

    problem = fem_ad.petsc.LinearProblem(
        a,
        L,
        u=uh,
        petsc_options_prefix="test_constant_measures_",
        graph=graph_,
        adjoint_petsc_options={
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-14,
            "ksp_error_if_not_converged": True,
        },
    )
    problem.solve(graph=graph_)
    seed = -2.5
    J = fem_ad.assemble_scalar(
        fem_ad.form(seed * uh * ufl.dx, graph=graph_), graph=graph_
    )

    (gradient,) = graph_.backprop(J, c)

    expected = seed * 2.0 * value * measure_size
    if value.ndim == 0:
        assert np.isscalar(gradient) and np.isclose(gradient, expected)
    else:
        assert gradient.getSize() == expected.size
        if gradient.getLocalSize() > 0:
            np.testing.assert_allclose(gradient.array_r, expected.ravel())


def test_problem_records_a_solution_that_is_not_in_the_graph(
    unit_square_mesh: mesh.Mesh,
) -> None:
    """Catch a problem built on a solution the graph does not know."""
    domain = unit_square_mesh
    graph_ = Graph()
    c = fem_ad.Constant(domain, ScalarType(2.0), graph=graph_)

    V = fem.functionspace(domain, ("Lagrange", 1))
    uh = fem_ad.Function(V, name="uh")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    direct_solver = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = fem_ad.petsc.LinearProblem(
        c * ufl.inner(u, v) * ufl.dx,
        ufl.conj(v) * ufl.dx,
        u=uh,
        petsc_options_prefix="test_problem_untracked_u_",
        petsc_options=direct_solver,
        graph=graph_,
        adjoint_petsc_options=direct_solver,
    )
    problem.solve(graph=graph_)

    J_form = ufl.inner(uh, uh) * ufl.dx
    J = fem_ad.assemble_scalar(fem_ad.form(J_form, graph=graph_), graph=graph_)

    (gradient,) = graph_.backprop(J, c)

    # P1 contains the exact solution u = c⁻¹, so J = c⁻² on the unit square and
    # dJ/dc = -2 c⁻³ is -0.25.
    assert np.isclose(gradient, -0.25)


def test_problem_edges_compile_the_adjoint_with_the_recorded_arguments(
    unit_square_mesh: mesh.Mesh,
    left_half_unit_square_mesh: tuple[mesh.Mesh, mesh.EntityMap],
) -> None:
    """Catch the coefficient edge of a problem compiling its derivative bare."""
    submesh, cell_map = left_half_unit_square_mesh
    graph_ = Graph()

    V = fem.functionspace(unit_square_mesh, ("Lagrange", 1))
    f = fem_ad.Function(V, name="f")
    f.x.array[:] = 1.0
    graph_.track(f)

    W = fem.functionspace(submesh, ("Lagrange", 1))
    uh = fem_ad.Function(W, name="uh", graph=graph_)
    u = ufl.TrialFunction(W)
    v = ufl.TestFunction(W)
    dx = ufl.Measure("dx", domain=submesh)

    direct_solver = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = fem_ad.petsc.LinearProblem(
        ufl.inner(u, v) * dx,
        ufl.inner(f, v) * dx,
        u=uh,
        entity_maps=[cell_map],
        petsc_options_prefix="test_problem_entity_maps_",
        petsc_options=direct_solver,
        graph=graph_,
        adjoint_petsc_options=direct_solver,
    )
    problem.solve(graph=graph_)

    J_form = ufl.inner(uh, uh) * dx
    J = fem_ad.assemble_scalar(
        fem_ad.form(J_form, entity_maps=[cell_map], graph=graph_), graph=graph_
    )

    (gradient,) = graph_.backprop(J, f)

    # uh is the projection of f onto the half, so with f = 1 it is 1 and J is the area.
    # dJ/df is 2 f on the half, whose entries sum to twice that area.
    assert np.isclose(gradient.sum(), 1.0)


def _gradient_of_a_time_loop(domain: mesh.Mesh, order: str, reuse: bool) -> np.ndarray:
    """Step u' = Δu three times and differentiate ∫ u² with respect to the initial value.

    Args:
        domain: The mesh.
        order: Whether each step solves and then assigns the solution to the previous
            state, or assigns first.
        reuse: Whether one problem is set up before the loop and solved in every step,
            or a problem is set up in every step.

    Returns:
        The gradient.
    """
    graph_ = Graph()
    V = fem.functionspace(domain, ("Lagrange", 1))
    initial = fem_ad.Function(V, name="initial")
    initial.interpolate(lambda x: 1.0 + x[0] * x[1])
    graph_.track(initial)
    u_prev = fem_ad.Function(V, name="u_prev")
    u_prev.assign(initial, graph=graph_)
    u_next = fem_ad.Function(V, name="u_next")
    u_next.assign(initial, graph=graph_)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx + 0.1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_prev, v) * ufl.dx

    def set_up():
        return fem_ad.petsc.LinearProblem(
            a,
            L,
            u=u_next,
            petsc_options_prefix="test_problem_time_loop_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            graph=graph_,
        )

    problem = set_up() if reuse else None
    for _ in range(3):
        if order == "assign_then_solve":
            u_prev.assign(u_next, graph=graph_)
        (problem if reuse else set_up()).solve(graph=graph_)
        if order == "solve_then_assign":
            u_prev.assign(u_next, graph=graph_)

    J = fem_ad.assemble_scalar(
        fem_ad.form(ufl.inner(u_next, u_next) * ufl.dx, graph=graph_), graph=graph_
    )
    (gradient,) = graph_.backprop(J, initial)
    return gradient.array.copy()


@pytest.mark.parametrize("order", ["solve_then_assign", "assign_then_solve"])
def test_problem_solved_in_a_loop_is_differentiated_like_one_set_up_per_step(
    unit_square_mesh: mesh.Mesh, order: str
) -> None:
    """Catch a reused problem whose solves all depend on the values it was set up with."""
    reused = _gradient_of_a_time_loop(unit_square_mesh, order, reuse=True)
    per_step = _gradient_of_a_time_loop(unit_square_mesh, order, reuse=False)

    assert np.allclose(reused, per_step)


def _gradient_of_two_independent_solves(domain: mesh.Mesh, reuse: bool) -> np.ndarray:
    """Solve for two conductivities k and differentiate ∫ u₁² + u₂ with respect to f.

    Args:
        domain: The mesh.
        reuse: Whether one problem is solved for both conductivities, or a problem is
            set up for the second one.

    Returns:
        The gradient.
    """
    graph_ = Graph()
    V = fem.functionspace(domain, ("Lagrange", 1))
    f = fem_ad.Function(V, name="f")
    f.interpolate(lambda x: 1.0 + x[0])
    graph_.track(f)
    k = fem_ad.Function(V, name="k")
    k.interpolate(lambda x: 1.0 + x[0] * x[1])
    graph_.track(k)
    uh = fem_ad.Function(V, name="uh", graph=graph_)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (k * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)) * ufl.dx
    L = k * ufl.inner(f, v) * ufl.dx
    direct_solver = {"ksp_type": "preonly", "pc_type": "lu"}

    def set_up():
        return fem_ad.petsc.LinearProblem(
            a,
            L,
            u=uh,
            petsc_options_prefix="test_problem_independent_solves_",
            petsc_options=direct_solver,
            graph=graph_,
            adjoint_petsc_options=direct_solver,
        )

    problem = set_up()
    problem.solve(graph=graph_)
    first = fem_ad.Function(V, name="first")
    first.assign(uh, graph=graph_)
    new_k = fem_ad.Function(V)
    new_k.interpolate(lambda x: 2.0 + x[0] * x[1])
    k.assign(new_k, graph=graph_)
    (problem if reuse else set_up()).solve(graph=graph_)

    J = fem_ad.assemble_scalar(
        fem_ad.form((ufl.inner(first, first) + uh) * ufl.dx, graph=graph_), graph=graph_
    )
    (gradient,) = graph_.backprop(J, f)
    return gradient.array.copy()


def test_problem_solved_twice_is_differentiated_like_one_set_up_per_solve(
    unit_square_mesh: mesh.Mesh,
) -> None:
    """Catch the solves of one problem that share the workspace of their adjoint."""
    reused = _gradient_of_two_independent_solves(unit_square_mesh, reuse=True)
    per_solve = _gradient_of_two_independent_solves(unit_square_mesh, reuse=False)

    assert np.allclose(reused, per_solve)


@pytest.mark.parametrize("control_name", ["coefficient", "constant", "boundary"])
def test_problem_edges_ignore_writes_after_the_solve(
    controlled_problem, control_name: str
) -> None:
    """Catch a problem edge that differentiates at the values its inputs hold at backprop."""
    values = {**controlled_problem.default_values(), "constant": (2.0,)}
    reference = controlled_problem.evaluate(values, graph=Graph())
    (expected,) = reference.graph.backprop(
        reference.J, reference.controls[control_name]
    )

    evaluation = controlled_problem.evaluate(values, graph=Graph())
    f, c, g = (
        evaluation.controls[name] for name in ("coefficient", "constant", "boundary")
    )
    for function in (f, g, evaluation.uh):
        function.x.array[:] += 1.0
    c.value = (3.0,)
    (gradient,) = evaluation.graph.backprop(
        evaluation.J, evaluation.controls[control_name]
    )

    np.testing.assert_allclose(gradient.array, expected.array)


@pytest.mark.parametrize(
    ("operator", "source", "expected"),
    [
        (lambda u: 1 + u**2, lambda u: 1.0, -0.5),
        (lambda u: 1.0, lambda u: 2 * u, 2.0),
    ],
    ids=["operator", "source"],
)
def test_linear_problem_reading_its_solution_depends_on_its_previous_version(
    unit_square_mesh_per_comm: mesh.Mesh, operator, source, expected: float
) -> None:
    """Catch a solve whose forms read its solution differentiated as a fixed point."""
    domain = unit_square_mesh_per_comm
    graph_ = Graph()
    V = fem.functionspace(domain, ("DG", 0))
    m = fem_ad.Function(V, name="m")
    m.x.array[:] = 1.0
    graph_.track(m)
    uh = fem_ad.Function(V, name="uh", graph=graph_)
    uh.assign(m, graph=graph_)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    direct_solver = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = fem_ad.petsc.LinearProblem(
        operator(uh) * ufl.inner(u, v) * ufl.dx,
        source(uh) * ufl.conj(v) * ufl.dx,
        u=uh,
        petsc_options_prefix="test_linear_problem_reading_its_solution_",
        petsc_options=direct_solver,
        graph=graph_,
        adjoint_petsc_options=direct_solver,
    )
    problem.solve(graph=graph_)
    J = fem_ad.assemble_scalar(fem_ad.form(uh * ufl.dx, graph=graph_), graph=graph_)

    (gradient,) = graph_.backprop(J, m)

    assert np.isclose(gradient.sum(), expected)


@pytest.mark.parametrize("kind", ["linear", "nonlinear"])
def test_problem_set_up_outside_the_graph_is_not_solved_into_it(
    unit_interval_mesh: mesh.Mesh, kind: str
) -> None:
    """Catch a solve recorded without the PDE term of its problem."""
    V = fem.functionspace(unit_interval_mesh, ("Lagrange", 1))
    m, u = fem_ad.Function(V), fem_ad.Function(V)
    trial, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    prefix = f"test_{kind}_problem_set_up_outside_the_graph_"
    if kind == "linear":
        problem = fem_ad.petsc.LinearProblem(
            trial * v * ufl.dx, m * v * ufl.dx, u=u, petsc_options_prefix=prefix
        )
    else:
        problem = fem_ad.petsc.NonlinearProblem(
            (u - m) * v * ufl.dx, u, petsc_options_prefix=prefix
        )
    graph_ = Graph()
    graph_.track(m)
    with pytest.raises(ValueError, match="not recorded in this graph"):
        problem.solve(graph=graph_)
    graph_.release()
