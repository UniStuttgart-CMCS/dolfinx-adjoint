import numpy as np
import pytest
import ufl
from dolfinx import mesh
from dolfinx.fem.petsc import create_vector
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph, fem
from dolfinx_adjoint.fem.petsc import AdjointProblem
from dolfinx_adjoint.graph import Edge


@pytest.fixture(params=["coefficient", "constant", "boundary"])
def adjoint_edge(request, option) -> Edge:
    """An edge of a problem that has been given options for its adjoint equations."""
    forward_options_prefix = "test_adjoint_solver_options_forward_"
    adjoint_options = {"ksp_type": "cg", "pc_type": "jacobi"}
    adjoint_options[option] = "invalid_adjoint_solver_for_test"

    graph_ = Graph()

    domain = mesh.create_unit_interval(MPI.COMM_SELF, 4)
    V = fem.functionspace(domain, ("Lagrange", 1))
    W = fem.functionspace(domain, ("DG", 0))

    uh = fem.Function(V, name="uh", graph=graph_)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Function(W, name="f", graph=graph_)
    f.x.array[:] = 1.0
    c = fem.Constant(domain, ScalarType(2.0), name="c", graph=graph_)

    uD = fem.Function(V, name="uD", graph=graph_)
    uD.x.array[:] = 1.0
    boundary_dofs = fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    )
    bcs = [fem.dirichletbc(uD, boundary_dofs, graph=graph_)]

    problem = fem.petsc.LinearProblem(
        c * ufl.inner(u, v) * ufl.dx,
        ufl.inner(f, v) * ufl.dx,
        u=uh,
        bcs=bcs,
        petsc_options_prefix=forward_options_prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        adjoint_petsc_options=adjoint_options,
        graph=graph_,
    )
    # The solve provides the updated solution that the edge of the coefficient reads.
    problem.solve(graph=graph_)

    predecessor = {"coefficient": f, "constant": c, "boundary": bcs[0]}[request.param]
    edge = graph_.get_edge(
        graph_.get_node(id(predecessor)), graph_.get_node(id(problem))
    )

    edge.input_value = create_vector(V)

    return edge


@pytest.fixture
def adjoint_problem_data():
    """Form, vector, and solution for configuring an adjoint problem."""
    domain = mesh.create_unit_interval(MPI.COMM_SELF, 4)
    V = fem.functionspace(domain, ("Lagrange", 1))
    a = ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * ufl.dx
    b = create_vector(V)
    b.set(0.0)
    try:
        yield a, b, fem.Function(V)
    finally:
        b.destroy()


def test_adjoint_solver_is_configured_by_the_given_options(adjoint_problem_data):
    """The caller's options configure the adjoint solver."""
    problem = AdjointProblem(
        *adjoint_problem_data,
        petsc_options={"ksp_type": "cg", "pc_type": "lu"},
        petsc_options_prefix="test_adjoint_solver_options_",
    )
    assert problem.solver.getType() == "cg"
    assert problem.solver.getPC().getType() == "lu"


@pytest.mark.parametrize("option", ["ksp_type", "pc_type"])
def test_adjoint_solver_receives_the_options_given_to_the_problem(adjoint_edge):
    """Each edge lets PETSc reject invalid adjoint KSP and PC options."""
    with pytest.raises(PETSc.Error, match="invalid_adjoint_solver_for_test"):
        adjoint_edge.calculate_adjoint()


def test_adjoint_solver_removes_its_options_from_the_database(adjoint_problem_data):
    """The solver takes the options it sets out of the global database again."""
    prefix = "test_adjoint_solver_database_"
    problem = AdjointProblem(
        *adjoint_problem_data,
        petsc_options={"ksp_type": "cg"},
        petsc_options_prefix=prefix,
    )
    del problem

    assert prefix + "ksp_type" not in PETSc.Options()


def test_adjoint_solver_is_destroyed_after_use(adjoint_problem_data):
    """The solver of the adjoint equations is destroyed when it is no longer used."""
    problem = AdjointProblem(
        *adjoint_problem_data, petsc_options_prefix="test_adjoint_solver_lifetime_"
    )
    solver = problem.solver
    del problem

    assert solver.handle == 0
