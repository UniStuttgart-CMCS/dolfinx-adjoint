from contextlib import contextmanager

import numpy as np
import pytest
import ufl
from dolfinx import mesh
from dolfinx.fem.petsc import create_vector
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph, fem
from dolfinx_adjoint.fem.petsc import (
    DEFAULT_ADJOINT_PETSC_OPTIONS,
    _create_adjoint_solver,
)
from dolfinx_adjoint.graph import Edge


@pytest.fixture(params=["coefficient", "constant", "boundary"])
def adjoint_edge(request) -> tuple[Edge, dict[str, str], str]:
    """An edge of a problem that has been given options for its adjoint equations."""
    forward_options_prefix = "test_adjoint_solver_options_forward_"
    adjoint_options = {"ksp_type": "cg", "pc_type": "jacobi"}
    assert adjoint_options != DEFAULT_ADJOINT_PETSC_OPTIONS

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

    return edge, adjoint_options, forward_options_prefix + "adjoint_"


def test_adjoint_solver_is_configured_by_the_given_options():
    """The caller's options configure the adjoint solver, and nothing else does."""
    A = PETSc.Mat().createAIJ((2, 2), comm=MPI.COMM_SELF)
    A.setUp()
    A.assemble()

    with _create_adjoint_solver(
        A,
        petsc_options={"ksp_type": "cg", "pc_type": "lu"},
        petsc_options_prefix="test_adjoint_solver_options_",
    ) as solver:
        # Catches a solver that ignores the given options and keeps its own type.
        assert solver.getType() == "cg"
        # Catches a hardcoded factorisation surviving next to the given options,
        # whether they are ignored outright or merged on top of the defaults.
        assert solver.getPC().getFactorSolverType() != "mumps"


def test_adjoint_solver_receives_the_options_given_to_the_problem(
    adjoint_edge, monkeypatch
):
    """Each edge creates its adjoint solver with the options of its problem."""
    edge, petsc_options, petsc_options_prefix = adjoint_edge

    created_with = []

    @contextmanager
    def _record(A, petsc_options=None, petsc_options_prefix=None):
        created_with.append((petsc_options, petsc_options_prefix))
        with _create_adjoint_solver(A, petsc_options, petsc_options_prefix) as solver:
            yield solver

    monkeypatch.setattr(fem.petsc, "_create_adjoint_solver", _record)

    edge.calculate_adjoint()

    assert created_with == [(petsc_options, petsc_options_prefix)]


def test_adjoint_solver_removes_its_options_from_the_database():
    """The solver takes the options it sets out of the global database again."""
    A = PETSc.Mat().createAIJ((2, 2), comm=MPI.COMM_SELF)
    A.setUp()
    A.assemble()

    prefix = "test_adjoint_solver_database_"
    with _create_adjoint_solver(
        A,
        petsc_options={"ksp_type": "cg"},
        petsc_options_prefix=prefix,
    ):
        pass

    assert prefix + "ksp_type" not in PETSc.Options()
