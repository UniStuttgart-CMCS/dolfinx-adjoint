"""Shared fixtures for unit tests."""

from __future__ import annotations

from dataclasses import dataclass

import dolfinx
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_graph_ad import Graph
from dolfinx_graph_ad import fem as fem_ad


@pytest.fixture(scope="module")
def unit_square_mesh() -> mesh.Mesh:
    """Create one unit-square mesh on COMM_SELF per test module."""
    return mesh.create_unit_square(MPI.COMM_SELF, 4, 4)


@pytest.fixture(
    scope="module",
    params=[MPI.COMM_SELF] + ([MPI.COMM_WORLD] if MPI.COMM_WORLD.size > 1 else []),
    ids=lambda comm: "COMM_SELF" if comm is MPI.COMM_SELF else "COMM_WORLD",
)
def unit_square_mesh_per_comm(request: pytest.FixtureRequest) -> mesh.Mesh:
    """Create one unit-square mesh per communicator and test module."""
    return mesh.create_unit_square(request.param, 4, 4)


@pytest.fixture(scope="module")
def unit_interval_mesh() -> mesh.Mesh:
    """Create one unit-interval mesh on COMM_SELF per test module."""
    return mesh.create_unit_interval(MPI.COMM_SELF, 4)


@pytest.fixture(scope="module")
def left_half_unit_square_mesh(
    unit_square_mesh: mesh.Mesh,
) -> tuple[mesh.Mesh, mesh.EntityMap]:
    """The left half of the unit square, with the map back to the cells it was cut from."""
    tdim = unit_square_mesh.topology.dim
    cells = mesh.locate_entities(unit_square_mesh, tdim, lambda x: x[0] <= 0.5)
    submesh, cell_map, _, _ = mesh.create_submesh(unit_square_mesh, tdim, cells)
    return submesh, cell_map


@dataclass
class ControlledEvaluation:
    """Controls, state and functional from one forward execution."""

    graph: Graph | None
    controls: dict  # The coefficient f, the constant c and the boundary value g.
    uh: fem_ad.Function
    J: float


class ControlledProblem:
    """Set up -∇·(k∇u) + c φ(u) = f with u = g on the left boundary, and J = ∫ u².

    φ(u) = u for a linear problem and φ(u) = u + u³ for a nonlinear one. The coefficient
    f, the constant c and the boundary value g are the controls, and each takes a
    different edge out of the problem. The coefficient k = 1 + xy is not tracked.
    """

    def __init__(self, domain: mesh.Mesh, problem_type: str):
        self.domain = domain
        self.problem_type = problem_type
        self.V = fem.functionspace(domain, ("Lagrange", 1))
        self.dofs = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0.0))
        self.petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

    def default_values(self) -> dict:
        """f = 1 + x, c = 2 and g = 1/2 + y, with the ghost entries of f and g."""
        f = dolfinx.fem.Function(self.V)
        f.interpolate(lambda x: 1.0 + x[0])
        g = dolfinx.fem.Function(self.V)
        g.interpolate(lambda x: 0.5 + x[1])
        return {
            "coefficient": f.x.array.copy(),
            "constant": 2.0,
            "boundary": g.x.array.copy(),
        }

    def evaluate(
        self, values: dict, graph: Graph | None = None
    ) -> ControlledEvaluation:
        """Solve the problem and assemble J at the given control values.

        Args:
            values: The values of the controls, see :py:meth:`default_values`. A
                constant given with a shape creates a Constant of that shape, whose
                first entry enters the problem.
            graph: A fresh graph to record into, or None for plain DOLFINx.

        Returns:
            The controls, state and functional of this evaluation.
        """
        f = fem_ad.Function(self.V, name="f")
        f.x.array[:] = values["coefficient"]
        constant = np.asarray(values["constant"], dtype=ScalarType)
        c = fem_ad.Constant(self.domain, constant, graph=graph)
        g = fem_ad.Function(self.V, name="g")
        g.x.array[:] = values["boundary"]
        if graph is not None:
            graph.track(f)
            graph.track(g)
        k = fem_ad.Function(self.V, name="k")
        k.interpolate(lambda x: 1.0 + x[0] * x[1])
        uh = fem_ad.Function(self.V, name="uh", graph=graph)
        bcs = [fem_ad.dirichletbc(g, self.dofs, graph=graph)]

        c_ = c if c.ufl_shape == () else c[0]
        v = ufl.TestFunction(self.V)
        derivative_options = dict(
            adjoint_petsc_options=self.petsc_options,
        )
        if self.problem_type == "nonlinear":
            F = (
                k * ufl.inner(ufl.grad(uh), ufl.grad(v))
                + c_ * ufl.inner(uh + uh**3, v)
                - ufl.inner(f, v)
            ) * ufl.dx
            problem = fem_ad.petsc.NonlinearProblem(
                F,
                uh,
                bcs=bcs,
                petsc_options_prefix="test_controlled_problem_",
                petsc_options={
                    **self.petsc_options,
                    "snes_atol": 1e-12,
                    "snes_rtol": 1e-12,
                    "snes_error_if_not_converged": True,
                },
                graph=graph,
                **derivative_options,
            )
        else:
            u = ufl.TrialFunction(self.V)
            problem = fem_ad.petsc.LinearProblem(
                (k * ufl.inner(ufl.grad(u), ufl.grad(v)) + c_ * ufl.inner(u, v))
                * ufl.dx,
                ufl.inner(f, v) * ufl.dx,
                u=uh,
                bcs=bcs,
                petsc_options_prefix="test_controlled_problem_",
                petsc_options=self.petsc_options,
                graph=graph,
                **derivative_options,
            )
        problem.solve(graph=graph)

        J = fem_ad.assemble_scalar(
            fem_ad.form(ufl.inner(uh, uh) * ufl.dx, graph=graph), graph=graph
        )
        return ControlledEvaluation(
            graph=graph,
            controls={"coefficient": f, "constant": c, "boundary": g},
            uh=uh,
            J=J,
        )


@pytest.fixture(scope="module", params=["linear", "nonlinear"])
def controlled_problem(unit_square_mesh, request) -> ControlledProblem:
    """Share the space and the boundary dofs across evaluations."""
    return ControlledProblem(unit_square_mesh, request.param)


def _central_difference(evaluate, control, direction, step=1e-6):
    """The central difference (J(m + εh) - J(m - εh)) / 2ε of a functional.

    The control is moved in place, so that evaluate reads it, and restored afterwards.
    For a functional quadratic in the control, it is the directional derivative up to
    round-off for any step.

    Args:
        evaluate: The functional, called without arguments. It may return an array,
            e.g. a gradient, whose central difference is a Hessian action.
        control: A Function or a Constant.
        direction: A Function in the space of a Function control, including consistent
            ghost entries, or a number or array of the shape of a Constant control.
        step: The step ε.
    """
    if isinstance(control, dolfinx.fem.Function):
        base = control.x.array.copy()

        def move(step):
            control.x.array[:] = base + step * direction.x.array

    else:
        base = np.array(control.value, copy=True)

        def move(step):
            control.value = base + step * np.asarray(direction)

    values = []
    try:
        for sign in (1.0, -1.0):
            move(sign * step)
            values.append(np.array(evaluate(), copy=True))
    finally:
        move(0.0)
    return (values[0] - values[1]) / (2.0 * step)


@pytest.fixture(scope="session")
def central_difference():
    """The central difference of a functional, with the control moved in place.

    Returns:
        A function ``(evaluate, control, direction, step=1e-6)``, see
        :py:func:`_central_difference`.
    """
    return _central_difference
