"""The heat equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import ufl
from dolfinx import mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph, fem


@dataclass
class HeatEquationEvaluation:
    """Controls, states and functional from one forward execution."""

    problem: HeatEquationProblem
    graph: Graph | None
    initial_guess: fem.Function
    u_prev: fem.Function
    u_next: fem.Function
    u_iterations: list[fem.Function]  # States of all time steps, for the test.
    F: ufl.Form
    J_form: ufl.Form
    J: float  # Rank-local assembled object, retained for graph lookup by id.
    value: float  # Functional summed over the mesh communicator.


class HeatEquationProblem:
    """Set up the heat equation problem that will be used in all tests."""

    def __init__(self, domain: mesh.Mesh):
        self.domain = domain
        self.V = fem.functionspace(domain, ("Lagrange", 1))
        self.dt = 0.01
        self.T = 0.05
        self.dt_constant = fem.Constant(domain, ScalarType(self.dt))

        # We use a direct solve for both the forward and adjoint problems with MUMPS to keep the same factorisation across rank counts.
        self.petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        }

        # Create true data set
        true_initial = fem.Function(self.V, name="u_true_initial")
        true_initial.interpolate(
            lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])
        )
        u_prev = true_initial.copy()
        u_next = true_initial.copy()

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        a = (
            ufl.inner(u / self.dt_constant, v) * ufl.dx
            + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        )
        L = ufl.inner(u_prev / self.dt_constant, v) * ufl.dx

        problem = fem.petsc.LinearProblem(
            a,
            L,
            u=u_next,
            petsc_options_prefix="forward_linear",
            petsc_options=self.petsc_options,
        )

        t = 0.0
        while t < self.T:
            problem.solve()
            u_prev.x.array[:] = u_next.x.array[:]
            t += self.dt
        self.true_data = u_next.copy()

    def evaluate(
        self,
        *,
        initial_guess: fem.Function | None = None,
        graph: Graph | None = None,
    ) -> HeatEquationEvaluation:
        """Step the problem through time and assemble J at the given control value.

        Args:
            initial_guess: Initial temperature Function in this problem's state space,
                copied for the solve. Defaults to the interpolation of
                15 x[0] (1 - x[0]) x[1] (1 - x[1]).
            graph: A fresh graph to record into, or None for plain DOLFINx.

        Returns:
            The states, functional and forms of this evaluation.

        """

        initial = fem.Function(self.V, name="initial_guess", graph=graph)
        if initial_guess is None:
            initial.interpolate(
                lambda x: 15.0 * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
            )
        else:
            initial_guess.x.petsc_vec.copy(initial.x.petsc_vec)
        initial.x.scatter_forward()

        u_prev = initial.copy(graph=graph, name="u_prev")
        u_next = fem.Function(self.V, name="u_next", graph=graph)
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = (
            ufl.inner(u / self.dt_constant, v) * ufl.dx
            + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        )
        L = ufl.inner(u_prev / self.dt_constant, v) * ufl.dx

        # Write the residual directly to avoid creating a UFL Replacer cycle.
        F = (
            ufl.inner(u_next / self.dt_constant, v) * ufl.dx
            + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
            - L
        )

        t = 0.0
        i = 0

        # Store the states for the whole time-domain
        u_iterations = [initial.copy()]
        while t < self.T:
            i += 1
            problem = fem.petsc.LinearProblem(
                a,
                L,
                u=u_next,
                petsc_options_prefix="forward_linear",
                petsc_options=self.petsc_options,
                adjoint_petsc_options=self.petsc_options,
                graph=graph,
            )
            problem.solve(graph=graph, version=i)
            t += self.dt
            u_prev.assign(u_next, graph=graph, version=i)

            # Storing the iterations for later visualization and testing
            u_iterations.append(u_next.copy())

        alpha = fem.Constant(self.domain, ScalarType(1.0e-6))
        J_form = (
            ufl.inner(self.true_data - u_next, self.true_data - u_next) * ufl.dx
            + alpha * ufl.inner(ufl.grad(initial), ufl.grad(initial)) * ufl.dx
        )
        J = fem.assemble_scalar(fem.form(J_form, graph=graph), graph=graph)
        value = self.domain.comm.allreduce(J, op=MPI.SUM)
        return HeatEquationEvaluation(
            problem=self,
            graph=graph,
            initial_guess=initial,
            u_prev=u_prev,
            u_next=u_next,
            u_iterations=u_iterations,
            F=F,
            J_form=J_form,
            J=J,
            value=value,
        )


@pytest.fixture(scope="module")
def heat_equation_problem(unit_square_mesh) -> HeatEquationProblem:
    """Share the space and the true data across evaluations."""
    return HeatEquationProblem(unit_square_mesh)


@pytest.fixture(scope="module")
def heat_equation_evaluation(heat_equation_problem) -> HeatEquationEvaluation:
    """Record one evaluation per problem configuration within each test module."""
    return heat_equation_problem.evaluate(graph=Graph())
