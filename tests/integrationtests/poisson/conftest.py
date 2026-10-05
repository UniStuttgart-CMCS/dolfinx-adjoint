"""The Poisson problem, with a boundary condition on the inflow or on the whole boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_graph_ad import Graph
from dolfinx_graph_ad import fem as fem_ad


@pytest.fixture(
    scope="module",
    params=["inflow", "full_boundary"],
    ids=["inflow", "full-boundary"],
)
def boundary_condition(request):
    """Return the selected Poisson boundary-condition type."""
    return request.param


@dataclass
class PoissonEvaluation:
    """Controls, state and functional from one forward execution."""

    problem: PoissonProblem
    graph: Graph | None
    u: fem_ad.Function
    f: fem_ad.Function
    nu: fem_ad.Constant
    u_D: fem_ad.Function
    F: ufl.Form
    J_form: ufl.Form
    J: float  # Rank-local assembled object, retained for graph lookup by id.
    value: float  # Functional summed over the mesh communicator.


class PoissonProblem:
    """Set up the Poisson problem that will be used in all tests."""

    def __init__(self, domain: mesh.Mesh, solver: str, boundary_condition: str):
        self.domain = domain
        self.solver = solver
        self.V = fem.functionspace(domain, ("Lagrange", 1))
        self.W = fem.functionspace(domain, ("DG", 0))
        domain.topology.create_connectivity(
            domain.topology.dim - 1, domain.topology.dim
        )

        if boundary_condition == "inflow":
            self.control_dofs = fem.locate_dofs_geometrical(
                self.V,
                lambda x: np.isclose(x[0], 0.0)
                & ~np.isclose(x[1], 0.0)
                & ~np.isclose(x[1], 1.0),
            )
            boundary_dofs_r = fem.locate_dofs_geometrical(
                self.V, lambda x: np.isclose(x[0], 1.0)
            )
            boundary_dofs_t = fem.locate_dofs_geometrical(
                self.V, lambda x: np.isclose(x[1], 1.0)
            )
            boundary_dofs_b = fem.locate_dofs_geometrical(
                self.V, lambda x: np.isclose(x[1], 0.0)
            )
            self.bcs_dofs = np.unique(
                np.concatenate(
                    [
                        self.control_dofs,
                        boundary_dofs_r,
                        boundary_dofs_t,
                        boundary_dofs_b,
                    ]
                )
            )
            fixed_value = fem_ad.Function(self.V, name="u_D_fixed")
            fixed_value.x.array[:] = 1.0
            self.fixed_bcs = [
                fem_ad.dirichletbc(fixed_value, boundary_dofs_r),
                fem_ad.dirichletbc(fixed_value, boundary_dofs_t),
                fem_ad.dirichletbc(fixed_value, boundary_dofs_b),
            ]
        elif boundary_condition == "full_boundary":
            exterior_facets = mesh.exterior_facet_indices(domain.topology)
            self.control_dofs = fem.locate_dofs_topological(
                self.V, domain.topology.dim - 1, exterior_facets
            )
            self.bcs_dofs = self.control_dofs
            self.fixed_bcs = []
        else:
            raise ValueError(f"Unknown boundary condition: {boundary_condition}")

        # We use a direct solve for both the forward and adjoint problems with MUMPS to keep the same factorisation across rank counts.
        self.petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        }

    def evaluate(
        self,
        *,
        f: fem_ad.Function | None = None,
        nu: float = 1.0,
        u_D: fem_ad.Function | None = None,
        graph: Graph | None = None,
    ) -> PoissonEvaluation:
        """Solve the problem and assemble J at the given control values.

        Args:
            f: Forcing Function in this problem's forcing space, copied for the solve.
                Defaults to the interpolation of x[0] + x[1].
            nu: Diffusion coefficient, common to all ranks. Defaults to 1.0.
            u_D: Boundary-value Function in this problem's state space, copied for the solve.
                Defaults to 1.0 throughout the space.
            graph: A fresh graph to record into, or None for plain DOLFINx.

        Returns:
            The state, functional and forms of this evaluation.

        """
        uh = fem_ad.Function(self.V, name="uₕ", graph=graph)
        forcing = fem_ad.Function(self.W, name="f")

        if f is None:
            forcing.interpolate(lambda x: x[0] + x[1])
        else:
            f.x.petsc_vec.copy(forcing.x.petsc_vec)
        forcing.x.scatter_forward()

        diffusion = fem_ad.Constant(self.domain, ScalarType(nu), name="ν", graph=graph)

        boundary_value = fem_ad.Function(self.V, name="u_D")
        if u_D is None:
            boundary_value.x.array[:] = 1.0
        else:
            u_D.x.petsc_vec.copy(boundary_value.x.petsc_vec)
        boundary_value.x.scatter_forward()
        if graph is not None:
            graph.track(forcing)
            graph.track(boundary_value)

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        a = diffusion * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = forcing * v * ufl.dx

        # Write the residual directly to avoid creating a UFL Replacer cycle.
        F = diffusion * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - L
        bcs = [
            fem_ad.dirichletbc(boundary_value, self.control_dofs, graph=graph),
            *self.fixed_bcs,
        ]

        if self.solver == "nonlinear":
            problem = fem_ad.petsc.NonlinearProblem(
                F,
                uh,
                bcs=bcs,
                petsc_options_prefix="forward_nonlinear",
                petsc_options={
                    **self.petsc_options,
                    "snes_atol": 1e-12,
                    "snes_rtol": 1e-12,
                    "snes_error_if_not_converged": True,
                },
                adjoint_petsc_options=self.petsc_options,
                graph=graph,
            )
        else:
            problem = fem_ad.petsc.LinearProblem(
                a,
                L,
                u=uh,
                bcs=bcs,
                petsc_options_prefix="forward_linear",
                petsc_options=self.petsc_options,
                adjoint_petsc_options=self.petsc_options,
                graph=graph,
            )
        problem.solve(graph=graph)

        x = ufl.SpatialCoordinate(self.domain)
        g = (1 / (2 * np.pi**2)) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        alpha = fem_ad.Constant(self.domain, ScalarType(1e-6), name="α")
        J_form = (
            0.5 * ufl.inner(uh - g, uh - g) * ufl.dx
            + alpha * ufl.inner(forcing, forcing) * ufl.dx
        )
        J = fem_ad.assemble_scalar(fem_ad.form(J_form, graph=graph), graph=graph)
        value = self.domain.comm.allreduce(J, op=MPI.SUM)
        return PoissonEvaluation(
            problem=self,
            graph=graph,
            u=uh,
            f=forcing,
            nu=diffusion,
            u_D=boundary_value,
            F=F,
            J_form=J_form,
            J=J,
            value=value,
        )


@pytest.fixture(scope="module")
def poisson_problem(unit_square_mesh, solver, boundary_condition) -> PoissonProblem:
    """Share the spaces and boundary conditions across evaluations."""
    return PoissonProblem(unit_square_mesh, solver, boundary_condition)


@pytest.fixture(scope="module")
def poisson_evaluation(poisson_problem) -> PoissonEvaluation:
    """Record one evaluation per problem configuration within each test module."""
    return poisson_problem.evaluate(graph=Graph())
