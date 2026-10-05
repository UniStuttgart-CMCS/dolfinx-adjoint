"""Plane elasticity with a controlled Dirichlet boundary, and linear elasticity of a beam."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph
from dolfinx_adjoint import fem as fem_ad


@pytest.fixture(scope="module")
def beam_mesh() -> tuple[mesh.Mesh, float, float]:
    """Create the beam mesh and return it with its length and width."""
    L = 1
    W = 0.1
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([L, W, W])],
        [30, 10, 10],
        cell_type=mesh.CellType.hexahedron,
    )
    return domain, L, W


@dataclass
class PlaneElasticityEvaluation:
    """Controls, state and functional from one forward execution."""

    problem: PlaneElasticityProblem
    graph: Graph | None
    u: fem_ad.Function
    u_D: fem_ad.Function
    J: float  # Rank-local assembled object, retained for graph lookup by id.
    value: float  # Functional summed over the mesh communicator.


class PlaneElasticityProblem:
    """Set up a plane elasticity problem with a controlled Dirichlet boundary."""

    def __init__(self, domain: mesh.Mesh):
        self.domain = domain
        self.V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
        domain.topology.create_connectivity(
            domain.topology.dim - 1, domain.topology.dim
        )
        self.control_dofs = fem.locate_dofs_geometrical(
            self.V, lambda x: np.isclose(x[1], 0.0)
        )

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
        u_D: fem_ad.Function | None = None,
        graph: Graph | None = None,
    ) -> PlaneElasticityEvaluation:
        """Solve the problem and assemble J at the given control value.

        Args:
            u_D: Boundary-value Function in this problem's state space, copied for the solve.
                Defaults to (0.5, 0.25) throughout the space.
            graph: A fresh graph to record into, or None for plain DOLFINx.

        Returns:
            The state and functional of this evaluation.

        """
        uh = fem_ad.Function(self.V, name="u", graph=graph)

        uD_control = fem_ad.Function(self.V, name="u_D", graph=graph)
        if u_D is None:
            uD_control.interpolate(
                lambda x: np.stack((0.5 + 0.0 * x[0], 0.25 + 0.0 * x[1]))
            )
        else:
            u_D.x.petsc_vec.copy(uD_control.x.petsc_vec)
        uD_control.x.scatter_forward()

        mu = fem_ad.Constant(self.domain, ScalarType(1.0), name="μ")
        lambda_ = fem_ad.Constant(self.domain, ScalarType(1.25), name="λ")
        rho = 1.0
        g = 0.016

        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(
                u
            )

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        f = fem_ad.Constant(self.domain, ScalarType((0.0, -rho * g)))
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.dot(f, v) * ufl.dx
        bcs = [fem_ad.dirichletbc(uD_control, self.control_dofs, graph=graph)]

        problem = fem_ad.petsc.LinearProblem(
            a,
            L,
            u=uh,
            bcs=bcs,
            petsc_options_prefix="plane_elasticity_",
            petsc_options=self.petsc_options,
            adjoint_petsc_options=self.petsc_options,
            graph=graph,
        )
        problem.solve(graph=graph)

        J_form = 0.5 * ufl.inner(uh, uh) * ufl.dx
        J = fem_ad.assemble_scalar(fem_ad.form(J_form, graph=graph), graph=graph)
        value = self.domain.comm.allreduce(J, op=MPI.SUM)
        return PlaneElasticityEvaluation(
            problem=self,
            graph=graph,
            u=uh,
            u_D=uD_control,
            J=J,
            value=value,
        )


@pytest.fixture(scope="module")
def plane_elasticity_problem(unit_square_mesh) -> PlaneElasticityProblem:
    """Share the space and controlled boundary dofs across evaluations."""
    return PlaneElasticityProblem(unit_square_mesh)


@dataclass
class LinearElasticityEvaluation:
    """Controls, state and functional from one forward execution."""

    problem: LinearElasticityProblem
    graph: Graph | None
    u: fem_ad.Function
    lambda_: fem_ad.Constant
    mu: fem_ad.Constant
    F: ufl.Form
    J_form: ufl.Form
    J: float  # Rank-local assembled object, retained for graph lookup by id.
    value: float  # Functional summed over the mesh communicator.


class LinearElasticityProblem:
    """Set up the linear elasticity problem that will be used in all tests."""

    def __init__(self, domain: mesh.Mesh, length: float, width: float):
        self.domain = domain

        # Scaled variable
        rho = 1
        delta = width / length
        gamma = 0.4 * delta**2
        g = gamma

        vector_element = element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        self.V = fem.functionspace(domain, vector_element)
        self.f = fem_ad.Constant(domain, ScalarType((0, 0, -rho * g)))
        self.T = fem_ad.Constant(domain, ScalarType((0, 0, 0)))

        # Boundary condition describing the clamped left side of the beam
        boundary_facets = mesh.locate_entities_boundary(
            domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], 0)
        )
        self.bcs_dofs = fem.locate_dofs_topological(
            self.V, domain.topology.dim - 1, boundary_facets
        )
        self.fixed_bcs = [
            fem_ad.dirichletbc(
                np.array([0, 0, 0], dtype=ScalarType), self.bcs_dofs, self.V
            )
        ]

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
        lambda_: float = 1.0,
        mu: float = 1.25,
        graph: Graph | None = None,
    ) -> LinearElasticityEvaluation:
        """Solve the problem and assemble J at the given control values.

        Args:
            lambda_: First Lamé parameter, common to all ranks. Defaults to 1.0.
            mu: Second Lamé parameter, common to all ranks. Defaults to 1.25.
            graph: A fresh graph to record into, or None for plain DOLFINx.

        Returns:
            The state, functional and forms of this evaluation.

        """
        uh = fem_ad.Function(self.V, name="Deformation", graph=graph)
        lame_lambda = fem_ad.Constant(self.domain, ScalarType(lambda_), graph=graph)
        lame_mu = fem_ad.Constant(self.domain, ScalarType(mu), graph=graph)

        def sigma(w):
            return lame_lambda * ufl.nabla_div(w) * ufl.Identity(
                len(w)
            ) + 2 * lame_mu * ufl.sym(ufl.grad(w))

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        ds = ufl.Measure("ds", domain=self.domain)
        a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
        L = ufl.dot(self.f, v) * ufl.dx + ufl.dot(self.T, v) * ds

        # Write the residual directly to avoid creating a UFL Replacer cycle.
        F = ufl.inner(sigma(uh), ufl.sym(ufl.grad(v))) * ufl.dx - L

        problem = fem_ad.petsc.LinearProblem(
            a,
            L,
            u=uh,
            bcs=self.fixed_bcs,
            petsc_options=self.petsc_options,
            petsc_options_prefix="linear_elasticity",
            adjoint_petsc_options=self.petsc_options,
            graph=graph,
        )
        problem.solve(graph=graph)

        J_form = ufl.inner(uh, uh) * ufl.dx
        J = fem_ad.assemble_scalar(fem_ad.form(J_form, graph=graph), graph=graph)
        value = self.domain.comm.allreduce(J, op=MPI.SUM)
        return LinearElasticityEvaluation(
            problem=self,
            graph=graph,
            u=uh,
            lambda_=lame_lambda,
            mu=lame_mu,
            F=F,
            J_form=J_form,
            J=J,
            value=value,
        )


@pytest.fixture(scope="module")
def linear_elasticity_problem(beam_mesh) -> LinearElasticityProblem:
    """Share the space and clamped boundary across evaluations."""
    return LinearElasticityProblem(*beam_mesh)


@pytest.fixture(scope="module")
def linear_elasticity_evaluation(
    linear_elasticity_problem,
) -> LinearElasticityEvaluation:
    """Record one evaluation per problem configuration within each test module."""
    return linear_elasticity_problem.evaluate(graph=Graph())
