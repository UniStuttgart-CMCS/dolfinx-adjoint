"""
Integration tests for the Poisson equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np
import pytest
import ufl
from basix.ufl import element, mixed_element
from dolfinx import mesh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph, fem


@pytest.fixture(
    scope="module",
    params=[mesh.CellType.triangle, mesh.CellType.quadrilateral],
    ids=["triangle", "quadrilateral"],
)
def unit_square_mesh(request) -> mesh.Mesh:
    """Create a 64-by-64 unit-square mesh per cell type and test module."""
    return mesh.create_unit_square(MPI.COMM_WORLD, 64, 64, request.param)


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


@pytest.fixture(scope="module")
def dfg_2d_mesh() -> tuple[gmshio.MeshData, float]:
    """Create the tagged DFG 2D cylinder geometry and return its channel height."""

    # Mesh parameters
    gmsh.initialize()
    L = 2.2
    H = 0.41
    c_x = 0.2
    c_y = 0.2
    r = 0.05
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    fluid_marker = 1
    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
    inflow, outflow, walls, obstacle = [], [], [], []

    # Mesh generation
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        circle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
        fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, circle)])
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H / 2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H / 2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(
                center_of_mass, [L / 2, 0, 0]
            ):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, wall_marker)
        gmsh.model.setPhysicalName(1, wall_marker, "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
        gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
        gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
        gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", obstacle)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", 0.01)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 0.04)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(2, "DistMax", H)
        gmsh.model.mesh.field.add("Min", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
        gmsh.model.mesh.generate(2)

    mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    mesh_data.facet_tags.name = "Facet markers"
    gmsh.finalize()
    return mesh_data, H


@pytest.fixture(
    scope="module",
    params=["nonlinear", "linear"],
)
def solver(request):
    return request.param


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
    u: fem.Function
    f: fem.Function
    nu: fem.Constant
    u_D: fem.Function
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
            fixed_value = fem.Function(self.V, name="u_D_fixed")
            fixed_value.x.array[:] = 1.0
            self.fixed_bcs = [
                fem.dirichletbc(fixed_value, boundary_dofs_r),
                fem.dirichletbc(fixed_value, boundary_dofs_t),
                fem.dirichletbc(fixed_value, boundary_dofs_b),
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
        f: fem.Function | None = None,
        nu: float = 1.0,
        u_D: fem.Function | None = None,
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
        uh = fem.Function(self.V, name="uₕ", graph=graph)
        forcing = fem.Function(self.W, name="f", graph=graph)

        if f is None:
            forcing.interpolate(lambda x: x[0] + x[1])
        else:
            f.x.petsc_vec.copy(forcing.x.petsc_vec)
        forcing.x.scatter_forward()

        diffusion = fem.Constant(self.domain, ScalarType(nu), name="ν", graph=graph)

        boundary_value = fem.Function(self.V, name="u_D", graph=graph)
        if u_D is None:
            boundary_value.x.array[:] = 1.0
        else:
            u_D.x.petsc_vec.copy(boundary_value.x.petsc_vec)
        boundary_value.x.scatter_forward()

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        a = diffusion * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = forcing * v * ufl.dx

        # Write the residual directly to avoid creating a UFL Replacer cycle.
        F = diffusion * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - L
        bcs = [
            fem.dirichletbc(boundary_value, self.control_dofs, graph=graph),
            *self.fixed_bcs,
        ]

        if self.solver == "nonlinear":
            problem = fem.petsc.NonlinearProblem(
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
            problem = fem.petsc.LinearProblem(
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
        alpha = fem.Constant(self.domain, ScalarType(1e-6), name="α")
        J_form = (
            0.5 * ufl.inner(uh - g, uh - g) * ufl.dx
            + alpha * ufl.inner(forcing, forcing) * ufl.dx
        )
        J = fem.assemble_scalar(fem.form(J_form, graph=graph), graph=graph)
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


@dataclass
class PlaneElasticityEvaluation:
    """Controls, state and functional from one forward execution."""

    problem: PlaneElasticityProblem
    graph: Graph | None
    u: fem.Function
    u_D: fem.Function
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
        u_D: fem.Function | None = None,
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
        uh = fem.Function(self.V, name="u", graph=graph)

        uD_control = fem.Function(self.V, name="u_D", graph=graph)
        if u_D is None:
            uD_control.interpolate(
                lambda x: np.stack((0.5 + 0.0 * x[0], 0.25 + 0.0 * x[1]))
            )
        else:
            u_D.x.petsc_vec.copy(uD_control.x.petsc_vec)
        uD_control.x.scatter_forward()

        mu = fem.Constant(self.domain, ScalarType(1.0), name="μ")
        lambda_ = fem.Constant(self.domain, ScalarType(1.25), name="λ")
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
        f = fem.Constant(self.domain, ScalarType((0.0, -rho * g)))
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.dot(f, v) * ufl.dx
        bcs = [fem.dirichletbc(uD_control, self.control_dofs, graph=graph)]

        problem = fem.petsc.LinearProblem(
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
        J = fem.assemble_scalar(fem.form(J_form, graph=graph), graph=graph)
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
    u: fem.Function
    lambda_: fem.Constant
    mu: fem.Constant
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
        self.f = fem.Constant(domain, ScalarType((0, 0, -rho * g)))
        self.T = fem.Constant(domain, ScalarType((0, 0, 0)))

        # Boundary condition describing the clamped left side of the beam
        boundary_facets = mesh.locate_entities_boundary(
            domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], 0)
        )
        self.bcs_dofs = fem.locate_dofs_topological(
            self.V, domain.topology.dim - 1, boundary_facets
        )
        self.fixed_bcs = [
            fem.dirichletbc(
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
        uh = fem.Function(self.V, name="Deformation", graph=graph)
        lame_lambda = fem.Constant(self.domain, ScalarType(lambda_), graph=graph)
        lame_mu = fem.Constant(self.domain, ScalarType(mu), graph=graph)

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

        problem = fem.petsc.LinearProblem(
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
        J = fem.assemble_scalar(fem.form(J_form, graph=graph), graph=graph)
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


@dataclass
class StokesEvaluation:
    """Controls, state and functional from one forward execution."""

    problem: StokesProblem
    graph: Graph | None
    up: fem.Function
    g: fem.Function
    nu: fem.Constant
    F: ufl.Form
    J_form: ufl.Form
    J: float  # Rank-local assembled object, retained for graph lookup by id.
    value: float  # Functional summed over the mesh communicator.


class StokesProblem:
    """Set up the Stokes problem that will be used in all tests."""

    def __init__(self, mesh_data: gmshio.MeshData, height: float):
        self.domain = mesh_data.mesh
        ft = mesh_data.facet_tags
        inlet_marker = mesh_data.physical_groups["Inlet"].tag
        outlet_marker = mesh_data.physical_groups["Outlet"].tag
        wall_marker = mesh_data.physical_groups["Walls"].tag
        obstacle_marker = mesh_data.physical_groups["Obstacle"].tag

        # Function spaces as Mixed Element space
        u_elem = element("Lagrange", self.domain.basix_cell(), 2, shape=(2,))
        p_elem = element("Lagrange", self.domain.basix_cell(), 1)
        self.V = fem.functionspace(self.domain, mixed_element([u_elem, p_elem]))
        self.V_u, (self.V_u_map,) = self.V.sub(0).collapse()
        V_p, _ = self.V.sub(1).collapse()

        # Boundary conditions
        h = fem.Function(self.V_u, name="f")
        speed = 0.3
        h.interpolate(
            lambda x: np.stack(
                (speed * 4 * x[1] * (height - x[1]) / (height * height), 0.0 * x[0])
            )
        )
        noslip = fem.Function(self.V_u, name="noslip")
        outflow = fem.Function(V_p, name="outflow")

        dofs_walls = fem.locate_dofs_topological(
            (self.V.sub(0), self.V_u), 1, ft.indices[ft.values == wall_marker]
        )
        dofs_inflow = fem.locate_dofs_topological(
            (self.V.sub(0), self.V_u), 1, ft.indices[ft.values == inlet_marker]
        )
        dofs_outflow = fem.locate_dofs_topological(
            (self.V.sub(1), V_p), 1, ft.indices[ft.values == outlet_marker]
        )
        self.dofs_obstacle = fem.locate_dofs_topological(
            (self.V.sub(0), self.V_u), 1, ft.indices[ft.values == obstacle_marker]
        )

        self.bcs_dofs = np.unique(
            np.concatenate(
                [
                    dofs_walls[0],
                    dofs_inflow[0],
                    dofs_outflow[0],
                    self.dofs_obstacle[0],
                ]
            )
        )
        self.fixed_bcs = [
            fem.dirichletbc(h, dofs_inflow, self.V.sub(0)),
            fem.dirichletbc(noslip, dofs_walls, self.V.sub(0)),
            fem.dirichletbc(outflow, dofs_outflow, self.V.sub(1)),
        ]
        self.dObs = ufl.Measure(
            "ds", domain=self.domain, subdomain_data=ft, subdomain_id=obstacle_marker
        )

        # We use a direct solve for both the forward and adjoint problems with MUMPS to keep the same factorisation across rank counts.
        # As the Stokes system is a saddle point system, the pressure block of the Jacobian is zero. MUMPS reorders around it, while the built-in LU factorisation of PETSc does not pivot dynamically and raises on such a pivot.
        self.petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        }

    def evaluate(
        self,
        *,
        g: fem.Function | None = None,
        nu: float = 1.0,
        graph: Graph | None = None,
    ) -> StokesEvaluation:
        """Solve the problem and assemble J at the given control values.

        Args:
            g: Obstacle velocity Function in this problem's collapsed velocity space,
                copied for the solve. Defaults to zero.
            nu: Viscosity, common to all ranks. Defaults to 1.0.
            graph: A fresh graph to record into, or None for plain DOLFINx.

        Returns:
            The state, functional and forms of this evaluation.

        """
        up = fem.Function(self.V, name="up", graph=graph)
        u, p = ufl.split(up)

        obstacle_value = fem.Function(self.V_u, name="g", graph=graph)
        if g is not None:
            g.x.petsc_vec.copy(obstacle_value.x.petsc_vec)
            obstacle_value.x.scatter_forward()

        viscosity = fem.Constant(self.domain, ScalarType(nu), name="ν", graph=graph)

        # Parameters
        alpha = 10.0
        beta = 1.0e-3
        f = fem.Function(self.V_u, name="f")

        # Variational formulation
        v, q = ufl.split(ufl.TestFunction(self.V))
        a = (
            viscosity * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.div(v) * p * ufl.dx
            + q * ufl.div(u) * ufl.dx
        )
        L = ufl.inner(f, v) * ufl.dx
        F = a - L
        bcs = [
            fem.dirichletbc(
                obstacle_value, self.dofs_obstacle, self.V.sub(0), graph=graph
            ),
            *self.fixed_bcs,
        ]

        problem = fem.petsc.NonlinearProblem(
            F,
            up,
            bcs=bcs,
            petsc_options_prefix="forward_nonlinear",
            petsc_options={
                **self.petsc_options,
                "snes_error_if_not_converged": True,
            },
            adjoint_petsc_options=self.petsc_options,
            graph=graph,
        )
        problem.solve(graph=graph)

        # Define the objective function
        J_form = (
            0.5 * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
            + alpha / 2 * ufl.inner(obstacle_value, obstacle_value) * self.dObs
            + beta / 2 * ufl.inner(p, p) * ufl.dx
        )
        J = fem.assemble_scalar(fem.form(J_form, graph=graph), graph=graph)
        value = self.domain.comm.allreduce(J, op=MPI.SUM)
        return StokesEvaluation(
            problem=self,
            graph=graph,
            up=up,
            g=obstacle_value,
            nu=viscosity,
            F=F,
            J_form=J_form,
            J=J,
            value=value,
        )


@pytest.fixture(scope="module")
def stokes_problem(dfg_2d_mesh) -> StokesProblem:
    """Share the spaces and boundary conditions across evaluations."""
    return StokesProblem(*dfg_2d_mesh)


@pytest.fixture(scope="module")
def stokes_evaluation(stokes_problem) -> StokesEvaluation:
    """Record one evaluation per problem configuration within each test module."""
    return stokes_problem.evaluate(graph=Graph())


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
