"""The Stokes problem on the DFG 2D cylinder geometry."""

from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np
import pytest
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_graph_ad import Graph
from dolfinx_graph_ad import fem as fem_ad


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


@dataclass
class StokesEvaluation:
    """Controls, state and functional from one forward execution."""

    problem: StokesProblem
    graph: Graph | None
    up: fem_ad.Function
    g: fem_ad.Function
    nu: fem_ad.Constant
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
        h = fem_ad.Function(self.V_u, name="f")
        speed = 0.3
        h.interpolate(
            lambda x: np.stack(
                (speed * 4 * x[1] * (height - x[1]) / (height * height), 0.0 * x[0])
            )
        )
        noslip = fem_ad.Function(self.V_u, name="noslip")
        outflow = fem_ad.Function(V_p, name="outflow")

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
            fem_ad.dirichletbc(h, dofs_inflow, self.V.sub(0)),
            fem_ad.dirichletbc(noslip, dofs_walls, self.V.sub(0)),
            fem_ad.dirichletbc(outflow, dofs_outflow, self.V.sub(1)),
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
        g: fem_ad.Function | None = None,
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
        up = fem_ad.Function(self.V, name="up", graph=graph)
        u, p = ufl.split(up)

        obstacle_value = fem_ad.Function(self.V_u, name="g", graph=graph)
        if g is not None:
            g.x.petsc_vec.copy(obstacle_value.x.petsc_vec)
            obstacle_value.x.scatter_forward()

        viscosity = fem_ad.Constant(self.domain, ScalarType(nu), name="ν", graph=graph)

        # Parameters
        alpha = 10.0
        beta = 1.0e-3
        f = fem_ad.Function(self.V_u, name="f")

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
            fem_ad.dirichletbc(
                obstacle_value, self.dofs_obstacle, self.V.sub(0), graph=graph
            ),
            *self.fixed_bcs,
        ]

        problem = fem_ad.petsc.NonlinearProblem(
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
        J = fem_ad.assemble_scalar(fem_ad.form(J_form, graph=graph), graph=graph)
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
