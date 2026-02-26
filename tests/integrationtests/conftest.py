"""
Integration tests for the Poisson equation adjoint gradients.

This test module verifies that the automatic differentiation implementation
correctly computes gradients using the adjoint method by comparing against
explicit adjoint calculations.
"""

import gmsh
import numpy as np
import pytest
import ufl
from basix.ufl import element
from dolfinx import mesh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph, fem, nls


@pytest.fixture(
    scope="module",
    params=[mesh.CellType.triangle, mesh.CellType.quadrilateral],
    ids=["triangle", "quadrilateral"],
)
def cell_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=["nonlinear", "nonlinear_newton", "linear"],
)
def solver(request):
    return request.param


@pytest.fixture(scope="module")
def poisson_problem(cell_type, solver: bool):
    """Set up the Poisson problem that will be used in all tests."""
    # Create graph object to store the computational graph
    graph_ = Graph()

    print(f"Testing with cell type: {cell_type}")
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64, cell_type)
    V = fem.functionspace(domain, ("Lagrange", 1))
    W = fem.functionspace(domain, ("DG", 0))

    # Define the basis functions and parameters
    uh = fem.Function(V, name="uₕ", graph=graph_)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Function(W, name="f", graph=graph_)
    nu = fem.Constant(domain, ScalarType(1.0), name="ν", graph=graph_)

    f.interpolate(lambda x: x[0] + x[1])

    # Define the variational form and the residual equation
    a = nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    F = a - L
    if not solver == "linear":
        F = ufl.replace(F, {u: uh})

    # Define the boundary and the boundary conditions
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    uD_L = fem.Function(V, name="u_D", graph=graph_)
    uD_L.interpolate(lambda x: 1.0 + 0.0 * x[0])
    uD_R = fem.Function(V, name="u_D")
    uD_R.interpolate(lambda x: 1.0 + 0.0 * x[0])
    uD_T = fem.Function(V, name="u_D")
    uD_T.interpolate(lambda x: 1.0 + 0.0 * x[1])
    uD_B = fem.Function(V, name="u_D")
    uD_B.interpolate(lambda x: 1.0 + 0.0 * x[1])

    boundary_dofs_L = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    boundary_dofs_R = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
    boundary_dofs_T = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 1.0))
    boundary_dofs_B = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0))

    bcs = [
        fem.dirichletbc(uD_L, boundary_dofs_L, graph=graph_),
        fem.dirichletbc(uD_R, boundary_dofs_R),
        fem.dirichletbc(uD_T, boundary_dofs_T),
        fem.dirichletbc(uD_B, boundary_dofs_B),
    ]

    # Boundary conditions of adjoint must be set to zero since there cannot be any contribution from the boundary to the gradient of J with respect to variables except for the boundary condition itself.
    bcs_adjoint = fem.dirichletbc(
        ScalarType(0.0),
        np.concatenate(
            [boundary_dofs_L, boundary_dofs_R, boundary_dofs_T, boundary_dofs_B]
        ),
        V,
    )

    # Define the problem solver and solve it
    if solver == "nonlinear":
        problem = fem.petsc.NonlinearProblem(
            F,
            uh,
            bcs=bcs,
            petsc_options_prefix="forward_nonlinear",
            graph=graph_,
        )
        problem.solve(graph=graph_)
    elif solver == "nonlinear_newton":
        problem = fem.petsc.NewtonSolverNonlinearProblem(
            F,
            uh,
            bcs=bcs,
            graph=graph_,
        )
        solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem, graph=graph_)
        solver.solve(uh, graph=graph_)
    elif solver == "linear":
        problem = fem.petsc.LinearProblem(
            *ufl.system(F),
            u=uh,
            bcs=bcs,
            petsc_options_prefix="forward_linear",
            graph=graph_,
        )
        problem.solve(graph=graph_)

    # Define profile g
    x = ufl.SpatialCoordinate(domain)
    g = (1 / (2 * np.pi**2)) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

    # Define the objective function
    alpha = fem.Constant(domain, ScalarType(1e-6), name="α")
    J_form = 0.5 * ufl.inner(uh - g, uh - g) * ufl.dx + alpha * ufl.inner(f, f) * ufl.dx
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    # For testing, the form F needs to be replaced by the form with the solution function uh instead of the TrialFunction u, since the dependencies of uh on the coefficients and constants in the form need to be tracked in the graph for the adjoint calculation. In the linear case, this is not done in the forward solve and thus needs to be done here.
    if solver == "linear":
        F = ufl.replace(F, {u: uh})

    # Return all components as a dictionary
    return {
        "graph_": graph_,
        "domain": domain,
        "uh": uh,
        "f": f,
        "nu": nu,
        "F": F,
        "uD_L": uD_L,
        "boundary_dofs_L": boundary_dofs_L,
        "J_form": J_form,
        "J": J,
        "bcs_adjoint": [bcs_adjoint],
    }


@pytest.fixture(scope="module")
def linear_elasticity_problem():
    """Set up the linear elasticity problem that will be used in all tests."""
    # Scaled variable
    L = 1
    W = 0.1
    rho = 1
    delta = W / L
    gamma = 0.4 * delta**2
    g = gamma

    graph_ = Graph()

    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([L, W, W])],
        [30, 10, 10],
        cell_type=mesh.CellType.hexahedron,
    )

    vector_element = element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    V = fem.functionspace(domain, vector_element)
    ds = ufl.Measure("ds", domain=domain)

    u = fem.Function(V, name="Deformation", graph=graph_)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, ScalarType((0, 0, -rho * g)))
    T = fem.Constant(domain, ScalarType((0, 0, 0)))
    lambda_ = fem.Constant(domain, ScalarType(1.0), graph=graph_)
    mu = fem.Constant(domain, ScalarType(1.25), graph=graph_)

    a = (
        ufl.inner(
            lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u))
            + 2 * mu * ufl.sym(ufl.grad(u)),
            ufl.sym(ufl.grad(v)),
        )
        * ufl.dx
    )
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
    F = a - L

    # Boundary condition describing the clamped left side of the beam
    u_D = np.array([0, 0, 0], dtype=ScalarType)
    boundary_facets = mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], 0)
    )
    bc = fem.dirichletbc(
        u_D, fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets), V
    )

    problem = fem.petsc.NonlinearProblem(
        F,
        u,
        bcs=[bc],
        petsc_options_prefix="forward_nonlinear",
        graph=graph_,
    )
    problem.solve(graph=graph_)

    J_form = ufl.inner(u, u) * ufl.dx
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    return {
        "graph_": graph_,
        "domain": domain,
        "u": u,
        "lambda_": lambda_,
        "mu": mu,
        "F": F,
        "J_form": J_form,
        "J": J,
        "bc": bc,
    }


@pytest.fixture(scope="module")
def stokes_problem():
    """Set up the Stokes problem that will be used in all tests."""
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
    mesh = mesh_data.mesh
    ft = mesh_data.facet_tags
    ft.name = "Facet markers"

    graph_ = Graph()

    # Function spaces as Mixed Element space
    from basix.ufl import element, mixed_element

    u_elem = element("Lagrange", mesh.basix_cell(), 2, shape=(2,))
    p_elem = element("Lagrange", mesh.basix_cell(), 1)

    v_elem = mixed_element([u_elem, p_elem])
    V = fem.functionspace(mesh, v_elem)
    V_u, V_u_map = V.sub(0).collapse()
    V_p, V_p_map = V.sub(1).collapse()

    up = fem.Function(V, name="up", graph=graph_)
    u, p = ufl.split(up)

    vq = ufl.TestFunction(V)
    v, q = ufl.split(vq)

    # Boundary conditions
    h = fem.Function(V_u, name="f")
    speed = 0.3
    h.interpolate(
        lambda x: np.stack((speed * 4 * x[1] * (H - x[1]) / (H * H), 0.0 * x[0]))
    )
    g = fem.Function(V_u, name="g", graph=graph_)
    noslip = fem.Function(V_u, name="noslip")
    outflow = fem.Function(V_p, name="outflow")
    outflow.interpolate(lambda x: 0.0 * x[0] + 0.0)

    dofs_walls = fem.locate_dofs_topological(
        (V.sub(0), V_u), 1, ft.indices[ft.values == wall_marker]
    )
    dofs_inflow = fem.locate_dofs_topological(
        (V.sub(0), V_u), 1, ft.indices[ft.values == inlet_marker]
    )
    dofs_outflow = fem.locate_dofs_topological(
        (V.sub(1), V_p), 1, ft.indices[ft.values == outlet_marker]
    )
    dofs_obstacle = fem.locate_dofs_topological(
        (V.sub(0), V_u), 1, ft.indices[ft.values == obstacle_marker]
    )

    bc_dofs_total = np.concatenate(
        [dofs_walls[0], dofs_inflow[0], dofs_outflow[0], dofs_obstacle[0]]
    )

    bcs = [
        fem.dirichletbc(h, dofs_inflow, V.sub(0)),
        fem.dirichletbc(g, dofs_obstacle, V.sub(0), graph=graph_, map=V_u_map),
        fem.dirichletbc(noslip, dofs_walls, V.sub(0)),
        fem.dirichletbc(outflow, dofs_outflow, V.sub(1)),
    ]

    # Parameters
    nu = fem.Constant(mesh, ScalarType(1.0), name="ν", graph=graph_)
    alpha = 10.0
    f = fem.Function(V_u, name="f")
    f.interpolate(lambda x: (0.0 * x[0], 0.0 + 0.0 * x[1]))

    # Variational formulation
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.div(v) * p * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    F = a - L

    # Define the problem solver
    problem = fem.petsc.NonlinearProblem(
        F,
        up,
        bcs=bcs,
        petsc_options_prefix="forward_nonlinear",
        graph=graph_,
    )
    problem.solve(graph=graph_)

    # Define the objective function
    dObs = ufl.Measure(
        "ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker
    )
    J_form = (
        0.5 * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
        + alpha / 2 * ufl.inner(g, g) * dObs
    )

    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    gmsh.finalize()

    return {
        "graph_": graph_,
        "mesh": mesh,
        "V": V,
        "V_u_map": V_u_map,
        "up": up,
        "g": g,
        "nu": nu,
        "F": F,
        "J_form": J_form,
        "J": J,
        "bcs": bcs,
        "bc_dofs_total": bc_dofs_total,
        "dofs_obstacle": dofs_obstacle,
        "dObs": dObs,
    }


@pytest.fixture(scope="module")
def heat_equation_problem():
    """Set up the heat equation problem that will be used in all tests."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    dt = 0.01
    T = 0.05

    # Create true data set
    true_initial = fem.Function(V, name="u_true_initial")
    true_initial.interpolate(
        lambda x: np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    )
    u_prev = true_initial.copy()
    u_next = true_initial.copy()

    v = ufl.TestFunction(V)
    dt_constant = fem.Constant(domain, ScalarType(dt))

    # Set dirichlet boundary conditions
    uD = fem.Function(V)
    uD.interpolate(lambda x: 0.0 + 0.0 * x[0])
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    F = (
        ufl.inner((u_next - u_prev) / dt_constant, v) * ufl.dx
        + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
    )
    problem = fem.petsc.NewtonSolverNonlinearProblem(F, u_next)
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    t = 0.0
    while t < T:
        solver.solve(u_next)
        u_prev.x.array[:] = u_next.x.array[:]
        t += dt
    true_data = u_next.copy()

    # Create test data
    graph_ = Graph()

    # Set the initial values of the temperature variable u
    initial_guess = fem.Function(V, name="initial_guess", graph=graph_)
    initial_guess.interpolate(
        lambda x: 15.0 * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    )
    u_prev = initial_guess.copy(graph=graph_, name="u_prev")
    u_next = initial_guess.copy(graph=graph_, name="u_next")

    F = (
        ufl.inner((u_next - u_prev) / dt_constant, v) * ufl.dx
        + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
    )
    t = 0.0
    i = 0

    # Store the states for the whole time-domain
    u_iterations = [u_next.copy()]
    while t < T:
        i += 1
        F = (
            ufl.inner((u_next - u_prev) / dt_constant, v) * ufl.dx
            + ufl.inner(ufl.grad(u_next), ufl.grad(v)) * ufl.dx
        )
        problem = fem.petsc.NonlinearProblem(
            F,
            u_next,
            petsc_options_prefix="forward_nonlinear",
            graph=graph_,
        )
        problem.solve(graph=graph_, version=i)
        t += dt
        u_prev.assign(u_next, graph=graph_, version=i)

        # Storing the iterations for later visualization and testing
        u_iterations.append(u_next.copy())

    alpha = fem.Constant(domain, ScalarType(1.0e-6))

    J_form = (
        ufl.inner(true_data - u_next, true_data - u_next) * ufl.dx
        + alpha * ufl.inner(ufl.grad(initial_guess), ufl.grad(initial_guess)) * ufl.dx
    )
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    return {
        "graph_": graph_,
        "domain": domain,
        "J_form": J_form,
        "J": J,
        "initial_guess": initial_guess,
        "u_next": u_next,
        "u_prev": u_prev,
        "F": F,
        "u_iterations": u_iterations,
    }
