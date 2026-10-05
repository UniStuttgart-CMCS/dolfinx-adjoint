"""Unit tests for the derivative with respect to the value of a Dirichlet condition."""

import dolfinx
import numpy as np
import pytest
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_scalar_type, fem, mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph
from dolfinx_adjoint import fem as fem_ad

_DIRECT = {"ksp_type": "preonly", "pc_type": "lu"}


def _distinct_function(V: fem.FunctionSpace, reverse: bool = False) -> fem_ad.Function:
    """A function on V whose entries are distinct over all of the ranks, addressed by the global index of their dof."""
    function = dolfinx.fem.Function(V)
    block_size = V.dofmap.index_map_bs
    owned = V.dofmap.index_map.size_local * block_size
    total = V.dofmap.index_map.size_global * block_size

    indices = V.dofmap.index_map.local_range[0] * block_size + np.arange(owned)
    if reverse:
        indices = total - 1 - indices

    function.x.array[:owned] = 1.0 + indices / total
    function.x.scatter_forward()
    return function


def _boundary(domain: mesh.Mesh, boundary: str = "exterior") -> tuple:
    """The dimension and the local indices of the entities the condition is set on.

    The exterior facets, or the vertices at the two corners of the left boundary.
    """

    tdim = domain.topology.dim
    if boundary == "exterior":
        domain.topology.create_connectivity(tdim - 1, tdim)
        return tdim - 1, mesh.exterior_facet_indices(domain.topology)
    domain.topology.create_connectivity(0, tdim)
    corners = mesh.locate_entities(
        domain,
        0,
        lambda x: np.isclose(x[0], 0.0)
        & (np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)),
    )
    return 0, corners


def _extension(V: fem.FunctionSpace, bc, graph: Graph | None = None) -> float:
    """J = ∫ u·u on the calling rank, for the extension u of the value of the condition."""
    uh = fem_ad.Function(V, graph=graph)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    zero = dolfinx.fem.Function(V)
    problem = fem_ad.petsc.LinearProblem(
        ufl.inner(u, v) * ufl.dx,
        ufl.inner(zero, v) * ufl.dx,
        u=uh,
        bcs=[bc],
        petsc_options_prefix="test_dirichletbc_",
        petsc_options=_DIRECT,
        graph=graph,
        adjoint_petsc_options=_DIRECT,
    )
    problem.solve(graph=graph)
    return fem_ad.assemble_scalar(
        fem_ad.form(ufl.inner(uh, uh) * ufl.dx, graph=graph), graph=graph
    )


@pytest.mark.parametrize(
    "value_shape, boundary",
    [((), "exterior"), ((2,), "exterior"), ((), "corners")],
    ids=["scalar space", "blocked space", "two dofs"],
)
def test_function_value_is_differentiated_like_the_solve(
    unit_square_mesh_per_comm: mesh.Mesh,
    value_shape: tuple,
    boundary: str,
    mode: str,
    central_difference,
    directional_derivative,
):
    """Catch a value on the space it constrains paired with other dofs than DOLFINx sets."""
    domain = unit_square_mesh_per_comm
    V = fem.functionspace(domain, ("Lagrange", 1, value_shape))
    dofs = fem.locate_dofs_topological(V, *_boundary(domain, boundary))
    value = _distinct_function(V)
    direction = _distinct_function(V, reverse=True)

    graph_ = Graph()
    g = fem_ad.Function(V)
    g.x.array[:] = value.x.array
    graph_.track(g)
    J = _extension(V, fem_ad.dirichletbc(g, dofs, graph=graph_), graph_)
    derivative = directional_derivative(graph_, J, g, direction, mode, domain.comm)

    bc = dolfinx.fem.dirichletbc(value, dofs)
    difference = central_difference(
        lambda: domain.comm.allreduce(_extension(V, bc), op=MPI.SUM),
        value,
        direction,
        step=1.0,
    )
    assert np.isclose(derivative, difference)


@pytest.mark.parametrize(
    "sub_space", [0, 1], ids=["vector sub space", "scalar sub space"]
)
def test_value_on_a_collapsed_space_is_differentiated_like_the_solve(
    unit_square_mesh_per_comm: mesh.Mesh,
    sub_space: int,
    mode: str,
    central_difference,
    directional_derivative,
):
    """Catch a value on a collapsed sub space moved into the mixed space along other
    pairs of dofs than those the condition is given."""

    domain = unit_square_mesh_per_comm
    u_elem = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    p_elem = element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, mixed_element([u_elem, p_elem]))
    V_sub, _ = V.sub(sub_space).collapse()
    dofs = fem.locate_dofs_topological((V.sub(sub_space), V_sub), *_boundary(domain))
    value = _distinct_function(V_sub)
    direction = _distinct_function(V_sub, reverse=True)

    graph_ = Graph()
    g = fem_ad.Function(V_sub)
    g.x.array[:] = value.x.array
    graph_.track(g)
    J = _extension(
        V, fem_ad.dirichletbc(g, dofs, V.sub(sub_space), graph=graph_), graph_
    )
    derivative = directional_derivative(graph_, J, g, direction, mode, domain.comm)

    bc = dolfinx.fem.dirichletbc(value, dofs, V.sub(sub_space))
    difference = central_difference(
        lambda: domain.comm.allreduce(_extension(V, bc), op=MPI.SUM),
        value,
        direction,
        step=1.0,
    )
    assert np.isclose(derivative, difference)


@pytest.mark.parametrize(
    "value",
    [1.0, (1.0, 2.0), ((1.0, 2.0), (3.0, 4.0))],
    ids=["scalar constant", "vector constant", "tensor constant"],
)
def test_constant_value_is_differentiated_like_the_solve(
    unit_square_mesh_per_comm: mesh.Mesh,
    value: float | tuple,
    mode: str,
    central_difference,
    directional_derivative,
):
    """Catch a constant broadcast onto the dofs otherwise than DOLFINx does it, or its
    derivative summed over the dofs of one rank only."""

    domain = unit_square_mesh_per_comm
    value = np.asarray(value, dtype=default_scalar_type)
    direction = np.linspace(0.5, -1.5, value.size, dtype=default_scalar_type).reshape(
        value.shape
    )
    V = fem.functionspace(domain, ("Lagrange", 1, value.shape))
    dofs = fem.locate_dofs_topological(V, *_boundary(domain))

    graph_ = Graph()
    c = fem_ad.Constant(domain, value, graph=graph_)
    J = _extension(V, fem_ad.dirichletbc(c, dofs, V, graph=graph_), graph_)
    derivative = directional_derivative(graph_, J, c, direction, mode, domain.comm)

    constant = dolfinx.fem.Constant(domain, value)
    bc = dolfinx.fem.dirichletbc(constant, dofs, V)
    difference = central_difference(
        lambda: domain.comm.allreduce(_extension(V, bc), op=MPI.SUM),
        constant,
        direction,
        step=1.0,
    )
    assert np.isclose(derivative, difference)


def _gradients_of_a_boundary_control_per_step(
    domain: mesh.Mesh, reuse: bool
) -> list[np.ndarray]:
    """Solve -Δu + u = 1 twice, with u = g on the left boundary.

    Before each solve, g is assigned a control of its own, and each solve contributes
    ∫ u² to the functionals, which are differentiated with respect to both controls.

    Args:
        domain: The mesh.
        reuse: Whether one boundary condition and one problem serve both solves, as in
            DOLFINx, or both are set up after each assignment.

    Returns:
        The gradient with respect to each control.
    """
    graph_ = Graph()
    V = fem.functionspace(domain, ("Lagrange", 1))
    dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    controls = []
    for step in range(2):
        control = fem_ad.Function(V, name=f"g{step}")
        control.interpolate(lambda x, step=step: 1.0 + (step + 1) * x[1])
        graph_.track(control)
        controls.append(control)
    g = fem_ad.Function(V, name="g", graph=graph_)
    uh = fem_ad.Function(V, name="uh", graph=graph_)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(u, v)) * ufl.dx
    L = ufl.conj(v) * ufl.dx
    direct_solver = {"ksp_type": "preonly", "pc_type": "lu"}

    def set_up():
        bc = fem_ad.dirichletbc(g, dofs, graph=graph_)
        return fem_ad.petsc.LinearProblem(
            a,
            L,
            u=uh,
            bcs=[bc],
            petsc_options_prefix="test_problem_boundary_control_per_step_",
            petsc_options=direct_solver,
            graph=graph_,
            adjoint_petsc_options=direct_solver,
        )

    problem = set_up() if reuse else None
    J = fem_ad.form(ufl.inner(uh, uh) * ufl.dx, graph=graph_)
    functionals = []
    for control in controls:
        g.assign(control, graph=graph_)
        (problem if reuse else set_up()).solve(graph=graph_)
        functionals.append(fem_ad.assemble_scalar(J, graph=graph_))

    gradients = graph_.backprop(functionals, controls)
    return [gradient.array.copy() for gradient in gradients]


def test_boundary_condition_reused_in_a_loop_applies_the_latest_value(
    unit_square_mesh: mesh.Mesh,
) -> None:
    """Catch a boundary condition whose edge starts at the version it was created with."""
    reused = _gradients_of_a_boundary_control_per_step(unit_square_mesh, reuse=True)
    per_step = _gradients_of_a_boundary_control_per_step(unit_square_mesh, reuse=False)

    assert all(np.allclose(a, b) for a, b in zip(reused, per_step, strict=True))


@pytest.mark.skipif(
    dolfinx.__version__ < "0.12",
    reason="DOLFINx 0.11 keeps a scalar value without a Constant to control.",
)
def test_boundary_condition_given_as_a_scalar_is_a_control_through_its_value(
    unit_square_mesh_per_comm: mesh.Mesh,
) -> None:
    """Catch a condition given as a scalar whose value cannot become a control."""
    domain = unit_square_mesh_per_comm
    graph_ = Graph()
    V = fem.functionspace(domain, ("Lagrange", 1))
    dofs = fem.locate_dofs_geometrical(
        V,
        lambda x: np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0),
    )
    native_bc = dolfinx.fem.dirichletbc(ScalarType(3.0), dofs, V)
    graph_.track(native_bc.g)
    bc = fem_ad.dirichletbc(native_bc.g, dofs, V, graph=graph_)
    uh = fem_ad.Function(V, name="uh", graph=graph_)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    zero = fem_ad.Constant(domain, ScalarType(0.0))
    direct_solver = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = fem_ad.petsc.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        zero * ufl.conj(v) * ufl.dx,
        u=uh,
        bcs=[bc],
        petsc_options_prefix="test_boundary_condition_given_as_a_scalar_",
        petsc_options=direct_solver,
        graph=graph_,
        adjoint_petsc_options=direct_solver,
    )
    problem.solve(graph=graph_)
    J = fem_ad.assemble_scalar(fem_ad.form(uh * ufl.dx, graph=graph_), graph=graph_)

    (gradient,) = graph_.backprop(J, bc.g)

    assert np.isclose(gradient, 1.0)
