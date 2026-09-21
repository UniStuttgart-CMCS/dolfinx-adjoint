"""
The edge is the transpose of the operation the condition performs, so every test compares
the two through the identity

    <edge(u), v> = <u, condition(v)>,

for an accumulated gradient u on the space the condition constrains and a value v of the
condition.
"""

import dolfinx
import numpy as np
import pytest
from basix.ufl import element, mixed_element
from dolfinx import default_scalar_type, mesh

from dolfinx_adjoint import Graph, fem


def _distinct_function(V: fem.FunctionSpace, reverse: bool = False) -> fem.Function:
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


@pytest.mark.parametrize(
    "value_shape", [(), (2,)], ids=["scalar space", "blocked space"]
)
def test_gradient_of_a_function_value(
    value_shape: tuple[int, ...], unit_square_mesh_per_comm: mesh.Mesh
):
    """The gradient of a function value on the space the condition constrains restricts
    the accumulated gradient to the entries the condition sets."""
    graph_ = Graph()
    domain = unit_square_mesh_per_comm
    facet_dim = domain.topology.dim - 1
    domain.topology.create_connectivity(facet_dim, domain.topology.dim)
    facets = mesh.exterior_facet_indices(domain.topology)

    V = fem.functionspace(domain, ("Lagrange", 1, value_shape))
    dofs = fem.locate_dofs_topological(V, facet_dim, facets)

    # A value on the space it constrains is passed without the space itself.
    g = fem.Function(V, name="g", graph=graph_)
    bc = fem.dirichletbc(g, dofs, graph=graph_)

    u = _distinct_function(V)
    v = _distinct_function(V, reverse=True)

    # What the condition does with v, performed by DOLFINx.
    condition_of_v = dolfinx.fem.Function(V)
    dolfinx.fem.dirichletbc(v, dofs).set(condition_of_v.x.array)

    edge = graph_.get_edge(graph_.get_node(id(g)), graph_.get_node(id(bc)))
    edge.input_value = u.x.petsc_vec

    assert np.isclose(
        edge.calculate_adjoint().dot(v.x.petsc_vec),
        u.x.petsc_vec.dot(condition_of_v.x.petsc_vec),
    )


def test_gradient_of_a_condition_on_two_dofs(unit_square_mesh: mesh.Mesh):
    """A condition holding exactly two dofs is transposed like any other.

    Its dofs are a flat array of two entries, which catches an overload taking them for
    the pair of arrays a value on a different space is given with.
    """

    graph_ = Graph()
    V = fem.functionspace(unit_square_mesh, ("Lagrange", 1))
    dofs = fem.locate_dofs_geometrical(
        V,
        lambda x: np.isclose(x[0], 0.0)
        & (np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)),
    )
    # The two corners of the left boundary, which the condition is then built on.
    assert dofs.size == 2

    g = fem.Function(V, name="g", graph=graph_)
    bc = fem.dirichletbc(g, dofs, graph=graph_)

    u = _distinct_function(V)
    v = _distinct_function(V, reverse=True)

    # What the condition does with v, performed by DOLFINx.
    condition_of_v = dolfinx.fem.Function(V)
    dolfinx.fem.dirichletbc(v, dofs).set(condition_of_v.x.array)

    edge = graph_.get_edge(graph_.get_node(id(g)), graph_.get_node(id(bc)))
    edge.input_value = u.x.petsc_vec

    assert np.isclose(
        edge.calculate_adjoint().dot(v.x.petsc_vec),
        u.x.petsc_vec.dot(condition_of_v.x.petsc_vec),
    )


@pytest.mark.parametrize(
    "sub_space", [0, 1], ids=["vector sub space", "scalar sub space"]
)
def test_gradient_of_a_value_on_a_collapsed_space(
    sub_space: int, unit_square_mesh_per_comm: mesh.Mesh
):
    """The gradient of a function value on the collapsed sub space a condition constraint is moved into that space along the pairs of dofs the condition is given."""

    graph_ = Graph()
    domain = unit_square_mesh_per_comm
    facet_dim = domain.topology.dim - 1
    domain.topology.create_connectivity(facet_dim, domain.topology.dim)
    facets = mesh.exterior_facet_indices(domain.topology)

    u_elem = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    p_elem = element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, mixed_element([u_elem, p_elem]))
    V_sub, _ = V.sub(sub_space).collapse()
    dofs = fem.locate_dofs_topological((V.sub(sub_space), V_sub), facet_dim, facets)

    # A value on a different space than the one it constrains is passed with it.
    g = fem.Function(V_sub, name="g", graph=graph_)
    bc = fem.dirichletbc(g, dofs, V.sub(sub_space), graph=graph_)

    u = _distinct_function(V)
    v = _distinct_function(V_sub, reverse=True)

    # What the condition does with v, performed by DOLFINx.
    condition_of_v = dolfinx.fem.Function(V)
    dolfinx.fem.dirichletbc(v, dofs, V.sub(sub_space)).set(condition_of_v.x.array)

    edge = graph_.get_edge(graph_.get_node(id(g)), graph_.get_node(id(bc)))
    edge.input_value = u.x.petsc_vec

    assert np.isclose(
        edge.calculate_adjoint().dot(v.x.petsc_vec),
        u.x.petsc_vec.dot(condition_of_v.x.petsc_vec),
    )


@pytest.mark.parametrize(
    "value",
    [1.0, (1.0, 2.0), ((1.0, 2.0), (3.0, 4.0))],
    ids=["scalar constant", "vector constant", "tensor constant"],
)
def test_gradient_of_a_constant_value(
    value: float | tuple, unit_square_mesh_per_comm: mesh.Mesh
):
    """The gradient of a constant value adds up the accumulated gradient over the entries the condition sets."""
    graph_ = Graph()
    domain = unit_square_mesh_per_comm
    facet_dim = domain.topology.dim - 1
    domain.topology.create_connectivity(facet_dim, domain.topology.dim)
    facets = mesh.exterior_facet_indices(domain.topology)

    value = np.asarray(value, dtype=default_scalar_type)
    V = fem.functionspace(domain, ("Lagrange", 1, np.shape(value)))
    dofs = fem.locate_dofs_topological(V, facet_dim, facets)

    # A constant is broadcast onto the space it constrains, which it is passed with.
    c = fem.Constant(domain, value, name="c", graph=graph_)
    bc = fem.dirichletbc(c, dofs, V, graph=graph_)

    u = _distinct_function(V)

    # Determine every component of the transpose using DOLFINx's forward operation.
    # Every rank visits the same directions; each dot counts owned entries globally.
    expected = np.zeros_like(value)
    for component in np.ndindex(value.shape):
        direction = np.zeros_like(value)
        direction[component] = 1.0
        condition_of_direction = dolfinx.fem.Function(V)
        direction_bc = dolfinx.fem.dirichletbc(direction, dofs, V)
        direction_bc.set(condition_of_direction.x.array)
        expected[component] = u.x.petsc_vec.dot(condition_of_direction.x.petsc_vec)

    edge = graph_.get_edge(graph_.get_node(id(c)), graph_.get_node(id(bc)))
    edge.input_value = u.x.petsc_vec

    gradient = edge.calculate_adjoint()

    if value.ndim == 0:
        assert np.isscalar(gradient) and np.isclose(gradient, expected)
    else:
        assert gradient.getSize() == expected.size
        if gradient.getLocalSize() > 0:
            np.testing.assert_allclose(gradient.array_r, expected.ravel())
