"""Unit tests for graph-tracked forms and their derivatives."""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph
from dolfinx_adjoint import fem as fem_ad


def test_form_constant_edge_gradient_without_coefficient(
    unit_square_mesh_per_comm: mesh.Mesh,
) -> None:
    """A constant-only form must not reference an undefined coefficient.

    The exact gradient also catches invalid mesh access and scalar assembly
    of a derivative with a test argument. The seed catches ignored input values.
    """
    domain = unit_square_mesh_per_comm
    graph_ = Graph()
    c = fem_ad.Constant(domain, ScalarType(3.0), graph=graph_)

    J_form = 0.5 * c**2 * ufl.dx(domain=domain)
    J = fem_ad.assemble_scalar(fem_ad.form(J_form, graph=graph_), graph=graph_)

    seed = -2.5
    (gradient,) = graph_.backprop(J, c, grad_outputs=seed)

    # The unit square has area 1, so dJ/dc is 3.
    assert np.isclose(gradient, seed * 3.0)


@pytest.mark.parametrize("measure_name", ["dx", "ds", "dS"])
@pytest.mark.parametrize(
    "value",
    [2.0, (2.0,), (2.0, 3.0, 5.0), ((2.0, 3.0), (5.0, 7.0))],
    ids=["scalar", "vector1", "vector3", "tensor"],
)
def test_form_constant_gradient_on_measures(
    unit_square_mesh_per_comm: mesh.Mesh, measure_name: str, value: float | tuple
) -> None:
    """Catch incorrect Constant components and lost explicit restrictions on dS.

    The constant edge must capture c rather than the tracked coefficient u.
    The dx and ds cases guard against a correction that changes their gradients.
    """

    domain = unit_square_mesh_per_comm
    graph_ = Graph()
    value = np.asarray(value, dtype=ScalarType)
    c = fem_ad.Constant(domain, value, graph=graph_)
    measure = ufl.Measure(measure_name, domain=domain)

    V = fem.functionspace(domain, ("Lagrange", 1))
    u = fem_ad.Function(V)
    u.interpolate(lambda x: 1.0 + x[0])
    u.x.scatter_forward()
    graph_.track(u)
    weight = ufl.avg(u) if measure_name == "dS" else u
    weighted_measure = domain.comm.allreduce(
        fem_ad.assemble_scalar(fem_ad.form(weight * measure)), op=MPI.SUM
    )

    restricted_c = c("+") if measure_name == "dS" else c
    J_form = ufl.inner(restricted_c, restricted_c) * weight * measure
    J = fem_ad.assemble_scalar(fem_ad.form(J_form, graph=graph_), graph=graph_)
    seed = -2.5
    (gradient,) = graph_.backprop(J, c, grad_outputs=seed)

    expected = seed * 2.0 * value * weighted_measure
    if value.ndim == 0:
        assert np.isscalar(gradient) and np.isclose(gradient, expected)
    else:
        assert gradient.getSize() == expected.size
        if gradient.getLocalSize() > 0:
            np.testing.assert_allclose(gradient.array_r, expected.ravel())


@pytest.mark.parametrize("control_name", ["coefficient", "constant"])
def test_form_edges_compile_the_derivative_with_the_recorded_arguments(
    unit_square_mesh: mesh.Mesh,
    left_half_unit_square_mesh: tuple[mesh.Mesh, mesh.EntityMap],
    control_name: str,
) -> None:
    """Catch either edge of a form compiling its derivative bare."""
    submesh, cell_map = left_half_unit_square_mesh
    graph_ = Graph()

    V = fem.functionspace(unit_square_mesh, ("Lagrange", 1))
    f = fem_ad.Function(V)
    f.x.array[:] = 3.0
    graph_.track(f)
    c = fem_ad.Constant(
        unit_square_mesh, np.asarray((3.0,), dtype=ScalarType), graph=graph_
    )

    J = fem_ad.assemble_scalar(
        fem_ad.form(
            (ufl.inner(f, f) + ufl.inner(c, c)) * ufl.dx(domain=submesh),
            entity_maps=[cell_map],
            graph=graph_,
        ),
        graph=graph_,
    )
    control = {"coefficient": f, "constant": c}[control_name]

    (gradient,) = graph_.backprop(J, control)

    # Both derivatives integrate 2 * 3 over the left half of the unit square.
    assert np.isclose(gradient.sum(), 2.0 * 3.0 * 0.5)


def test_function_assigned_between_assemblies_depends_on_each_value(
    unit_square_mesh: mesh.Mesh,
) -> None:
    """Catch a form differentiated at other values than those it was assembled with."""
    graph_ = Graph()
    V = fem.functionspace(unit_square_mesh, ("Lagrange", 1))
    f = fem_ad.Function(V, name="f")
    f.x.array[:] = 1.0
    graph_.track(f)
    g = fem_ad.Function(V, name="g")
    g.x.array[:] = 3.0
    graph_.track(g)
    original = graph_.get_node(f)

    M = fem_ad.form(ufl.inner(f, f) * ufl.dx, graph=graph_)
    first = fem_ad.assemble_scalar(M, graph=graph_)
    f.assign(g, graph=graph_)
    second = fem_ad.assemble_scalar(M, graph=graph_)
    gradients = graph_.backprop(
        [first, second], [original, g], grad_outputs=[1.0, 10.0]
    )

    # Each gradient summed over its entries is its integral, by the partition of unity.
    assert np.allclose([gradient.sum() for gradient in gradients], (2.0, 60.0))


@pytest.mark.parametrize("control_name", ["coefficient", "constant"])
def test_form_edges_ignore_writes_after_the_assembly(
    unit_square_mesh_per_comm: mesh.Mesh, control_name: str
) -> None:
    """Catch a form edge that differentiates at the values its inputs hold at backprop."""
    domain = unit_square_mesh_per_comm
    graph_ = Graph()

    V = fem.functionspace(domain, ("Lagrange", 1))
    u = fem_ad.Function(V)
    u.interpolate(lambda x: 1.0 + x[0])
    u.x.scatter_forward()
    graph_.track(u)
    c = fem_ad.Constant(domain, np.asarray((2.0,), dtype=ScalarType), graph=graph_)
    w = fem_ad.Function(V)
    w.interpolate(lambda x: 1.0 + x[1])
    w.x.scatter_forward()

    J_form = ufl.inner(c, c) * w * ufl.inner(u, u) * ufl.dx
    J = fem_ad.assemble_scalar(fem_ad.form(J_form, graph=graph_), graph=graph_)

    # dJ/du summed over its entries, which the partition of unity turns into an integral,
    # and dJ/dc, both at the assembled values.
    derivative_form = {
        "coefficient": 2.0 * ufl.inner(c, c) * w * u * ufl.dx,
        "constant": 2.0 * c[0] * w * ufl.inner(u, u) * ufl.dx,
    }[control_name]
    expected = domain.comm.allreduce(
        fem_ad.assemble_scalar(fem_ad.form(derivative_form)), op=MPI.SUM
    )

    u.x.array[:] += 1.0
    c.value = (3.0,)

    control = {"coefficient": u, "constant": c}[control_name]
    (gradient,) = graph_.backprop(J, control)

    assert np.isclose(gradient.sum(), expected)
