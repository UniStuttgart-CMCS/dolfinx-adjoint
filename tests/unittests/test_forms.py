"""Unit tests for graph-tracked forms and their derivatives."""

import numpy as np
import pytest
import ufl
from dolfinx import mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_adjoint import Graph, fem


def test_form_constant_edge_gradient_without_coefficient(
    unit_square_mesh_per_comm: mesh.Mesh,
) -> None:
    """A constant-only form must not reference an undefined coefficient.

    The exact gradient also catches invalid mesh access and scalar assembly
    of a derivative with a test argument. The seed catches ignored input values.
    Updating c catches evaluation at its constructor value.
    """
    domain = unit_square_mesh_per_comm
    graph_ = Graph()
    c = fem.Constant(domain, ScalarType(1.0), graph=graph_)
    c.value = 3.0

    J_form = 0.5 * c**2 * ufl.dx(domain=domain)
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

    seed = -2.5
    gradient = graph_.backprop(id(J), id(c), seed=seed)

    # The unit square has area 1, so dJ/dc is 3; the constructor value gives 1.
    assert np.isclose(gradient, seed * 3.0)


@pytest.mark.parametrize("measure_name", ["dx", "ds", "dS"])
@pytest.mark.parametrize(
    "value",
    [2.0, (2.0,), (2.0, 3.0), (2.0, 3.0, 5.0), ((2.0, 3.0), (5.0, 7.0))],
    ids=["scalar", "vector1", "vector2", "vector3", "tensor"],
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
    c = fem.Constant(domain, value, graph=graph_)
    measure = ufl.Measure(measure_name, domain=domain)

    V = fem.functionspace(domain, ("Lagrange", 1))
    u = fem.Function(V, graph=graph_)
    u.interpolate(lambda x: 1.0 + x[0])
    u.x.scatter_forward()
    weight = ufl.avg(u) if measure_name == "dS" else u
    weighted_measure = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(weight * measure)), op=MPI.SUM
    )

    restricted_c = c("+") if measure_name == "dS" else c
    J_form = ufl.inner(restricted_c, restricted_c) * weight * measure
    J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)
    seed = -2.5
    gradient = graph_.backprop(id(J), id(c), seed=seed)

    expected = seed * 2.0 * value * weighted_measure
    if value.ndim == 0:
        assert np.isscalar(gradient) and np.isclose(gradient, expected)
    else:
        assert gradient.getSize() == expected.size
        if gradient.getLocalSize() > 0:
            np.testing.assert_allclose(gradient.array_r, expected.ravel())
