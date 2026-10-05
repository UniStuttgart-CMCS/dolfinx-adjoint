.. _differentiation:

Differentiation
===============

The computational graph is a directed acyclic graph (DAG) where:

- **Nodes** represent DOLFINx objects (Functions, Constants, boundary conditions, solver states)
- **Edges** represent operations and store the corresponding adjoint equations

During backpropagation, edges compute derivatives using UFL automatic differentiation
for forms and implicit differentiation for PDE solves:

.. math::

   \frac{du}{df} = -\left(\frac{\partial F}{\partial u}\right)^{-1} \frac{\partial F}{\partial f}

Selecting outputs and controls
------------------------------

Pass objects or exact recorded nodes to ``Graph.backprop``:

.. code-block:: python

   dJdf, dJdnu = graph.backprop(J, [f, nu])
   initial_f = graph.get_node(f, version=0)
   (dJdf_initial,) = graph.backprop(J, initial_f)

Objects select their latest recorded version. Nodes must belong to this graph
and select that particular version. A single control still returns a tuple;
omitting controls stores leaf gradients and returns ``None``. An empty control
sequence also selects the dependency leaves. Both ``backprop`` and ``get_node`` accept
objects or exact nodes; integer arguments are treated as recorded values, not object
IDs.

Multiple objectives produce one gradient of their weighted sum per control. To keep
each objective's gradients separate, call ``backprop`` once per objective with all
controls. See :ref:`multiple-objectives-and-controls` for examples. All ranks must request
corresponding outputs and controls in the same order.

.. _multiple-objectives-and-controls:

Multiple objectives and controls
--------------------------------

``Graph.backprop`` returns one gradient per requested control, in control order.
Multiple outputs are differentiated together as a weighted vector–Jacobian product,
like ``torch.autograd.grad``. For scalar objectives :math:`J_j` and seeds
:math:`s_j`, the returned gradient for control :math:`m_i` is

.. math::

   g_i = \sum_j s_j \frac{\partial J_j}{\partial m_i}.

A single seed applies to every output; the default ``1.0`` differentiates the sum of
scalar objectives. Alternatively, supply a seed sequence of the same length and order
as the outputs.

For two recorded scalar objectives and three recorded controls:

.. code-block:: python

   controls = (m1, m2, m3)
   summed = graph_.backprop((J1, J2), controls)  # three gradients of J1 + J2
   weighted = graph_.backprop((J1, J2), controls, grad_outputs=(1.0, 0.5))

To obtain each objective's gradients separately, request all controls in one call per
objective, using the same graph.
