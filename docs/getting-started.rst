.. _getting-started:

Getting started
===============

Installation
------------

Follow the installation instructions in the
`project README <https://github.com/unistuttgart-cmcs/dolfinx-adjoint#installation>`_,
including the prerequisites and optional dependencies for demos and tests.

Record a forward solve
----------------------

Use DOLFINx for mesh, space and UFL setup, and the overloaded ``fem_ad`` API
for the operations you want to record. The following excerpt assumes a mesh
``domain``, scalar spaces ``V`` and ``W`` on it, and boundary conditions ``bcs``
for the solution in ``V`` from your Poisson setup.
See :ref:`demos` for complete notebooks.

Compute :math:`dJ/df` by recording both the problem constructor and its solve:

.. code-block:: python

   import ufl

   from dolfinx_adjoint import Graph
   from dolfinx_adjoint import fem as fem_ad

   graph_ = Graph()

   # Set the control value before recording its initial snapshot.
   f = fem_ad.Function(W, name="f")
   f.x.array[:] = 1.0
   f.x.scatter_forward()
   graph_.track(f)
   uh = fem_ad.Function(V, name="u", graph=graph_)

   u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
   a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
   L = ufl.inner(f, v) * ufl.dx
   problem = fem_ad.petsc.LinearProblem(
       a, L, u=uh, bcs=bcs, petsc_options_prefix="poisson_", graph=graph_
   )
   problem.solve(graph=graph_)

   J_form = ufl.inner(uh, uh) * ufl.dx
   J = fem_ad.assemble_scalar(fem_ad.form(J_form, graph=graph_), graph=graph_)

   # Compute the gradient
   (dJdf,) = graph_.backprop(J, f)

Pass ``graph=`` on every call you want recorded, including constructors, form
compilation and assembly. Register a Function or Constant that exists already, e.g.
one whose values are set before it becomes a control, with ``graph.track(f)`` before
its first recorded use.

``backprop`` returns a tuple even for one control. For a Function control, its
derivative is a PETSc vector whose owned entries are valid.

``assemble_scalar`` returns a rank-local value. Differentiate that recorded value
before reducing it; use ``domain.comm.allreduce(J, op=MPI.SUM)`` for the global
objective. Every participating rank must record and differentiate corresponding
operations in the same order.

Continue with :doc:`recording` for supported operations,
:ref:`derivative-solvers` for adjoint solver options, and :ref:`differentiation` for
multiple outputs and controls.
