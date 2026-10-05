Recording operations
====================

All overloaded objects accept an optional ``graph=`` keyword argument. When omitted,
they behave identically to their DOLFINx counterparts. Everything else, e.g.
``functionspace``, is used from DOLFINx and is not recorded. Import both next to each
other, so that every call shows whether it is recorded:

.. code-block:: python

   from dolfinx import fem
   from dolfinx_adjoint import fem as fem_ad

   V = fem.functionspace(domain, ("Lagrange", 1))  # DOLFINx, not recorded
   u = fem_ad.Function(V, graph=graph_)  # recorded

Create a new ``Graph`` for each forward evaluation. A single evaluation can contain
many solves, including a complete time loop. Construct each overloaded problem with
the graph that will record its solves, and create a new problem for a new graph.
Record form compilation, constructors and subsequent operations explicitly with
``graph=``. Register existing controls with ``graph.track(function)`` or
``graph.track(constant)`` before recording operations. No global tape is installed.

Supported operations
--------------------

The following DOLFINx objects and functions have drop-in replacements that record
operations on the computational graph:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - DOLFINx object
     - Adjoint capability
   * - ``fem.Function``
     - Graph-tracked initialization, ``copy()``, and ``assign()``
   * - ``fem.Constant``
     - Graph-tracked constants with adjoint support
   * - ``fem.form``
     - UFL differentiation w.r.t. coefficients and constants
   * - ``fem.assemble_scalar``
     - Tracked scalar assembly for objective functions
   * - ``fem.dirichletbc``
     - Boundary condition adjoint via DOF restriction
   * - ``fem.petsc.LinearProblem``
     - Implicit adjoint solve for linear systems
   * - ``fem.petsc.NonlinearProblem``
     - Implicit adjoint solve for nonlinear systems

See :ref:`derivative-solvers` for the options of the adjoint solvers.


Saved values and versions
-------------------------

Each solve and scalar assembly differentiates at its saved inputs and records the
latest versions it reads. A version is saved by the recorded call that creates it.
Set a leaf control before ``graph.track(control)``; subsequent Function assignments
must be recorded with ``graph=``. Raw array writes do not create versions and are
not detected. Untracked values are read in place and must remain unchanged until
differentiation finishes.

An output Function need not be tracked before its first solve. Without a recorded
initial value, the first solve produces version 0; otherwise it produces the next
version. Later solves record subsequent versions automatically unless ``version=``
is supplied explicitly. A linear problem whose forms read its own solution, such
as a Picard step, depends on the value before the solve as a separate input.

Boundary conditions follow DOLFINx's list order: at overlapping degrees of freedom,
the last condition supplies the value and its derivative. Untracked conditions
also overwrite earlier ones. Each solve retains its own precedence. Reverse
contributions use owned entries and accumulate mapped ghost contributions to their
owners. Native DOLFINx Constants can be registered with ``graph.track`` and passed
to recorded forms, problems and boundary-condition constructors.

Scalar assembly still returns the rank-local value. Keep that object for
``backprop`` and reduce a separate value over the mesh communicator for reporting.
Derivatives retain the forward quadrature degree when it is inferred without an
explicit degree or supplied as a nonnegative degree. Explicit negative degrees
remain a known limitation of the current quadrature preservation.
