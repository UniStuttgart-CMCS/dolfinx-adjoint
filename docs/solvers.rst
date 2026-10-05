.. _derivative-solvers:

Derivative solver configuration
===============================

Adjoint solver options can be passed to the overloaded problem's constructor. They
configure the solver of the adjoint equations of the problem's recorded solves:

.. code-block:: python

   problem = fem_ad.petsc.LinearProblem(
       a, L,
       petsc_options_prefix="forward_",
       adjoint_petsc_options=adjoint_options,
       graph=graph_,
   )
   problem.solve(graph=graph_)

The constructor also accepts a separate ``adjoint_petsc_options_prefix``, which
defaults to the forward prefix followed by ``adjoint_``. Without options the adjoint
solver keeps PETSc defaults; forward options and null spaces are never inherited. You
assert its convergence, e.g. with ``"ksp_error_if_not_converged": True``.
