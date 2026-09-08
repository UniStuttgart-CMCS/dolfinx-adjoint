DOLFINx-ADJOINT
================

Automatic differentiation for `DOLFINx <https://github.com/FEniCS/dolfinx>`_ using the adjoint method.
Efficient sensitivity analysis and gradient-based optimization for finite element simulations.

----

Key Features
------------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: :octicon:`git-branch` Automatic Differentiation

      Computational graph construction that automatically tracks operations and dependencies.

   .. grid-item-card:: :octicon:`iterations` Adjoint Method

      Efficient gradient computation through backpropagation on the computational graph.

   .. grid-item-card:: :octicon:`plug` DOLFINx Integration

      Seamless integration into existing DOLFINx workflows with minimal code changes.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: :octicon:`infinity` Complex PDEs

      Support for nonlinear problems and coupled multi-physics simulations.

   .. grid-item-card:: :octicon:`project` Graph Visualization

      Built-in tools for visualizing and debugging the computational graph.

   .. grid-item-card:: :octicon:`diff` Minimal Changes

      Drop-in replacements for DOLFINx objects -- just add a ``graph`` argument.

----

How It Works
------------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: 1. Build the graph

      Create a ``Graph`` object and pass it to DOLFINx operations.
      The graph automatically records every Function, Constant,
      and boundary condition.

   .. grid-item-card:: 2. Solve forward

      Run your simulation as usual. Solvers, forms, and assemblies
      are recorded as edges connecting the nodes in the graph.

   .. grid-item-card:: 3. Backpropagate

      Call ``graph.backprop(J, x)`` to compute the gradient of any
      quantity *J* with respect to any parameter *x*.

The computational graph is a directed acyclic graph (DAG) where:

- **Nodes** represent DOLFINx objects (Functions, Constants, boundary conditions, solver states)
- **Edges** represent operations and store the corresponding adjoint equations

During backpropagation, edges compute derivatives using UFL automatic differentiation
for forms and implicit differentiation for PDE solves:

.. math::

   \frac{du}{df} = -\left(\frac{\partial F}{\partial u}\right)^{-1} \frac{\partial F}{\partial f}

----

Supported Operations
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

All overloaded objects accept an optional ``graph=`` keyword argument. When omitted,
they behave identically to their DOLFINx counterparts.

----

Quick Start
-----------

**Install**

.. code-block:: bash

   git clone https://github.com/unistuttgart-cmcs/dolfinx-adjoint.git
   cd dolfinx-adjoint && pip install -e .

Requires Python >= 3.12 and `DOLFINx <https://github.com/FEniCS/dolfinx>`_ >= 0.10.0.
CI targets DOLFINx 0.11.0. Compatibility with the declared minimum, 0.10.0,
is not currently checked in CI.

For demos and tests, use ``pip install -e ".[all]"``.

**Usage** -- compute dJ/df for a Poisson problem:

.. code-block:: python

   from dolfinx_adjoint import Graph, fem

   graph_ = Graph()

   # Set up your DOLFINx problem, passing graph= to track operations
   f  = fem.Function(W, name="f", graph=graph_)
   uh = fem.Function(V, name="u", graph=graph_)

   problem = fem.petsc.LinearProblem(a, L, u=uh, bcs=bcs, graph=graph_)
   problem.solve(graph=graph_)

   J = fem.assemble_scalar(fem.form(J_form, graph=graph_), graph=graph_)

   # Compute the gradient
   dJdf = graph_.backprop(id(J), id(f))

----

Demos
-----

Jupyter notebook examples in the ``demos/`` directory:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`beaker` Poisson Equation
      :link: https://github.com/unistuttgart-cmcs/dolfinx-adjoint/blob/main/demos/poisson.ipynb

      Stationary Poisson equation on a unit square. Computes gradients of a
      tracking-type objective w.r.t. the source term *f*, the diffusion
      coefficient :math:`\nu`, and the Dirichlet boundary values.

   .. grid-item-card:: :octicon:`flame` Heat Equation
      :link: https://github.com/unistuttgart-cmcs/dolfinx-adjoint/blob/main/demos/heat_equation.ipynb

      Time-dependent heat equation solved with backward Euler. Demonstrates
      adjoint backpropagation through multiple time steps to compute
      sensitivities w.r.t. the initial condition.

   .. grid-item-card:: :octicon:`north-star` Linear Elasticity
      :link: https://github.com/unistuttgart-cmcs/dolfinx-adjoint/blob/main/demos/linear_elasticity.ipynb

      3D cantilever beam under gravity. Computes gradients of the deformation
      energy w.r.t. the Lame parameters :math:`\lambda` and :math:`\mu`.

   .. grid-item-card:: :octicon:`milestone` Stokes Flow
      :link: https://github.com/unistuttgart-cmcs/dolfinx-adjoint/blob/main/demos/stokes.ipynb

      Stokes flow around a cylindrical obstacle using P2-P1 Taylor-Hood
      elements. Computes gradients of viscous dissipation w.r.t. the viscosity
      and the obstacle boundary condition.

----

Acknowledgments
---------------

This library builds upon the `FEniCS Project <https://fenicsproject.org/>`_ and uses
`DOLFINx <https://github.com/FEniCS/dolfinx>`_ as its foundation.

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/modules
