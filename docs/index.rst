DOLFINx-GraphAD
===============

Automatic differentiation for `DOLFINx <https://github.com/FEniCS/dolfinx>`_
using the adjoint method. Compute sensitivities and optimize finite element
simulations with an explicit computational graph that you own.

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

Start with :ref:`getting-started` or explore a complete notebook in :ref:`demos`.

Featured demos
--------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`beaker` Poisson Equation
      :link: https://github.com/unistuttgart-cmcs/dolfinx-graph-ad/blob/main/demos/poisson.ipynb

      Stationary Poisson equation on a unit square. Computes gradients of a
      tracking-type objective w.r.t. the source term *f*, the diffusion
      coefficient :math:`\nu`, and the Dirichlet boundary values.

   .. grid-item-card:: :octicon:`flame` Heat Equation
      :link: https://github.com/unistuttgart-cmcs/dolfinx-graph-ad/blob/main/demos/heat_equation.ipynb

      Time-dependent heat equation solved with backward Euler. Demonstrates
      adjoint backpropagation through multiple time steps to compute
      sensitivities w.r.t. the initial condition.

See :ref:`all demos <demos>` for linear elasticity and Stokes flow, alongside these
examples.

.. toctree::
   :maxdepth: 2
   :caption: User guide
   :hidden:

   getting-started
   recording
   solvers
   differentiation
   demos

.. include:: api/dolfinx_graph_ad.rst

Acknowledgments
---------------

This library builds upon the `FEniCS Project <https://fenicsproject.org/>`_
and uses DOLFINx as its foundation.
