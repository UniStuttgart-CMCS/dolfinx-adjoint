.. _demos:

Demos
=====

Jupyter notebook examples in the ``demos/`` directory:

.. grid:: 2
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

   .. grid-item-card:: :octicon:`north-star` Linear Elasticity
      :link: https://github.com/unistuttgart-cmcs/dolfinx-graph-ad/blob/main/demos/linear_elasticity.ipynb

      3D cantilever beam under gravity. Computes gradients of the deformation
      energy w.r.t. the Lame parameters :math:`\lambda` and :math:`\mu`.

   .. grid-item-card:: :octicon:`milestone` Stokes Flow
      :link: https://github.com/unistuttgart-cmcs/dolfinx-graph-ad/blob/main/demos/stokes.ipynb

      Stokes flow around a cylindrical obstacle using P2-P1 Taylor-Hood
      elements. Computes gradients of viscous dissipation w.r.t. the viscosity
      and the obstacle boundary condition.
