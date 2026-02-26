# DOLFINx-ADJOINT

[![CI](https://github.com/unistuttgart-cmcs/dolfinx-adjoint/actions/workflows/ci.yml/badge.svg)](https://github.com/unistuttgart-cmcs/dolfinx-adjoint/actions/workflows/ci.yml)
[![Docs](https://github.com/unistuttgart-cmcs/dolfinx-adjoint/actions/workflows/docs-publish.yml/badge.svg)](https://unistuttgart-cmcs.github.io/dolfinx-adjoint/)
![DOLFINx >= 0.10.0](https://img.shields.io/badge/DOLFINx-%3E%3D0.10.0-orange)

Automatic differentiation for [DOLFINx](https://github.com/FEniCS/dolfinx) using the adjoint method.
Efficient sensitivity analysis and gradient-based optimization for finite element simulations.

## Documentation

Full documentation is available at [unistuttgart-cmcs.github.io/dolfinx-adjoint](https://unistuttgart-cmcs.github.io/dolfinx-adjoint/).

## Installation

```bash
git clone https://github.com/unistuttgart-cmcs/dolfinx-adjoint.git
cd dolfinx-adjoint
pip install -e .
```

Requires Python >= 3.12 and [DOLFINx](https://github.com/FEniCS/dolfinx) >= 0.10.0. For demos and tests, use `pip install -e ".[all]"`.

## Quick Example

Compute dJ/df for a Poisson problem:

```python
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
```

## Demos

The `demos/` directory contains comprehensive Jupyter notebook examples demonstrating various PDE problems:

- **[poisson.ipynb](demos/poisson.ipynb)** - Poisson equation with gradient computation w.r.t. source term, diffusion coefficient, and boundary conditions
- **[heat_equation.ipynb](demos/heat_equation.ipynb)** - Time-dependent heat equation with sensitivity analysis
- **[linear_elasticity.ipynb](demos/linear_elasticity.ipynb)** - Linear elasticity problem with parameter sensitivities
- **[stokes.ipynb](demos/stokes.ipynb)** - Stokes flow with obstacle, computing gradients w.r.t. viscosity and boundary conditions

## Acknowledgments

This library builds upon the excellent work of the [FEniCS Project](https://fenicsproject.org/) and uses [DOLFINx](https://github.com/FEniCS/dolfinx) as its foundation.
