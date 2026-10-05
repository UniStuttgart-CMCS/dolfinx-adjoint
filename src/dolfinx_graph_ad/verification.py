from typing import Callable

import numpy as np
from dolfinx import fem


def _perturbed(
    control: fem.Function | fem.Constant,
    direction: fem.Function | float | np.ndarray,
    step_size: float,
) -> fem.Function | fem.Constant:
    """Move a control by a step in a direction, returning a copy of it."""
    if isinstance(control, fem.Function):
        copy = fem.Function.copy(control)
        copy.x.petsc_vec.axpy(step_size, direction.x.petsc_vec)
        copy.x.scatter_forward()
        return copy
    return fem.Constant(
        control.ufl_domain(),
        control.value + step_size * np.asarray(direction, dtype=control.dtype),
    )


def _remainders(
    evaluate: Callable,
    value: float,
    derivative: float,
    step_sizes: list,
) -> tuple:
    """The Taylor remainders with orders 0 and 1, at every step size."""
    remainders = ([], [])
    for step_size in step_sizes:
        difference = evaluate(step_size) - value
        remainders[0].append(abs(difference))
        remainders[1].append(abs(difference - step_size * derivative))
    return remainders


def _rates(residuals: list, step_sizes: list) -> list:
    """The convergence rates, the log-slopes of the residuals between adjacent steps."""
    return [
        (
            float(np.log(coarse / fine) / np.log(coarse_step / fine_step))
            if coarse > 0 and fine > 0
            else float("nan")
        )
        for coarse, fine, coarse_step, fine_step in zip(
            residuals[:-1], residuals[1:], step_sizes[:-1], step_sizes[1:]
        )
    ]
