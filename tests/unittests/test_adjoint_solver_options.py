from mpi4py import MPI
from petsc4py import PETSc

from dolfinx_adjoint.fem.petsc import _create_adjoint_solver


def test_adjoint_solver_is_configured_by_the_given_options():
    """The caller's options configure the adjoint solver, and nothing else does.

    ``pc_type`` is requested as a factorisation so that the factor solver type is
    observable at all; it is not asserted on itself.
    """
    A = PETSc.Mat().createAIJ((2, 2), comm=MPI.COMM_SELF)
    A.setUp()
    A.assemble()

    with _create_adjoint_solver(
        A,
        petsc_options={"ksp_type": "cg", "pc_type": "lu"},
        petsc_options_prefix="test_adjoint_solver_options_",
    ) as solver:
        # Catches a solver that ignores the given options and keeps its own type.
        assert solver.getType() == "cg"
        # Catches a hardcoded factorisation surviving next to the given options,
        # whether they are ignored outright or merged on top of the defaults.
        assert solver.getPC().getFactorSolverType() != "mumps"
