"""Record each linear or nonlinear solve at its own dependencies and values."""

import inspect

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem as LinearProblemBase
from dolfinx.fem.petsc import NonlinearProblem as NonlinearProblemBase
from dolfinx.fem.petsc import assign, set_bc
from petsc4py import PETSc

import dolfinx_graph_ad.graph as graph
from dolfinx_graph_ad.fem import _calculus as ad
from dolfinx_graph_ad.fem._recording import bind_arguments
from dolfinx_graph_ad.fem.bcs import DirichletBCNode
from dolfinx_graph_ad.fem.function import FunctionNode
from dolfinx_graph_ad.graph.dolfinx_helpers import resolve_capture, scalar, zeros


class LinearProblem(LinearProblemBase):
    """OVERLOADS: :py:class:`dolfinx.fem.petsc.LinearProblem`.
    Linear problem class for solving the linear problem

    The overloaded class modifies the initialization of the LinearProblem to keep track of the dependencies
    in the computational graph and the adjoint equations. The original functionality is kept.

    """

    def __init__(self, *args, **kwargs):
        """OVERLOADS: :py:class:`dolfinx.fem.petsc.LinearProblem`.
        Initialize solver for solving a linear problem

        Args:
            args: Arguments to :py:class:`dolfinx.fem.petsc.LinearProblem`.
            kwargs: Keyword arguments to :py:class:`dolfinx.fem.petsc.LinearProblem`.
            graph: An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.
            adjoint_petsc_options: An additional keyword argument with the PETSc options
                configuring the solver of the adjoint equations of the problem.
            adjoint_petsc_options_prefix: An additional keyword argument with the
                options prefix of that solver. Defaults to the options prefix of the
                problem followed by ``adjoint_``.

        """
        _graph = kwargs.pop("graph", None)
        options = {
            "adjoint_petsc_options": kwargs.pop("adjoint_petsc_options", None),
            "adjoint_petsc_options_prefix": kwargs.pop(
                "adjoint_petsc_options_prefix", None
            ),
        }
        super().__init__(*args, **kwargs)
        if _graph is None:
            return
        arguments = bind_arguments(LinearProblemBase.__init__, self, *args, **kwargs)
        del arguments["self"]
        forms = (arguments.pop("a"), arguments.pop("L"))
        if not isinstance(forms[0], ufl.Form):
            raise NotImplementedError(
                "A blocked problem cannot be recorded. Without the graph, the "
                "problem behaves exactly like the one of DOLFINx."
            )
        # Record the native output Function, including one allocated by DOLFINx.
        arguments["u"] = self.u
        problem_node = LinearProblemNode(self, arguments, forms, options)
        _graph.add_node(problem_node)

    def solve(self, *args, **kwargs):
        """OVERLOADS: :py:meth:`dolfinx.fem.petsc.LinearProblem.solve`
        Solve the linear problem into the function u, which is returned.

        Args:
            args: Arguments to :py:meth:`dolfinx.fem.petsc.LinearProblem.solve`
            kwargs: Keyword arguments to :py:meth:`dolfinx.fem.petsc.LinearProblem.solve`
            graph (graph, optional): An additional keyword argument to specify whether the
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.
            version (int, optional): The version of the solution in the graph. Defaults
                to the version following its latest recorded one, or 0 if unrecorded.

        Returns:
            fem.Function: The solution function u after solving the linear problem.

        Raises:
            ValueError: If the problem was not recorded in the graph by its constructor,
                since the solve then has no residual to differentiate.
        """

        _graph = kwargs.pop("graph", None)
        version = kwargs.pop("version", None)
        if _graph is None:
            return super().solve(*args, **kwargs)
        problem_node: ProblemNode = _graph.get_node(self)
        if problem_node is None:
            raise ValueError(
                f"{type(self).__name__} is not recorded in this graph, so its solve has no residual to differentiate. Pass the graph to its constructor."
            )
        if problem_node._residual is not None:
            # Already prepared for a solve: retain its derivative data and record
            # this solve as a new version of the same native problem.
            problem_node = LinearProblemNode(
                problem_node.object,
                problem_node._recorded_kwargs,
                problem_node._forms,
                problem_node._options,
                version=problem_node.version + 1,
            )
            _graph.add_node(problem_node)
        solve_node = problem_node._prepare_solve(_graph, version)
        solution = super().solve(*args, **kwargs)
        _graph.save(solve_node)
        problem_node._solve_node = _graph.capture(solve_node)
        return solution


class NonlinearProblem(NonlinearProblemBase):
    """OVERLOADS: :py:class:`dolfinx.fem.petsc.NonlinearProblem`.
    Nonlinear problem class for solving the non-linear problem

    The overloaded class modifies the initialization of the NonlinearProblem to keep track of the dependencies
    in the computational graph and the adjoint equations. The original functionality is kept.

    """

    def __init__(self, *args, **kwargs):
        """OVERLOADS: :py:class:`dolfinx.fem.petsc.NonlinearProblem`.
        Initialize solver for solving a non-linear problem using Newton's method

        Args:
            args: Arguments to :py:class:`dolfinx.fem.petsc.NonlinearProblem`.
            kwargs: Keyword arguments to :py:class:`dolfinx.fem.petsc.NonlinearProblem`.
            graph: An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.
            adjoint_petsc_options: An additional keyword argument with the PETSc options
                configuring the solver of the adjoint equations of the problem.
            adjoint_petsc_options_prefix: An additional keyword argument with the
                options prefix of that solver. Defaults to the options prefix of the
                problem followed by ``adjoint_``.

        """

        _graph = kwargs.pop("graph", None)
        options = {
            "adjoint_petsc_options": kwargs.pop("adjoint_petsc_options", None),
            "adjoint_petsc_options_prefix": kwargs.pop(
                "adjoint_petsc_options_prefix", None
            ),
        }
        super().__init__(*args, **kwargs)
        if _graph is None:
            return
        arguments = bind_arguments(NonlinearProblemBase.__init__, self, *args, **kwargs)
        del arguments["self"]
        F = arguments.pop("F")
        if not isinstance(F, ufl.Form):
            raise NotImplementedError(
                "A blocked problem cannot be recorded. Without the graph, the "
                "problem behaves exactly like the one of DOLFINx."
            )
        problem_node = NonlinearProblemNode(self, arguments, F, options)
        _graph.add_node(problem_node)

    def solve(self, *args, **kwargs):
        """OVERLOADS: :py:meth:`dolfinx.fem.petsc.NonlinearProblem.solve`

        Solve the non-linear problem into the function u, which is returned.

        Args:
            args: Arguments to :py:meth:`dolfinx.fem.petsc.NonlinearProblem.solve`
            kwargs: Keyword arguments to :py:meth:`dolfinx.fem.petsc.NonlinearProblem.solve`
            graph (graph, optional): An additional keyword argument to specify whether the
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.
            version (int, optional): The version of the solution in the graph. Defaults
                to the version following its latest recorded one, or 0 if unrecorded.

        Returns:
            fem.Function: The solution function u after solving the nonlinear problem.

        Raises:
            ValueError: If the problem was not recorded in the graph by its constructor,
                since the solve then has no residual to differentiate.
        """

        _graph = kwargs.pop("graph", None)
        version = kwargs.pop("version", None)
        if _graph is None:
            return super().solve(*args, **kwargs)
        problem_node: ProblemNode = _graph.get_node(self)
        if problem_node is None:
            raise ValueError(
                f"{type(self).__name__} is not recorded in this graph, so its solve has no residual to differentiate. Pass the graph to its constructor."
            )
        if problem_node._solve_node is not None:
            # Already prepared for a solve: retain its derivative data and record
            # this solve as a new version of the same native problem.
            problem_node = NonlinearProblemNode(
                problem_node.object,
                problem_node._recorded_kwargs,
                problem_node._forms,
                problem_node._options,
                version=problem_node.version + 1,
            )
            _graph.add_node(problem_node)
        solve_node = problem_node._prepare_solve(_graph, version)
        solution = super().solve(*args, **kwargs)
        _graph.save(solve_node)
        problem_node._solve_node = _graph.capture(solve_node)
        return solution


class ProblemNode(graph.AbstractNode):
    """A recorded problem version with its own dependencies and captured values.

    Attributes:
        residual (ufl.Form): The residual, built on first access during preparation.
        values (dict): Captured input versions, or None before preparation.
        adjoint_form_kwargs (dict): Arguments used to compile derivative forms.
    """

    def __init__(self, object, arguments, forms, options, **kwargs):
        """Create a problem version from the recorded constructor inputs.

        Args:
            object: The forward problem whose solve versions are recorded.
            arguments: The arguments of the recorded DOLFINx constructor.
            forms: Symbolic forms defining the forward problem.
            options: The ``adjoint_petsc_options`` and ``adjoint_petsc_options_prefix``
                the constructor was given for the derivative equations.
            **kwargs: Arguments of :py:class:`dolfinx_graph_ad.graph.AbstractNode`.
        """

        super().__init__(object, **kwargs)
        self._recorded_kwargs = arguments
        self._recorded_state = arguments["u"]
        self._recorded_bcs = arguments.get("bcs")
        self._forms = forms
        self._options = options
        self._residual = None
        self._previous_states = None
        self.values = None
        self._solve_node = None
        form_parameters = inspect.signature(fem.form).parameters
        self.adjoint_form_kwargs = {
            key: value for key, value in arguments.items() if key in form_parameters
        }
        if options["adjoint_petsc_options_prefix"] is None:
            prefix = arguments["petsc_options_prefix"]
            options["adjoint_petsc_options_prefix"] = f"{prefix}adjoint_"

    def _record_input_edges(self, _graph, previous_states):
        """Record the edges into this problem version from the latest versions of the
        recorded values its residual depends on.

        Each boundary condition is bound to its current value version before
        recording its edge into the problem.
        """
        u = self._recorded_state

        bcs = self._recorded_bcs

        residual = self.residual

        boundary_dofs = {}
        # Native lifting and set_bc apply conditions in order: later values win.
        # Include untracked conditions; adjoints consume owned entries.
        covered = np.empty(0, dtype=np.int32)
        for bc in reversed(bcs or ()):
            dofs, owned = bc.dof_indices()
            boundary_dofs.setdefault(id(bc), np.setdiff1d(dofs[:owned], covered))
            covered = np.union1d(covered, dofs[:owned])

        for coefficient in residual.coefficients():
            if coefficient == u:
                continue
            # A stand-in depends on the version of the solution the solve starts from,
            # which is the latest one before the solve is recorded.
            coefficient_node = _graph.get_node(
                previous_states.get(coefficient, coefficient)
            )
            if coefficient_node is not None:
                coefficient_edge = Problem_Coefficient_Edge(
                    coefficient_node, self, ctx=coefficient
                )
                _graph.add_edge(coefficient_edge)
                self.grad_fns.append(coefficient_edge)
        for constant in residual.constants():
            constant_node = _graph.get_node(constant)
            if constant_node is not None:
                function = ad.real_function(constant)
                constant_edge = Problem_Constant_Edge(
                    constant_node, self, ctx=[constant, function]
                )
                _graph.add_edge(constant_edge)
                self.grad_fns.append(constant_edge)
        for bc in bcs or ():
            # A recorded condition is bound to the value version this solve reads.
            bc_node = _graph.get_node(bc)
            if isinstance(bc_node, DirichletBCNode):
                bc_node = bc_node.bind(_graph)
            if bc_node is not None:
                bc_edge = Problem_Boundary_Edge(
                    bc_node, self, ctx=boundary_dofs[id(bc)]
                )
                _graph.add_edge(bc_edge)
                self.grad_fns.append(bc_edge)

    def _prepare_solve(self, _graph, version):
        """Record input edges, capture input versions and create the exact output node.

        Return this problem version's output node. The native solve then writes
        the output, which is saved before its version is captured for derivatives.
        """

        # Build collectively on the mesh ranks only when recording a solve.
        residual = self.residual
        previous_states = self._previous_states
        self._record_input_edges(_graph, previous_states)

        # Capture the old state before adding the version the solve will write.
        state = self._recorded_state
        previous = _graph.get_node(state)
        previous_values = {}
        if previous_states:
            initial = (
                _graph.capture(previous)
                if previous is not None
                else fem.Function.copy(state)
            )
            previous_values = {stand_in: initial for stand_in in previous_states}

        if version is None:
            version = 0 if previous is None else previous.version + 1
        solve_node = SolveNode(state, version=version, name=state.name)
        _graph.add_node(solve_node)
        function_edge = graph.Edge(self, solve_node)
        solve_node.grad_fns = [function_edge]
        _graph.add_edge(function_edge)
        self.capture_inputs(_graph, previous_values)
        return solve_node

    def capture_inputs(self, _graph, previous_values):
        """Capture controls and boundary data before the native solve.

        Args:
            _graph (Graph): The graph the solve is recorded in.
            previous_values: Captured pre-solve values keyed by stand-in coefficients.
        """

        inputs = [
            *self.residual.coefficients(),
            *self.residual.constants(),
            # A recorded condition's value as passed: DOLFINx 0.11 returns its C++
            # object from ``bc.g``. An untracked condition stays constant.
            *(
                node.arguments["value"]
                for bc in self._recorded_bcs or ()
                if (node := _graph.get_node(bc)) is not None
            ),
        ]
        self.values = {
            value: (
                previous_values[value]
                if value in previous_values
                else _graph.capture(value)
            )
            for value in inputs
            if value is not self._recorded_state
        }

    def solve_adjoint(self, seed):
        """Solve (∂F/∂u)ᵀ λ = −seed at this forward solve's saved values.

        Each control edge creates its own adjoint problem. Construction and release
        are collective in the same edge order on every mesh rank; solution ghosts
        are updated before the edge assembles its sensitivity.

        Args:
            seed (PETSc.Vec): The adjoint value of the solution. It is not modified.

        Returns:
            tuple: The residual at saved values, adjoint solution and saved value mapping.

        Raises:
            RuntimeError: If the solve has not been performed, so that no values were
                copied.
        """

        if self.values is None:
            raise RuntimeError(
                "The adjoint of a solve that has not been performed has no values "
                "to be evaluated at."
            )
        values = {
            coefficient: resolve_capture(captured)
            for coefficient, captured in self.values.items()
        }
        if self._solve_node is not None:
            values[self._recorded_state] = resolve_capture(self._solve_node)
        residual = ufl.replace(self.residual, values)
        solution = fem.Function(
            self._recorded_state.function_space,
            dtype=self._recorded_state.dtype,
            name="adjoint",
        )
        # The adjoint Jacobian (∂F/∂u)ᵀ of the residual.
        state = values.get(self._recorded_state, self._recorded_state)
        jacobian = ufl.adjoint(
            ufl.derivative(residual, state, ufl.TrialFunction(state.function_space))
        )
        problem = AdjointProblem(
            jacobian,
            seed,
            solution,
            bcs=self._recorded_bcs,
            petsc_options=self._options["adjoint_petsc_options"],
            petsc_options_prefix=self._options["adjoint_petsc_options_prefix"],
            **self.adjoint_form_kwargs,
        )
        try:
            # AdjointProblem owns a copy of the seed, so the incoming seed is unchanged.
            problem.b.scale(-1.0)
            problem.solve()
        finally:
            # DOLFINx destroys the PETSc objects here, collectively on the mesh ranks.
            del problem
        return residual, solution, values

    def release(self):
        """Release the problem, its recording metadata and its adjoint."""

        super().release()
        self._residual = None
        self._previous_states = None
        self.values = None
        self._solve_node = None
        self.adjoint_form_kwargs = None
        self._recorded_kwargs = None
        self._options = None
        self._recorded_state = None
        self._recorded_bcs = None
        self._forms = None


class LinearProblemNode(ProblemNode):
    """A recorded version of :py:class:`dolfinx.fem.petsc.LinearProblem`."""

    @property
    def residual(self):
        """The residual F(u) = a(u) - L."""
        if self._residual is None:
            u = self._recorded_state
            a, L = self._forms
            stand_ins = {}
            if u in a.coefficients() or u in L.coefficients():
                stand_ins[u] = fem.Function(
                    u.function_space, dtype=u.dtype, name=f"{u.name}_previous"
                )
            if stand_ins:
                a, L = ufl.replace(a, stand_ins), ufl.replace(L, stand_ins)
            self._residual = ufl.replace(a - L, {a.arguments()[1]: u})
            self._previous_states = {
                stand_in: state for state, stand_in in stand_ins.items()
            }
        return self._residual


class NonlinearProblemNode(ProblemNode):
    """A recorded nonlinear problem version."""

    @property
    def residual(self):
        """The supplied residual F(u) = 0, with no previous-state stand-ins."""
        if self._residual is None:
            self._residual = self._forms
            self._previous_states = {}
        return self._residual


class SolveNode(FunctionNode):
    """The saved solution of one recorded solve.

    Its incoming edge identifies the problem version that produced the solution.
    """


class Problem_Coefficient_Edge(graph.Edge):
    """Sensitivity of a solve to a recorded coefficient."""

    def calculate_adjoint(self, value):
        """Return the coefficient derivative of the pairing λᵀF.

        Args:
            value: The accumulated solution seed.

        Returns:
            PETSc.Vec: The gradient, with valid owned entries.
        """

        m = self.ctx
        residual, adjoint, values = self.successor.solve_adjoint(value)
        pairing = ufl.action(residual, adjoint)
        derivative = ufl.derivative(pairing, values[m])
        gradient = ad.assemble_vector(derivative, self.successor.adjoint_form_kwargs)
        return zeros(m.x.petsc_vec) if gradient is None else gradient


class Problem_Constant_Edge(graph.Edge):
    """Sensitivity of a solve to a recorded Constant."""

    def calculate_adjoint(self, value):
        """Differentiate λᵀF through a real-element stand-in for the Constant.

        Args:
            value: The accumulated solution seed.

        Returns:
            float or complex or PETSc.Vec: The scalar or component gradient.
        """

        constant, function = self.ctx
        residual, adjoint, values = self.successor.solve_adjoint(value)
        # The constant is differentiated as a Function on a real element.
        pairing = ufl.action(residual, adjoint)
        derivative = ad.ufl_derivative_constant(
            pairing, values[constant], stand_in_function=function
        )
        gradient = ad.assemble_vector(derivative, self.successor.adjoint_form_kwargs)
        if gradient is None:
            gradient = zeros(function.x.petsc_vec)
        return scalar(constant, gradient)


class Problem_Boundary_Edge(graph.Edge):
    """Sensitivity on the owned dofs where this boundary condition survives."""

    def calculate_adjoint(self, value):
        """Return the lifting sensitivity plus the direct constrained-state seed.

        Differentiate λᵀF with respect to the solved state, then add the seed because
        u = g on constrained dofs. Restrict to owned dofs surviving later conditions;
        the boundary-condition edge maps these entries back to its control.

        Args:
            value: The accumulated solution seed, left unchanged.

        Returns:
            PETSc.Vec: The state-layout gradient restricted to surviving owned dofs.
        """

        dofs = self.ctx
        residual, adjoint, values = self.successor.solve_adjoint(value)
        state = values[self.successor._recorded_state]
        pairing = ufl.action(residual, adjoint)
        derivative = ufl.derivative(pairing, state)
        gradient = ad.assemble_vector(derivative, self.successor.adjoint_form_kwargs)
        if gradient is None:
            gradient = zeros(value)
        gradient.axpy(1.0, value)
        values = gradient.array_r[dofs].copy()
        gradient.array_w[:] = 0.0
        gradient.array_w[dofs] = values
        return gradient


class AdjointProblem(LinearProblemBase):
    """OVERLOADS: :py:class:`dolfinx.fem.petsc.LinearProblem`.

    Solve an adjoint bilinear form against a vector with homogeneous Dirichlet
    conditions. DOLFINx owns and destroys the matrix, vectors, and KSP.
    """

    def __init__(
        self,
        a: ufl.Form,
        b: PETSc.Vec,
        u: fem.Function,
        bcs=None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        **kwargs,
    ):
        """Initialize the adjoint problem.

        Args:
            a: The bilinear form of the adjoint equation.
            b: The right-hand side vector. Its owned entries are copied.
            u: The solution function.
            bcs: Boundary conditions whose constrained entries are set to zero.
            petsc_options: Options configuring the adjoint KSP.
            petsc_options_prefix: Solver options prefix, defaulting to
                ``dolfinx_graph_ad_``.
            kwargs: The arguments of :py:class:`dolfinx.fem.petsc.LinearProblem` the
                forms of the adjoint equation are compiled with, which are the ones the
                problem it belongs to was set up with.

        Note:
            The solver of the adjoint equation is configured through ``adjoint_petsc_options`` instead.

        """

        if petsc_options_prefix is None:
            petsc_options_prefix = "dolfinx_graph_ad_"

        # The zero form supplies the vector layout without an unused RHS kernel.
        super().__init__(
            a,
            ufl.ZeroBaseForm((a.arguments()[0],)),
            u=u,
            bcs=bcs,
            petsc_options_prefix=petsc_options_prefix,
            **kwargs,
        )
        b.copy(self.b)

        options = PETSc.Options(petsc_options_prefix)
        petsc_options = {} if petsc_options is None else petsc_options
        try:
            for key, value in petsc_options.items():
                options[key] = value
            self.solver.setFromOptions()
        finally:
            for key in petsc_options:
                del options[key]

    def solve(self) -> fem.Function:
        """Solve using the supplied vector, without RHS assembly or lifting.

        Returns:
            The solution function with updated ghost entries.
        """

        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.a, bcs=self.bcs)
        self.A.assemble()
        set_bc(self.b, self.bcs, alpha=0.0)
        self.solver.solve(self.b, self.x)
        self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        assign(self.x, self.u)
        return self.u
