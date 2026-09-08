from typing import Any

import ufl
from dolfinx import fem, la
from dolfinx.fem.petsc import LinearProblem as LinearProblemBase
from dolfinx.fem.petsc import (
    NewtonSolverNonlinearProblem as NewtonSolverNonlinearProblemBase,
)
from dolfinx.fem.petsc import NonlinearProblem as NonlinearProblemBase
from dolfinx.fem.petsc import assign, create_vector, set_bc
from petsc4py import PETSc

import dolfinx_adjoint.graph as graph


class LinearProblem(LinearProblemBase):
    """OVERLOADS: :py:class:`dolfinx.fem.petsc.LinearProblem`.
    Linear problem class for solving the linear problem

    The overloaded class modifies the initialization of the LinearProblem to keep track of the dependencies
    in the computational graph and the adjoint equations. The original functionality is kept.

    """

    def __init__(self, *args, **kwargs):
        """OVERLOADS: :py:func:`dolfinx.fem.petsc.LinearProblem.__init__`.
        Initialize solver for solving a linear problem

        Args:
            args: Arguments to :py:func:`dolfinx.fem.petsc.LinearProblem.__init__`.
            kwargs: Keyword arguments to :py:func:`dolfinx.fem.petsc.LinearProblem.__init__`.
            graph: An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.

        """
        if not "graph" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)

            a = args[0]
            L = args[1]
            F_form = a - L

            u = kwargs.get("u")
            if u == None:
                raise ValueError(
                    "The solution function u needs to be provided as a keyword argument for the LinearProblem when using the graph functionalities."
                )

            problem_node = LinearProblemNode(self, a, L, **kwargs)
            _graph.add_node(problem_node)

            u_node = _graph.get_node(id(u))

            # Replace Trial Function in the form with the solution function to be able to track the dependencies of the solution function on the coefficients and constants in the form.
            # By definition, the trial function is always the second argument in the form, thus F_form.arguments()[1] is used to identify the trial function.
            F_form = ufl.replace(F_form, {F_form.arguments()[1]: u})

            # Creating and adding edges to the graph if the coefficients are in the graph
            for coefficient in F_form.coefficients():
                if coefficient == u:
                    continue
                coefficient_node = _graph.get_node(id(coefficient))
                if not coefficient_node == None:
                    ctx = [F_form, u_node, coefficient, kwargs.get("bcs"), _graph]
                    coefficient_edge = NonlinearProblem_Coefficient_Edge(
                        coefficient_node, problem_node, ctx=ctx
                    )
                    _graph.add_edge(coefficient_edge)
                    problem_node.append_gradFuncs(coefficient_edge)
                    coefficient_edge.set_next_functions(
                        coefficient_node.get_gradFuncs()
                    )

            # Creating and adding edges to the graph if the constants are in the graph
            for constant in F_form.constants():
                constant_node = _graph.get_node(id(constant))
                if not constant_node == None:
                    ctx = [F_form, u_node, constant, kwargs.get("bcs")]
                    constant_edge = NonlinearProblem_Constant_Edge(
                        constant_node, problem_node, ctx=ctx
                    )
                    _graph.add_edge(constant_edge)
                    problem_node.append_gradFuncs(constant_edge)
                    constant_edge.set_next_functions(constant_node.get_gradFuncs())

            # Creating and adding edges to the graph if the boundary conditions are in the graph
            if "bcs" in kwargs.keys() and not kwargs.get("bcs") == None:
                for bc in kwargs.get("bcs"):
                    bc_node = _graph.get_node(id(bc))
                    if not bc_node == None:
                        # For linear problems, dF/dbc is represented by the bilinear form a.
                        ctx = [F_form, u_node, kwargs.get("bcs"), self._a]
                        bc_edge = NonlinearProblem_Boundary_Edge(
                            bc_node, problem_node, ctx=ctx
                        )
                        _graph.add_edge(bc_edge)
                        problem_node.append_gradFuncs(bc_edge)
                        bc_edge.set_next_functions(bc_node.get_gradFuncs())

    def solve(self, *args, **kwargs):
        """OVERLOADS: :py:func:`dolfinx.fem.petsc.LinearProblem.solve`
        Solve linear problem into function u. Returns the number of iterations and if the solver converged.

        Args:
            args: Arguments to :py:func:`dolfinx.fem.petsc.LinearProblem.solve`
            kwargs: Keyword arguments to :py:func:`dolfinx.fem.petsc.LinearProblem.solve`
            graph (graph, optional): An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.

        Returns:
            fem.Function: The solution function u after solving the linear problem.

        """
        # Add the edge from the LinearProblem to the Function
        if "graph" not in kwargs:
            output = super().solve(*args, **kwargs)
        else:
            if "version" in kwargs:
                version = kwargs["version"]
                del kwargs["version"]
            else:
                version = 1
            _graph = kwargs["graph"]
            del kwargs["graph"]

            problem_node = _graph.get_node(id(self))
            solve_node = SolveNode(
                self._u, problem_node, version=version, name=self._u.name
            )
            _graph.add_node(solve_node)

            # Creating and adding the edge to the graph
            if not problem_node == None:
                function_edge = graph.Edge(problem_node, solve_node)
                solve_node.set_gradFuncs([function_edge])
                _graph.add_edge(function_edge)
                function_edge.set_next_functions(problem_node.get_gradFuncs())

            output = super().solve(*args, **kwargs)

        return output


class NonlinearProblem(NonlinearProblemBase):
    """OVERLOADS: :py:class:`dolfinx.fem.petsc.NonlinearProblem`.
    Nonlinear problem class for solving the non-linear problem

    The overloaded class modifies the initialization of the NonlinearProblem to keep track of the dependencies
    in the computational graph and the adjoint equations. The original functionality is kept.

    """

    def __init__(self, *args, **kwargs):
        """OVERLOADS: :py:func:`dolfinx.fem.petsc.NewtonSolverNonlinearProblem.__init__`.
        Initialize solver for solving a non-linear problem using Newton's method

        Args:
            args: Arguments to :py:func:`dolfinx.fem.petsc.NewtonSolverNonlinearProblem.__init__`.
            kwargs: Keyword arguments to :py:func:`dolfinx.fem.petsc.NewtonSolverNonlinearProblem.__init__`.
            graph: An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.

        """
        if not "graph" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)

            F_form = args[0]
            u = args[1]

            problem_node = NonlinearProblemNode(self, F_form, u, **kwargs)
            _graph.add_node(problem_node)

            u_node = _graph.get_node(id(u))

            # Creating and adding edges to the graph if the coefficients are in the graph
            for coefficient in F_form.coefficients():
                if coefficient == u:
                    continue
                coefficient_node = _graph.get_node(id(coefficient))
                if not coefficient_node == None:
                    ctx = [F_form, u_node, coefficient, kwargs.get("bcs"), _graph]
                    coefficient_edge = NonlinearProblem_Coefficient_Edge(
                        coefficient_node, problem_node, ctx=ctx
                    )
                    _graph.add_edge(coefficient_edge)
                    problem_node.append_gradFuncs(coefficient_edge)
                    coefficient_edge.set_next_functions(
                        coefficient_node.get_gradFuncs()
                    )

            # Creating and adding edges to the graph if the constants are in the graph
            for constant in F_form.constants():
                constant_node = _graph.get_node(id(constant))
                if not constant_node == None:
                    ctx = [F_form, u_node, constant, kwargs.get("bcs")]
                    constant_edge = NonlinearProblem_Constant_Edge(
                        constant_node, problem_node, ctx=ctx
                    )
                    _graph.add_edge(constant_edge)
                    problem_node.append_gradFuncs(constant_edge)
                    constant_edge.set_next_functions(constant_node.get_gradFuncs())

            # Creating and adding edges to the graph if the boundary conditions are in the graph
            if "bcs" in kwargs.keys() and not kwargs.get("bcs") == None:
                for bc in kwargs.get("bcs"):
                    bc_node = _graph.get_node(id(bc))
                    if not bc_node == None:
                        ctx = [F_form, u_node, kwargs.get("bcs"), self._J]
                        bc_edge = NonlinearProblem_Boundary_Edge(
                            bc_node, problem_node, ctx=ctx
                        )
                        _graph.add_edge(bc_edge)
                        problem_node.append_gradFuncs(bc_edge)
                        bc_edge.set_next_functions(bc_node.get_gradFuncs())

    def solve(self, *args, **kwargs):
        """OVERLOADS: :py:func:`dolfinx.fem.petsc.NonlinearProblem.solve`
        Solve non-linear problem into function u. Returns the number of iterations and if the solver converged.

        Args:
            args: Arguments to :py:func:`dolfinx.fem.petsc.NonlinearProblem.solve`
            kwargs: Keyword arguments to :py:func:`dolfinx.fem.petsc.NonlinearProblem.solve`
            graph (graph, optional): An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.

        Returns:
            fem.Function: The solution function u after solving the nonlinear problem.

        """
        # Add the edge from the NonlinearProblem to the Function
        if "graph" not in kwargs:
            output = super().solve(*args, **kwargs)
        else:
            if "version" in kwargs:
                version = kwargs["version"]
                del kwargs["version"]
            else:
                version = 1
            _graph = kwargs["graph"]
            del kwargs["graph"]

            problem_node = _graph.get_node(id(self))
            solve_node = SolveNode(
                self._u, problem_node, version=version, name=self._u.name
            )
            _graph.add_node(solve_node)

            # Creating and adding the edge to the graph
            if not problem_node == None:
                function_edge = graph.Edge(problem_node, solve_node)
                solve_node.set_gradFuncs([function_edge])
                _graph.add_edge(function_edge)
                function_edge.set_next_functions(problem_node.get_gradFuncs())

            output = super().solve(*args, **kwargs)

        return output


class NewtonSolverNonlinearProblem(NewtonSolverNonlinearProblemBase):
    """OVERLOADS: :py:class:`dolfinx.fem.petsc.NewtonSolverNonlinearProblem`.
    Nonlinear problem class for solving the non-linear problem

    The overloaded class modifies the initialization of the NewtonSolverNonlinearProblem to keep track of the dependencies
    in the computational graph and the adjoint equations. The original functionality is kept.

    """

    def __init__(self, *args, **kwargs):
        """OVERLOADS: :py:func:`dolfinx.fem.petsc.NewtonSolverNonlinearProblem.__init__`.
        Initialize solver for solving a non-linear problem using Newton's method

        Args:
            args: Arguments to :py:func:`dolfinx.fem.petsc.NewtonSolverNonlinearProblem.__init__`.
            kwargs: Keyword arguments to :py:func:`dolfinx.fem.petsc.NewtonSolverNonlinearProblem.__init__`.
            graph: An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.

        """
        if not "graph" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            _graph = kwargs["graph"]
            del kwargs["graph"]
            super().__init__(*args, **kwargs)

            F_form = args[0]
            u = args[1]

            problem_node = NewtonSolverNonlinearProblemNode(self, F_form, u, **kwargs)
            _graph.add_node(problem_node)

            u_node = _graph.get_node(id(u))

            # Creating and adding edges to the graph if the coefficients are in the graph
            for coefficient in F_form.coefficients():
                if coefficient == u:
                    continue
                coefficient_node = _graph.get_node(id(coefficient))
                if not coefficient_node == None:
                    ctx = [F_form, u_node, coefficient, kwargs.get("bcs"), _graph]
                    coefficient_edge = NonlinearProblem_Coefficient_Edge(
                        coefficient_node, problem_node, ctx=ctx
                    )
                    _graph.add_edge(coefficient_edge)
                    problem_node.append_gradFuncs(coefficient_edge)
                    coefficient_edge.set_next_functions(
                        coefficient_node.get_gradFuncs()
                    )

            # Creating and adding edges to the graph if the constants are in the graph
            for constant in F_form.constants():
                constant_node = _graph.get_node(id(constant))
                if not constant_node == None:
                    ctx = [F_form, u_node, constant, kwargs.get("bcs")]
                    constant_edge = NonlinearProblem_Constant_Edge(
                        constant_node, problem_node, ctx=ctx
                    )
                    _graph.add_edge(constant_edge)
                    problem_node.append_gradFuncs(constant_edge)
                    constant_edge.set_next_functions(constant_node.get_gradFuncs())

            # Creating and adding edges to the graph if the boundary conditions are in the graph
            if "bcs" in kwargs.keys() and not kwargs.get("bcs") == None:
                for bc in kwargs.get("bcs"):
                    bc_node = _graph.get_node(id(bc))
                    if not bc_node == None:
                        ctx = [F_form, u_node, kwargs.get("bcs"), self._a]
                        bc_edge = NonlinearProblem_Boundary_Edge(
                            bc_node, problem_node, ctx=ctx
                        )
                        _graph.add_edge(bc_edge)
                        problem_node.append_gradFuncs(bc_edge)
                        bc_edge.set_next_functions(bc_node.get_gradFuncs())


class LinearProblemNode(graph.AbstractNode):
    """
    Node for the initialization of :py:class:`dolfinx.fem.petsc.LinearProblem`.
    """

    def __init__(
        self,
        object: Any,
        a: ufl.form.Form,
        L: ufl.form.Form,
        **kwargs,
    ):
        """
        Constructor for the LinearProblemNode.

        In order to create the LinearProblem in the forward pass,
        ufl form and the function of the linear problem are needed.

        Args:
            object (Any): The LinearProblem object.
            a (ufl.form.Form): The bilinear form of the linear problem.
            L (ufl.form.Form): The linear form of the linear problem.
            u (fem.Function): The solution of the linear problem.
            kwargs: Additional keyword arguments to be passed to the super class.

        """
        super().__init__(object, name="LinearProblem")
        self.a = a
        self.L = L
        self.kwargs = kwargs

    def __call__(self):
        """
        The initialization of the LinearProblem object.

        """
        output = LinearProblemBase(self.a, self.L, **self.kwargs)
        self.object = output
        return output


class NonlinearProblemNode(graph.AbstractNode):
    """
    Node for the initialization of :py:class:`dolfinx.fem.petsc.NonlinearProblem`.
    """

    def __init__(self, object: Any, F: ufl.form.Form, u: fem.Function, **kwargs):
        """
        Constructor for the NonlinearProblemNode.

        In order to create the NonlinearProblem in the forward pass,
        ufl form and the function of the nonlinear problem are needed.

        Args:
            object (Any): The NonlinearProblem object.
            F (ufl.form.Form): The form of the nonlinear problem.
            u (fem.Function): The solution of the nonlinear problem.
            kwargs: Additional keyword arguments to be passed to the super class.

        """
        super().__init__(object, name="NonlinearProblem")
        self.F = F
        self.u = u
        self.kwargs = kwargs

    def __call__(self):
        """
        The initialization of the NonlinearProblem object.

        """
        output = NonlinearProblemBase(self.F, self.u, **self.kwargs)
        self.object = output
        return output


class SolveNode(graph.Node):
    """
    Node for the operation :py:func:`dolfinx.nls.petsc.NewtonSolver.solve`

    """

    def __init__(self, object: Any, problemNode: graph.Node, name="solve", **kwargs):
        """
        Constructor for the SolveNode

        In order to solve the non-linear problem associated with the NewtonSolver,
        the solverNode storing the NewtonSolver and the non-linear problem are needed.

        Args:
            object (Any): The object to be wrapped in the node
            solverNode (graph.Node): The node storing the NewtonSolver
            name (str, optional): The name of the node
            kwargs (optional): Additional keyword arguments to be passed to the :py:class:`dolfinx_adjoint.graph.AbstractNode` constructor

        """
        super().__init__(object, name=name, **kwargs)
        self.problemNode = problemNode
        self.initial_values = object.x.array.copy()

    def __call__(self):
        """
        The call method to solve the non-linear problem associated with the NewtonSolver

        """

        self.object.x.array[:] = self.initial_values[:]
        self.problemNode.object.solve()


class NewtonSolverNonlinearProblemNode(graph.AbstractNode):
    """
    Node for the initialization of :py:class:`dolfinx.fem.petsc.NewtonSolverNonlinearProblem`.
    """

    def __init__(self, object: Any, F: ufl.form.Form, u: fem.Function, **kwargs):
        """
        Constructor for the NewtonSolverNonlinearProblemNode.

        In order to create the NewtonSolverNonlinearProblem in the forward pass,
        ufl form and the function of the nonlinear problem are needed.

        Args:
            object (Any): The NewtonSolverNonlinearProblem object.
            F (ufl.form.Form): The form of the nonlinear problem.
            u (fem.Function): The solution of the nonlinear problem.
            kwargs: Additional keyword arguments to be passed to the super class.

        """
        super().__init__(object, name="NewtonSolverNonlinearProblem")
        self.F = F
        self.u = u
        self.kwargs = kwargs

    def __call__(self):
        """
        The initialization of the NewtonSolverNonlinearProblem object.

        """
        output = NewtonSolverNonlinearProblemBase(self.F, self.u, self.kwargs)
        self.object = output
        return output


class NonlinearProblem_Coefficient_Edge(graph.Edge):
    """
    Edge providing the adjoint equation for the derivative of the solution to the nonlinear problem with respect to the coefficient.

    """

    def calculate_adjoint(self):
        """
        The method provides the adjoint equation for the derivative of the solution to the nonlinear problem with respect to the coefficient.

        By taking the derivative of F(u) = 0 with respect to a coefficient f, we obtain a representation of du/df:
            dF/df = ∂F/∂u * du/df + ∂F/∂f = 0
            => du/df = -(∂F/∂u)^-1 * ∂F/∂f

        By using the accumulated input gradient x the adjoint equation is calculated as:
            (∂F/∂u)ᵀ λ = -xᵀ

        The accumulated gradient is defined by:
            λᵀ * ∂F/∂f

        Returns:
            (PETSc.Vec): The accumulated gradient up to this point in the computational graph.

        """
        # Extract variables from contextvariable ctx
        F, u_node, m, bcs, _graph = self.ctx

        m_node = self.predecessor

        u = u_node.get_object()
        u_next = _graph.get_node(u_node.id, version=u_node.version + 1)

        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        F_manipulated = ufl.replace(F, {u: u_next.data, m: m_node.data})
        J = fem.petsc.assemble_matrix(
            fem.form(ufl.derivative(F_manipulated, u_node.data, du)), bcs=bcs
        )
        J.assemble()

        # Solve (J⁻¹)ᵀ λ = -x where x is the input with a sparse linear solver
        adjoint_solution = AdjointProblemSolver(
            J.transpose(), -self.input_value, fem.Function(V), bcs=bcs
        )

        # Calculate ∂F/∂m
        dFdm = fem.petsc.assemble_matrix(
            fem.form(ufl.derivative(F_manipulated, m_node.data))
        )
        dFdm.assemble()

        # Calculate λᵀ * ∂F/∂m
        return dFdm.transpose() * adjoint_solution.x.petsc_vec


class NonlinearProblem_Constant_Edge(graph.Edge):
    """
    Edge providing the adjoint equation for the derivative of the solution to the nonlinear problem with respect to the constant.

    """

    def calculate_adjoint(self):
        """
        The method provides the adjoint equation for the derivative of the solution to the nonlinear problem with respect to the constant.

        By taking the derivative of F(u) = 0 with respect to a constant c, we obtain a representation of du/dc:
            dF/dc = ∂F/∂u * du/dc + ∂F/∂c = 0
            => du/dc = -(∂F/∂u)^-1 * ∂F/∂c

        By using the accumulated input gradient x the adjoint equation is calculated as:
            (∂F/∂u)ᵀ λ = -xᵀ

        The accumulated gradient is defined by:
            λᵀ * ∂F/∂c

        Returns:
            (PETSc.Vec): The accumulated gradient up to this point in the computational graph.

        """

        # Extract variables from contextvariable ctx
        F, u_node, m, bcs = self.ctx

        u = u_node.get_object()

        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        J = fem.petsc.assemble_matrix(fem.form(ufl.derivative(F, u, du)), bcs=bcs)
        J.assemble()

        # Solve (J⁻¹)ᵀ λ = -x where x is the input with a sparse linear solver
        adjoint_solution = AdjointProblemSolver(
            J.transpose(), -self.input_value, fem.Function(V), bcs=bcs
        )

        # Create a function based on the constant in order to use ufl.derivative
        # to calculate ∂F/∂m
        domain = m.domain
        DG0 = fem.functionspace(domain, ("DG", 0))
        function = fem.Function(DG0)
        function.x.array[:] = m.c
        replaced_form = ufl.replace(F, {m: function})
        dFdm = fem.petsc.assemble_vector(
            fem.form(ufl.derivative(replaced_form, function))
        )
        dFdm.assemble()

        # Calculate λᵀ * ∂F/∂m
        return adjoint_solution.x.petsc_vec.dot(dFdm)


class NonlinearProblem_Boundary_Edge(graph.Edge):
    """
    Edge providing the adjoint equation for the derivative of the solution to the nonlinear problem with respect to the boundary condition.

    """

    def calculate_adjoint(self):
        """
        The method provides the adjoint equation for the derivative of the solution to the nonlinear problem with respect to the boundary condition.

        By taking the derivative of F(u) = 0 with respect to a boundary condition g, we obtain a representation of du/dg:
            dF/dg = ∂F/∂u * du/dg + ∂F/∂g = 0
            => du/dg = -(∂F/∂u)^-1 * ∂F/∂g

        By using the accumulated input gradient x the adjoint equation is calculated as:
            (∂F/∂u)ᵀ λ = -xᵀ

        The accumulated gradient is defined by:
            λᵀ * ∂F/∂g

        Returns:
            (PETSc.Vec): The accumulated gradient up to this point in the computational graph.

        """

        # Extract variables from contextvariable ctx
        F, u_node, bcs, dFdbc_form = self.ctx

        u = u_node.get_object()

        # Construct the Jacobian J = ∂F/∂u
        V = u.function_space
        du = ufl.TrialFunction(V)
        J = ufl.derivative(F, u, du)
        J = fem.petsc.assemble_matrix(fem.form(ufl.derivative(F, u, du)), bcs=bcs)
        J.assemble()

        # Solve (J⁻¹)ᵀ λ = -x where x is the input with a sparse linear solver
        adjoint_solution = AdjointProblemSolver(
            J.transpose(), -self.input_value, fem.Function(V), bcs=bcs
        )

        # ∂F/∂m = dFdbc defined in the nonlinear problem as a fem.Form
        dFdbc = fem.petsc.assemble_matrix(dFdbc_form)
        dFdbc.assemble()

        return dFdbc.transpose() * adjoint_solution.x.petsc_vec


def AdjointProblemSolver(A: PETSc.Mat, b: PETSc.Vec, x: fem.Function, bcs=None):
    """
    Linear solver using PETSc as a linear algebra backend for the adjoint equations.

    Args:
        A (PETSc.Mat): The matrix of the adjoint equation.
        b (PETSc.Vec): The right-hand side of the adjoint equation.
        x (fem.Function): The solution of the adjoint equation.
        bcs (list): The boundary conditions of the adjoint equation.

    Returns:
        (fem.Function): The solution of the adjoint equation.

    """

    _x = create_vector(x.function_space)
    _solver = PETSc.KSP().create(x.function_space.mesh.comm)
    _solver.setOperators(A)

    _b = PETSc.Vec().createWithArray(b)
    if bcs is not None:
        set_bc(_b, bcs, alpha=0.0)

    _solver.setType("preonly")
    _solver.getPC().setType("lu")
    _solver.getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()
    opts["mat_mumps_icntl_24"] = (
        1  # Option to support solving a singular matrix (pressure nullspace)
    )
    opts["mat_mumps_icntl_25"] = (
        0  # Option to support solving a singular matrix (pressure nullspace)
    )
    opts["ksp_error_if_not_converged"] = 1
    _solver.setFromOptions()
    _solver.solve(_b, _x)

    assign(_x, x)

    return x
