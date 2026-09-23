from dolfinx import fem, la

import dolfinx_adjoint.graph as graph
from dolfinx_adjoint.utils import bind_arguments


class Function(fem.Function):
    """OVERLOADS: :py:class:`dolfinx.fem.Function`
    Initialize a finite element Function.

    The overloaded class modifies the initialization of the Function to keep track of the dependencies
    in the computational graph and the adjoint equations. The original functionality is kept.
    Two additional methods are added to the class: copy and assign. The copy method creates a new function
    and adds the corresponding node to the graph. The assign method assign values to the function and adds the
    corresponding operation to the graph.

    """

    def __init__(self, *args, **kwargs):
        """OVERLOADS: :py:class:`dolfinx.fem.Function`
        Initialize a finite element Function.

        Args:
            args: Arguments to :py:class:`dolfinx.fem.Function`
            kwargs: Keyword arguments to :py:class:`dolfinx.fem.Function`
            graph (graph, optional): An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.
            name (str, optional): An additional keyword argument to specify the name of the function. If not present, the name is set to "f".
            map (optional): An additional keyword argument to specify the map of the function. If not present, the map is set to None.

        Note:
            The map is used to keep track of relations in mixed element spaces.

        """
        _graph = kwargs.pop("graph", None)
        map = kwargs.pop("map", None)

        super().__init__(*args, **kwargs)
        if _graph is None:
            return

        self.map = map
        function_node = graph.Node(self, name=self.name)
        _graph.add_node(function_node)

    def copy(self, **kwargs):
        """Creates a new dolfinx.fem.Function with the same function space and a copy of the PETsc vector.

        Args:
            name (str, optional): Keyword arguments to specify the name of the function.
            graph (graph, optional): An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.

        Returns:
            Function: A new dolfinx.fem.Function with the same function space and a copy of the PETsc vector.

        """
        _graph = kwargs.pop("graph", None)
        result = Function(
            self.function_space, la.Vector(type(self.x._cpp_object)(self.x._cpp_object))
        )
        if "name" in kwargs:
            result.name = kwargs["name"]
        if _graph is None:
            return result

        function_node = graph.Node(result, name=result.name)
        _graph.add_node(function_node)

        copied_node = _graph.get_node(id(self))
        copy_edge = graph.Edge(copied_node, function_node)
        function_node.set_gradFuncs([copy_edge])
        _graph.add_edge(copy_edge)
        copy_edge.set_next_functions(copied_node.get_gradFuncs())

        return result

    def assign(self, function: fem.Function, **kwargs):
        """
        Assign values of a different dolfinx.fem.Function to the function and adds the corresponding operation to the graph.

        Args:
            function (dolfinx.fem.Function): The function to assign values from.
            graph (graph, optional): An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.
            version (int, optional): An additional keyword argument to specify the version of the operation in the graph. If not present, the version is set to 0.

        Note:
            The node is added in any case, since it is the version of this function the
            subsequent operations depend on.

        """
        self.x.array[:] = function.x.array[:]

        _graph = kwargs.pop("graph", None)
        version = kwargs.pop("version", 0)
        if _graph is None:
            return

        function_node = _graph.get_node(id(function))

        assign_node = graph.Node(self, name=self.name, version=version)
        _graph.add_node(assign_node)

        if function_node is None:
            return

        assign_edge = graph.Edge(function_node, assign_node)
        assign_node.set_gradFuncs([assign_edge])
        _graph.add_edge(assign_edge)
        assign_edge.set_next_functions(function_node.get_gradFuncs())


class Constant(fem.Constant):
    """OVERLOADS: :py:class:`dolfinx.fem.Constant`
    Initialize a constant function.

    The overloaded class modifies the initialization of the Constant to keep track of the dependencies
    in the computational graph and the adjoint equations. The original functionality is kept.

    """

    def __init__(self, *args, **kwargs):
        """OVERLOADS: :py:class:`dolfinx.fem.Constant`
        Initialize a constant function.

        Args:
            args: Arguments to :py:class:`dolfinx.fem.Constant`
            kwargs: Keyword arguments to :py:class:`dolfinx.fem.Constant`
            graph (graph, optional): An additional keyword argument to specifier whether the assemble
                operation should be added to the graph. If not present, the original functionality
                of dolfinx is used without any additional functionalities.
            name (str, optional): An additional keyword argument to specify the name of the function. If not present, the name is set to "Constant".

        """
        _graph = kwargs.pop("graph", None)
        name = kwargs.pop("name", "Constant")

        super().__init__(*args, **kwargs)
        if _graph is None:
            return

        Constant_node = graph.Node(self, name=name)
        _graph.add_node(Constant_node)
        arguments = bind_arguments(fem.Constant.__init__, self, *args, **kwargs)
        self.domain = arguments["domain"]
