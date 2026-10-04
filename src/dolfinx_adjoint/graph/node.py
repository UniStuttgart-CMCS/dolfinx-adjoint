import copy
from typing import Any

import petsc4py.PETSc as PETSc


class AbstractNode:
    """
    The base class for nodes in the graph.

    AbstractNodes provide the functionality to represent the object and operations in the graph.
    To create a new node, the user should inherit from this class and implement the __call__ method or
    use the this AbstractNode class to directly create a new node without any additional functionalities.

    The AbstractNode class is used to represent objects without any numerical value.

    Attributes:
        id (int): The python id of the object
        version (int, optional): The version of the object, indicates if an object is an updated version of an
            already existing object. Defaults to 0.
        object (Any): The object that the node represents
        grad_fns (list): A list of the edges into the node, whose adjoint values it receives
        _name (str): The name of the node

    """

    def __init__(self, object: Any, version=0, **kwargs):
        """
        Constructor for the AbstractNode class.

        Args:
            object (Any): The object that the node represents
            version (int, optional): The version of the object, indicates if an object is an updated version of an
                already existing object. Defaults to 0.
            **kwargs: Additional keyword arguments

        """
        self.id = id(object)
        self.version = version
        self.object = object
        self.grad_fns = []
        if "name" in kwargs:
            self._name = kwargs["name"]
        else:
            self._name = str(object.__class__.__name__)

    @property
    def name(self):
        """
        Returns the name of the node.

        Returns:
            str: The name of the node
        """
        if self.version != 0:
            return f"{self._name} [{str(self.version)}]"
        else:
            return str(self._name)

    def set_object(self, object: Any):
        """
        Sets the object of the node.

        Args:
            object (Any): The object that the node represents
        """
        self.object = object

    def release(self):
        """
        Releases the values saved in the node.

        """
        self.object = None

    def __call__(self, *args, **kwargs):
        """
        This method is a placeholder for the computation of the node.

        The __call__ method is used to compute the node with the given arguments.
        The method should be overwritten by the user to implement the computation of the node.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        """
        pass

    def __str__(self):
        """
        Returns the string representation of the node.

        Returns:
            str: The string representation of the node
        """
        return str(self.name)


class Node(AbstractNode):
    """
    The Node class is used to represent objects with a numerical value.

    The Node class inherits from the AbstractNode class and provides the functionality to represent the object and operations in the graph.

    Attributes:
        data (Any): The object that the node represents
        grad (float or PETSc.Vec): The gradient of the object

    """

    def __init__(self, object: Any, **kwargs):
        super().__init__(object, **kwargs)
        self.data = copy.copy(object)
        self.grad = None

    def get_object(self):
        """The object the node represents, or None once the node is released."""

        return self.object

    def reset_grad(self):
        self.grad = None

    def accumulate_grad(self, value: float | PETSc.Vec):
        """
        Accumulate a gradient contribution in the node.

        Note:
            Contributions relying on shared data types (e.g. PETSc.Vec) should be accumulated without modifying the contribution. This is ensured for PETSc.Vec by using the copy method. Contributions relying on non-shared data types (e.g. float) can be accumulated directly.

        Args:
            value (float or PETSc.Vec): The gradient contribution to accumulate

        """

        if self.grad is None:
            self.grad = value.copy() if isinstance(value, PETSc.Vec) else value
        else:
            self.grad += value

    def release(self):
        """
        Releases the object, the data and the gradient of the node.

        """
        super().release()
        self.data = None
        self.grad = None
