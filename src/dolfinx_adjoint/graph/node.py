from typing import Any

import numpy as np
import petsc4py.PETSc as PETSc


class AbstractNode:
    """
    The base class for nodes in the graph.

    AbstractNodes provide the functionality to represent the object and operations in the graph.
     To create a new node, the user can inherit from this class or use it directly.

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

    @property
    def snapshot(self) -> None:
        """None, since a node without a numerical value saves nothing, see Node."""
        return None

    def release(self):
        """
        Releases the values saved in the node.

        """
        self.object = None

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
        grad (float or PETSc.Vec): The gradient of the object
        snapshot (Any or None): The saved copy of this version's value, or None when
            no copy is held. Subclasses implement ``save`` to create it.

    """

    def __init__(self, object: Any, **kwargs):
        super().__init__(object, **kwargs)
        self.grad = None
        self._snapshot = None

    @property
    def snapshot(self) -> Any | None:
        """The snapshot owned by this version, or None; only the node replaces it."""
        return self._snapshot

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
            self.grad = (
                value.copy() if isinstance(value, (PETSc.Vec, np.ndarray)) else value
            )
        else:
            self.grad += value

    def release(self):
        """
        Releases the object, the snapshot and the gradient of the node.

        """

        super().release()
        self.grad = None
        self._snapshot = None
