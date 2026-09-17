import inspect
from collections.abc import Callable
from typing import Any


def bind_arguments(function: Callable, *args, **kwargs) -> dict[str, Any]:
    """Bind a call by name against the signature of the overloaded DOLFINx function.

    Overloads forward the call of the user unchanged to DOLFINx and read the arguments
    they need from the result, so they neither depend on the order of the parameters
    nor on whether an argument was passed positionally or as a keyword.

    Args:
        function: The DOLFINx function or method whose signature is used. Methods are
            bound with the instance as the first positional argument.
        args: The positional arguments of the call.
        kwargs: The keyword arguments of the call, without the ones added by the overload.

    Returns:
        The arguments passed in the call, keyed by parameter name. Parameters that were
        not passed are absent.

    Raises:
        TypeError: If the call does not match the signature.

    """
    return inspect.signature(function).bind(*args, **kwargs).arguments
