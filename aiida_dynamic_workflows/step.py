# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import copy
from typing import Any, Callable, Dict, Optional, Tuple, Union

import toolz

from .data import PyFunction

__all__ = ["step"]


@toolz.curry
def step(
    f: Callable,
    *,
    returns: Union[str, Tuple[str]] = "_return_value",
    resources: Optional[Dict[str, Any]] = None,
) -> PyFunction:
    """Construct a PyFunction from a Python function.

    This function is commonly used as a decorator.

    Parameters
    ----------
    f
        The function to transform into a PyFunction
    returns
        The name of the output of this function.
        If multiple names are provided, then 'f' is assumed to return
        as many values (as a tuple) as there are names.
    resources
        Optional specification of computational resources that this
        function needs. Possible resources are: "memory", "cores".
        "memory" must be a string containing an integer value followed
        by one of the following suffixes: "kB", "MB", "GB".
        "cores" must be a positive integer.

    Examples
    --------
    >>> f = step(lambda x, y: x + y, returns="sum")
    >>>
    >>> @step(returns="other_sum", resources={"memory": "10GB", cores=2})
    ... def g(x: int, y: int) -> int:
    ...     return x + y
    ...
    >>> @step(returns=("a", "b"))
    ... def h(x):
    ...     return (x + 1, x + 2)
    ...
    >>>
    """
    # TODO: First query the Aiida DB to see if this function already exists.
    #       This will require having a good hash for Python functions.
    #       This is a hard problem.
    if resources:
        _validate_resources(resources)

    node = PyFunction(func=f, returns=returns, resources=resources)
    node.store()
    return node


def _validate_resources(resources) -> Dict:
    resources = copy.deepcopy(resources)
    if "memory" in resources:
        _validate_memory(resources.pop("memory"))
    if "cores" in resources:
        _validate_cores(resources.pop("cores"))
    if resources:
        raise ValueError(f"Unexpected resource specifications: {list(resources)}")


def _validate_memory(memory: str):
    mem, unit = memory[:-2], memory[-2:]
    if not mem.isnumeric():
        raise ValueError(f"Expected an integer amount of memory, got: '{mem}'")
    elif int(mem) == 0:
        raise ValueError("Cannot specify zero memory")
    valid_units = ("kB", "MB", "GB")
    if unit not in valid_units:
        raise ValueError(
            f"Invalid memory unit: '{unit}' (expected one of {valid_units})."
        )


def _validate_cores(cores: int):
    if int(cores) != cores:
        raise ValueError(f"Expected an integer number of cores, got: {cores}")
    elif cores <= 0:
        raise ValueError(f"Expected a positive number of cores, got: {cores}")
