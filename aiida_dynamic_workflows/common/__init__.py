# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# Common code used both by the plugin and by the runtime that wraps usercode.

import importlib.resources

from .array import FileBasedObjectArray
from .mapspec import MapSpec
from .serialize import dump, load

__all__ = ["dump", "load", "FileBasedObjectArray", "MapSpec", "package_module_contents"]


def package_module_contents():
    """Yield (filename, contents) pairs for each module in this subpackage."""
    for filename in importlib.resources.contents(__package__):
        if filename.endswith(".py"):
            yield filename, importlib.resources.read_text(__package__, filename)
