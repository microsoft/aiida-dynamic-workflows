# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Aiida data plugins for running arbitrary Python functions."""

from concurrent.futures import ThreadPoolExecutor
import functools
import inspect
import io
from itertools import repeat
import operator
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Optional

import aiida.orm
import cloudpickle
import numpy as np
import toolz

# To get Aiida's caching to be useful we need to have a stable way to hash Python
# functions. The "default" is to hash the cloudpickle blob, but this is not
# typically stable for functions defined in a Jupyter notebook.
# TODO: insert something useful here.
function_hasher = None


class PyFunction(aiida.orm.Data):
    """Aiida representation of a Python function."""

    def __init__(self, **kwargs):
        # TODO: basic typechecks on these
        func = kwargs.pop("func")
        assert callable(func)
        returns = kwargs.pop("returns")
        if isinstance(returns, str):
            returns = [returns]
        resources = kwargs.pop("resources", None)
        if resources is None:
            resources = dict()

        super().__init__(**kwargs)

        self.put_object_from_filelike(
            path="function.pickle",
            handle=io.BytesIO(cloudpickle.dumps(func)),
        )
        self.set_attribute("resources", resources)
        self.set_attribute("returns", returns)
        self.set_attribute("parameters", _parameters(func))

        # If 'function_hasher' is available then we store the
        # function hash directly, and _get_objects_to_hash will
        # _not_ use the pickle blob (which is not stable e.g.
        # for functions defined in a notebook).
        if callable(function_hasher):
            self.set_attribute("_function_hash", function_hasher(func))

        try:
            source = inspect.getsource(func)
        except Exception:
            pass
        else:
            self.set_attribute("source", source)

        name = getattr(func, "__name__", None)
        if name:
            self.set_attribute("name", name)

    @property
    def resources(self) -> dict[str, str]:
        """Resources required by this function."""
        return self.get_attribute("resources")

    @property
    def source(self) -> str:
        """Source code of this function."""
        return self.get_attribute("source")

    @property
    def name(self) -> str:
        """Name of this function."""
        return self.get_attribute("name")

    @property
    def parameters(self) -> list[str]:
        """Parameters of this function."""
        return self.get_attribute("parameters")

    @property
    def returns(self) -> Optional[list[str]]:
        """List of names returned by this function."""
        return self.get_attribute("returns")

    # TODO: use better caching for this (maybe on the class level?)
    @functools.cached_property
    def pickle(self) -> bytes:
        """Pickled function."""
        return self.get_object_content("function.pickle", "rb")

    @functools.cached_property
    def callable(self) -> Callable:
        """Return the function stored in this object."""
        return cloudpickle.loads(self.pickle)

    @property
    def __signature__(self):
        return inspect.signature(self.callable)

    def __call__(self, *args: Any, **kwargs: Any):
        """Call the function stored in this object."""
        return self.callable(*args, **kwargs)

    def _get_objects_to_hash(self) -> list[Any]:
        objects = super()._get_objects_to_hash()

        # XXX: this depends on the specifics of the implementation
        #      of super()._get_objects_to_hash(). The second-to-last
        #      elements in 'objects' is the hash of the file repository.
        #      For 'PyFunction' nodes this contains the cloudpickle blob,
        #      which we _do not_ want hashed.
        if "_function_hash" in self.attributes:
            *a, _, x = objects
            return [*a, x]
        else:
            return objects


def _parameters(f: Callable) -> list[str]:
    valid_kinds = [
        getattr(inspect.Parameter, k) for k in ("POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY")
    ]
    params = inspect.signature(f).parameters.values()
    if any(p.kind not in valid_kinds for p in params):
        raise TypeError("Invalid signature")
    return [p.name for p in params]


class Nil(aiida.orm.Data):
    """Trivial representation of the None type in Aiida."""


# TODO: make this JSON serializable so it can go directly in the DB
class PyOutline(aiida.orm.Data):
    """Naive Aiida representation of a workflow outline."""

    def __init__(self, **kwargs):
        outline = kwargs.pop("outline")
        super().__init__(**kwargs)

        self.put_object_from_filelike(
            path="outline.pickle",
            handle=io.BytesIO(cloudpickle.dumps(outline)),
        )

    @functools.cached_property
    def value(self):
        """Python object loaded from the stored pickle."""
        return cloudpickle.loads(self.get_object_content("outline.pickle", "rb"))


# TODO: Annotate these with the class name (useful for visualization)
class PyData(aiida.orm.Data):
    """Naive Aiida representation of an arbitrary Python object."""

    def __init__(self, **kwargs):
        pickle_path = kwargs.pop("pickle_path")

        super().__init__(**kwargs)
        self.put_object_from_file(filepath=pickle_path, path="object.pickle")

    # TODO: do caching more intelligently: we could attach a cache to the
    #       _class_ instead so that if we create 2 PyData objects that
    #       point to the _same_ database entry (pk) then we only have to
    #       load the data once.
    #       (does Aiida provide some tooling for this?)
    @functools.cached_property
    def value(self):
        """Python object loaded from the stored pickle."""
        return cloudpickle.loads(self.get_object_content("object.pickle", "rb"))


class PyRemoteData(aiida.orm.RemoteData):
    """Naive Aiida representation of an arbitrary Python object on a remote computer."""

    def __init__(self, **kwargs):
        pickle_path = str(kwargs.pop("pickle_path"))
        super().__init__(**kwargs)

        self.set_attribute("pickle_path", pickle_path)

    @property
    def pickle_path(self):
        """Return the remote path that contains the pickle."""
        return os.path.join(self.get_remote_path(), self.get_attribute("pickle_path"))

    def fetch_value(self):
        """Load Python object from the remote pickle."""
        with tempfile.NamedTemporaryFile(mode="rb") as f:
            self.getfile(self.get_attribute("pickle_path"), f.name)
            return cloudpickle.load(f)

    @classmethod
    def from_remote_data(cls, rd: aiida.orm.RemoteData, pickle_path: str):
        """Return a new PyRemoteData, given an existing RemoteData.

        Parameters
        ----------
        rd
            RemoteData folder.
        pickle_path
            Relative path in the RemoteData that contains pickle data.
        """
        return cls(
            remote_path=rd.get_remote_path(),
            pickle_path=pickle_path,
            computer=rd.computer,
        )


class PyRemoteArray(aiida.orm.RemoteData):
    """Naive Aiida representation of a remote array of arbitrary Python objects.

    Each object is stored in a separate file.
    """

    def __init__(self, **kwargs):
        shape = kwargs.pop("shape")
        filename_template = kwargs.pop("filename_template")
        super().__init__(**kwargs)
        self.set_attribute("shape", tuple(shape))
        self.set_attribute("filename_template", str(filename_template))

    def _file(self, i: int) -> str:
        return self.get_attribute("filename_template").format(i)

    @property
    def pickle_path(self):
        """Return the remote path that contains the pickle files."""
        return self.get_remote_path()

    def _fetch_buffer(self, local_files=False):
        """Return iterator over Python objects in this array."""

        def _load(dir: Path, pickle_file: str):
            path = dir / pickle_file
            if not path.is_file():
                return None
            else:
                with open(path, "rb") as f:
                    return cloudpickle.load(f)

        def _iter_files(dir):
            with ThreadPoolExecutor() as ex:
                file_gen = map(self._file, range(self.size))
                yield from ex.map(_load, repeat(dir), file_gen)

        if local_files:
            # If the array's directory does not exist then it's
            # not actually mounted locally.
            root_dir = Path(self.get_remote_path())
            if not root_dir.is_dir():
                raise FileNotFoundError(str(root_dir))
            else:
                yield from _iter_files(root_dir)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                dir = Path(os.path.join(temp_dir, "values"))
                # TODO: do this with chunks, rather than all files at once.
                with self.get_authinfo().get_transport() as transport:
                    transport.gettree(self.get_remote_path(), dir)
                yield from _iter_files(dir)

    def fetch_value(self, local_files=False) -> np.ma.core.MaskedArray:
        """Return a numpy array with dtype 'object' for this array."""
        # Objects that have a bogus '__array__' implementation fool
        # 'buff[:] = xs', so we need to manually fill the array.
        buff = np.empty((self.size,), dtype=object)
        for i, x in enumerate(self._fetch_buffer(local_files)):
            buff[i] = x
        buff = buff.reshape(self.shape)
        return np.ma.array(buff, mask=self.mask)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of this remote array."""
        return tuple(self.get_attribute("shape"))

    @property
    def is_masked(self) -> bool:
        """Return True if some elements of the array are 'masked' (missing)."""
        return np.any(self.mask)

    @property
    def mask(self) -> np.ndarray:
        """Return the mask for the missing elements of the array."""
        existing_files = {
            v["name"] for v in self.listdir_withattributes() if not v["isdir"]
        }
        return np.array(
            [self._file(i) not in existing_files for i in range(self.size)],
            dtype=bool,
        ).reshape(self.shape)

    @property
    def size(self) -> int:
        """Size of this remote array (product of the shape)."""
        return toolz.reduce(operator.mul, self.shape, 1)


class PyArray(PyData):
    """Wrapper around PyData for storing a single array."""

    def __init__(self, **kwargs):
        array = np.asarray(kwargs.pop("array"))
        with tempfile.NamedTemporaryFile() as handle:
            cloudpickle.dump(array, handle)
            handle.flush()
            handle.seek(0)
            super().__init__(pickle_path=handle.name, **kwargs)
        self.set_attribute("shape", array.shape)
        self.set_attribute("dtype", str(array.dtype))
        self._cached = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of this remote array."""
        return tuple(self.get_attribute("shape"))

    @property
    def dtype(self) -> tuple[int, ...]:
        """Shape of this remote array."""
        return np.dtype(self.get_attribute("dtype"))

    @property
    def size(self) -> int:
        """Size of this remote array (product of the shape)."""
        return toolz.reduce(operator.mul, self.shape, 1)

    def get_array(self) -> np.ndarray:
        """Return the array."""
        return self.value


class PyException(aiida.orm.Data):
    """Aiida representation of a Python exception."""

    # - Exception type
    # - message
    # - traceback
    ...


# Register automatic conversion from lists and numpy arrays
# to the appropriate Aiida datatypes


@aiida.orm.to_aiida_type.register(type(None))
def _(_: None):
    return Nil()


# Aiida Lists can only handle built-in types, which is not general
# enough for our purposes. We therefore convert Python lists into
# 1D PyArray types with 'object' dtype.
@aiida.orm.to_aiida_type.register(list)
def _(xs: list):
    arr = np.empty((len(xs),), dtype=object)
    # Objects that have a bogus '__array__' implementation fool
    # 'arr[:] = xs', so we need to manually fill the array.
    for i, x in enumerate(xs):
        arr[i] = x
    return PyArray(array=arr)


@aiida.orm.to_aiida_type.register(np.ndarray)
def _(x):
    return PyArray(array=x)


def ensure_aiida_type(x: Any) -> aiida.orm.Data:
    """Return a new Aiida value containing 'x', if not already of an Aiida datatype.

    If 'x' is already an Aiida datatype, then return 'x'.
    """
    if isinstance(x, aiida.orm.Data):
        return x
    else:
        r = aiida.orm.to_aiida_type(x)
        if not isinstance(r, aiida.orm.Data):
            raise RuntimeError(
                "Expected 'to_aiida_type' to return an Aiida data node, but "
                f"got an object of type '{type(r)}' instead (when passed "
                f"an object of type '{type(x)}')."
            )
        return r


# Register handlers for getting native Python objects from their
# Aiida equivalents


@functools.singledispatch
def from_aiida_type(x):
    """Turn Aiida types into their corresponding native Python types."""
    raise TypeError(f"Do not know how to convert {type(x)} to native Python type")


@from_aiida_type.register(Nil)
def _(_):
    return None


@from_aiida_type.register(aiida.orm.BaseType)
def _(x):
    return x.value


@from_aiida_type.register(PyData)
def _(x):
    return x.value


@from_aiida_type.register(PyArray)
def _(x):
    return x.get_array()


# Register handlers for figuring out array shapes for different datatypes


@functools.singledispatch
def array_shape(x) -> tuple[int, ...]:
    """Return the shape of 'x'."""
    try:
        return tuple(map(int, x.shape))
    except AttributeError:
        raise TypeError(f"No array shape defined for type {type(x)}")


@array_shape.register(aiida.orm.List)
def _(x):
    return (len(x),)


# Register handlers for figuring out array masks for different datatypes


@functools.singledispatch
def array_mask(x) -> np.ndarray:
    """Return the mask applied to 'x'."""
    try:
        return x.mask
    except AttributeError:
        raise TypeError(f"No array mask defined for type {type(x)}")


@array_mask.register(aiida.orm.List)
def _(x):
    return np.full((len(x),), False)


@array_mask.register(PyArray)
@array_mask.register(np.ndarray)
def _(x):
    return np.full(x.shape, False)
