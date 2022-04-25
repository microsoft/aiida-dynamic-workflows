# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Aiida Calculations for running arbitrary Python functions."""

import os
import textwrap
from typing import Any, Dict, Sequence

import aiida.common
import aiida.engine
import numpy as np
import toolz

from . import common
from .data import (
    Nil,
    PyArray,
    PyData,
    PyFunction,
    PyRemoteArray,
    PyRemoteData,
    array_mask,
    array_shape,
)


class PyCalcJob(aiida.engine.CalcJob):
    """CalcJob that runs a single Python function."""

    @aiida.common.lang.override
    def out(self, output_port, value=None) -> None:
        """Attach output to output port."""
        # This hack is necessary to work around a bug with output namespace naming.
        # Some parts of Aiida consider the namespace/port separator to be '__',
        # but others think it is '.'.
        return super().out(output_port.replace("__", "."), value)

    @classmethod
    def define(cls, spec: aiida.engine.CalcJobProcessSpec):  # noqa: D102
        super().define(spec)

        spec.input(
            "func",
            valid_type=PyFunction,
            help="The function to execute",
        )
        spec.input_namespace(
            "kwargs",
            dynamic=True,
            help="The (keyword) arguments to the function",
        )

        spec.output_namespace(
            "return_values",
            dynamic=True,
            help="The return value(s) of the function",
        )
        spec.output(
            "exception",
            required=False,
            help="The exception raised (if any)",
        )

        spec.inputs["metadata"]["options"][
            "parser_name"
        ].default = "dynamic_workflows.PyCalcParser"
        spec.inputs["metadata"]["options"]["resources"].default = dict(
            num_machines=1, num_mpiprocs_per_machine=1
        )

        # TODO: add error codes
        spec.exit_code(
            401,
            "USER_CODE_RAISED",
            invalidates_cache=True,
            message="User code raised an Exception.",
        )
        spec.exit_code(
            402,
            "NONZERO_EXIT_CODE",
            invalidates_cache=True,
            message="Script returned non-zero exit code.",
        )
        spec.exit_code(
            403,
            "MISSING_OUTPUT",
            invalidates_cache=True,
            message="Script returned zero exit code, but no output generated.",
        )

    # TODO: refactor this; it is a bit of a mess
    def prepare_for_submission(
        self,
        folder: aiida.common.folders.Folder,
    ) -> aiida.common.CalcInfo:  # noqa: D102

        # TODO: update "resources" given the resources specified on "py_func"
        codeinfo = aiida.common.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid

        calcinfo = aiida.common.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.remote_copy_list = []
        calcinfo.remote_symlink_list = []

        py_function = self.inputs.func
        computer = self.inputs.code.computer
        kwargs = getattr(self.inputs, "kwargs", dict())

        remaining_kwargs_file = "__kwargs__/__remaining__.pickle"
        kwargs_array_folder_template = "__kwargs__/{}"
        kwargs_filename_template = "__kwargs__/{}.pickle"
        function_file = "__func__.pickle"
        exception_file = "__exception__.pickle"
        return_value_files = [
            f"__return_values__/{r}.pickle" for r in py_function.returns
        ]

        folder.get_subfolder("__kwargs__", create=True)
        folder.get_subfolder("__return_values__", create=True)

        calcinfo.retrieve_list = [exception_file]

        # TODO: figure out how to do this with "folder.copy_file" or whatever
        with folder.open(function_file, "wb") as f:
            f.write(py_function.pickle)

        literal_kwargs = dict()
        local_kwargs = dict()
        remote_kwargs = dict()
        remote_array_kwargs = dict()
        for k, v in kwargs.items():
            # TODO: refactor this to allow more generic / customizable dispatch
            if isinstance(v, aiida.orm.BaseType):
                literal_kwargs[k] = v.value
            elif isinstance(v, PyArray):
                literal_kwargs[k] = v.get_array()
            elif isinstance(v, PyRemoteData):
                remote_kwargs[k] = v
            elif isinstance(v, PyRemoteArray):
                remote_array_kwargs[k] = v
            elif isinstance(v, Nil):
                literal_kwargs[k] = None
            elif isinstance(v, PyData):
                local_kwargs[k] = v
            else:
                raise ValueError(f"Unsure how to treat '{k}' ({type(v)})")

        for k, v in remote_kwargs.items():
            # TODO: move the data as needed.
            if v.computer.uuid != self.inputs.code.computer.uuid:
                raise ValueError(
                    f"Data passed as '{k}' to '{py_function.name}' is stored "
                    f"on '{v.computer.label}', which is not directly accessible "
                    f"from '{computer.label}'."
                )
            calcinfo.remote_symlink_list.append(
                (computer.uuid, v.pickle_path, kwargs_filename_template.format(k))
            )

        for k, v in remote_array_kwargs.items():
            # TODO: move the data as needed.
            if v.computer.uuid != self.inputs.code.computer.uuid:
                raise ValueError(
                    f"Data passed as '{k}' to '{py_function.name}' is stored "
                    f"on '{v.computer.label}', which is not directly accessible "
                    f"from '{computer.label}'."
                )
            calcinfo.remote_symlink_list.append(
                (computer.uuid, v.pickle_path, kwargs_array_folder_template.format(k))
            )

        assert not local_kwargs
        kwarg_filenames = [kwargs_filename_template.format(k) for k in remote_kwargs]
        kwarg_array_folders = [
            kwargs_array_folder_template.format(k) for k in remote_array_kwargs
        ]
        kwarg_array_shapes = [v.shape for v in remote_array_kwargs.values()]
        separate_kwargs = list(remote_kwargs.keys())
        separate_array_kwargs = list(remote_array_kwargs.keys())

        if literal_kwargs:
            common.dump(literal_kwargs, remaining_kwargs_file, opener=folder.open)

        # Add the '.common' subpackage as a package called 'common'.
        # This can therefore be used directly from the script.
        common_package_folder = folder.get_subfolder("common", create=True)
        for filename, contents in common.package_module_contents():
            with common_package_folder.open(filename, "w") as f:
                f.write(contents)

        # TODO: factor this out
        script = textwrap.dedent(
            f"""\
            import os
            import sys
            import cloudpickle

            import common

            # Define paths for I/O

            function_file = "{function_file}"
            separate_kwargs = {separate_kwargs}
            separate_kwarg_filenames = {kwarg_filenames}
            separate_array_kwargs = {separate_array_kwargs}
            separate_array_folders = {kwarg_array_folders}
            separate_array_shapes = {kwarg_array_shapes}
            remaining_kwargs_file = "{remaining_kwargs_file}"
            exception_file = "{exception_file}"
            return_value_files = {return_value_files}
            assert return_value_files

            # Load code

            func = common.load(function_file)

            # Load kwargs

            kwargs = dict()
            # TODO: hard-code this when we switch to a Jinja template
            # TODO: parallel load using a threadpool
            for pname, fname in zip(separate_kwargs, separate_kwarg_filenames):
                kwargs[pname] = common.load(fname)
            for pname, fname, shape in zip(
                separate_array_kwargs, separate_array_folders, separate_array_shapes,
            ):
                kwargs[pname] = common.FileBasedObjectArray(fname, shape=shape)
            if os.path.exists(remaining_kwargs_file):
                kwargs.update(common.load(remaining_kwargs_file))

            # Execute

            try:
                return_values = func(**kwargs)
            except Exception as e:
                common.dump(e, exception_file)
                sys.exit(1)

            # Output

            if len(return_value_files) == 1:
                common.dump(return_values, return_value_files[0])
            else:
                for r, f in zip(return_values, return_value_files):
                    common.dump(r, f)
        """
        )

        with folder.open("__in__.py", "w", encoding="utf8") as handle:
            handle.write(script)
        codeinfo.stdin_name = "__in__.py"

        return calcinfo


class PyMapJob(PyCalcJob):
    """CalcJob that maps a Python function over (a subset of) its parameters."""

    @classmethod
    def define(cls, spec: aiida.engine.CalcJobProcessSpec):  # noqa: D102
        super().define(spec)

        spec.input(
            "metadata.options.mapspec",
            valid_type=str,
            help=(
                "A specification for which parameters to map over, "
                "and how to map them"
            ),
        )

        # Setting 1 as the default means people won't accidentally
        # overload the cluster with jobs.
        spec.input(
            "metadata.options.max_concurrent_machines",
            valid_type=int,
            default=1,
            help="How many machines to use for this map, maximally.",
        )
        spec.input(
            "metadata.options.cores_per_machine",
            valid_type=int,
            help="How many cores per machines to use for this map.",
        )
        spec.inputs["metadata"]["options"][
            "parser_name"
        ].default = "dynamic_workflows.PyMapParser"

    @property
    def mapspec(self) -> common.MapSpec:
        """Parameter and shape specification for this map job."""
        return common.MapSpec.from_string(self.metadata.options.mapspec)

    # TODO: refactor / merge this with PyCalcJob
    def prepare_for_submission(  # noqa: C901
        self, folder: aiida.common.folders.Folder
    ) -> aiida.common.CalcInfo:  # noqa: D102
        # TODO: update "resources" given the resources specified on "py_func"
        codeinfo = aiida.common.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid

        calcinfo = aiida.common.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.remote_copy_list = []
        calcinfo.remote_symlink_list = []

        py_function = self.inputs.func
        kwargs = self.inputs.kwargs
        computer = self.inputs.code.computer

        spec = self.mapspec
        mapped_kwargs = {
            k: v for k, v in self.inputs.kwargs.items() if k in spec.parameters
        }
        mapped_kwarg_shapes = toolz.valmap(array_shape, mapped_kwargs)
        # This will raise an exception if the shapes are not compatible.
        spec.shape(mapped_kwarg_shapes)

        function_file = "__func__.pickle"
        exceptions_folder = "__exceptions__"
        remaining_kwargs_file = "__kwargs__/__remaining__.pickle"
        kwarg_file_template = "__kwargs__/{}.pickle"
        mapped_kwarg_folder_template = "__kwargs__/{}"
        return_value_folders = [f"__return_values__/{r}" for r in py_function.returns]

        calcinfo.retrieve_list = [exceptions_folder]

        # TODO: figure out how to do this with "folder.copy_file" or whatever
        with folder.open(function_file, "wb") as f:
            f.write(py_function.pickle)

        folder.get_subfolder(exceptions_folder, create=True)
        folder.get_subfolder("__kwargs__", create=True)
        folder.get_subfolder("__return_values__", create=True)

        folder.get_subfolder(exceptions_folder, create=True)
        for rv in return_value_folders:
            folder.get_subfolder(rv, create=True)

        valid_sequence_types = (
            aiida.orm.List,
            PyArray,
            PyRemoteArray,
        )
        for k in mapped_kwargs:
            v = kwargs[k]
            if not isinstance(v, valid_sequence_types):
                raise TypeError(
                    f"Expected one of {valid_sequence_types} for {k}, "
                    f"but received {type(v)}"
                )

        remaining_kwargs = dict()
        mapped_literal_kwargs = dict()
        remote_kwargs = dict()
        for k, v in kwargs.items():
            # TODO: refactor this to allow more generic / customizable dispatch
            if isinstance(v, (PyRemoteData, PyRemoteArray)):
                remote_kwargs[k] = v
            elif isinstance(v, aiida.orm.List) and k in mapped_kwargs:
                mapped_literal_kwargs[k] = v.get_list()
            elif isinstance(v, PyArray) and k in mapped_kwargs:
                mapped_literal_kwargs[k] = v.get_array()
            elif isinstance(v, aiida.orm.List):
                remaining_kwargs[k] = v.get_list()
            elif isinstance(v, PyArray):
                remaining_kwargs[k] = v.get_array()
            elif isinstance(v, Nil):
                remaining_kwargs[k] = None
            elif isinstance(v, PyData):
                assert False
            else:
                try:
                    remaining_kwargs[k] = v.value
                except AttributeError:
                    raise RuntimeError(f"Unsure how to treat values of type {type(v)}")

        if remaining_kwargs:
            common.dump(remaining_kwargs, remaining_kwargs_file, opener=folder.open)

        for k, v in mapped_literal_kwargs.items():
            common.dump(v, kwarg_file_template.format(k), opener=folder.open)

        for k, v in remote_kwargs.items():
            # TODO: move the data as needed.
            if v.computer.uuid != self.inputs.code.computer.uuid:
                raise ValueError(
                    f"Data passed as '{k}' to '{py_function.name}' is stored "
                    f"on '{v.computer.label}', which is not directly accessible "
                    f"from '{computer.label}'."
                )
            if k in mapped_kwargs:
                template = mapped_kwarg_folder_template
            else:
                template = kwarg_file_template
            calcinfo.remote_symlink_list.append(
                (computer.uuid, v.pickle_path, template.format(k))
            )

        separate_kwargs = [k for k in remote_kwargs if k not in mapped_kwargs]

        # Add the '.common' subpackage as a package called 'common'.
        # This can therefore be used directly from the script.
        common_package_folder = folder.get_subfolder("common", create=True)
        for filename, contents in common.package_module_contents():
            with common_package_folder.open(filename, "w") as f:
                f.write(contents)

        # TODO: factor this out
        script = textwrap.dedent(
            f"""\
            import functools
            import operator
            import os
            import sys
            import cloudpickle

            import common

            # hard-coded to 1 job per map element for now
            element_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

            def tails(seq):
                while seq:
                    seq = seq[1:]
                    yield seq

            def make_strides(shape):
                return tuple(functools.reduce(operator.mul, s, 1) for s in tails(shape))

            mapspec = common.MapSpec.from_string("{self.metadata.options.mapspec}")
            kwarg_shapes = {mapped_kwarg_shapes}
            map_shape = mapspec.shape(kwarg_shapes)
            output_key = mapspec.output_key(map_shape, element_id)
            input_keys = {{
                k: v[0] if len(v) == 1 else v
                for k, v in mapspec.input_keys(map_shape, element_id).items()
            }}

            # Define paths for I/O

            function_file = "{function_file}"
            mapped_kwargs = {spec.parameters}
            mapped_literal_kwargs = {list(mapped_literal_kwargs.keys())}
            separate_kwargs = {separate_kwargs}

            kwarg_file_template = "{kwarg_file_template}"
            mapped_kwarg_folder_template = "{mapped_kwarg_folder_template}"

            remaining_kwargs_file = "{remaining_kwargs_file}"
            exceptions_folder = "{exceptions_folder}"
            return_value_folders = {return_value_folders}
            assert return_value_folders

            # Load code

            func = common.load(function_file)

            # Load kwargs

            kwargs = dict()
            # TODO: hard-code this when we switch to a Jinja template
            # TODO: parallel load using a threadpool
            for pname in separate_kwargs:
                kwargs[pname] = common.load(kwarg_file_template.format(pname))
            for pname in mapped_kwargs:
                if pname in mapped_literal_kwargs:
                    values = common.load(kwarg_file_template.format(pname))
                else:
                    values = common.FileBasedObjectArray(
                        mapped_kwarg_folder_template.format(pname),
                        shape=kwarg_shapes[pname],
                    )
                kwargs[pname] = values[input_keys[pname]]
            if os.path.exists(remaining_kwargs_file):
                kwargs.update(common.load(remaining_kwargs_file))

            # Execute

            try:
                return_values = func(**kwargs)
            except Exception as e:
                exceptions = common.FileBasedObjectArray(
                    exceptions_folder, shape=map_shape
                )
                exceptions.dump(output_key, e)
                sys.exit(1)

            # Output

            if len(return_value_folders) == 1:
                return_values = (return_values,)

            for r, f in zip(return_values, return_value_folders):
                output_array = common.FileBasedObjectArray(f, shape=map_shape)
                output_array.dump(output_key, r)
        """
        )

        with folder.open("__in__.py", "w", encoding="utf8") as handle:
            handle.write(script)
        codeinfo.stdin_name = "__in__.py"

        return calcinfo


@aiida.engine.calcfunction
def merge_remote_arrays(**kwargs: PyRemoteArray) -> PyRemoteArray:
    """Merge several remote arrays into a single array.

    This is most commonly used for combining the results of
    several PyMapJobs, where each job only produced a subset of
    the results (e.g. some tasks failed).

    Parameters
    ----------
    **kwargs
        The arrays to merge. The arrays will be merged in the same
        order as 'kwargs' (i.e. lexicographically by key).

    Raises
    ------
    ValueError
        If the input arrays are not on the same computer.
        If the input arrays are not the same shape
    """
    arrays = [kwargs[k] for k in sorted(kwargs.keys())]

    computer, *other_computers = (x.computer for x in arrays)
    if any(computer.uuid != x.uuid for x in other_computers):
        raise ValueError("Need to be on same computer")

    shape, *other_shapes = (x.shape for x in arrays)
    if any(shape != x for x in other_shapes):
        raise ValueError("Arrays need to be same shape")

    output_array = PyRemoteArray(
        computer=computer,
        shape=shape,
        filename_template=common.array.filename_template,
    )

    with computer.get_transport() as transport:
        f = create_remote_folder(transport, computer.get_workdir(), output_array.uuid)
        for arr in arrays:
            array_files = os.path.join(arr.get_attribute("remote_path"), "*")
            transport.copy(array_files, f, recursive=False)

    output_array.attributes["remote_path"] = f
    return output_array


def create_remote_folder(transport, workdir_template, uuid):
    """Create a folder in the Aiida working directory on a remote computer.

    Params
    ------
    transport
        A transport to the remote computer.
    workdir_template
        Template string for the Aiida working directory on the computer.
        Must expect a 'username' argument.
    uuid
        A UUID uniquely identifying the remote folder. This will be
        combined with 'workdir_template' to provide a sharded folder
        structure.
    """
    path = workdir_template.format(username=transport.whoami())
    # Create a sharded path, e.g. 'ab1234ef...' -> 'ab/12/34ef...'.
    for segment in (uuid[:2], uuid[2:4], uuid[4:]):
        path = os.path.join(path, segment)
        transport.mkdir(path, ignore_existing=True)
    return path


def num_mapjob_tasks(p: aiida.orm.ProcessNode) -> int:
    """Return the number of tasks that will be executed by a mapjob."""
    mapspec = common.MapSpec.from_string(p.get_option("mapspec"))
    mapped_kwargs = {
        k: v for k, v in p.inputs.kwargs.items() if k in mapspec.parameters
    }
    return np.sum(~expected_mask(mapspec, mapped_kwargs))


def expected_mask(mapspec: common.MapSpec, inputs: dict[str, Any]) -> np.ndarray:
    """Return the result mask that one should expect, given a MapSpec and inputs.

    When executing a PyMapJob over inputs that have a mask applied, we expect the
    output to be masked also. This function returns the expected mask.

    Parameters
    ----------
    mapspec
        MapSpec that determines how inputs should be combined.
    inputs
        Inputs to map over
    """
    kwarg_shapes = toolz.valmap(array_shape, inputs)
    kwarg_masks = toolz.valmap(array_mask, inputs)
    # This will raise an exception if the shapes are incompatible.
    map_shape = mapspec.shape(kwarg_shapes)
    map_size = np.prod(map_shape)

    # We only want to run tasks for _unmasked_ map elements.
    # Additionally, instead of a task array specified like "0,1,2,...",
    # we want to group tasks into 'runs': "0-30,35-38,...".
    def is_masked(i):
        return any(
            kwarg_masks[k][v] for k, v in mapspec.input_keys(map_shape, i).items()
        )

    return np.array([is_masked(x) for x in range(map_size)]).reshape(map_shape)


def array_job_spec(mapspec: common.MapSpec, inputs: dict[str, Any]) -> str:
    """Return a job-array task specification, given a MapSpec and inputs.

    Parameters
    ----------
    mapspec
        MapSpec that determines how inputs should be combined.
    inputs
        Inputs to map over
    """
    # We only want tasks in the array job corresponding to the _unmasked_
    # elements in the map.
    unmasked_elements = ~expected_mask(mapspec, inputs).reshape(-1)
    return array_job_spec_from_booleans(unmasked_elements)


def array_job_spec_from_booleans(should_run_task: Sequence[bool]) -> str:
    """Return a job-array task specification, given a sequence of booleans.

    If element 'i' in the sequence is 'True', then task 'i' will be included
    in the job array spec

    Examples
    --------
    >>> array_job_spec_from_booleans([False, True, True, True, False, True])
    "1-3,5"
    """
    return ",".join(
        str(start) if start == stop else f"{start}-{stop}"
        for start, stop in _group_runs(should_run_task)
    )


def _group_runs(s: Sequence[bool]):
    """Yield (start, stop) pairs for runs of 'True' in 's'.

    Examples
    --------
    >>> list(_group_runs([True, True, True]))
    [(0,2)]
    >>> list(_group_runs(
    ...     [False, True, True, True, False, False, True, False, True, True]
    ... )
    ...
    [(1,3), (6, 6), (8,9)]
    """
    prev_unmasked = False
    start = None
    for i, unmasked in enumerate(s):
        if unmasked and not prev_unmasked:
            start = i
        if prev_unmasked and not unmasked:
            assert start is not None
            yield (start, i - 1)
            start = None
        prev_unmasked = unmasked

    if prev_unmasked and start is not None:
        yield (start, i)


def all_equal(seq):
    """Return True iff all elements of the input are equal."""
    fst, *rest = seq
    if not rest:
        return True
    return all(r == fst for r in rest)
