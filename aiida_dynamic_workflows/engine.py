# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import os
import sys
from typing import Any, Callable

import aiida.engine
import aiida.orm
import toolz

from .calculations import PyCalcJob, PyMapJob, array_job_spec
from .common import MapSpec
from .data import PyFunction, ensure_aiida_type
from .workchains import RestartedPyCalcJob, RestartedPyMapJob

__all__ = ["apply", "map_"]


@dataclass(frozen=True)
class ExecutionEnvironment:
    """An execution environment in which to run a PyFunction as a PyCalcJob."""

    code_label: str
    computer_label: str
    queue: tuple[str, int] | None = None

    @property
    def code(self):
        return aiida.orm.load_code("@".join((self.code_label, self.computer_label)))

    @property
    def computer(self):
        return aiida.orm.load_computer(self.computer_label)


def code_from_conda_env(conda_env: str, computer_name: str) -> aiida.orm.Code:
    """Create AiiDA Code for python interpreter from conda environment."""
    c = aiida.orm.load_computer(computer_name)
    with c.get_transport() as t:
        username = t.whoami()
        try:
            conda_dir = c.get_property("conda_dir").format(username=username)
        except AttributeError:
            raise RuntimeError(f"'conda_dir' is not set for {computer_name}.")

        conda_initscript = os.path.join(conda_dir, "etc", "profile.d", "conda.sh")
        python_path = os.path.join(conda_dir, "envs", conda_env, "bin", "python")

        prepend_text = "\n".join(
            [f"source {conda_initscript}", f"conda activate {conda_env}"]
        )

        r, _stdout, stderr = t.exec_command_wait(prepend_text)

        if r != 0:
            raise RuntimeError(
                f"Failed to find Conda environment '{conda_env}' on '{computer_name}':"
                f"\n{stderr}"
            )

    code = aiida.orm.Code((c, python_path), label=conda_env)
    code.set_prepend_text(prepend_text)
    code.store()
    return code


def current_conda_environment() -> str:
    """Return current conda environment name."""
    # from https://stackoverflow.com/a/57716519/3447047
    return sys.exec_prefix.split(os.sep)[-1]


def execution_environment(conda_env: str | None, computer: str, queue=None):
    if conda_env is None:
        conda_env = current_conda_environment()
    code_id = "@".join([conda_env, computer])
    try:
        aiida.orm.load_code(code_id)
    except aiida.common.NotExistent:
        code = code_from_conda_env(conda_env, computer)
        code.store()

    if queue and (queue[0] not in get_queues(computer)):
        raise ValueError(f"Queue '{queue[0]}' does not exist on '{computer}'")

    return ExecutionEnvironment(conda_env, computer, queue)


def get_queues(computer_name) -> list[str]:
    """Return a list of valid queue names for the named computer."""
    computer = aiida.orm.load_computer(computer_name)
    with computer.get_transport() as t:
        command = "sinfo --summarize"
        retval, stdout, stderr = t.exec_command_wait(command)
        if retval != 0:
            raise RuntimeError(
                f"'{command}' failed on on '{computer_name}' "
                f"with exit code {retval}: {stderr}"
            )
        _, *lines = stdout.splitlines()
        return [line.split(" ")[0] for line in lines]


def local_current_execution_environment() -> ExecutionEnvironment:
    return execution_environment(None, "localhost")


class ProcessBuilder(aiida.engine.ProcessBuilder):
    """ProcessBuilder that is serializable."""

    def on(
        self, env: ExecutionEnvironment, max_concurrent_machines: int | None = None
    ) -> ProcessBuilder:
        """Return a new ProcessBuilder, setting it up for execution on 'env'."""
        r = copy.deepcopy(self)

        r.code = env.code

        if env.queue is not None:
            queue_name, cores_per_machine = env.queue
            r.metadata.options.queue_name = queue_name

        if issubclass(r.process_class, (PyMapJob, RestartedPyMapJob)):
            # NOTE: We are using a feature of the scheduler (Slurm in our case) to
            #       use array jobs. We could probably figure a way to do this with
            #       the 'direct' scheduler (GNU parallel or sth), but that is out
            #       of scope for now.
            if env.computer.scheduler_type != "dynamic_workflows.slurm":
                raise NotImplementedError(
                    "Mapping is currently only supported in an environment that "
                    f"supports Slurm array jobs, but {env.computer.label} is "
                    f" configured to use '{env.computer.scheduler_type}'."
                )

            if env.queue is None:
                raise ValueError(
                    "A queue specification (e.g. ('my-queue', 24) ) is required"
                )

            r.metadata.options.cores_per_machine = cores_per_machine

            if max_concurrent_machines is not None:
                r.metadata.options.max_concurrent_machines = max_concurrent_machines

        return r

    def finalize(self, **kwargs) -> ProcessBuilder:
        """Return a new ProcessBuilder, setting its 'kwargs' to those provided."""
        r = copy.deepcopy(self)
        r.kwargs = toolz.valmap(ensure_aiida_type, kwargs)

        opts = r.metadata.options

        custom_scheduler_commands = ["#SBATCH --requeue"]

        if issubclass(r.process_class, (PyMapJob, RestartedPyMapJob)):
            mapspec = MapSpec.from_string(opts.mapspec)
            mapped_kwargs = {
                k: v for k, v in r.kwargs.items() if k in mapspec.parameters
            }

            cores_per_job = opts.resources.get(
                "num_cores_per_mpiproc", 1
            ) * opts.resources.get("num_mpiprocs_per_machine", 1)
            jobs_per_machine = opts.cores_per_machine // cores_per_job
            max_concurrent_jobs = jobs_per_machine * opts.max_concurrent_machines

            task_spec = array_job_spec(mapspec, mapped_kwargs)
            # NOTE: This assumes that we are running on Slurm.
            custom_scheduler_commands.append(
                f"#SBATCH --array={task_spec}%{max_concurrent_jobs}"
            )

        opts.custom_scheduler_commands = "\n".join(custom_scheduler_commands)

        return r

    def with_restarts(self, max_restarts: int) -> ProcessBuilder:
        """Return a new builder for a RestartedPyCalcJob or RestartedPyMapJob."""
        if issubclass(self.process_class, (PyMapJob, RestartedPyMapJob)):
            r = ProcessBuilder(RestartedPyMapJob)
        elif issubclass(self.process_class, (PyCalcJob, RestartedPyCalcJob)):
            r = ProcessBuilder(RestartedPyCalcJob)
        else:
            raise TypeError(f"Do not know how to add restarts to {self.process_class}")
        _copy_builder_contents(to=r, frm=self)
        r.metadata.options.max_restarts = max_restarts
        return r

    # XXX: This is a complete hack to be able to serialize "Outline".
    #      We should think this through more carefully when we come to refactor.

    def __getstate__(self):
        def serialized_aiida_nodes(x):
            if isinstance(x, aiida.orm.Data):
                if not x.is_stored:
                    x.store()
                return _AiidaData(x.uuid)
            else:
                return x

        serialized_data = traverse_mapping(serialized_aiida_nodes, self._data)
        return self._process_class, serialized_data

    def __setstate__(self, state):
        process_class, serialized_data = state
        self.__init__(process_class)

        def deserialize_aiida_nodes(x):
            if isinstance(x, _AiidaData):
                return aiida.orm.load_node(x.uuid)
            else:
                return x

        deserialized_data = traverse_mapping(deserialize_aiida_nodes, serialized_data)

        for k, v in deserialized_data.items():
            if isinstance(v, Mapping):
                getattr(self, k)._update(v)
            else:
                setattr(self, k, v)


# XXX: This is part of the __getstate__/__setstate__ hack for our custom ProcessBuilder
@dataclass(frozen=True)
class _AiidaData:
    uuid: str


def _copy_builder_contents(
    to: aiida.engine.ProcessBuilderNamespace,
    frm: aiida.engine.ProcessBuilderNamespace,
):
    """Recursively copy the contents of 'frm' into 'to'.

    This mutates 'to'.
    """
    for k, v in frm.items():
        if isinstance(v, aiida.engine.ProcessBuilderNamespace):
            _copy_builder_contents(to[k], v)
        else:
            setattr(to, k, v)


def traverse_mapping(f: Callable[[Any], Any], d: Mapping):
    """Traverse a nested Mapping, applying 'f' to all non-mapping values."""
    return {
        k: traverse_mapping(f, v) if isinstance(v, Mapping) else f(v)
        for k, v in d.items()
    }


def apply(f: PyFunction, *, max_restarts: int = 1, **kwargs) -> ProcessBuilder:
    """Apply f to **kwargs as a PyCalcJob or RestartedPyCalcJob.

    Parameters
    ----------
    f
        The function to apply
    max_restarts
        The number of times to run 'f'. If >1 then a builder
        for a RestartedPyCalcJob is returned, otherwise
        a builder for a PyCalcJob is returned.
    **kwargs
        Keyword arguments to pass to 'f'. Will be converted
        to Aiida types using "aiida.orm.to_aiida_type" if
        not already a subtype of "aiida.orm.Data".
    """
    # TODO: check that 'f' applies cleanly to '**kwargs'
    if max_restarts > 1:
        builder = ProcessBuilder(RestartedPyCalcJob)
        builder.metadata.options.max_restarts = int(max_restarts)
    else:
        builder = ProcessBuilder(PyCalcJob)

    builder.func = f
    builder.metadata.label = f.name
    if kwargs:
        builder.kwargs = toolz.valmap(ensure_aiida_type, kwargs)
    if f.resources:
        _apply_pyfunction_resources(f.resources, builder.metadata.options)
    return builder


def apply_some(f: PyFunction, *, max_restarts: int = 1, **kwargs) -> ProcessBuilder:
    """Apply f to **kwargs as a PyCalcJob or RestartedPyCalcJob.

    'kwargs' may contain _more_ inputs than what 'f' requires: extra
    inputs are ignored.

    Parameters
    ----------
    f
        The function to apply
    max_restarts
        The number of times to run 'f'. If >1 then a builder
        for a RestartedPyCalcJob is returned, otherwise
        a builder for a PyCalcJob is returned.
    **kwargs
        Keyword arguments to pass to 'f'. Will be converted
        to Aiida types using "aiida.orm.to_aiida_type" if
        not already a subtype of "aiida.orm.Data".
    """
    if max_restarts > 1:
        builder = ProcessBuilder(RestartedPyCalcJob)
        builder.metadata.options.max_restarts = int(max_restarts)
    else:
        builder = ProcessBuilder(PyCalcJob)

    builder.func = f
    builder.metadata.label = f.name
    relevant_kwargs = toolz.keyfilter(lambda k: k in f.parameters, kwargs)
    if relevant_kwargs:
        builder.kwargs = toolz.valmap(ensure_aiida_type, relevant_kwargs)
    if f.resources:
        _apply_pyfunction_resources(f.resources, builder.metadata.options)
    return builder


def map_(
    f: PyFunction,
    spec: str | MapSpec,
    *,
    max_concurrent_machines: int | None = None,
    max_restarts: int = 1,
    **kwargs,
) -> aiida.engine.ProcessBuilder:
    """Map 'f' over (a subset of) its inputs as a PyMapJob.

    Parameters
    ----------
    f
        Function to map over
    spec
        Specification for which parameters to map over, and how to map them.
    max_concurrent_machines
        The maximum number of machines to use concurrently.
    max_restarts
        The maximum number of times to restart the PyMapJob before returning
        a partial (masked) result and a non-zero exit code.
    **kwargs
        Keyword arguments to 'f'. Any arguments that are to be mapped over
        must by Aiida lists.

    Examples
    --------
    >>> from aiida.orm import List
    >>> import aiida_dynamic_workflows as flow
    >>>
    >>> f = flow.step(lambda x, y: x + y, returns="sum")
    >>>
    >>> # We can map over _all_ inputs
    >>> sums = flow.engine.map_(
    ...     f, "x[i], y[i] -> sum[i]", x=List([1, 2, 3]), y=List([4, 5, 6])
    ... )
    >>> # or we can map over a _subset_ of inputs
    >>> only_one = flow.engine.map_(f, "x[i] -> sum[i]", x=List([1, 2, 3]), y=5)
    >>> # or we can do an "outer product":
    >>> outer= flow.engine.map_(
    ...     f, "x[i], y[j] -> sum[i, j]", x=List([1, 2, 3]), y=List([4, 5, 6])
    ... )
    """
    if max_restarts > 1:
        builder = ProcessBuilder(RestartedPyMapJob)
        builder.metadata.options.max_restarts = int(max_restarts)
    else:
        builder = ProcessBuilder(PyMapJob)

    builder.func = f
    builder.metadata.label = f.name

    if isinstance(spec, str):
        spec = MapSpec.from_string(spec)
    elif not isinstance(spec, MapSpec):
        raise TypeError(f"Expected single string or MapSpec, got {spec}")
    if unknown_params := {x.name for x in spec.inputs} - set(f.parameters):
        raise ValueError(
            f"{f} cannot be mapped over parameters that "
            f"it does not take: {unknown_params}"
        )
    builder.metadata.options.mapspec = spec.to_string()

    if max_concurrent_machines is not None:
        builder.metadata.options.max_concurrent_machines = max_concurrent_machines

    if f.resources:
        _apply_pyfunction_resources(f.resources, builder.metadata.options)

    if not kwargs:
        return builder

    return builder.finalize(**kwargs)


def _apply_pyfunction_resources(
    resources: dict, options: aiida.engine.ProcessBuilderNamespace
) -> None:
    """Apply the resource specification in 'resources' to the CalcJob options 'options'.

    This mutates 'options'.
    """
    memory = resources.get("memory")
    if memory is not None:
        # The Aiida Slurm plugin erroneously uses the multiplyer "1024" when converting
        # to MegaBytes and passing to "--mem", so we must use it here also.
        multiplier = {"kB": 1, "MB": 1024, "GB": 1000 * 1024}
        amount, unit = memory[:-2], memory[-2:]
        options.max_memory_kb = int(amount) * multiplier[unit]

    cores = resources.get("cores")
    if cores is not None:
        # Re-assign the whole 'resources' input dict to avoid problems with
        # serialization (also, mutating it seems to change the 'resources' for
        # all other Builders, which is not good!).
        options.resources = toolz.assoc(
            options.resources, "num_cores_per_mpiproc", int(cores)
        )


def all_equal(seq):
    """Return True iff all elements of 'seq' are equal.

    Returns 'True' if the sequence contains 0 or 1 elements.
    """
    seq = list(seq)
    if len(seq) in (0, 1):
        return True
    fst, *rest = seq
    return all(r == fst for r in rest)
