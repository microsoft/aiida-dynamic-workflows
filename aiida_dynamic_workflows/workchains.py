# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from collections import defaultdict
from typing import Any, Dict, Optional

from aiida.engine import WorkChain, append_, if_, while_
import aiida.orm
import numpy as np
import toolz

from . import common
from .calculations import (
    PyCalcJob,
    PyMapJob,
    array_job_spec_from_booleans,
    expected_mask,
    merge_remote_arrays,
)


# Subclass needed for "option" getters/setters, so that a WorkChain
# can transparently wrap a CalcJob.
class WorkChainNode(aiida.orm.WorkChainNode):
    """ORM class for nodes representing the execution of a WorkChain."""

    def get_option(self, name: str) -> Optional[Any]:
        """Return the value of an option that was set for this CalcJobNode."""
        return self.get_attribute(name, None)

    def set_option(self, name: str, value: Any) -> None:
        """Set an option to the given value."""
        self.set_attribute(name, value)

    def get_options(self) -> Dict[str, Any]:
        """Return the dictionary of options set for this CalcJobNode."""
        options = {}
        for name in self.process_class.spec_options.keys():
            value = self.get_option(name)
            if value is not None:
                options[name] = value

        return options

    def set_options(self, options: Dict[str, Any]) -> None:
        """Set the options for this CalcJobNode."""
        for name, value in options.items():
            self.set_option(name, value)


# Hack to make this new node type use the Aiida logger.
# This is important so that WorkChains that use this node type also
# use the Aiida logger.
WorkChainNode._logger = aiida.orm.WorkChainNode._logger


class RestartedPyMapJob(WorkChain):
    """Workchain that resubmits a PyMapJob until all the tasks are complete.

    Tasks in the PyMapJob that succeeded on previous runs will not be resubmitted.
    """

    _node_class = WorkChainNode

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)
        spec.expose_inputs(PyMapJob)
        spec.expose_outputs(PyMapJob, include=["return_values", "exception"])
        spec.input(
            "metadata.options.max_restarts",
            valid_type=int,
            default=5,
            help=(
                "Maximum number of iterations the work chain will "
                "restart the process to finish successfully."
            ),
        )
        spec.exit_code(
            410,
            "MAXIMUM_RESTARTS_EXCEEDED",
            message="The maximum number of restarts was exceeded.",
        )

        spec.outline(
            cls.setup,
            while_(cls.should_run)(cls.run_mapjob, cls.inspect_result),
            if_(cls.was_restarted)(cls.merge_arrays, cls.extract_merged_arrays).else_(
                cls.pass_through_arrays
            ),
            cls.output,
        )

    def setup(self):  # noqa: D102
        self.report("Setting up")

        mapspec = common.MapSpec.from_string(self.inputs.metadata.options.mapspec)
        mapped_inputs = {
            k: v for k, v in self.inputs.kwargs.items() if k in mapspec.parameters
        }

        self.ctx.required_mask = expected_mask(mapspec, mapped_inputs)
        self.ctx.total_output_mask = np.full_like(self.ctx.required_mask, True)

        self.ctx.job_shape = self.ctx.required_mask.shape
        self.ctx.total_num_tasks = np.sum(~self.ctx.required_mask)

        self.ctx.iteration = 0
        self.ctx.launched_mapjobs = []

    @property
    def n_tasks_remaining(self) -> int:
        """Return the number of tasks that remain to be run."""
        return self.ctx.total_num_tasks - np.sum(~self.ctx.total_output_mask)

    @property
    def remaining_task_array(self) -> np.ndarray:
        """Return a boolean array indicating which tasks still need to be run."""
        return np.logical_xor(self.ctx.required_mask, self.ctx.total_output_mask)

    @property
    def has_all_results(self) -> bool:
        """Return True iff all the necessary outputs are present."""
        return np.all(self.ctx.total_output_mask == self.ctx.required_mask)

    def should_run(self):  # noqa: D102
        return (
            not self.has_all_results
            and self.ctx.iteration < self.inputs.metadata.options.max_restarts
        )

    def run_mapjob(self):  # noqa: D102
        # Run failed elements only, using custom
        # Slurm parameters: -A 1,3-10,20%24
        self.ctx.iteration += 1

        self.report(f"Running MapJob for {self.n_tasks_remaining} tasks")

        inputs = self.exposed_inputs(PyMapJob)

        # Modify "metadata.options.custom_scheduler_commands" so that the
        # correct tasks in the Slurm Job Array are run.
        # NOTE: This assumes we are running on Slurm
        options = inputs["metadata"]["options"]
        csc = options.custom_scheduler_commands
        # Remove the existing Array Job specification
        commands = [x for x in csc.split("\n") if "--array" not in x]
        # Add an updated Array Job specification
        task_spec = array_job_spec_from_booleans(self.remaining_task_array.reshape(-1))
        max_concurrent_jobs = (
            options.cores_per_machine * options.max_concurrent_machines
        )
        commands.append(f"#SBATCH --array={task_spec}%{max_concurrent_jobs}")
        inputs = toolz.assoc_in(
            inputs,
            ("metadata", "options", "custom_scheduler_commands"),
            "\n".join(commands),
        )

        # "max_restarts" does not apply to PyMapJobs
        del inputs["metadata"]["options"]["max_restarts"]

        fut = self.submit(PyMapJob, **inputs)
        return self.to_context(launched_mapjobs=append_(fut))

    def inspect_result(self):  # noqa: D102
        self.report("Inspecting result")

        job = self.ctx.launched_mapjobs[-1]

        m = result_mask(job, self.ctx.job_shape)
        self.ctx.total_output_mask[~m] = False

        self.report(
            f"{np.sum(~m)} tasks succeeded, "
            f"{self.n_tasks_remaining} / {self.ctx.total_num_tasks} remaining"
        )

    def was_restarted(self):  # noqa: D102
        return self.ctx.iteration > 1

    def merge_arrays(self):  # noqa: D102
        self.report(f"Gathering arrays from {self.ctx.iteration} mapjobs.")
        assert self.ctx.iteration > 1

        exception_arrays = []
        return_value_arrays = defaultdict(list)
        for j in self.ctx.launched_mapjobs:
            if "exception" in j.outputs:
                exception_arrays.append(j.outputs.exception)
            if "return_values" in j.outputs:
                for k, v in j.outputs.return_values.items():
                    return_value_arrays[k].append(v)

        # 'merge_remote_array' must take **kwargs (this is a limitation of Aiida), so
        # we convert a list of inputs into a dictionary with keys 'x0', 'x1' etc.
        def list_to_dict(lst):
            return {f"x{i}": x for i, x in enumerate(lst)}

        context_update = dict()

        # TODO: switch 'runner.run_get_node' to 'submit' once WorkChain.submit
        #       allows CalcFunctions (it should already; this appears to be a
        #       bug in Aiida).

        if exception_arrays:
            r = self.runner.run_get_node(
                merge_remote_arrays,
                **list_to_dict(exception_arrays),
            )
            context_update["exception"] = r.node

        for k, arrays in return_value_arrays.items():
            r = self.runner.run_get_node(
                merge_remote_arrays,
                **list_to_dict(arrays),
            )
            context_update[f"return_values.{k}"] = r.node

        return self.to_context(**context_update)

    def extract_merged_arrays(self):  # noqa: D102
        if "exception" in self.ctx:
            self.ctx.exception = self.ctx.exception.outputs.result
        if "return_values" in self.ctx:
            for k, v in self.ctx.return_values.items():
                self.ctx.return_values[k] = v.outputs.result

    def pass_through_arrays(self):  # noqa: D102
        self.report("Passing through results from single mapjob")
        assert self.ctx.iteration == 1
        (job,) = self.ctx.launched_mapjobs
        if "exception" in job.outputs:
            self.ctx.exception = job.outputs.exception
        if "return_values" in job.outputs:
            for k, v in job.outputs.return_values.items():
                self.ctx[f"return_values.{k}"] = v

    def output(self):  # noqa: D102
        self.report("Setting outputs")
        if "exception" in self.ctx:
            self.out("exception", self.ctx.exception)
        for k, v in self.ctx.items():
            if k.startswith("return_values"):
                self.out(k, v)

        max_restarts = self.inputs.metadata.options.max_restarts
        if not self.has_all_results and self.ctx.iteration >= max_restarts:
            self.report(f"Restarted the maximum number of times {max_restarts}")
            return self.exit_codes.MAXIMUM_RESTARTS_EXCEEDED


def result_mask(job, expected_shape) -> np.ndarray:
    """Return the result mask for a PyMapJob that potentially has multiple outputs."""
    if "return_values" not in job.outputs:
        return np.full(expected_shape, True)
    rvs = job.outputs.return_values
    masks = [getattr(rvs, x).mask for x in rvs]
    if len(masks) == 1:
        return masks[0]
    else:
        # If for some reason one of the outputs is missing elements (i.e. the
        # mask value is True) then we need to re-run the corresponding task.
        return np.logical_or(*masks)


class RestartedPyCalcJob(WorkChain):
    """Workchain that resubmits a PyCalcJob until it succeeds."""

    _node_class = WorkChainNode

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)
        spec.expose_inputs(PyCalcJob)
        spec.expose_outputs(PyCalcJob, include=["return_values", "exception"])
        spec.input(
            "metadata.options.max_restarts",
            valid_type=int,
            default=5,
            help=(
                "Maximum number of iterations the work chain will "
                "restart the process to finish successfully."
            ),
        )
        spec.exit_code(
            410,
            "MAXIMUM_RESTARTS_EXCEEDED",
            message="The maximum number of restarts was exceeded.",
        )
        spec.exit_code(
            411,
            "CHILD_PROCESS_EXCEPTED",
            message="The child process excepted.",
        )
        spec.outline(
            cls.setup,
            while_(cls.should_run)(cls.run_calcjob, cls.inspect_result),
            cls.output,
        )

    def setup(self):  # noqa: D102
        self.ctx.iteration = 0
        self.ctx.function_name = self.inputs.func.name
        self.ctx.children = []
        self.ctx.is_finished = False

    def should_run(self):  # noqa: D102
        return (
            not self.ctx.is_finished
            and self.ctx.iteration < self.inputs.metadata.options.max_restarts
        )

    def run_calcjob(self):  # noqa: D102
        self.ctx.iteration += 1
        inputs = self.exposed_inputs(PyCalcJob)
        del inputs["metadata"]["options"]["max_restarts"]
        node = self.submit(PyCalcJob, **inputs)

        self.report(
            f"Launching {self.ctx.function_name}<{node.pk}> "
            f"iteration #{self.ctx.iteration}"
        )

        return self.to_context(children=append_(node))

    def inspect_result(self):  # noqa: D102
        node = self.ctx.children[-1]

        if node.is_excepted:
            self.report(f"{self.ctx.function_name}<{node.pk}> excepted; aborting")
            return self.exit_codes.CHILD_PROCESS_EXCEPTED

        self.ctx.is_finished = node.exit_status == 0

    def output(self):  # noqa: D102
        node = self.ctx.children[-1]
        label = f"{self.ctx.function_name}<{node.pk}>"

        self.out_many(self.exposed_outputs(node, PyCalcJob))

        max_restarts = self.inputs.metadata.options.max_restarts
        if not self.ctx.is_finished and self.ctx.iteration >= max_restarts:
            self.report(
                f"Reached the maximum number of iterations {max_restarts}: "
                f"last ran {label}"
            )
            return self.exit_codes.MAXIMUM_RESTARTS_EXCEEDED
        else:
            self.report(
                f"Succeeded after {self.ctx.iteration} submissions: "
                f"last ran {label}"
            )
