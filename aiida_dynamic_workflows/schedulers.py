# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from collections.abc import Mapping
import datetime
from typing import List, Optional, T

from aiida.common.lang import type_check
from aiida.schedulers import JobInfo, JobState
from aiida.schedulers.plugins.slurm import SlurmScheduler
import toolz

__all__ = ["SlurmSchedulerWithJobArray"]


class SlurmSchedulerWithJobArray(SlurmScheduler):
    """A Slurm scheduler that reports only a single JobInfo for job arrays."""

    def _parse_joblist_output(self, retval, stdout, stderr):
        # Aiida assumes that there is a single job associated with each call
        # to 'sbatch', but this is not true in the case of job arrays.
        # In order to meet this requirement we merge the JobInfos for each job
        # in the array.
        return merge_job_arrays(super()._parse_joblist_output(retval, stdout, stderr))

    # Return only the necessary fields for 'parse_output' to do its job.
    # Our fat array jobs mean the response from 'sacct' can be pretty huge.
    _detailed_job_info_fields = [
        "JobID",
        "ExitCode",
        "State",
        "Reason",
        "CPUTime",
    ]

    def _get_detailed_job_info_command(self, job_id):
        fields = ",".join(self._detailed_job_info_fields)
        # --parsable2 separates fields with pipes, with no trailing pipe
        return f"sacct --format={fields} --parsable2 --jobs={job_id}"

    @classmethod
    def parse_detailed_job_info(cls, detailed_job_info):
        """Parse output from 'sacct', issued after the completion of the job."""
        type_check(detailed_job_info, dict)

        retval = detailed_job_info["retval"]
        if retval != 0:
            stderr = detailed_job_info["stderr"]
            raise ValueError(f"Error code {retval} returned by 'sacct': {stderr}")

        try:
            detailed_stdout = detailed_job_info["stdout"]
        except KeyError:
            raise ValueError(
                "the `detailed_job_info` does not contain the required key `stdout`."
            )

        type_check(detailed_stdout, str)

        lines = detailed_stdout.splitlines()

        try:
            fields, *job_infos = lines
        except IndexError:
            raise ValueError("`detailed_job_info.stdout` does not contain enough lines")
        fields = fields.split("|")

        if fields != cls._detailed_job_info_fields:
            raise ValueError(
                "Fields returned by 'sacct' do not match fields specified."
            )

        # Parse the individual job outputs
        job_infos = [dict(zip(fields, info.split("|"))) for info in job_infos]
        # Each job has a 'batch' entry also, which we ignore
        job_infos = [j for j in job_infos if not j["JobID"].endswith(".batch")]

        return job_infos

    def parse_output(self, detailed_job_info, stdout, stderr):
        """Parse output from 'sacct', issued after the completion of the job."""
        from aiida.engine import CalcJob

        job_infos = self.parse_detailed_job_info(detailed_job_info)

        # TODO: figure out how to return richer information to the calcjob, so
        #       that a workchain could in principle reschedule with only the
        #       failed jobs.
        if any(j["State"] == "OUT_OF_MEMORY" for j in job_infos):
            return CalcJob.exit_codes.ERROR_SCHEDULER_OUT_OF_MEMORY
        if any(j["State"] == "TIMEOUT" for j in job_infos):
            return CalcJob.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME


def merge_job_arrays(jobs: List[JobInfo]) -> List[JobInfo]:
    """Merge JobInfos from jobs in the same Slurm Array into a single JobInfo."""
    mergers = {
        "job_id": toolz.compose(job_array_id, toolz.first),
        "dispatch_time": min,
        "finish_time": toolz.compose(
            max, toolz.curried.map(with_default(datetime.datetime.min)),
        ),
        "job_state": total_job_state,
        "raw_data": toolz.identity,
    }

    job_array_id_from_info = toolz.compose(
        job_array_id, toolz.functoolz.attrgetter("job_id")
    )

    return [
        merge_with_functions(*jobs, mergers=mergers, factory=JobInfo)
        for jobs in toolz.groupby(job_array_id_from_info, jobs).values()
    ]


def total_job_state(states: List[JobState]) -> JobState:
    # Order is important here
    possible_states = [
        JobState.UNDETERMINED,
        JobState.RUNNING,
        JobState.SUSPENDED,
        JobState.QUEUED_HELD,
        JobState.QUEUED,
    ]
    for ps in possible_states:
        if any(state == ps for state in states):
            return ps

    if all(state == JobState.DONE for state in states):
        return JobState.DONE
    else:
        raise RuntimeError("Invalid state encountered")


def job_array_id(job_id: str) -> str:
    """Return the ID of the associated array job.

    If the provided job is not part of a job array then
    the job ID is returned.
    """
    return toolz.first(job_id.split("_"))


@toolz.curry
def with_default(default: T, v: Optional[T]) -> T:
    """Return 'v' if it is not 'None', otherwise return 'default'."""
    return default if v is None else v


def merge_with_functions(*dicts, mergers, factory=dict):
    """Merge 'dicts', using 'mergers'.

    Parameters
    ----------
    *dicts
        The dictionaries / mappings to merge
    mergers
        Mapping from keys in 'dicts' to functions. Each function
        accepts a list of values and returns a single value.
    factory
        Function that returns a new instance of the mapping
        type that we would like returned

    Examples
    --------
    >>> merge_with_functions(
    ...     {"a": 1, "b": 10, "c": "hello"},
    ...     {"a": 5, "b": 20, "c": "goodbye"},
    ...     mergers={"a": min, "b": max},
    ... )
    {"a": 1, "b": 20, "c": "goodbye"}
    """
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]

    result = factory()
    for d in dicts:
        for k, v in d.items():
            if k not in result:
                result[k] = [v]
            else:
                result[k].append(v)
    return toolz.itemmap(
        lambda kv: (kv[0], mergers.get(kv[0], toolz.last)(kv[1])), result, factory
    )
