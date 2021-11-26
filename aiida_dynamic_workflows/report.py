# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from collections import Counter
import textwrap
from typing import Union

from IPython.display import Image
import aiida.cmdline.utils.common as cmd
from aiida.cmdline.utils.query.formatting import format_relative_time
import aiida.orm
from aiida.tools.visualization import Graph
import graphviz

from . import query
from .calculations import PyCalcJob, PyMapJob, num_mapjob_tasks
from .data import PyRemoteArray, PyRemoteData
from .utils import render_png
from .workchains import RestartedPyCalcJob, RestartedPyMapJob
from .workflow import PyWorkChain

__all__ = [
    "log",
    "graph",
    "progress",
    "running_workflows",
    "recent_workflows",
]


ProcessType = Union[aiida.orm.ProcessNode, int, str]


def log(proc: ProcessType) -> str:
    """Return the output of 'verdi process report' for the given process.

    Parameters
    ----------
    proc
        The Aiida node for the process, or a numeric ID, or a UUID.
    """
    proc = _ensure_process_node(proc)
    if isinstance(proc, aiida.orm.CalcJobNode):
        return cmd.get_calcjob_report(proc)
    elif isinstance(proc, aiida.orm.WorkChainNode):
        return cmd.get_workchain_report(proc, levelname="REPORT")
    elif isinstance(proc, (aiida.orm.CalcFunctionNode, aiida.orm.WorkFunctionNode)):
        return cmd.get_process_function_report(proc)
    else:
        raise TypeError(f"Cannot get report for processes of type '{type(proc)}'")


def graph(
    proc: ProcessType, size=(20, 20), as_png=False
) -> Union[graphviz.Digraph, Image]:
    """Return a graph visualization of a calculation or workflow.

    Parameters
    ----------
    proc
        The Aiida node for the process, or a numeric ID, or a UUID.
    """
    proc = _ensure_process_node(proc)
    graph = Graph(
        graph_attr={"size": ",".join(map(str, size)), "rankdir": "LR"},
        node_sublabel_fn=_node_sublabel,
    )
    graph.recurse_descendants(proc, include_process_inputs=True)
    if as_png:
        return render_png(graph.graphviz)
    return graph.graphviz


def progress(proc: ProcessType) -> str:
    """Return a progress report of the given calculation or workflow.

    Parameters
    ----------
    proc
        The Aiida node for the process, or a numeric ID, or a UUID.
    """
    proc = _ensure_process_node(proc)
    if isinstance(proc, aiida.orm.CalcJobNode):
        return _calcjob_progress(proc)
    elif isinstance(proc, aiida.orm.WorkChainNode):
        if issubclass(proc.process_class, PyWorkChain):
            return _workflow_progress(proc)
        elif issubclass(proc.process_class, (RestartedPyCalcJob, RestartedPyMapJob)):
            return _restarted_calcjob_progress(proc)
    elif isinstance(proc, (aiida.orm.CalcFunctionNode, aiida.orm.WorkFunctionNode)):
        return _function_progress(proc)
    else:
        raise TypeError(
            "Cannot get a progress report for processes of type '{type(proc)}'"
        )


def running_workflows() -> str:
    """Return a progress report of the running workflows."""
    r = _flatten(query.running_workflows().iterall())
    return "\n\n".join(map(_workflow_progress, r))


def recent_workflows(days: int = 0, hours: int = 0, minutes: int = 0) -> str:
    """Return a progress report of all workflows that were started recently.

    This also includes workflows that are already complete.

    Parameters
    ----------
    days, hours, minutes
        Any workflows started more recently than this many days/minutes/hours
        will be included in the result of the query.
    """
    r = _flatten(query.recent_workflows(**locals()).iterall())
    return "\n\n".join(map(_workflow_progress, r))


def _flatten(xs):
    for ys in xs:
        yield from ys


def _workflow_progress(p: aiida.orm.WorkChainNode) -> str:
    assert issubclass(p.process_class, PyWorkChain)
    lines = [
        # This is a _single_ output line
        f"{p.label or '<No label>'} (pk: {p.id}) "
        f"[{_process_status(p)}, created {format_relative_time(p.ctime)}]"
    ]
    for c in p.called:
        lines.append(textwrap.indent(progress(c), "    "))

    return "\n".join(lines)


def _restarted_calcjob_progress(p: aiida.orm.WorkChainNode) -> str:
    assert issubclass(p.process_class, (RestartedPyCalcJob, RestartedPyMapJob))
    lines = [
        f"with_restarts({p.get_option('max_restarts')}) "
        f"(pk: {p.id}) [{_process_status(p)}]"
    ]
    for i, c in enumerate(p.called, 1):
        if c.label == p.label:
            # The launched process is the payload that we are running with restarts
            s = f"attempt {i}: {progress(c)}"
        else:
            # Some post-processing (for RestartedPyMapJob)
            s = progress(c)
        lines.append(textwrap.indent(s, "    "))

    return "\n".join(lines)


def _calcjob_progress(p: aiida.orm.CalcJobNode) -> str:
    assert issubclass(p.process_class, PyCalcJob)
    s = p.get_state() or p.process_state

    # Show more detailed info while we're waiting for the Slurm job.
    if s == aiida.common.CalcJobState.WITHSCHEDULER:
        sections = [
            f"created {format_relative_time(p.ctime)}",
        ]
        if p.get_scheduler_state():
            sections.append(f"{p.get_scheduler_state().value} job {p.get_job_id()}")

        # Show total number of tasks and states of remaining tasks in mapjobs.
        job_states = _slurm_job_states(p)
        if job_states:
            if issubclass(p.process_class, PyMapJob):
                task_counts = Counter(job_states)
                task_states = ", ".join(f"{k}: {v}" for k, v in task_counts.items())
                task_summary = f"{sum(task_counts.values())} / {num_mapjob_tasks(p)}"
                sections.extend(
                    [
                        f"remaining tasks ({task_summary})",
                        f"task states: {task_states}",
                    ]
                )
            else:
                sections.append(f"job state: {job_states[0]}")
        msg = ", ".join(sections)
    else:
        msg = _process_status(p)

    return f"{p.label} (pk: {p.id}) [{msg}]"


def _process_status(p: aiida.orm.ProcessNode) -> str:

    generic_failure = (
        f"failed, run 'aiida_dynamic_workflows.report.log({p.id})' "
        "for more information"
    )

    if p.is_finished and not p.is_finished_ok:
        # 's.value' is "finished", even if the process finished with a non-zero exit
        # code. We prefer the more informative 'failed' + next steps.
        msg = generic_failure
    elif p.is_killed:
        # Process was killed: 'process_status' includes the reason why.
        msg = f"killed, {p.process_status}"
    elif p.is_excepted:
        # Process failed, and the error occured in the Aiida layers
        msg = generic_failure
    elif p.is_created_from_cache:
        msg = (
            f"{p.process_state.value} "
            f"(created from cache, uuid: {p.get_cache_source()})"
        )
    elif p.is_finished_ok:
        msg = "success"
    else:
        try:
            # Calcjobs have 'get_state', which gives more fine-grained information
            msg = p.get_state().value
        except AttributeError:
            msg = p.process_state.value

    return msg


def _function_progress(
    p: Union[aiida.orm.CalcFunctionNode, aiida.orm.WorkFunctionNode]
) -> str:
    return f"{p.label} (pk: {p.id}) [{p.process_state.value}]"


def _slurm_job_states(process):
    info = process.get_last_job_info()
    if not info:
        return []
    else:
        return [x[1] for x in info.raw_data]


def _ensure_process_node(
    node_or_id: Union[aiida.orm.ProcessNode, int, str]
) -> aiida.orm.ProcessNode:
    if isinstance(node_or_id, aiida.orm.ProcessNode):
        return node_or_id
    else:
        return aiida.orm.load_node(node_or_id)


def _node_sublabel(node):
    if isinstance(node, aiida.orm.CalcJobNode) and issubclass(
        node.process_class, PyCalcJob
    ):
        labels = [f"function: {node.inputs.func.name}"]
        if state := node.get_state():
            labels.append(f"State: {state.value}")
        if (job_id := node.get_job_id()) and (state := node.get_scheduler_state()):
            labels.append(f"Job: {job_id} ({state.value})")
        if node.exit_status is not None:
            labels.append(f"Exit Code: {node.exit_status}")
        if node.exception:
            labels.append("excepted")
        return "\n".join(labels)
    elif isinstance(node, (PyRemoteData, PyRemoteArray)):
        try:
            create_link = node.get_incoming().one()
        except Exception:
            return aiida.tools.visualization.graph.default_node_sublabels(node)
        if create_link.link_label.startswith("return_values"):
            return create_link.link_label.split("__")[1]
        else:
            return create_link.link_label
    else:
        return aiida.tools.visualization.graph.default_node_sublabels(node)
