# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import datetime
import itertools
import multiprocessing
from pathlib import Path

import aiida.common
import aiida.engine
import aiida.manage.configuration
import aiida.orm

from .data import PyRemoteArray, PyRemoteData
from .workflow import PyWorkChain


def workflows() -> aiida.orm.QueryBuilder:
    """Return an Aiida database query that will return all workflows."""
    q = aiida.orm.QueryBuilder()
    q.append(cls=PyWorkChain, tag="flow")
    q.order_by({"flow": [{"ctime": {"order": "desc"}}]})
    return q


def running_workflows() -> aiida.orm.QueryBuilder:
    """Return an Aiida database query that will return all running workflows."""
    r = workflows()
    r.add_filter(
        "flow",
        {
            "attributes.process_state": {
                "in": [
                    aiida.engine.ProcessState.RUNNING.value,
                    aiida.engine.ProcessState.WAITING.value,
                ],
            }
        },
    )
    return r


def recent_workflows(
    days: int = 0, hours: int = 0, minutes: int = 0
) -> aiida.orm.QueryBuilder:
    """Return an Aiida database query for all recently started workflows.

    Parameters
    ----------
    days, hours, minutes
        Any workflows started more recently than this many days/minutes/hours
        will be included in the result of the query.
    """
    delta = aiida.common.timezone.now() - datetime.timedelta(
        days=days, hours=hours, minutes=minutes
    )
    r = workflows()
    r.add_filter("flow", {"ctime": {">": delta}})
    return r


def remote_files(
    profile: str | None = None,
    root: str | Path | None = None,
) -> set[Path]:
    """Return the paths of all RemoteData for the given profile.

    Parameters
    ----------
    profile
        The profile name for which to return the UUIDs.
        If not provided, runs on the currently loaded profile.
    root
        If provided, return only sub-paths of this root path.

    Notes
    -----
    As Paths are returned without any information about what computer
    the path refers to, this function is only useful in environments
    where the Paths are globally unique.
    """
    if profile:
        aiida.load_profile(profile)

    # PyRemoteData and PyRemoteArray are not in the 'data.core.remote'
    # plugin path, so 'query.append' does not include them when querying
    # for 'aiida.orm.RemoteData', despite the fact that they do subclass it.
    remote_data = [aiida.orm.RemoteData, PyRemoteArray, PyRemoteData]

    query = aiida.orm.QueryBuilder()
    query.append(cls=remote_data, project="attributes.remote_path", tag="files")
    if root:
        root = Path(root).absolute()
        query.add_filter("files", {"attributes.remote_path": {"like": f"{root}%"}})

    return {Path(p) for p, in query.iterall()}


# Needs to be importable to be used with multiprocessing in 'referenced_remote_files'
def _run_on_q(f, q, *args):
    try:
        r = f(*args)
    except Exception as e:
        q.put(("error", e))
    else:
        q.put(("ok", r))


def referenced_remote_files(root: str | Path | None = None) -> set[Path]:
    """Return the paths of all RemoteData for all profiles.

    Parameters
    ----------
    root
        If provided, return only sub-paths of this root path.

    Notes
    -----
    As Paths are returned without any information about what computer
    the path refers to, this function is only useful in environments
    where the Paths are globally unique.
    """
    # Loading different AiiDA profiles requires starting a fresh Python interpreter.
    # For this reason we cannot use concurrent.futures, and must use bare
    # multiprocessing.
    # TODO: revisit whether this is necessary when AiiDA 2.0 is released
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    profiles = aiida.manage.configuration.get_config().profile_names
    procs = [
        ctx.Process(target=_run_on_q, args=(remote_files, q, p, root)) for p in profiles
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    results = [q.get() for _ in range(q.qsize())]
    if errors := [e for status, e in results if status != "ok"]:
        raise ValueError(f"One or more processes errored: {errors}")

    return set(itertools.chain.from_iterable(r for _, r in results))


def referenced_work_directories(root: str | Path) -> set[Path]:
    """Return all calcjob working directories referenced in the AiiDA database.

    Notes
    -----
    As Paths are returned without any information about what computer
    the path refers to, this function is only useful in environments
    where the Paths are globally unique.
    """
    root = Path(root).absolute()
    # aiiDA shards working directory paths like '/path/to/.aiida_run/ab/cd/1234-...'
    # so we add 3 subdirectories onto the root to get to the working directories.
    n = len(root.parts) + 3
    return {Path(*p.parts[:n]) for p in referenced_remote_files(root)}


def existing_work_directories(root: str | Path) -> set[Path]:
    """Return all calcjob working directories under 'root' that exist on disk.

    Notes
    -----
    As Paths are returned without any information about what computer
    the path refers to, this function is only useful in environments
    where the Paths are globally unique.

    Examples
    --------
    >>> work_directories("/path/to/my-user/.aiida_run")
    {PosixPath('/path/to/my-user/.aiida_run/00/24/ab.c2-899c-4106-8c8e-74638dbdd71c')}
    """
    root = Path(root).absolute()
    # aiiDA shards working directory paths like '/path/to/.aiida_run/ab/cd/1234-...'
    # so we add glob 3 subdirectories onto the root to get to the working directories.
    return {Path(p) for p in root.glob("*/*/*")}


def unreferenced_work_directories(root: str | Path) -> set[Path]:
    """Return all unreferenced calcjob working directories under 'root'.

    i.e. return all calcjob working directories that exist on disk, but are
    not referenced in the AiiDA database.

    Notes
    -----
    As Paths are returned without any information about what computer
    the path refers to, this function is only useful in environments
    where the Paths are globally unique.

    Examples
    --------
    >>> unreferenced_work_directories("/path/to/my-user/.aiida_run")
    {PosixPath('/path/to/my-user/.aiida_run/00/24/abc2-899c-4106-8c8e-74638dbdd71c')}
    """
    root = Path(root).absolute()

    return existing_work_directories(root) - referenced_work_directories(root)


def computer_work_directory(computer: str | aiida.orm.Computer) -> Path:
    """Return the work directory for 'computer'.

    Like 'computer.get_workdir()', except that '{username}' template
    parameters are replaced with actual usernames.

    Parameters
    ----------
    computer
        A Computer instance, or a computer label.
    """
    if not isinstance(computer, aiida.orm.Computer):
        computer = aiida.orm.load_computer(computer)

    with computer.get_transport() as t:
        return Path(computer.get_workdir().format(username=t.whoami()))
