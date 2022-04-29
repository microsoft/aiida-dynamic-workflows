# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import subprocess
import time
from typing import Optional, Union

from aiida import get_config_option
from aiida.cmdline.commands.cmd_process import process_kill, process_pause, process_play
from aiida.cmdline.utils import common, daemon, echo
from aiida.engine.daemon.client import get_daemon_client
from aiida.orm import ProcessNode, load_node


def kill(process: Union[ProcessNode, int, str], timeout: int = 5) -> bool:
    """Kill the specified process.

    Params
    ------
    process
        The process to kill.
    timeout
        Timeout (in seconds) to wait for confirmation that the process was killed.

    Returns
    -------
    True only if the process is now terminated.
    """
    process = _ensure_process_node(process)
    process_kill.callback([process], timeout=timeout, wait=True)
    return process.is_terminated


def pause(process: Union[ProcessNode, int, str], timeout: int = 5) -> bool:
    """Pause the specified process.

    Paused processes will not continue execution, and can be unpaused later.

    Params
    ------
    process
        The process to kill.
    timeout
        Timeout (in seconds) to wait for confirmation that the process was killed.

    Returns
    -------
    True only if the process is now paused.
    """
    process = _ensure_process_node(process)
    if process.is_terminated:
        raise RuntimeError("Cannot pause terminated process {process.pk}.")
    process_pause.callback([process], all_entries=False, timeout=timeout, wait=True)
    return process.paused


def unpause(process: Union[ProcessNode, int, str], timeout: int = 5) -> bool:
    """Unpause the specified process.

    Params
    ------
    process
        The process to kill.
    timeout
        Timeout (in seconds) to wait for confirmation that the process was killed.

    Returns
    -------
    True only if the process is now unpaused.
    """
    process = _ensure_process_node(process)
    if process.is_terminated:
        raise RuntimeError("Cannot unpause terminated process {process.pk}.")
    process_play.callback([process], all_entries=False, timeout=timeout, wait=True)
    return not process.paused


def ensure_daemon_stopped():
    """Stop the daemon (if it is running)."""
    client = get_daemon_client()

    if client.is_daemon_running:
        echo.echo("Stopping the daemon...", nl=False)
        response = client.stop_daemon(wait=True)
        retcode = daemon.print_client_response_status(response)
        if retcode:
            raise RuntimeError(f"Problem stopping Aiida daemon: {response['status']}")

    assert not client.is_daemon_running


def ensure_daemon_restarted(n_workers: Optional[int] = None):
    """Restart the daemon (if it is running), or start it (if it is stopped).

    Parameters
    ----------
    n_workers
        The number of daemon workers to start. If not provided, the default
        number of workers for this profile is used.

    Notes
    -----
    If the daemon is running this is equivalent to running
    'verdi daemon restart --reset', i.e. we fully restart the daemon, including
    the circus controller. This ensures that any changes in the environment are
    properly picked up by the daemon.
    """
    client = get_daemon_client()
    n_workers = n_workers or get_config_option("daemon.default_workers")

    if client.is_daemon_running:
        echo.echo("Stopping the daemon...", nl=False)
        response = client.stop_daemon(wait=True)
        retcode = daemon.print_client_response_status(response)
        if retcode:
            raise RuntimeError(f"Problem restarting Aiida daemon: {response['status']}")

    echo.echo("Starting the daemon...", nl=False)

    # We have to run this in a subprocess because it daemonizes, and we do not
    # want to daemonize _this_ process.
    command = [
        "verdi",
        "-p",
        client.profile.name,
        "daemon",
        "start-circus",
        str(n_workers),
    ]
    try:
        currenv = common.get_env_with_venv_bin()
        subprocess.check_output(command, env=currenv, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exception:
        echo.echo("FAILED", fg="red", bold=True)
        raise RuntimeError("Failed to start the daemon") from exception

    time.sleep(1)
    response = client.get_status()

    retcode = daemon.print_client_response_status(response)
    if retcode:
        raise RuntimeError(f"Problem starting Aiida daemon: {response['status']}")


def _ensure_process_node(node_or_id: Union[ProcessNode, int, str]) -> ProcessNode:
    if isinstance(node_or_id, ProcessNode):
        return node_or_id
    else:
        return load_node(node_or_id)
