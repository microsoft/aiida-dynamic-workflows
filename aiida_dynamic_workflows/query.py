# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import datetime

import aiida.common
import aiida.engine
import aiida.orm

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
