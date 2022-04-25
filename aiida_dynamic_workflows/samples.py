# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import itertools
from typing import Dict, Iterable, Optional, Tuple

import aiida.orm
import toolz

from .calculations import PyCalcJob, PyMapJob
from .common import MapSpec
from .data import PyRemoteArray, from_aiida_type


def input_samples(result: PyRemoteArray) -> Iterable[dict]:
    """Return an iterable of samples, given a result from a PyMapJob.

    Parameters
    ----------
    result
        The array resulting from the execution of a PyMapJob.

    Returns
    -------
    An iterable of dictionaries, ordered as 'result' (flattened, if
    'result' is a >1D array). Each dictionary has the same keys (the
    names of the parameters that produced 'result').

    Examples
    --------
    >>> import pandas as pd
    >>> # In the following we assume 'charge' is a PyRemoteArray output from a PyMapJob.
    >>> df = pd.DataFrame(input_samples(charge))
    >>> # Add a 'charge' column showing the result associated with each sample.
    >>> df.assign(charge=charge.reshape(-1))
    """
    if result.creator is None:
        raise ValueError(
            "Cannot generate sample plan from data that was not produced from a CalcJob"
        )
    job = result.creator
    if not issubclass(job.process_class, PyMapJob):
        raise TypeError("Expected data that was produced from a MapJob")
    output_axes = MapSpec.from_string(job.attributes["mapspec"]).output.axes
    sp = _parameter_spec(result)

    consts = {k: from_aiida_type(v) for k, (v, axes) in sp.items() if axes is None}
    mapped = {
        k: (from_aiida_type(v), axes) for k, (v, axes) in sp.items() if axes is not None
    }

    # This could be done more efficiently if we return instead a dictionary of arrays.

    for el in itertools.product(*map(range, result.shape)):
        el = dict(zip(output_axes, el))
        d = {k: v[tuple(el[ax] for ax in axes)] for k, (v, axes) in mapped.items()}
        yield toolz.merge(consts, d)


def _parameter_spec(result: aiida.orm.Data, axes: Optional[tuple[str]] = None) -> dict:
    """Return a dictionary specifying the parameters that produced a given 'result'.

    Parameters
    ----------
    result
        Data produced from a PyCalcJob or PyMapJob.
    axes
        Labels for each axis of 'result', used to rename input axis labels.

    Returns
    -------
    Dictionary mapping parameter names (strings) to pairs: (Aiida node, axis names).
    """
    job = result.creator
    job_type = job.process_class

    if not issubclass(job_type, PyCalcJob):
        raise TypeError(f"Don't know what to do with {job_type}")

    if issubclass(job_type, PyMapJob):
        mapspec = MapSpec.from_string(job.attributes["mapspec"])
        if axes:
            assert len(axes) == len(mapspec.output.axes)
            translation = dict(zip(mapspec.output.axes, axes))
        else:
            translation = dict()
        input_axes = {
            spec.name: [translation.get(ax, ax) for ax in spec.axes]
            for spec in mapspec.inputs
        }
    else:
        input_axes = dict()
        assert axes is None

    kwargs = job.inputs.kwargs if hasattr(job.inputs, "kwargs") else {}
    # Inputs that were _not_ created by another CalcJob are the parameters we seek.
    parameters = {k: (v, input_axes.get(k)) for k, v in kwargs.items() if not v.creator}
    # Inputs that _were_ created by another Calcjob need to have
    # _their_ inputs inspected, in turn.
    other_inputs = [(v, input_axes.get(k)) for k, v in kwargs.items() if v.creator]
    upstream_params = [_parameter_spec(v, ax) for v, ax in other_inputs]

    return toolz.merge(parameters, *upstream_params)
