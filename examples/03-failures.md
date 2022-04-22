# Handling failures


This notebook demonstrates how workflows can be built to handle common failure modes:

1. Persistent errors (e.g. a few samples in the sample plan are ill-defined)
2. Transient errors (e.g. meshing failed due to some random failure)

To explore this we will take the workflows developed in [02-workflows.md](./02-workflows.md) and make a few modifications.


First we do the usual imports and define an execution environment

```python
from dataclasses import dataclass
import random
import time

import numpy as np
import toolz
```

```python
import aiida
aiida.load_profile()

aiida.__version__
```

```python
import aiida_dynamic_workflows as flows
from aiida_dynamic_workflows import step

flows.control.ensure_daemon_restarted()
flows.__version__
```

```python
cluster_env = flows.engine.execution_environment(
    "py39",   # conda environment
    "my-cluster",  # computer name
    queue=("some-queue", 24),  # queue and num. cores per machine
)
```

## Defining the steps and workflows


This is copied verbatim from [02-workflows.md](./02-workflows.md).

In principle we could put this in a separate module, but this won't quite work until cloudpickle gets [this new feature](https://github.com/cloudpipe/cloudpickle/pull/417).

```python
@dataclass(frozen=True)
class Geometry:
    x : float
    y : float

@dataclass(frozen=True)
class Mesh:
    geometry : Geometry
    mesh_size : float

@dataclass(frozen=True)
class Materials:
    geometry: Geometry
    materials: list[str]

@dataclass(frozen=True)
class Electrostatics:
    mesh: Mesh
    materials: Materials
    voltages: list[float]
```

```python
@step(returns="geo")
def make_geometry(x: float, y: float) -> Geometry:
    time.sleep(5)  # do some work
    return Geometry(x, y)


@step(returns=("mesh", "coarse_mesh"))
def make_mesh(
    geo: Geometry,
    mesh_size: float,
    coarse_mesh_size: float,
) -> tuple[Mesh, Mesh]:
    time.sleep(5)  # do some work
    return Mesh(geo, mesh_size), Mesh(geo, coarse_mesh_size)


@step(returns="materials")
def make_materials(geo: Geometry) -> Materials:
    time.sleep(5)  # do some work
    return Materials(geo, ["a", "b", "c"])


@step(returns="electrostatics")
def run_electrostatics(
    mesh: Mesh, materials: Materials, V_left: float, V_right: float
) -> Electrostatics:
    time.sleep(10)  # do some work
    return Electrostatics(mesh, materials, [V_left, V_right])


@step(returns="charge")
def get_charge(electrostatics: Electrostatics) -> float:
    # obviously not actually the charge; but we should return _some_ number that
    # is "derived" from the electrostatics.
    return sum(electrostatics.voltages)


@step(returns="average_charge")
def average_charge(charge: "FileBasedObjectArray") -> float:
    # .to_array() is a bit dumb; it loads in _all_ the data at once, but
    # this is the simplest way, and in this example the data is not so large.
    return np.mean(charge.to_array())
```

```python
from aiida_dynamic_workflows.workflow import first, concurrently, map_, new_workflow

model_flow = (
    new_workflow(name="model_flow")
    .then(make_geometry)
    .then(
        # These 2 steps will be done at the same time
        concurrently(make_mesh, make_materials)
    )
)

electrostatics_flow = (
    new_workflow(name="electrostatics_flow")
    .then(
        map_(
            run_electrostatics,
            "V_left[a], V_right[b] -> electrostatics[a, b]",
        )
    ).then(
        map_(
            get_charge,
            "electrostatics[i, j] -> charge[i, j]"
        )
    ).then(average_charge)
)

total_flow = (
    new_workflow(name="total_electrostatics")
    .join(model_flow)
    .join(electrostatics_flow)
    .returning("electrostatics", average_charge="avg_electrostatic_charge")
)
```

## Modifying steps


Now we make new meshing and electrostatics steps with the following modifications:

+ If the `mesh_error` parameter is True, then the meshing step always raises a `ValueError`.
+ If `V_left` or `V_right` is outside the bounds set by `V_limits` then the electrostatics step raises a `ValueError`.
+ The charge-extracting step will randomly fail with probability `failure_probability`.

```python
# Inside the modified steps we should only reference the raw Python function, _not_ the
# object in the Aiida database (which we will not be able to resolve, given that the code
# will eventually be run in a job on the cluster).
original_make_mesh = make_mesh.callable
original_electrostatics = run_electrostatics.callable
original_get_charge = get_charge.callable


@flows.step(returns=("mesh", "coarse_mesh"))
def modified_make_mesh(geo, mesh_size, coarse_mesh_size, mesh_error):
    if mesh_error:
        raise ValueError("Meshing step failed")
    else:
        return original_make_mesh(geo, mesh_size, coarse_mesh_size)


@flows.step(returns="electrostatics")
def modified_electrostatics(geo, mesh, materials, V_left, V_right, V_limits: tuple):
    a, b = V_limits
    if not (a < V_left < b and a < V_right < b):
        raise ValueError(f"Voltages ({V_left}, {V_right}) out of acceptable range {V_limits}")
    else:
        return original_electrostatics(mesh, materials, V_left, V_right)

@flows.step(returns="charge")
def modified_get_charge(electrostatics, failure_probability):
    import random
    if random.random() < failure_probability:
        raise ValueError("Randomly failed!")
    else:
        return original_get_charge(electrostatics)
```

## Modifying workflows


We now use the `replace_steps` method of the `total_flow` defined in `basic_electrostatics`.

This allows us to easily replace the `make_mesh` and `run_electrostatics` steps with their modified versions that we defined above:

```python
new_flow = (
    total_flow
    .rename("total_flow_with_failures")
    .replace_steps({
        make_mesh: modified_make_mesh,
        run_electrostatics: modified_electrostatics,
        get_charge: modified_get_charge,
    })
)

new_flow.visualize(as_png=True)
```

## Running the workflow


Let's first run the workflow with `mesh_error=True`, and see what happens:

```python
inputs = dict(
    mesh_size=0.015,
    V_left=np.linspace(0, 1, 10),
    V_right=np.linspace(-0.5, 0.5, 20),
    x=0.15,
    y=0.25,
    coarse_mesh_size=0.05,
    # Extra parameters; needed for the modified steps
    V_limits=[-0.4, 0.4],
    failure_probability=0.2,
    mesh_error=True,
)
```

```python
running_workflow = aiida.engine.submit(flows.workflow.build(
    new_flow.on(cluster_env),
    **inputs,
))
```


```python
print(flows.report.progress(running_workflow))
flows.report.graph(running_workflow, as_png=True)
```

We see that the `make_geometry` and `make_mat_data` steps completed successfully (Exit Code 0), but `modified_make_mesh` failed with exit code 401.

We can use `flows.report.log` to figure out what happened:

```python
modified_mesh_calc = running_workflow.called[1]
print(flows.report.log(modified_mesh_calc))
```

We see that `User code raised an Exception`, and that `modified_make_mesh` returned an `exception` output.

We can inspect the exception to see what happened:

```python
modified_mesh_calc.outputs.exception.fetch_value()
```

We can get more insight into what happened by printing the log from the _workflow_:

```python
print(flows.report.log(running_workflow))
```

We see that the workflow detected the failure of `modified_make_mesh`.

It tried to carry on anyway, but `modified_electrostatics` requires `mesh`, _so the step is skipped_.
The remaining steps are also skipped for similar reasons.


**The default workflow behaviour is to try to execute all steps, skipping steps for which there is not sufficient input.**


## Persistent errors in Map elements


Now we will flip the `mesh_error` flag so that the mesh step completes successfully.

Note, however, that some elements of the `modified_electrostatics` map will raise an exception
because `V_left` or `V_right` are outside of the specified limits.

We will see how the workflow handles such an error condition.


We can easily make a small modification to the parameters before resubmitting by using `get_builder_restart()` on the previously executed workflow:

```python
no_mesh_error = running_workflow.get_builder_restart()
no_mesh_error.kwargs.mesh_error = aiida.orm.to_aiida_type(False)
```

Then submit the workflow with the modified parameters:

```python
running_workflow2 = aiida.engine.submit(no_mesh_error)
```

```python
print(flows.report.progress(running_workflow2))
flows.report.graph(running_workflow2, as_png=True)
```

We see that the `modified_electrostatics` step returned exit code 401, indicating that our code raised a Python exception; we also see that the `exception` output was produced.


However, we also see that an `electrostatics` output was produced, despite the non-zero exit code.


Let's load in the exception to see what is going on:

```python
electrostatics_step = running_workflow2.called[3]
```

```python
exceptions = electrostatics_step.outputs.exception.fetch_value()
```

```python
exceptions[:4, :4]
```

As `modified_electrostatics` was run as a `PyMapJob`, `exceptions` is a _masked array_, that contains the exception raised by the given element in the map (and masked for elements that did not raise an exception).


Similarly, `electrostatics` will be a masked array, with the map elements that raised an exception _masked out_:

```python
electrostatics_array = electrostatics_step.outputs.return_values.electrostatics
electrostatics_array.mask[:4, :4]
```

Nevertheless, the workflow can continue, even with this "partial" output.


The downstream `PyMapJob` that runs `get_charge` detects that the input(s) are "masked" and only runs `get_charge` for the data that actually exists.

We can "see" this by inspecting the `--array` specification that was passed to Slurm by the job:

```python
get_charge_step = running_workflow2.called[4]
print(get_charge_step.attributes["custom_scheduler_commands"])
```

We see that only array elements 2-17, 22-37 etc. are submitted, as the other elements of `electrostatics` are missing.


## Mitigating transient errors


Our modified `get_charge` step randomly fails with a certain probability: a "transient" error.

We can see this, as the array of charges output by the step does not have the same mask as the electrostatics that were used as input:

```python
charge_array = get_charge_step.outputs.return_values.charge

np.sum(charge_array.mask != electrostatics_array.mask)
```

A simple way to mitigate transient errors is to specify that steps should be restarted.

For MapJobs we can do this by specifying `max_restarts` to `map_`:

```python
electrostatics_flow_with_restarts = (
    first(
        map_(
            run_electrostatics,
            "V_left[i], V_right[j] -> electrostatics[i, j]",
        )
    ).then(
        map_(
            get_charge,
            "electrostatics[i, j] -> charge[i, j]",
            max_restarts=5,  # <-- specify the max number of restarts here
        )
    ).then(average_charge)
)

```

Alternatively, we can use the `with_restarts` method of an existing workflow to add restarts to the named steps:

```python
total_flow_with_restarts = (
    new_flow
    .rename(name="flow_with_restarts")
    .with_restarts({modified_get_charge: 5})
)
```

```python
running_workflow_with_restarts = aiida.engine.submit(flows.workflow.build(
    total_flow_with_restarts.on(cluster_env),
    **toolz.assoc(inputs, "mesh_error", False),
))
```

```python
print(flows.report.progress(running_workflow_with_restarts))
flows.report.graph(running_workflow_with_restarts, as_png=True)
```

We see that after `modified_electrostatics` there is a "`RestartedPyMapJob`" that sequentially launches several `PyMapJob`s that each run `modified_get_charge`.

We can see what is going on by printing the log of this `RestartedMapJob`:

```python
restarted_mapjob = running_workflow_with_restarts.called[-2]
print(flows.report.log(restarted_mapjob))
```

The first time `modified_get_charge` is run, it is run over 64 tasks, the number of unmasked `electrostatics` in the input:

```python
electrostatics_mapjob = running_workflow_with_restarts.called[-3]
np.sum(~electrostatics_mapjob.outputs.return_values.electrostatics.mask)
```

This run results in a few failures, so the failed tasks are submitted again and so on until all 64 results have been obtained (or the maximum number of restarts has been exceeded).

```python
for j in restarted_mapjob.called:
    if 'PyMapJob' not in j.process_type:
        continue
    print(j.get_option("custom_scheduler_commands"))
```

The `RestartedPyMapJob` then merges the outputs from the different runs together into a single array.


Finally, we see that even the `average_charge` step completed successfully, as it was written in such a way that it transparently handles masked arrays:

```python
running_workflow_with_restarts.outputs.return_values.avg_electrostatic_charge.fetch_value()
```
