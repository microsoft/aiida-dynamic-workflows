# Dynamic workflows

This notebook shows how to compose several steps into a workflow and launch them all at once.

Contrast this to [01-calculations.md](./01-calculations.md), where we waited until calculations were finished before passing their data to the next calculation.


First we do the usual imports and define an execution environment

```python
from dataclasses import dataclass
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

## Step definitions


Next we define a bunch of individual "steps" from Python functions.

as we saw in [01-calculations.md](./01-calculations.md), this will save the pickled function in the Aiida database

```python
from aiida_dynamic_workflows import step
```

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
    return Mesh(geometry, mesh_size), Mesh(geometry, coarse_mesh_size)


@step(returns="materials")
def make_materials(geo: Geometry) -> MatData:
    time.sleep(5)  # do some work
    return Materials(geometry, ["a", "b", "c"])


@step(returns="electrostatics")
def run_electrostatics(
    mesh: MeshData, materials: Materials, V_left: float, V_right: float
) -> Electrostatics:
    time.sleep(10)  # do some work
    return Electrostatics(mesh, materials, [V_left, V_right])

@step(returns="charge")
def get_charge(electrostatics: Electrostatics) -> float:
    # obviously not actually the charge; but we should return _some_ number that
    # is "derived" from the electrostatics.
    return sum(electrostatics.voltages)
```

This final step is a little special.

As we shall see in a couple cell's time this step will be used on the output `get_charge`, which will be "mapped" over its inputs.

As a consequence, `average_charge` will be passed a reference to an "array" of values, where each value in the array is actually stored in a separate file on disk, hence the strance type signature.

```python
@step(returns="average_charge")
def average_charge(charge: "FileBasedObjectArray") -> float:
    # .to_array() is a bit dumb; it loads in _all_ the data at once, but
    # this is the simplest way, and in this example the data is not so large.
    return np.mean(charge.to_array())
```

## Composing workflows


Here we compose up 2 "workflows": `model_flow` and `electrostatics_flow`:

```python
from aiida_dynamic_workflows.workflow import first, concurrently, map_, new_workflow

model_flow = (
    new_workflow(name="model_flow")
    .then(make_geometry)
    .then(
        # These 2 steps will be done at the same time
        concurrently(make_mesh, make_mat_data)
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
```

We see that `electrostatics_flow` makes use of the `map_` function, which takes the step to execute, as well as a specification for how to map the inputs to the outputs.

In the above example we see that `electrostatics_flow` expects `V_left` and `V_right` to be 1D arrays, and it will `run_electrostatics` for each pair of values in these two arrays (an "outer product"), producing a 2D array of values.

The next step takes each of the elements in this 2D array and runs `get_charge` on them.

The final step of `electrostatics_flow` (`average_charge`) takes the _whole 2D `charge` array_ and produces a single value.


We can inspect what parameters and what outputs are produced by each flow:

```python
model_flow.parameters, model_flow.all_outputs
```

```python
electrostatics_flow.parameters, electrostatics_flow.all_outputs
```

Note that the `mat_data` and `mesh` outputs from `model_flow` "line up" with parameters of the same name of `electrostatics_flow`.

This enables us to `join` the two flows together:

```python
total_flow = (
    new_workflow(name="total_electrostatics")
    .join(model_flow)
    .join(electrostatics_flow)
    .returning("electrostatics", average_charge="avg_electrostatic_charge")
)
```

Invoking `returning` allows us to declare which of all the outputs should be considered "return values" of the workflow:

```python
total_flow.returns
```

This is purely a convenience; all outputs produced by `total_flow` will be inspectable.


We can finally visualize the workflow with `.visualize()`:

```python
total_flow.visualize(as_png=True)
```

Ovals represent **data** and rectangles represent **calculations**.

**grey** ovals represent _inputs_, while **white** ovals represent "intermediate" data.

Any **red** rectangles indicate "map" calculations. **red** ovals represent data that is being mapped over / produced by a "map" step.


## Running the workflow


Firstly we create a dictionary of all the inputs required by `total_flow`:

```python
total_flow.parameters
```

```python
inputs = dict(
    mesh_size=0.01,
    V_left=np.linspace(0, 2, 10),
    V_right=np.linspace(-0.5, 0.5, 20),
    x=0.1,
    y=0.2,
    coarse_mesh_size=0.05,
)
```

then we combine the workflow and the inputs into a specification that Aiida can run:

```python
ready = flows.workflow.build(
    total_flow.on(cluster_env),
    **inputs,
)
```

Note that similarly to single calculations, the workflow has an `on` method that can be used to specify where the calculations in the workflow should be run.


Finally we submit the workflow to the Aiida daemon

```python
running_flow = aiida.engine.submit(ready)
```

## Seeing what's happening


We can print a progress report of what's going on:

```python
print(flows.report.progress(running_flow))
```

And visualize the workflow graph:

```python
flows.report.graph(running_flow)
```

### If you restart your notebook


As soon as you `submit`, your workflow run is recorded in the Aiida database, so even if you restart your notebook you will not "lose" the running workflow.

You can use `running_workflows()` to get a summary of the workflows that are currently running:

```python
print(flows.report.running_workflows())
```

You can also get a summary of all the workflows started recently, e.g.:

```python
print(flows.report.recent_workflows(days=2))  # All workflows started in the last 2 days.
```

## Viewing results


Once the workflow has completed we can get the returned values by inspecting `outputs.return_values`:

```python
running_flow.outputs.return_values
```

Note that to get an inspectable value back we use `fetch_value()`, which pulls the cloudpickle blob from the cluster filesystem and loads it:

```python
%%time
running_flow.outputs.return_values.avg_electrostatic_charge.fetch_value()
```

We can also inspect any intermediate results by loading the appropriate data:

```python
%%time
running_flow.called[-2].outputs.return_values.charge.fetch_value(local_files=True)[:2, :2]
```

## Viewing anything else


We can always load any object that is stored in the database by querying for it's "primary key" or "UUID".

For example, if we wanted the database node corresponding to the step `make_geometry` from the above run, we could:

```python
## NB: change the "5269" to the "primary key" of the "make_geometry" step
##     You can get this information from the call-graph above.
executed_geometry_step = aiida.orm.load_node(5269)
```

We can get, for example, the output from `sacct` from the completed job:

```python
executed_geometry_step.get_detailed_job_info()
```

Or the contents of `stdout` and `stderr` from the job:

```python
executed_geometry_step.get_scheduler_stdout()
```

In a pinch we can also get the directory on the cluster where the job ran.
We can use this to manually inspect input/output files for sanity.

```python
executed_geometry_step.get_remote_workdir()
```

## Inspecting "sample plans" for results


Often, given the result of a simulation we will want to be able to see the parameters that produced it.

For example, the above workflow produces an intermediate result `charges`, and we might want to know what values of the inputs `x`, `y`, `V_left` etc. correspond to each values in the `charges` array.

We can query this using `input_samples`:

```python
import pandas as pd

charges = running_flow.called[-2].outputs.return_values.charge

df = pd.DataFrame(flows.input_samples(charges))

df
```

We see that we can feed the output into `pd.DataFrame` to get a dataframe of samples.

Even though `charges` is a 2D array:

```python
charges.shape
```

The samples are still presented as a (1D) dataframe.

The rows of the dataframe are ordered in the same way as a _flattened_ `charges`.

We can add another column to the dataframe so that the result is reported along with the inputs:

```python
df_with_results = df.assign(charge=charges.fetch_value(local_files=True).reshape(-1))
df_with_results
```
