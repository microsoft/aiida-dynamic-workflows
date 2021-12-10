# Running individual calculations with aiida-dynamic-workflows


This notebook shows how to define and run individual calculations with aiida-dynamic-workflows, and how to _manually chain the results_ from one calculation into the next one. Chaining individual calculations together in a _workflow_ will be shown in the next notebook.


### This example assumes you already have Aiida set up, as well as the relevant codes/computers


If that's not your case, check out the `example_computer_setup` directory.


### The imports


First things first we must import `aiida` and call `aiida.load_profile`.

This loads the default Aiida profile. Each Aiida profile has a separate database for storing calculations and data,
as well as separate daemons for submitting calculations.

```python
import aiida

aiida.load_profile()
aiida.__version__
```

Next we must import the plugin.

Additionally we call `ensure_daemon_restarted()` to ensure that the Aiida daemon has loaded the latest version of the plugin.
Failing to restart the daemon when aiida-dynamic-workflows is updated can give strange results, as the environment in the notebook and the environment on the daemon will differ. After a restart the daemon will continue processing any running calculations (so nothing will be lost).

```python
import aiida_dynamic_workflows as flows

flows.control.ensure_daemon_restarted()
flows.__version__
```

# First define the execution environment


We create an execution environment that uses the Conda environment `py39` on `my-cluster`, and will submit calculations to the `some-queue` queue.

```python
cluster_env = flows.engine.execution_environment(
    "py39",   # conda environment
    "my-cluster",  # computer name
    queue=("some-queue", 24),  # queue and num. cores per machine
)
```

We can also create an execution environment that uses the Conda environment on _this_ machine:

```python
local_env = flows.engine.execution_environment("py39", "localhost")
```

Let's use the cluster execution environment going forward.

```python
env = cluster_env
```

## Then define some functions to run

```python
@flows.step(returns="x_plus_y")
def add(x: int, y: int):
    return x + y
```

```python
@flows.step(returns="z")
def multiply(x: int, y: int) -> int:
    return x * y
```

### Can be used as ordinary Python functions

```python
add(1, 2)
```

```python
multiply(1, 2)
```

```python
multiply(add(3, 4), 5)
```

### But they are really objects in the Aiida data store

```python
add
```

## We can submit them using Aiida


We first build the calculation:

```python
z = flows.engine.apply(add, x=1, y=2)
z
```

We see that `engine.apply` produces a kind of specification for what to run.

This is not yet enough to be able to run the thing: we need to specify _where_ to run it.

We do this with the `on` method, which expects an execution environment:

```python
z.on(env)
```

Note that the specification returned from the `on` method now contains a `queue_name`, and a `code` (which includes a reference to the cluster to run on).


Finally we will actually run this specification.

Even though the notebook is blocked, execution of `add` is actually happening _on the cluster_.

```python
r = aiida.engine.run(z.on(env))
```

If the execution of the cell above is hanging for too long, you may want to drop to the command line and inspect the running processes, e.g. using via `verdi process list`.
The (verbose) daemon logs should be showing "copying file/folder" + Slurm-related stuff.

If you're having trouble with remote execution, feel free to continue through the rest of the tutorial on your local computer by setting `env=local_env`.

```python
%%time
r["return_values"]["x_plus_y"].fetch_value()
```

This is good for debugging, but typically you don't want to block the notebook.

Instead of `run` you can use `submit` to get a daemon worker to do the waiting for you:

```python
r_submitted = aiida.engine.submit(z.on(env))
```

```python
r_submitted
```

```python
print(flows.report.progress(r_submitted))
flows.report.graph(r_submitted)
```

Only a _reference to a file on the cluster_ is returned:

```python
remote_value = r_submitted.outputs.return_values.x_plus_y
remote_value
```

```python
remote_value.pickle_path
```

```python
%%time
remote_value.fetch_value()
```

## We can pass the output `PyRemoteData` as an _input_ to the next calculation

```python
r_pass_as_remote_value = aiida.engine.run(
    flows.engine
    .apply(multiply, x=remote_value, y=2)
    .on(env)
)
```

```python
actual_return_value = r_pass_as_remote_value["return_values"]["z"]
```

```python
%%time
actual_return_value.fetch_value()
```

## We can also do maps, which will make use of Slurm Job arrays

```python
import numpy as np

xs = np.arange(100).reshape(10, 10)
ys = np.arange(100, 200).reshape(10, 5, 2)
```

```python
z = (
    flows.engine
    .map_(add, "x[i, j], y[j, k, l] -> z[i, j, k, l]")
    .on(env, max_concurrent_machines=2)
)
```

```python
%%time
r_map = aiida.engine.submit(z.finalize(x=xs, y=ys))
```

```python
print(flows.report.progress(r_map))
flows.report.graph(r_map)
```

```python
remote_mapped_values = r_map.outputs.return_values.x_plus_y
remote_mapped_values.shape
```

Each element in the `map` is in its own Slurm job (in a single job array), _and they all write to separate files_.

```python
%%time
a = remote_mapped_values.fetch_value()
a
```

`.fetch_value()` uses the default Aiida transport, and so is quite inefficient for loading many files (as in this example)

Passing `local_files=True` is useful when the Aiida working directory on `my-cluster` is actually mounted locally on the machine where this notebook is running.

```python
remote_mapped_values.get_remote_path()
```

```python
%%time
a = remote_mapped_values.fetch_value(local_files=True)
a
```

The loading operation is, consequently, several times faster.


## We can use the output of _that_ map as the input to another, of course!

```python
chained_map = (
    flows.engine
    .map_(
        multiply,
        "x[i, j, k, l] -> z[i, j, k, l]",  # Now we only map over 'x'; 'y' is treated single value
        max_concurrent_machines=1,
    ).on(env)
    .finalize(x=remote_mapped_values, y=5)
)
```

```python
chained_map_job = aiida.engine.submit(chained_map)
```

```python
print(flows.report.progress(chained_map_job))
flows.report.graph(chained_map_job)
```

```python
rv = chained_map_job.outputs.return_values.z
print(rv.shape)
%time rv.fetch_value(local_files=True)
```

## And then perform a reduction

```python
@flows.step
def reduce(xs: "FileBasedObjectArray"):
    return np.sum(xs.to_array())
```

```python
r = aiida.engine.submit(flows.engine.apply(reduce, xs=rv).on(env))
```

```python
print(flows.report.progress(r))
flows.report.graph(r)
```

```python
r.outputs.return_values._return_value.fetch_value()
```

# Defining the resource requirements for functions


You can specify that functions need a certain amount of resources to run by passing a `resources` dictionary to `step`.


Currently only `memory` and `cores` may be specified; these are passed to Slurm using the `--mem` and `--cpus-per-task` flags.

For example, the following function declares that it requires 6 cores to run, and a total of `25GB` of memory (for the whole thing, not per core).


A single instance of the function will run on this allocation, so we may use whatever method we wish to distribute work over the cores. In this example we are using `loky` to perform a simple map-reduce, but you could also use, e.g. an OpenMP-enabled BLAS to distribute a matrix computation over the cores.

```python
import loky
import time

@flows.step(returns=("z", "elapsed_time"), resources=dict(memory="25GB", cores=6))
def f_on_several_cores(xs: list) -> list:

    def go(x):
        time.sleep(5)
        return x ** 2

    with loky.ProcessPoolExecutor(6) as ex:
        start = time.time()
        r = sum(ex.map(go, xs))
        return r, f"execution time: {time.time() - start:.2f}s"
```

```python
r = aiida.engine.submit(
    flows.engine.apply(
        f_on_several_cores, xs=list(range(18))
    ).on(cluster_env)
)
```

```python
print(flows.report.progress(r))
flows.report.graph(r)
```

```python
r.outputs.return_values.z.fetch_value()
```

```python
r.outputs.return_values.elapsed_time.fetch_value()
```
