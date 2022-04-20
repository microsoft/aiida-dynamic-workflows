# Deleting data


This notebook provides guidance on how to delete data that you no longer need.


As usual we first import AiiDA and aiida_dynamic_workflows:

```python
import aiida
aiida.load_profile()

aiida.__version__
```

```python
import aiida_dynamic_workflows
import aiida_dynamic_workflows.workflow
import aiida_dynamic_workflows.report

aiida_dynamic_workflows.control.ensure_daemon_restarted()
aiida_dynamic_workflows.__version__
```

Next we define a utility function for watching processes as they evolve:

```python
import datetime
import time

import ipywidgets as widgets

def wait(p, timeout=2):
    out = widgets.Output()
    while not p.is_terminated:
        out.clear_output(wait=True)
        print(f"last updated @ {datetime.datetime.now()}")
        print(aiida_dynamic_workflows.report.progress(p))
        time.sleep(timeout)
    out.clear_output(wait=True)
    print(f"Finished @ {p.mtime}")
    print(aiida_dynamic_workflows.report.progress(p))
```

Now we create a small workflow, for illustrative purposes:

```python
@aiida_dynamic_workflows.step(returns="c")
def add(a, b):
    return a + b

@aiida_dynamic_workflows.step(returns="z")
def mul(c, y):
    return c * y


workflow = (
    aiida_dynamic_workflows.workflow
    .new_workflow("test")
    .then(add)
    .then(mul)
    .returning("c", "z")
)

local = aiida_dynamic_workflows.engine.execution_environment("py39", "localhost")
```

```python
from functools import partial
import random

rand = partial(random.randint, 0, 1000)

flow = aiida_dynamic_workflows.workflow.build(workflow.on(local), a=rand(), b=rand(), y=rand())
```

And we run it:

```python
run = aiida.engine.submit(flow)
wait(run)
```

## Deleting nodes from the AiiDA database


Let's say that that you wish to delete the two runs from the database.

AiiDA provides the following functionality of deleting the nodes from the database:

```python
marked_pks, are_deleted = aiida.tools.delete_nodes([run.id])
```

This function returns two things:
1. The first is a set containing the IDs of the nodes that were deleted (or not)
2. The second is a boolean value that is True if the nodes were actually deleted


The first thing to notice is that `marked_pks` contains many more nodes than the ones we explicitly marked for deletion:

```python
len(marked_pks)
```

This is because AiiDA tries to maintain the integrity of the provenance graph.

If we delete the Workflow nodes then the calculation nodes that were created by the workflow, as well as all the produced data nodes, must also be deleted.


We see that the above invocation did not actually delete anything:

```python
are_deleted
```

This is a safety feature; to have `delete_nodes` actually delete, we must pass `dry_run=False`:

```python
marked_pks, are_deleted = aiida.tools.delete_nodes([run.id], dry_run=False)
```

```python
are_deleted
```

## Deleting the remote data


Deleting the nodes from the AiiDA database is a good first step, however a typical workflow has all the intermediate data stored as `PyRemoteData` and `PyRemoteArray`. This means that the actual data is stored in a file on some remote filesystem (cluster NFS); only a _reference_ to the file is stored in the AiiDA database.

Once we have deleted the nodes from the database we also need to ensure we remove the data from the remote filsystem, to avoid filling up our disk with unwanted data.

Pyiida provides the following tools for achieving this.


#### `aiida_dynamic_workflows.query.unreferenced_work_directories`


This function returns any CalcJob working directories that are unreference by any RemoteData in _any profile_ in the AiiDA database.

It expects a path that will be used as a root directory for the search (i.e. only paths under this root will be returned).

To help with this there is `computer_work_directory`, which returns the CalcJob working directory root for the named computer:

```python
from aiida_dynamic_workflows.query import unreferenced_work_directories, computer_work_directory
```

```python
unreferenced_paths = unreferenced_work_directories(computer_work_directory("localhost"))
```

We see that there are a few paths that are unreferenced by the AiiDA database:

```python
unreferenced_paths
```

As these paths are not referenced by any RemoteData in the AiiDA database, they may safely be removed without invalidating the AiiDA provenance graph.

As these are just plain old paths, they may be removed by any method you wish (e.g. export to a file `to-remove.txt` and run `cat to-remove.txt | parallel rm -r {}`)
However, aiida_dynamic_workflows has a useful tool for just this:

```python
aiida_dynamic_workflows.utils.parallel_rmtree(unreferenced_paths)
```

After having removed these paths, we should see that there are no more unreferenced work directories:

```python
unreferenced_work_directories(computer_work_directory("localhost"))
```

## Preserving cached data


Let's run the calculation twice, again:

```python
original_run = aiida.engine.submit(flow)
wait(original_run)
```

```python
cached_run = aiida.engine.submit(flow)
wait(cached_run)
```

We see that the calculations in the second run are created from the calculations in the first run:

```python
for c in original_run.called:
    print(c.inputs.func.name, c.uuid)
```

Indeed, we see that the data nodes for the two runs point to the same location on the remote storage:

```python
original_data_paths = {k: v.get_remote_path() for k, v in original_run.outputs.return_values.items()}
print(original_data_paths)
```

```python
cached_data_paths = {k: v.get_remote_path() for k, v in cached_run.outputs.return_values.items()}
print(cached_data_paths)
```

```python
assert original_data_paths == cached_data_paths
```

If we delete the original run only we therefore need to keep the remote data around, as it is still referenced by the cached run.

**Let's verify that this is what happens.**


First let's check that there is not any unreferenced data already:

```python
assert not unreferenced_work_directories(computer_work_directory("localhost"))
```

and let's check that removing the original run is not going to remove any nodes associated with the cached run:

```python
marked_pks, are_deleted = aiida.tools.delete_nodes([original_run.id])
for n in marked_pks:
    print(repr(aiida.orm.load_node(n)))
```

```python
aiida_dynamic_workflows.report.graph(cached_run, as_png=True)
```

We indeed see that there is no overlap; only the nodes from `original_run` are going to be deleted.


Let's actually delete them:

```python
marked_pks, are_deleted = aiida.tools.delete_nodes([original_run.id], dry_run=False)
assert are_deleted
```

Let's now check that, indeed, the data is still referenced:

```python
assert not unreferenced_work_directories(computer_work_directory("localhost"))
```

Success!


If we now delete the cached run:

```python
_, are_deleted = aiida.tools.delete_nodes([cached_run.id], dry_run=False)
assert are_deleted
```

We should see that the data is now unreferenced:

```python
unrefd = unreferenced_work_directories(computer_work_directory("localhost"))
print(unrefd)
assert unrefd
assert {str(x) for x in unrefd} == set(cached_data_paths.values())
```

And so we can safely delete them:

```python
aiida_dynamic_workflows.utils.parallel_rmtree(unrefd)
```
