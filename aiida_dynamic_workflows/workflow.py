# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import annotations

import abc
import copy
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import aiida.engine
import graphviz
import toolz

from . import common, engine
from .calculations import PyCalcJob, PyMapJob
from .data import PyFunction, PyOutline, ensure_aiida_type
from .utils import render_png

# TODO: this will all need to be refactored when we grok
#       Aiida's 'Process' and 'Port' concepts.


class Step(metaclass=abc.ABCMeta):
    """Abstract base class for steps."""

    pass


class Single(Step):
    """A single workflow step."""

    pass


class Action(Single):
    """Step that will be run with the current workchain passed as argument."""

    def do(self, workchain):
        """Do the action on the workchain."""
        pass


@dataclass(frozen=True)
class Concurrent(Step):
    """Step consisting of several concurrent steps."""

    steps: list[Step]


@dataclass(frozen=True)
class Sequential(Step):
    """Step consisting of several sequential steps."""

    steps: list[Step]


@dataclass(frozen=True)
class Process(Single):
    """Step consisting of a single Aiida Process."""

    builder: aiida.engine.ProcessBuilder
    parameters: tuple[str]
    returns: tuple[str]

    def __str__(self):
        kind = self.builder.process_class
        if issubclass(kind, PyCalcJob):
            func = self.builder.func
            return f"{kind.__name__}[{func.name}(pk: {func.pk})]"
        else:
            return kind.__name__


@dataclass(frozen=True)
class OutputAction(Action):
    """Action step that outputs values from the workflow context."""

    outputs: dict[str, str]

    def do(self, workchain):
        """Return the named outputs from this workflow."""
        for from_name, to_name in self.outputs.items():
            if from_name in workchain.ctx:
                workchain.out(f"return_values.{to_name}", workchain.ctx[from_name])
            else:
                workchain.report(
                    f"Failed to set output '{to_name}': '{from_name}' "
                    "does not exist on the workchain context (did "
                    "the step that produces this output fail?)"
                )


class PyAction(Action):
    """Action step defined by a PyFunction."""

    action: PyFunction

    def do(self, workchain):
        """Do the action on the workchain."""
        self.action(workchain)


def single_steps(step: Step) -> Iterator[Single]:
    """Yield all Single steps in a given step."""
    if isinstance(step, Single):
        yield step
    elif isinstance(step, (Concurrent, Sequential)):
        yield from toolz.mapcat(single_steps, step.steps)
    else:
        assert False, f"Unknown step type {type(step)}"


def single_processes(step: Step) -> Iterator[Process]:
    """Yield all Process steps in a given step."""
    return filter(lambda s: isinstance(s, Process), single_steps(step))


def _check_valid_pyfunction(f: PyFunction):
    """Check that the provided PyFunction may be used as part of a workflow."""
    if not isinstance(f, PyFunction):
        raise TypeError()
    if any(r.startswith("_") for r in f.returns):
        raise ValueError(
            "Cannot use functions with return names containing underscores "
            "in workflows."
        )
    if set(f.parameters).intersection(f.returns):
        raise ValueError(
            "Function has outputs that are named identically to its input(s)."
        )


def _check_pyfunctions_compatible(a: PyFunction, b: PyFunction):
    """Check that Pyfunction 'b' has enough inputs/outputs to be compatible with 'a'."""
    _check_valid_pyfunction(a)
    _check_valid_pyfunction(b)
    if missing_parameters := set(a.parameters) - set(b.parameters):
        raise ValueError(f"'{b.name}' is missing parameters: {missing_parameters}")
    if missing_returns := set(a.returns) - set(b.returns):
        raise ValueError(f"'{b.name}' is missing return values: {missing_returns}")


def from_pyfunction(f: PyFunction) -> Step:
    """Construct a Step corresponding to applying a PyFunction."""
    _check_valid_pyfunction(f)
    return Process(
        builder=engine.apply(f),
        parameters=f.parameters,
        returns=f.returns,
    )


def map_(f: PyFunction, *args, **kwargs) -> Step:
    """Construct a Step corresponding to mapping a PyFunction.

    Parameters
    ----------
    *args, **kwargs
        Positional/keyword arguments to pass to 'aiida_dynamic_workflows.engine.map_'.

    See Also
    --------
    aiida_dynamic_workflows.engine.map_
    """
    _check_valid_pyfunction(f)
    return Process(
        builder=engine.map_(f, *args, **kwargs),
        parameters=f.parameters,
        returns=f.returns,
    )


def concurrently(*fs: PyFunction | Step) -> Step:
    """Construct a Step for several tasks executing concurrently."""
    if len(fs) < 2:
        raise ValueError("Expected at least 2 steps")

    for i, f in enumerate(fs):
        for g in fs[i + 1 :]:
            if set(f.returns).intersection(g.returns):
                raise ValueError("Steps return values that are named the same")

    returns = [set(f.returns) for f in fs]

    parameters = [set(f.parameters) for f in fs]
    if any(a.intersection(b) for a in parameters for b in returns):
        raise ValueError("Steps cannot be run concurrently")

    def ensure_single(f):
        if isinstance(f, PyFunction):
            return from_pyfunction(f)
        elif isinstance(f, Single):
            return f
        else:
            raise TypeError(f"Expected PyFunction or Single, got {type(f)}")

    return Concurrent([ensure_single(f) for f in fs])


def new_workflow(name: str) -> Outline:
    """Return an Outline with no steps , and the given name."""
    return Outline(steps=(), label=name)


def first(s: PyFunction | Step) -> Outline:
    """Return an Outline consisting of a single Step."""
    return Outline(steps=(ensure_step(s),))


def ensure_step(s: Step | PyFunction) -> Step:
    """Return a Step, given a Step or a PyFunction."""
    if isinstance(s, Step):
        return s
    elif isinstance(s, PyFunction):
        return from_pyfunction(s)
    elif isinstance(s, Outline):
        return Sequential(s.steps)
    else:
        raise TypeError(f"Expected PyFunction, Step, or Outline, got {type(s)}")


def output(*names: str, **mappings: str) -> OutputAction:
    """Return an OutputAction that can be used in an outline."""
    outputs = {name: name for name in names}
    outputs.update({from_: to_ for from_, to_ in mappings.items()})

    return OutputAction(outputs)


@dataclass(frozen=True)
class Outline:
    """Outline of the steps to be executed.

    Each step kicks off either a _single_ process, or several processes
    concurrently.
    """

    steps: tuple[Step]
    #: Sequence of steps constituting the workflow
    label: str | None = None
    #: Optional label identifying the workflow

    def rename(self, name: str) -> Outline:
        """Return a new outline with a new name."""
        return replace(self, label=name)

    def then(self, step: PyFunction | Step | Outline) -> Outline:
        """Add the provided Step to the outline.

        If a PyFunction is provided it is added as a single step.
        """
        return replace(self, steps=self.steps + (ensure_step(step),))

    def join(self, other: Outline) -> Outline:
        """Return a new outline consisting of this and 'other' joined together."""
        return replace(self, steps=self.steps + other.steps)

    def returning(self, *names, **mappings) -> Outline:
        """Return the named values from this workflow."""
        possible_names = self.parameters.union(self.all_outputs)
        existing_names = self.returns
        requested_names = set(names).union(mappings.keys())

        if invalid_names := requested_names - possible_names:
            raise ValueError(
                f"Cannot return any of {invalid_names}; "
                "they do not appear in this outline."
            )

        if already_returned := requested_names.intersection(existing_names):
            raise ValueError(
                "The following names are already returned "
                f"by this outline: {already_returned}."
            )

        return replace(self, steps=self.steps + (output(*names, **mappings),))

    @property
    def _single_processes(self) -> Iterator[Process]:
        for step in self.steps:
            yield from single_processes(step)

    @property
    def _single_steps(self) -> Iterator[Single]:
        for step in self.steps:
            yield from single_steps(step)

    @property
    def parameters(self) -> set[str]:
        """Parameters of the Outline."""
        raw_parameters = toolz.reduce(
            set.union,
            (s.parameters for s in self._single_processes),
            set(),
        )
        return raw_parameters - self.all_outputs

    @property
    def returns(self) -> set[str]:
        """Values returned by this Outline."""
        ret = set()
        for step in self._single_steps:
            if isinstance(step, OutputAction):
                ret.update(step.outputs.values())
        return ret

    @property
    def all_outputs(self) -> set[str]:
        """All outputs of this outline."""
        return toolz.reduce(
            set.union,
            (s.returns for s in self._single_processes),
            set(),
        )

    def visualize(self, as_png=False) -> graphviz.Digraph:
        """Return a Graphviz visualization of this outline."""
        g = graphviz.Digraph(graph_attr=dict(rankdir="LR"))

        mapped_inputs = set()

        for proc in self._single_processes:
            proc_id = str(id(proc))
            is_mapjob = issubclass(proc.builder.process_class, PyMapJob)

            opts = dict(shape="rectangle")
            output_opts = dict()
            if is_mapjob:
                for d in (opts, output_opts):
                    d["style"] = "filled"
                    d["fillcolor"] = "#ffaaaaaa"

            g.node(proc_id, label=proc.builder.func.name, **opts)

            if is_mapjob:
                spec = common.MapSpec.from_string(proc.builder.metadata.options.mapspec)
                for p in spec.parameters:
                    mapped_inputs.add(p)
                    g.node(p, **output_opts)

            for r in proc.returns:
                g.node(r, **output_opts)
                g.edge(proc_id, r)

        for p in self.parameters - mapped_inputs:
            g.node(p, style="filled", fillcolor="#aaaaaa")

        for proc in self._single_processes:
            proc_id = str(id(proc))
            for p in proc.parameters:
                g.edge(p, proc_id)
        if as_png:
            return render_png(g)
        return g

    def traverse(self, f: Callable[[Single], Single]) -> Outline:
        """Return a copy of this Outline, with 'f' applied to all Single steps."""

        def transform(x: Step) -> Step:
            if isinstance(x, Single):
                return f(x)
            elif isinstance(x, (Concurrent, Sequential)):
                return type(x)(steps=tuple(map(transform, x.steps)))
            else:
                raise TypeError(f"Unknown step type {type(x)}")

        return replace(self, steps=tuple(map(transform, self.steps)))

    def with_restarts(self, step_restarts: dict[PyFunction, int]) -> Outline:
        """Return a copy of this Outline with restarts added to all specified steps.

        Examples
        --------
        >>> # Set up the original flow
        >>> import aiida_dynamic_workflows as flows
        >>> a = flows.step(lambda x, y: x + y, returning="z")
        >>> b = flows.step(lambda z: 2 * z)
        >>> flow = flows.workflow.first(a).then(b)
        >>> # Apply restarts: a restarted up to 2 times, b up to 3.
        >>> new_flow = flow.with_restarts({a: 2, b: 3})
        """

        def mapper(step):
            try:
                max_restarts = step_restarts[step.builder.func]
            except (AttributeError, KeyError):
                return step
            else:
                return replace(step, builder=step.builder.with_restarts(max_restarts))

        return self.traverse(mapper)

    def replace_steps(self, step_map: dict[PyFunction, PyFunction]) -> Outline:
        """Return a copy of this Outline, replacing the step functions specified.

        Any steps that are PyCalcJobs or PyMapJobs executing a PyFunction specified
        in 'step_map' will have the function executed replaced by the corresponding
        value in 'step_map'.

        See Also
        --------
        traverse

        Examples
        --------
        >>> # Set up the original flow
        >>> import aiida_dynamic_workflows as flows
        >>> a = flows.step(lambda x, y: x + y, returning="z")
        >>> b = flows.step(lambda z: 2 * z)
        >>> flow = flows.workflow.first(a).then(b)
        >>> # Create the new steps
        >>> new_a = flows.step(lambda x, y: x * y, returning="z")
        >>> new_b = flows.step(lambda z: 5 * z
        >>> # Replace the old steps with new ones!
        >>> new_flow = flow.replacing_steps({a: new_a, b: new_b})
        """
        for a, b in step_map.items():
            _check_pyfunctions_compatible(a, b)

        def mapper(step):
            try:
                new_func = step_map[step.builder.func]
            except (AttributeError, KeyError):
                return step
            else:
                b = copy.deepcopy(step.builder)
                b.func = new_func
                return Process(
                    builder=b, parameters=new_func.parameters, returns=new_func.returns
                )

        return self.traverse(mapper)

    def on(
        self,
        env: engine.ExecutionEnvironment,
        max_concurrent_machines: int | None = None,
    ) -> Outline:
        """Return a new Outline with the execution environment set for all steps."""

        def transform(s: Single):
            if not isinstance(s, Process):
                return s
            return replace(s, builder=s.builder.on(env, max_concurrent_machines))

        return self.traverse(transform)


# TODO: See if we can come up with a cleaner separation of "logical data flow"
#       and "error handling flow".

# TODO: see if we can do this more "directly" with the Aiida/Plumpy
#       "process" interface. As-is we are running our own "virtual machine"
#       on top of Aiida's!.
class PyWorkChain(aiida.engine.WorkChain):
    """WorkChain for executing Outlines."""

    @classmethod
    def define(cls, spec):  # noqa: D102
        super().define(spec)
        spec.input("outline", valid_type=PyOutline)
        spec.input_namespace("kwargs", dynamic=True)
        spec.output_namespace("return_values", dynamic=True)
        spec.outline(
            cls.setup,
            aiida.engine.while_(cls.is_not_done)(cls.do_step, cls.check_output),
            cls.finalize,
        )

        spec.exit_code(401, "INVALID_STEP", message="Invalid step definition")
        spec.exit_code(
            450, "STEP_RETURNED_ERROR_CODE", message="A step returned an error code"
        )

    @classmethod
    def get_builder(cls):  # noqa: D102
        return engine.ProcessBuilder(cls)

    # TODO: have the outline persisted into "self.ctx"; this way
    #       we don't need to reload it from the DB on every step.

    def setup(self):  # noqa: D102
        """Set up the state for the workchain."""
        outline = self.inputs.outline.value
        self.ctx._this_step = 0
        self.ctx._num_steps = len(outline.steps)
        self.ctx._had_errors = False

        if "kwargs" in self.inputs:
            self.ctx.update(self.inputs.kwargs)

    def finalize(self):
        """Finalize the workchain."""
        if self.ctx._had_errors:
            return self.exit_codes.STEP_RETURNED_ERROR_CODE

    def is_not_done(self) -> bool:
        """Return True when there are no more steps in the workchain."""
        return self.ctx._this_step < self.ctx._num_steps

    def do_step(self):
        """Execute the current step in the workchain."""
        this_step = self.ctx._this_step
        self.report(f"doing step {this_step} of {self.ctx._num_steps}")
        step = self.inputs.outline.value.steps[this_step]

        if isinstance(step, (Single, Sequential)):
            concurrent_steps = [step]
        elif isinstance(step, Concurrent):
            concurrent_steps = list(step.steps)
        else:
            self.report(f"Unknown step type {type(step)}")
            return self.exit_codes.INVALID_STEP

        for s in concurrent_steps:
            self._base_step(s)

        self.ctx._this_step += 1

    def _base_step(self, s: Step):
        if isinstance(s, Process):
            try:
                inputs = get_keys(self.ctx, s.parameters)
            except KeyError as err:
                self.report(f"Skipping step {s} due to missing inputs: {err.args}")
                self.ctx._had_errors = True
                return

            finalized_builder = s.builder.finalize(**inputs)

            fut = self.submit(finalized_builder)
            self.report(f"Submitted {s} (pk: {fut.pk})")
            self.to_context(_futures=aiida.engine.append_(fut))
        elif isinstance(s, Sequential):
            ol = Outline(steps=tuple(s.steps))
            try:
                inputs = get_keys(self.ctx, ol.parameters)
            except KeyError as err:
                self.report(f"Skipping step {s} due to missing inputs: {err.args}")
                self.ctx._had_errors = True
                return

            builder = PyWorkChain.get_builder()
            builder.outline = PyOutline(outline=ol)
            builder.kwargs = inputs
            fut = self.submit(builder)
            self.report(f"Submitted sub-workchain: {fut.pk}")
            self.to_context(_futures=aiida.engine.append_(fut))
        elif isinstance(s, Action):
            return s.do(self)

    def check_output(self):
        """Check the output of the current step in the workchain."""
        if "_futures" not in self.ctx:
            return

        for step in self.ctx._futures:
            if step.exit_status != 0:
                self.report(f"Step {step} reported a problem: {step.exit_message}")
                self.ctx._had_errors = True
            for name, value in return_values(step):
                self.ctx[name] = value

        del self.ctx["_futures"]


def get_keys(dictionary, keys):
    """Select all keys in 'keys' from 'dictionary'."""
    missing = []
    r = dict()
    for k in keys:
        if k in dictionary:
            r[k] = dictionary[k]
        else:
            missing.append(k)
    if missing:
        raise KeyError(*missing)
    return r


# XXX: This is all very tightly coupled to the definitions of "PyCalcJob"
#      and "PyMapJob".
def return_values(calc: aiida.orm.ProcessNode):
    """Yield (name, node) tuples of return values of the given ProcessNode.

    This assumes an output port namespace called "return_values".
    """
    try:
        return calc.outputs.return_values.items()
    except AttributeError:
        return ()


def build(outline: Outline, **kwargs) -> PyWorkChain:
    """Return a ProcessBuilder for launching the given Outline."""
    # TODO: validate that all ProcessBuilders in 'outline' are fully specified
    _check_outline(outline)
    builder = PyWorkChain.get_builder()
    builder.outline = PyOutline(outline=outline)
    if outline.label:
        builder.metadata.label = outline.label
    if missing := set(outline.parameters) - set(kwargs):
        raise ValueError(f"Missing parameters: {missing}")
    if superfluous := set(kwargs) - set(outline.parameters):
        raise ValueError(f"Too many parameters: {superfluous}")
    builder.kwargs = toolz.valmap(ensure_aiida_type, kwargs)
    return builder


def _check_outline(outline: Outline):
    for proc in outline._single_processes:
        if proc.builder.code is None:
            raise ValueError(
                f"Execution environment not specified for {proc.builder.func.name}. "
                "Did you remember to call 'on(env)' on the workflow?"
            )
