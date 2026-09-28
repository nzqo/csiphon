"""A measured batch run: what each step cost, from one pass over the recording.

`measure` is `pour` with a stopwatch (and, on request, tracemalloc) around every
node. It returns the outlets together with one `StepCost` per node in run order,
and can combine consecutive steps into a named `GroupCost` for the stages you
think of as one unit.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from csiphon.core.signal import Signal
from csiphon.pipeline.merges import Junction
from csiphon.pipeline.runners import (
    Outlets,
    _run_node,
    _seed_inlets,
    last_readers,
    release_spent_lines,
)

if TYPE_CHECKING:
    from csiphon.pipeline.pipeline import Node, Siphon


# A step is named by its 1-based number (as describe() prints it) or its op name.
StepRef = int | str
GroupSpans = Mapping[str, tuple[StepRef, StepRef]]


@dataclass(frozen=True, slots=True)
class StepCost:
    """What one node cost during a measured run."""

    # fmt: off
    number      : int              # 1-based, as describe() numbers the steps
    name        : str              # op name (a merge shows its strategy)
    output      : str              # line id the node wrote
    seconds     : float            # wall time of the node's process()
    shape       : tuple[int, ...]  # shape of the array it produced
    peak_bytes  : int | None       # high-water mark above the level at start
    added_bytes : int | None       # live bytes after minus before
    # fmt: on


@dataclass(frozen=True, slots=True)
class GroupCost:
    """The combined cost of consecutive steps `first`..`last` (inclusive)."""

    # fmt: off
    name        : str
    first       : int              # step number of the first member
    last        : int              # step number of the last member
    seconds     : float            # summed over the members
    shape       : tuple[int, ...]  # the last member's output shape
    peak_bytes  : int | None       # high-water mark above the group's start level
    added_bytes : int | None       # live bytes after the group minus before it
    # fmt: on


def combine(name: str, steps: Sequence[StepCost]) -> GroupCost:
    """Fold consecutive step costs into one group cost."""

    seconds = sum(step.seconds for step in steps)
    first, last, shape = steps[0].number, steps[-1].number, steps[-1].shape

    # Each step's peak is relative to its own start, which sits `added` bytes of
    # earlier members above the group's start; the group peak is the highest of
    # those re-based peaks. Memory is all-or-nothing per run, so one unmeasured
    # step means none were.
    level = 0
    peak = 0
    for step in steps:
        if step.peak_bytes is None or step.added_bytes is None:
            return GroupCost(name, first, last, seconds, shape, None, None)
        peak = max(peak, level + step.peak_bytes)
        level += step.added_bytes
    return GroupCost(name, first, last, seconds, shape, peak, level)


@dataclass(frozen=True, slots=True)
class MeasuredRun:
    """The outlets of one batch run plus what every step (and group) cost."""

    # fmt: off
    outlets : Outlets
    steps   : tuple[StepCost, ...]   # one per node, in run order
    groups  : dict[str, GroupCost]   # the named spans the caller asked for
    # fmt: on

    def total(self) -> GroupCost:
        """The whole run as one group."""

        return combine("total", self.steps)

    def __str__(self) -> str:
        """Render the per-step table, then the groups and the total."""

        return self.render()

    def render(self) -> str:
        """A plain-text table of every step's cost, the groups, and the total."""

        has_memory = any(step.peak_bytes is not None for step in self.steps)
        header = ["step", "name", "time", "shape"]
        rows = [
            [str(step.number), step.name, _seconds(step.seconds), str(step.shape)]
            for step in self.steps
        ]
        for group in (*self.groups.values(), self.total()):
            span = f"{group.first}-{group.last}"
            rows.append([span, group.name, _seconds(group.seconds), str(group.shape)])

        if has_memory:
            header += ["peak", "added"]
            costs: list[StepCost | GroupCost] = [
                *self.steps,
                *self.groups.values(),
                self.total(),
            ]
            for row, cost in zip(rows, costs, strict=True):
                row += [_bytes(cost.peak_bytes), _bytes(cost.added_bytes)]

        widths = [
            max(len(row[i]) for row in [header, *rows]) for i in range(len(header))
        ]
        lines = [_row(header, widths), _row(["-" * w for w in widths], widths)]
        lines += [_row(row, widths) for row in rows]
        return "\n".join(lines)


def _row(cells: list[str], widths: list[int]) -> str:
    """One table line; the time column is right-aligned, the rest left."""

    padded = [
        cell.rjust(width) if i == 2 else cell.ljust(width)
        for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
    ]
    return "  " + "  ".join(padded).rstrip()


def _seconds(seconds: float) -> str:
    """Wall time in the unit that keeps it readable (µs, ms, or s)."""

    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _bytes(count: int | None) -> str:
    """A byte count in binary units (B, KiB, MiB, GiB); '-' when not measured."""

    if count is None:
        return "-"
    size = float(count)
    for unit in ("B", "KiB", "MiB"):
        if abs(size) < 1024.0:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def measure(
    siphon: Siphon,
    signals: Signal | Mapping[str, Signal],
    *,
    memory: bool,
    groups: GroupSpans | None,
) -> MeasuredRun:
    """Run a whole recording through a siphon, timing (and sizing) every node.

    `memory=True` wraps each node in tracemalloc. That slows the run (so the times
    are inflated) and only sees allocations made through Python, numpy's
    included; memory taken inside compiled extensions (numba, scipy internals)
    is invisible to it.
    """

    # Resolve the group spans before running, so a bad span fails fast.
    spans = {
        name: _resolve_span(siphon.nodes, name, span)
        for name, span in (groups or {}).items()
    }

    # Only start (and stop) tracing when nobody else is tracing already.
    tracing = memory and not tracemalloc.is_tracing()
    if tracing:
        tracemalloc.start()
    try:
        env = _seed_inlets(siphon, signals, allow_partial=False)
        last_reader = last_readers(siphon)
        steps: list[StepCost] = []
        for number, node in enumerate(siphon.nodes, start=1):
            inputs = [env[line] for line in node.inputs]
            env[node.output], cost = _measure_node(number, node, inputs, memory)
            steps.append(cost)
            del inputs
            release_spent_lines(env, siphon, node, number - 1, last_reader)
    finally:
        if tracing:
            tracemalloc.stop()

    outlets = Outlets({display: env[line] for display, line in siphon.outlets.items()})
    named = {
        name: combine(name, steps[first - 1 : last])
        for name, (first, last) in spans.items()
    }
    return MeasuredRun(outlets, tuple(steps), named)


def _measure_node(
    number: int, node: Node, inputs: list[Signal], memory: bool
) -> tuple[Signal, StepCost]:
    """Run one node and record its time, output shape, and (optionally) memory."""

    before = 0
    if memory:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

    started = time.perf_counter()
    output = _run_node(node, inputs)
    seconds = time.perf_counter() - started

    peak_bytes = added_bytes = None
    if memory:
        after, peak = tracemalloc.get_traced_memory()
        peak_bytes = peak - before
        added_bytes = after - before

    name = node.op.display_name if isinstance(node.op, Junction) else node.op.name
    cost = StepCost(
        number, name, node.output, seconds, output.values.shape, peak_bytes, added_bytes
    )
    return output, cost


def _resolve_span(
    nodes: Sequence[Node], group: str, span: tuple[StepRef, StepRef]
) -> tuple[int, int]:
    """Turn a `(first, last)` pair of step numbers or names into step numbers."""

    first = _resolve_step(nodes, group, span[0])
    last = _resolve_step(nodes, group, span[1])
    if first > last:
        raise ValueError(
            f"Group '{group}' runs backwards: step {first} comes after step {last}."
        )
    return first, last


def _resolve_step(nodes: Sequence[Node], group: str, ref: StepRef) -> int:
    """A step number as given, or the number of the one step with that name."""

    if isinstance(ref, int):
        if not 1 <= ref <= len(nodes):
            raise ValueError(
                f"Group '{group}' names step {ref}; steps are numbered 1 to "
                f"{len(nodes)}."
            )
        return ref

    names = [
        node.op.display_name if isinstance(node.op, Junction) else node.op.name
        for node in nodes
    ]
    matches = [number for number, name in enumerate(names, start=1) if name == ref]
    if len(matches) != 1:
        raise ValueError(
            f"Group '{group}' names step '{ref}', which matches {len(matches)} steps "
            f"(available: {names}); use the step number to pick one."
        )
    return matches[0]
