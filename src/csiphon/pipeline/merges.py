"""Junctions: combine several branches back into one line.

A junction is the only N->1 operation in a pipeline. Regular steps take one line
in and give one out; a junction takes several branches, lines them up in time,
and combines them. It has two independent knobs:

- how to combine them (the `strategy`): Stack (side by side on a new axis),
  Concatenate (grow one existing axis), Mean / Sum (reduce across branches), or
  Fuse (lay differently-typed branches into one feature vector).
- how to line them up in time first (the `align` strategy): Exact (keep the
  timestamps every branch shares) or Hold (carry each branch's last value onto a
  reference branch's timeline, for branches on different time grids).

Both have a whole-recording form and a streaming form; `Trap` runs the streaming
side. Two neighbouring modules keep this one focused: `_align_ops` holds the
low-level per-signal helpers (hold / keep / concatenate), and `sequence` holds the
packet-number alignment feature (`Sequence`, `SequenceTrap`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_real_array, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import DataError, LayoutError, StreamingError
from csiphon.core.layout import Layout
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline._align_ops import (
    align_on_keys,
    align_on_time,
    concat_time,
    ensure_not_clogged,
    hold_on_keys,
    keep_times,
    sample_hold,
)
from csiphon.pipeline.sequence import Sequence, SequenceTrap, reject_sequence_streaming
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming

# Samples a branch may hold unaligned before a ClogError.
DEFAULT_MAX_HOLD = 100_000


class MergeStrategy(ABC):
    """How a junction combines its branches once they are aligned on time."""

    @abstractmethod
    def output_layout(self, inputs: SequenceABC[Layout]) -> Layout:
        """Validate the branch layouts and return the combined layout."""

    @abstractmethod
    def combine(
        self, arrays: SequenceABC[SignalArray], inputs: SequenceABC[Layout]
    ) -> SignalArray:
        """Combine the already time-aligned branch arrays into one array."""


def _require_matching(
    inputs: SequenceABC[Layout], *, ignore: AxisName | None = None
) -> None:
    """Fail unless every branch shares axes, value kind, and representation.

    With `ignore` set, that one axis may differ in size (used by Concatenate).
    """

    first = inputs[0]
    first_axes = [name.value for name in first.axis_names]
    join = f" (except '{ignore.value}', the axis being joined)" if ignore else ""
    for position, other in enumerate(inputs[1:], start=1):
        # Same set of axes, by name: a merge lines the branches up axis by axis, so
        # every branch must carry the same axes (Concatenate may still differ in size
        # on the one it joins, handled below).
        if other.axis_names != first.axis_names:
            other_axes = [name.value for name in other.axis_names]
            missing = [name for name in first_axes if name not in other_axes]
            extra = [name for name in other_axes if name not in first_axes]
            difference = "; ".join(
                part
                for part in (
                    f"branch {position} is missing {missing}" if missing else "",
                    f"branch {position} has extra {extra}" if extra else "",
                    "same axes, different order" if not missing and not extra else "",
                )
                if part
            )
            raise LayoutError(
                f"Merge branches must have the same axes. Branch 0 has {first_axes}, "
                f"branch {position} has {other_axes} ({difference}). Note a size-1 "
                "axis is dropped from a raw-CSI layout, so e.g. a 1-antenna receiver "
                "has no rx_antenna axis to merge on. Give it that axis first."
            )

        # Same value kind and representation, so the branch values are comparable.
        if other.values != first.values or other.representation != first.representation:
            first_kind = f"{first.values.value}/{first.representation.value}"
            other_kind = f"{other.values.value}/{other.representation.value}"
            raise LayoutError(
                "Merge branches must have the same value kind and representation: "
                f"branch 0 is {first_kind}, branch {position} is {other_kind}."
            )

        # Same length on every shared axis (the joined axis is allowed to differ).
        for name in first.axis_names:
            if name == ignore:
                continue
            if other.axis(name).size != first.axis(name).size:
                raise LayoutError(
                    f"Merge branches must match in size on every axis{join}. On "
                    f"'{name.value}', branch 0 has {first.axis(name).size} but branch "
                    f"{position} has {other.axis(name).size}."
                )


@dataclass(frozen=True, slots=True)
class Stack(MergeStrategy):
    """Put the branches side by side on a new axis of length = number of branches."""

    into: AxisName = AxisName.FEATURE

    def output_layout(self, inputs: SequenceABC[Layout]) -> Layout:
        """Require identical branches and append the new stacking axis."""

        _require_matching(inputs)
        first = inputs[0]
        if first.has_axis(self.into):
            raise LayoutError(
                f"Stack axis '{self.into}' already exists on the branches."
            )
        return first.set_axes((*first.axes, Axis.sized(self.into, len(inputs))))

    def combine(
        self, arrays: SequenceABC[SignalArray], inputs: SequenceABC[Layout]
    ) -> SignalArray:
        """Stack the branches along a new trailing axis."""

        return as_signal_array(np.stack(arrays, axis=-1))


@dataclass(frozen=True, slots=True)
class Concatenate(MergeStrategy):
    """Grow one existing axis by laying the branches end to end along it."""

    axis: AxisName

    def output_layout(self, inputs: SequenceABC[Layout]) -> Layout:
        """Require branches that match everywhere except `axis`; sum that axis."""

        first = inputs[0]
        first.require_static_axis(self.axis)
        _require_matching(inputs, ignore=self.axis)
        total = 0
        for layout in inputs:
            # require_static_axis on `first` plus the matching check above mean
            # every branch has a concrete size on this axis.
            size = layout.axis(self.axis).size
            assert size is not None
            total += size
        return first.replace_axis(self.axis, Axis.sized(self.axis, total))

    def combine(
        self, arrays: SequenceABC[SignalArray], inputs: SequenceABC[Layout]
    ) -> SignalArray:
        """Concatenate the branches along the chosen axis."""

        position = inputs[0].axis_position(self.axis)
        return as_signal_array(np.concatenate(arrays, axis=position))


@dataclass(frozen=True, slots=True)
class Mean(MergeStrategy):
    """Average the branches element by element (they must be identical in shape)."""

    def output_layout(self, inputs: SequenceABC[Layout]) -> Layout:
        """Require identical branches; the layout is unchanged."""

        _require_matching(inputs)
        return inputs[0]

    def combine(
        self, arrays: SequenceABC[SignalArray], inputs: SequenceABC[Layout]
    ) -> SignalArray:
        """Mean across the branches (inputs are identical, so no layout is needed)."""

        del inputs
        return as_signal_array(np.mean(np.stack(arrays, axis=0), axis=0))


@dataclass(frozen=True, slots=True)
class Sum(MergeStrategy):
    """Add the branches element by element (they must be identical in shape)."""

    def output_layout(self, inputs: SequenceABC[Layout]) -> Layout:
        """Require identical branches; the layout is unchanged."""

        _require_matching(inputs)
        return inputs[0]

    def combine(
        self, arrays: SequenceABC[SignalArray], inputs: SequenceABC[Layout]
    ) -> SignalArray:
        """Sum across the branches (inputs are identical, so no layout is needed)."""

        del inputs
        return as_signal_array(np.sum(np.stack(arrays, axis=0), axis=0))


def _unify_values(kinds: SequenceABC[ValueKind]) -> ValueKind:
    """The value kind of fused branches: keep it if shared, else a common numeric kind.

    Fusing turns the branches into one feature vector, so a mix of kinds is fine:
    if any branch is complex the result is complex (real branches sit in the real
    part); otherwise the different real kinds (magnitude, power, ...) become plain
    real-valued features.
    """

    unique = set(kinds)
    if len(unique) == 1:
        return kinds[0]
    if ValueKind.COMPLEX in unique:
        return ValueKind.COMPLEX
    return ValueKind.REAL


def _fuse_combine_names(inputs: SequenceABC[Layout]) -> list[AxisName]:
    """The name of each branch's axis that is laid into the feature axis.

    Axes are matched by name, not position (order is irrelevant here, as
    everywhere in csiphon). An axis name present in every branch with the same
    size is "shared" and must line up; each branch must have exactly one axis that
    is not shared, a differently named one (e.g. `subcarrier` vs `delay`) or the
    same name at a different size. That odd axis out is the one that is fused.
    """

    common = set.intersection(*(set(layout.axis_names) for layout in inputs))
    shared = {
        name
        for name in common
        if len({layout.axis(name).size for layout in inputs}) == 1
    }
    combine_names: list[AxisName] = []
    for layout in inputs:
        extras = [axis.name for axis in layout.axes if axis.name not in shared]
        if len(extras) != 1:
            listed = ", ".join(extras) if extras else "none"
            raise LayoutError(
                "Fuse needs each branch to share every axis but one; this branch "
                f"has {len(extras)} unshared axes ({listed}). The branches must "
                "line up on all axes except the single one being fused."
            )
        combine_names.append(extras[0])
    return combine_names


@dataclass(frozen=True, slots=True)
class Fuse(MergeStrategy):
    """Combine branches that differ on one axis into a single feature vector.

    Unlike the other strategies, Fuse does not require the branches to be the same
    kind of quantity. It lays each branch's one unshared axis (say a `subcarrier`
    axis on one branch and a `delay` axis on another, of any sizes) end to end
    into a generic `feature` axis, and unifies the value kinds (see
    `_unify_values`). The result is a feature vector: representation
    `FEATURE_VECTOR`, one feature axis of length equal to the summed sizes.

    All the other axes must match across the branches by name and size (order does
    not matter, branches are transposed to line up). Use this to assemble a
    feature matrix from heterogeneous features, e.g. a per-subcarrier statistic
    beside a delay- or frequency-domain one.
    """

    into: AxisName = AxisName.FEATURE

    def output_layout(self, inputs: SequenceABC[Layout]) -> Layout:
        """Replace the fused axis with one feature axis; unify the value kind."""

        combine_names = _fuse_combine_names(inputs)
        template = inputs[0]
        if self.into != combine_names[0] and template.has_axis(self.into):
            raise LayoutError(f"Fuse feature axis '{self.into}' already exists.")

        sizes = [inputs[i].axis(combine_names[i]).size for i in range(len(inputs))]
        # A dynamic (runtime-sized) branch makes the feature axis dynamic too;
        # otherwise the feature length is the sum of the branch sizes.
        total = None if any(s is None for s in sizes) else sum(s for s in sizes if s)

        # Keep the first branch's axis order, with its fused axis turned into the
        # feature axis; the other branches are transposed to match at run time.
        axes = [
            Axis(self.into, size=total) if axis.name == combine_names[0] else axis
            for axis in template.axes
        ]
        values = _unify_values([layout.values for layout in inputs])
        return Layout(tuple(axes), Representation.FEATURE_VECTOR, values)

    def combine(
        self, arrays: SequenceABC[SignalArray], inputs: SequenceABC[Layout]
    ) -> SignalArray:
        """Transpose each branch to the first branch's axis order, then concatenate."""

        combine_names = _fuse_combine_names(inputs)
        template = inputs[0]
        combine_index = template.axis_position(combine_names[0])

        aligned = []
        for index, (array, layout) in enumerate(zip(arrays, inputs, strict=True)):
            # Read each output slot from this branch: shared axes by their name,
            # and the fused slot from this branch's own unshared axis.
            order = [
                layout.axis_position(
                    combine_names[index] if name == combine_names[0] else name
                )
                for name in template.axis_names
            ]
            aligned.append(np.transpose(array, order))

        values = _unify_values([layout.values for layout in inputs])
        # A complex result carries any real-valued branch in its real part.
        if values == ValueKind.COMPLEX:
            aligned = [array.astype(np.complex128) for array in aligned]
        return as_signal_array(np.concatenate(aligned, axis=combine_index))


@dataclass(frozen=True, slots=True)
class Time:
    """Align branches on their timestamps (the default coordinate)."""

    def keys(self, signals: SequenceABC[Signal]) -> list[RealArray]:
        """One monotonic alignment key per branch: its timestamps."""

        return [as_real_array(signal.times) for signal in signals]


Coordinate = Time | Sequence


class Alignment(ABC):
    """How a merge lines up its branches in time before combining them.

    A merge combines the branches sample by sample, so their rows must first sit
    on one shared timeline. Branches can be on different time grids, a
    per-sample feature and a windowed one (whose timestamps are window centres)
    rarely share timestamps, and this decides how to reconcile that.

    `align` is the whole-recording (batch) version; `release` is the streaming
    one, called with each branch's buffered-so-far samples.
    """

    streaming: ClassVar[Streaming]

    # fmt: off
    on        : Coordinate  # how branches are matched up: by time, or by packet number
    reference : int = 0     # which branch's timeline leads (0 = the first branch)
    # fmt: on

    def validate_branches(self, _count: int) -> None:  # noqa: B027 (optional hook)
        """Check this alignment can run over a branch count (default: always can)."""

    @abstractmethod
    def align(
        self, signals: SequenceABC[Signal]
    ) -> tuple[list[SignalArray], RealArray]:
        """Line up the whole branch signals into aligned arrays and their times."""

    @abstractmethod
    def release(
        self, buffers: SequenceABC[Signal], *, final: bool
    ) -> tuple[list[SignalArray], RealArray, list[Signal]]:
        """Return the rows that are final now (aligned arrays and their times) plus
        the branch buffers to keep for later.

        `final` is set at end of stream (flush), when no more samples will arrive,
        so anything still waiting can be emitted with what is on hand.
        """


@dataclass(frozen=True, slots=True)
class Exact(Alignment):
    """Keep only the timestamps every branch actually shares (one common grid).

    Strict: if the branches share no timestamps at all, a whole-recording merge
    raises rather than silently emit nothing. This is the default; use it when
    every branch is on the same time grid.
    """

    on: Coordinate = field(default_factory=Time)
    streaming: ClassVar[Streaming] = Streaming.BATCH_EQUIVALENT

    def align(
        self, signals: SequenceABC[Signal]
    ) -> tuple[list[SignalArray], RealArray]:
        """Trim every branch to the keys (times or sequence numbers) they share."""

        arrays, times = align_on_keys(signals, self.on.keys(signals))
        # A frame signal has no time axis, so "no shared keys" cannot happen there;
        # only a real time axis with an empty overlap is the error.
        if signals[0].layout.dynamic_index is not None and times.size == 0:
            raise DataError(
                "Merge branches share no timestamps, so an exact merge would be "
                "empty. They are probably on different time grids (e.g. window-centre "
                "times vs per-sample times). Use align=Hold() to merge across grids, "
                "or bring the branches onto one grid first."
            )
        return arrays, times

    def release(
        self, buffers: SequenceABC[Signal], *, final: bool
    ) -> tuple[list[SignalArray], RealArray, list[Signal]]:
        """Emit the timestamps shared by every branch's buffer; hold on to the rest."""

        # Exact treats end-of-stream no differently: it only ever emits shared
        # timestamps, and anything unshared is simply dropped.
        del final
        reject_sequence_streaming(self.on)
        arrays, times = align_on_time(buffers)
        if times.size == 0:
            return arrays, times, list(buffers)

        # Keep each branch's timestamps that were not part of this emission.
        emitted = set(times.tolist())
        kept = [
            keep_times(buffer, np.array([t not in emitted for t in buffer.times], bool))
            for buffer in buffers
        ]
        return arrays, times, kept


@dataclass(frozen=True, slots=True)
class Hold(Alignment):
    """Sample-and-hold every branch onto one reference branch's timeline.

    Picks a reference branch (`reference`, the first by default). For each of its
    timestamps, every other branch supplies its most recent value at or before
    that time. This lets branches sampled at different times or rates merge, a
    per-sample feature beside a windowed one, say. It only ever looks backward, so
    it streams.
    """

    reference: int = 0
    on: Coordinate = field(default_factory=Time)
    streaming: ClassVar[Streaming] = Streaming.BATCH_EQUIVALENT

    def validate_branches(self, count: int) -> None:
        """The reference branch must be one of the `count` branches present."""

        if not 0 <= self.reference < count:
            raise LayoutError(
                f"Hold reference branch {self.reference} is out of range for "
                f"{count} branches."
            )

    def align(
        self, signals: SequenceABC[Signal]
    ) -> tuple[list[SignalArray], RealArray]:
        """Hold every branch onto the reference branch's key timeline."""

        keys = self.on.keys(signals)
        return hold_on_keys(signals, keys, self.reference)

    def release(
        self, buffers: SequenceABC[Signal], *, final: bool
    ) -> tuple[list[SignalArray], RealArray, list[Signal]]:
        """Emit reference times every other branch has reached; hold the rest.

        A reference time is final once every other branch has a sample at or past
        it, because then the value held at that time can no longer change. At end
        of stream (`final`) nothing more will arrive, so every remaining reference
        time is emitted with each branch's last value.
        """

        reject_sequence_streaming(self.on)
        nothing: tuple[list[SignalArray], RealArray, list[Signal]] = (
            [],
            as_real_array(np.zeros(0)),
            list(buffers),
        )
        if any(buffer.n_samples == 0 for buffer in buffers):
            return nothing

        others = [b for i, b in enumerate(buffers) if i != self.reference]
        # The cutoff is the earliest of the other branches' last times: every
        # reference time up to there has been seen by all of them. At end of
        # stream nothing more arrives, so treat everything as seen.
        reached = np.inf if final else min(float(b.times[-1]) for b in others)
        reference = buffers[self.reference]
        ready = reference.times[reference.times <= reached]
        if ready.size == 0:
            return nothing

        arrays = [sample_hold(buffer, as_real_array(ready)) for buffer in buffers]
        return arrays, as_real_array(ready), self._retain(buffers, float(ready[-1]))

    def _retain(
        self, buffers: SequenceABC[Signal], emitted_upto: float
    ) -> list[Signal]:
        """Trim each buffer, having emitted reference times up to `emitted_upto`.

        The next reference time will be later than `emitted_upto`, so each other
        branch only needs the value it held there onward: keep its last sample at
        or before `emitted_upto` (the carried value) and everything after.
        """

        kept: list[Signal] = []
        for index, buffer in enumerate(buffers):
            if index == self.reference:
                # Its emitted times are done with; keep only the later ones.
                keep = buffer.times > emitted_upto
            else:
                # Keep this branch's carried value (its last sample at or before
                # `emitted_upto`) and everything after it.
                carry = int(np.searchsorted(buffer.times, emitted_upto, side="right"))
                keep = np.zeros(buffer.n_samples, dtype=bool)
                keep[max(carry - 1, 0) :] = True
            kept.append(keep_times(buffer, keep))
        return kept


@dataclass(frozen=True, slots=True)
class Junction:
    """The N->1 join: line branches up in time (`align`), then combine (`strategy`)."""

    # fmt: off
    strategy : MergeStrategy
    align    : Alignment = field(default_factory=Exact)
    max_hold : int       = DEFAULT_MAX_HOLD  # unaligned samples allowed per branch
    # fmt: on

    spec: ClassVar[StepSpec] = StepSpec(
        name="merge",
        summary="align branches on time and combine them into one line",
        category=Category.RESTRUCTURING,
        admissible_values=None,
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(note="several branches combined into one line"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    @property
    def name(self) -> str:
        """This junction's short identifier (from its spec)."""

        return self.spec.name

    @property
    def display_name(self) -> str:
        """The merge's operation for display: its strategy, e.g. 'stack' or 'mean'."""

        return type(self.strategy).__name__.lower()

    def output_layout(self, inputs: SequenceABC[Layout]) -> Layout:
        """Validate the branch layouts and return the combined layout."""

        if len(inputs) < 2:
            raise LayoutError("A merge needs at least two branches.")
        self.align.validate_branches(len(inputs))
        return self.strategy.output_layout(inputs)

    def process(self, signals: SequenceABC[Signal], out_layout: Layout) -> Signal:
        """Align the branches on time and combine them (whole-recording)."""

        arrays, times = self.align.align(signals)
        combined = self.strategy.combine(arrays, [signal.layout for signal in signals])
        return Signal(values=combined, times=times, layout=out_layout)

    def trap(self, out_layout: Layout) -> Trap | SequenceTrap:
        """A streaming operator that buffers branches until they line up.

        Aligning on time uses the plain Trap. Aligning on sequence numbers uses a
        SequenceTrap, supported for Hold with a known wrap period; the other
        sequence cases (Exact, or no period) are batch-only for now.
        """

        on = self.align.on
        if isinstance(on, Sequence):
            if on.period is None:
                raise StreamingError(
                    "Streaming sequence alignment needs a wrap period; pass "
                    "Sequence(period=...), or pour the recording instead."
                )
            if not isinstance(self.align, Hold):
                raise StreamingError(
                    "Streaming sequence alignment supports Hold only; use "
                    "Hold(on=Sequence(...)), or pour the recording instead."
                )
            reference = self.align.reference
            return SequenceTrap(
                self.strategy.combine, self.max_hold, reference, on, out_layout
            )
        return Trap(self, out_layout)


class Trap:
    """A junction's streaming holding area: keep branches until they align, then emit.

    Each branch has its own holding area. On every push the new samples are added,
    then the timestamps now present in every branch are released (in time order)
    and removed from the holding areas; the rest waits. When branches run at the
    same delay the holding areas stay empty. `flush` releases whatever still lines
    up once the stream ends; anything that never lined up is dropped.

    If one branch keeps piling up without ever aligning (a stalled branch, or time
    grids that never intersect), its holding area would grow without bound; past
    the junction's `max_hold` samples that raises a `ClogError` instead.
    """

    def __init__(self, junction: Junction, out_layout: Layout) -> None:
        """Bind the junction and remember the compiled output layout."""

        self._junction = junction
        self._out_layout = out_layout
        self._held: list[Signal | None] = []

    def push(self, inputs: list[Signal]) -> Signal:
        """Add each branch's new samples, then release whatever now aligns."""

        if not self._held:
            self._held = [None] * len(inputs)
        for index, chunk in enumerate(inputs):
            if chunk.n_samples == 0:
                continue
            held = self._held[index]
            self._held[index] = chunk if held is None else concat_time(held, chunk)
        return self._release(final=False)

    def flush(self) -> Signal:
        """Release whatever still aligns at end of stream."""

        return self._release(final=True)

    def _release(self, *, final: bool) -> Signal:
        """Emit the rows the alignment says are ready; keep the rest held."""

        ensure_not_clogged(self._held, self._junction.max_hold)
        if not self._held or any(held is None for held in self._held):
            return empty_signal(self._out_layout)

        branches = [held for held in self._held if held is not None]
        arrays, times, kept = self._junction.align.release(branches, final=final)
        if times.size == 0:
            return empty_signal(self._out_layout)

        combined = self._junction.strategy.combine(
            arrays, [branch.layout for branch in branches]
        )
        self._held = list(kept)
        return Signal(values=combined, times=times, layout=self._out_layout)
