"""First-order time difference along the time axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.core.axes import AxisName
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import CONFIG_DEPENDENT, Category, LayoutEffect, StepSpec, Streaming


class Mode(StrEnum):
    """How each sample is combined with the one before it."""

    # fmt: off
    DIFFERENCE = "difference"
    CONJUGATE  = "conjugate"
    PHASE_ONLY = "phase-only"
    # fmt: on


# The two modes that build a conjugate product, which needs complex input.
_CONJUGATE_MODES = (Mode.CONJUGATE, Mode.PHASE_ONLY)


def _combine(current: SignalArray, previous: SignalArray, mode: Mode) -> SignalArray:
    """Combine each sample with the one before it, the way `mode` asks for."""

    if mode == Mode.DIFFERENCE:
        return as_signal_array(current - previous)

    # Multiplying by the conjugate subtracts the previous phase from the current
    # one, so the shared carrier phase cancels and only the change is left.
    product = current * np.conj(previous)
    if mode == Mode.CONJUGATE:
        return as_signal_array(product)

    # PHASE_ONLY keeps only the direction (unit phase) of that product.
    return as_signal_array(np.exp(1j * np.angle(product)))


@dataclass(frozen=True, slots=True)
class TimeDifference(Step):
    """Combine each sample with the one before it, along time.

    This turns a signal into how it changes from one time sample to the next. It
    is a simple way to bring out motion and drop whatever stays constant. There
    are three modes:

    DIFFERENCE - x[t] - x[t-1], the plain change between samples.
    CONJUGATE  - x[t] * conj(x[t-1]). For complex CSI this subtracts the previous
                 phase from the current one, so the shared (and unknown) carrier
                 phase cancels and only the change remains.
    PHASE_ONLY - the same conjugate product, but kept as unit phase only (the
                 magnitude is thrown away).

    The output has one fewer time sample than the input. When streaming, the last
    sample of each chunk is carried into the next, so the result matches batch.
    """

    mode: Mode = field(
        default=Mode.CONJUGATE,
        metadata={"doc": "how to combine each sample with the previous one"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="time-difference",
        summary="combine each sample with the previous one along time",
        category=Category.TEMPORAL_FEATURES,
        # The accepted value kinds depend on `mode`; see resolve_admissible_values.
        admissible_values=CONFIG_DEPENDENT,
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(note="drops the first time sample"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def resolve_admissible_values(self) -> tuple[ValueKind, ...] | None:
        """Conjugate modes need complex input; the plain difference accepts any."""

        if self.mode in _CONJUGATE_MODES:
            return (ValueKind.COMPLEX,)
        return None

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Same axes and semantics; the accepted value kinds depend on the mode."""

        self.require_inputs(layout)
        return layout

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Difference the whole recording (the operator handles it in one push)."""

        return _TimeDifferenceOperator(self.mode, out_layout).push(signal)

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Carry the last sample across chunks so differences span the boundaries."""

        return _TimeDifferenceOperator(self.mode, out_layout)


class _TimeDifferenceOperator(StreamOperator):
    """Differences consecutive samples, carrying the last one across chunks."""

    def __init__(self, mode: Mode, out_layout: Layout) -> None:
        """Bind the mode and output layout; start with no carried sample."""

        self._mode = mode
        self._out_layout = out_layout
        self._time_index = out_layout.dynamic_index
        self._previous: SignalArray | None = None

    def push(self, chunk: Signal) -> Signal:
        """Emit one difference per sample, once there is an earlier sample to use."""

        if self._time_index is None:
            raise ValueError("TimeDifference requires a signal with a time axis.")

        # Work with time on the first axis so we can pair each sample with the one
        # before it by simple slicing.
        data = as_signal_array(np.moveaxis(chunk.values, self._time_index, 0))
        if data.shape[0] == 0:
            return empty_signal(self._out_layout)

        # On the very first chunk there is no earlier sample, so the first output
        # is dropped. On later chunks we prepend the sample carried over from the
        # previous chunk, so the difference spans the chunk boundary.
        if self._previous is None:
            current, previous, out_times = data[1:], data[:-1], chunk.times[1:]
        else:
            padded = np.concatenate([self._previous[None], data], axis=0)
            current, previous, out_times = padded[1:], padded[:-1], chunk.times
        self._previous = data[-1]

        if current.shape[0] == 0:
            return empty_signal(self._out_layout)

        values = _combine(
            as_signal_array(current), as_signal_array(previous), self._mode
        )
        restored = as_signal_array(np.moveaxis(values, 0, self._time_index))
        return chunk.with_values(restored, self._out_layout, times=out_times)

    def flush(self) -> Signal:
        """Nothing is buffered beyond the carried sample."""

        return empty_signal(self._out_layout)
