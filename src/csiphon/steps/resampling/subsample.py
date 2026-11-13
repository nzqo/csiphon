"""Subsample a stream by keeping every Nth sample (plain decimation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from csiphon.core.arrays import SignalArray, as_real_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.sampling import divide_nominal_rate
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


def _keep_every(
    values: SignalArray, time_index: int, first: int, every: int
) -> SignalArray:
    """Return a view of every `every`-th sample along time, starting at `first`."""

    selector = [slice(None)] * values.ndim
    selector[time_index] = slice(first, None, every)
    return values[tuple(selector)]


@dataclass(frozen=True, slots=True)
class SubsampleEvery(Step):
    """Keep every Nth sample (`values[::every]`), carrying each one's own timestamp.

    A plain decimation: it selects a subset of the existing samples and keeps their
    real (possibly irregular) timestamps. To land on a regular clock instead (a
    hold or interpolation onto a uniform grid) use `Resample`. Causal, so streaming
    matches batch exactly.
    """

    every: int = field(default=1, metadata={"doc": "keep every Nth sample"})

    spec: ClassVar[StepSpec] = StepSpec(
        name="subsample-every",
        summary="keep every Nth sample (plain decimation)",
        category=Category.RESAMPLING,
        admissible_values=None,
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(note="every Nth time sample kept"),
        streaming=Streaming.BATCH_EQUIVALENT,
        streaming_note="causal subset selection; streams exactly",
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Time stays the (dynamic) axis, at a lower nominal rate."""

        self.require_inputs(layout)
        if self.every < 1:
            raise LayoutError(f"every must be >= 1, got {self.every}.")

        # Every Nth sample is kept, so the rate drops by that factor.
        return divide_nominal_rate(layout, self.every)

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Keep every Nth sample, along with its real timestamp."""

        time_index = signal.layout.dynamic_index
        if time_index is None or signal.n_samples == 0:
            return signal.with_values(signal.values, out_layout)

        # Views, not copies: the kept samples are not duplicated in memory.
        values = _keep_every(signal.values, time_index, 0, self.every)
        times = as_real_array(signal.times[:: self.every])
        return signal.with_values(values, out_layout, times=times)

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator:
        """A causal operator keeping the global every-Nth sample, chunks included."""

        return _EveryNOperator(self.every, out_layout)


class _EveryNOperator(StreamOperator):
    """Keep every Nth sample across the whole stream, chunk boundaries included."""

    def __init__(self, every: int, out_layout: Layout) -> None:
        """Track how many samples have gone by, to hold the global stride."""

        self._every = every
        self._out_layout = out_layout
        self._seen = 0

    def push(self, chunk: Signal) -> Signal:
        """Emit this chunk's samples that land on the global every-Nth grid."""

        time_index = chunk.layout.dynamic_index
        if time_index is None or chunk.n_samples == 0:
            return empty_signal(self._out_layout)
        # The first kept sample in this chunk is the one whose global index is the
        # next multiple of `every`; from there, every Nth sample.
        first = (-self._seen) % self._every
        self._seen += chunk.n_samples
        if first >= chunk.n_samples:
            return empty_signal(self._out_layout)
        values = _keep_every(chunk.values, time_index, first, self._every)
        times = as_real_array(chunk.times[first :: self._every])
        return chunk.with_values(values, self._out_layout, times=times)

    def flush(self) -> Signal:
        """Nothing is buffered: samples are emitted as they arrive."""

        return empty_signal(self._out_layout)
