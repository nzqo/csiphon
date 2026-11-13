"""Local least-squares slope over a trailing time window (a smooth derivative)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, assert_never

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_real_array, as_signal_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError, StreamingError
from csiphon.core.layout import Layout
from csiphon.core.sampling import effective_rate_hz
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming

_EPS = 1e-12


# The two ways to choose the window. Each holds only the parameters that apply to
# it, so a combination that makes no sense cannot be written down.
@dataclass(frozen=True, slots=True)
class FromNominalRate:
    """The last `window_size` samples, treated as evenly spaced.

    With `scale_by_dt` the slope is per second (the nominal sampling rate turns
    samples into seconds), otherwise it is per sample.
    """

    window_size: int = 10
    scale_by_dt: bool = True


@dataclass(frozen=True, slots=True)
class FromTimestamps:
    """The samples of the last `window_s` seconds, fitted against their timestamps.

    Lost samples and irregular timing are handled, because the fit uses the real
    time of each sample. The slope is per second.
    """

    window_s: float = 0.01


SlopeWindow = FromNominalRate | FromTimestamps


@dataclass(frozen=True, slots=True)
class WindowedSlope(Step):
    """Per-sample slope of a least-squares line over the trailing window.

    A steadier derivative than `x[t] - x[t-1]`: constant signals give ~0, linear
    trends give a constant, and noise is averaged down. `window` says which
    samples the line is fitted to (`FromNominalRate` or `FromTimestamps`).
    Streaming carries the window, so it matches batch exactly.
    """

    window: SlopeWindow = field(
        default=FromNominalRate(),
        metadata={
            "doc": "FromNominalRate(window_size, ...) or FromTimestamps(window_s)"
        },
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="windowed-slope",
        summary="fit a least-squares slope over a trailing window",
        category=Category.TEMPORAL_FEATURES,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(note="per-sample local slope"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout) -> Layout:
        """Shape and semantics are unchanged; each value becomes a local slope."""

        self.require_inputs(layout)
        match self.window:
            case FromNominalRate(window_size=window_size) if window_size < 2:
                raise LayoutError("WindowedSlope needs window_size >= 2.")
            case FromTimestamps(window_s=window_s) if window_s <= 0:
                raise LayoutError(f"window_s must be > 0, got {window_s}.")
        return layout

    def process(self, signal: Signal, out_layout: Layout) -> Signal:
        """Run the rolling slope over the whole recording."""

        # The whole recording is here, so if the profile gave no sampling rate we
        # can estimate it from the timestamps.
        rate_hz = None
        if self._scales_by_rate:
            rate_hz = effective_rate_hz(signal.times, signal.layout.nominal_rate_hz)
        return self._operator(rate_hz, out_layout).push(signal)

    def stream(self, in_layout: Layout, out_layout: Layout) -> StreamOperator | None:
        """Carry the window across chunks."""

        return self._operator(in_layout.nominal_rate_hz, out_layout)

    @property
    def _scales_by_rate(self) -> bool:
        """Whether the slope is turned into a per-second one with the sampling rate."""

        return isinstance(self.window, FromNominalRate) and self.window.scale_by_dt

    def _operator(self, rate_hz: float | None, out_layout: Layout) -> StreamOperator:
        """Build the operator that fits the window and scales the slope."""

        # The fitted slope is "change per sample". To report "change per second"
        # we multiply by the sampling rate (samples per second). Streaming cannot
        # estimate that rate, since the later timestamps have not arrived yet, so
        # it has to be known up front.
        scale = 1.0
        if self._scales_by_rate:
            if rate_hz is None:
                raise StreamingError(
                    f"'{self.name}' needs the sampling rate to report a slope per "
                    "second, and streaming cannot estimate it from timestamps it "
                    "has not seen yet. Set AcquisitionProfile.sampling_rate_hz, "
                    "or use FromTimestamps or FromNominalRate(scale_by_dt=False), "
                    "which do not need it."
                )
            scale = rate_hz
        return _WindowedSlopeOperator(self.window, scale, out_layout)


class _WindowedSlopeOperator(StreamOperator):
    """Computes the slope from sliding windows (fixed by sample count or time range).

    The recent samples that fall inside the window are kept in memory. Each new
    sample joins them, and the slope of a line fitted through them is its output.

    The memory persists between chunks, so streaming in pieces gives the same
    result as batch, which is a single chunk holding the whole recording.
    """

    def __init__(self, window: SlopeWindow, scale: float, out_layout: Layout) -> None:
        """Bind the window choice, the factor applied to each slope, and the layout."""

        self._window = window
        self._scale = scale
        self._out_layout = out_layout
        self._time_index = out_layout.dynamic_index

        # The memory: the recent samples and their timestamps, oldest first.
        self._recent_samples: deque[RealArray] = deque()
        self._recent_times: deque[float] = deque()

    def push(self, chunk: Signal) -> Signal:
        """Return the slope at every sample of the chunk, in order.

        A line needs two points, so a sample with fewer than two samples in its
        window (the very first one of a recording) gets slope 0.
        """

        if self._time_index is None:
            raise ValueError("WindowedSlope requires a signal with a time axis.")

        # Work with time on the first axis, so each sample is just data[index].
        data = as_real_array(np.moveaxis(chunk.values, self._time_index, 0))
        if data.shape[0] == 0:
            return empty_signal(self._out_layout)

        output = np.zeros_like(data)
        for index in range(data.shape[0]):
            # Add this sample to the memory, forgetting samples that are too old.
            self._remember(data[index], float(chunk.times[index]))

            # The line through everything in the memory gives this sample's slope.
            if len(self._recent_samples) >= 2:
                output[index] = self._slope_of_remembered_samples()

        restored = as_real_array(np.moveaxis(output, 0, self._time_index))
        return chunk.with_values(restored, self._out_layout)

    def flush(self) -> Signal:
        """Nothing is emitted at the end; every sample was already produced."""

        return empty_signal(self._out_layout)

    def _remember(self, sample: RealArray, time_s: float) -> None:
        """Add a sample to the memory, then forget the ones outside the window."""

        # Adding to memory: The sample and its time.
        self._recent_samples.append(sample)
        self._recent_times.append(time_s)

        # And remove the ones outside the slope window
        while self._oldest_is_outside_window():
            self._recent_samples.popleft()
            self._recent_times.popleft()

    def _oldest_is_outside_window(self) -> bool:
        """Tell whether the oldest remembered sample is outside the window.

        The window ends at the newest sample and reaches back `window_size`
        samples, or `window_s` seconds.
        """

        match self._window:
            # For nomial rates: Its outside if the memory exceeds the window size
            case FromNominalRate(window_size=window_size):
                return len(self._recent_samples) > window_size

            # For timestamps: Its outside if the last one is literally too old
            case FromTimestamps(window_s=window_s):
                return self._recent_times[-1] - self._recent_times[0] >= window_s
        assert_never(self._window)

    def _slope_of_remembered_samples(self) -> SignalArray:
        """Return the slope of the least-squares line through the remembered samples.

        The line is value = slope * position + offset, where the position of a
        sample is its index in the memory (FromNominalRate) or its timestamp
        (FromTimestamps). Centering the positions on their mean removes the
        offset, and the slope is then sum(position * value) / sum(position^2).
        """

        match self._window:
            case FromNominalRate():  # Synthetic times from nominal rate
                positions = np.arange(len(self._recent_samples), dtype=float)
            case FromTimestamps():  # True times
                positions = np.asarray(self._recent_times)
            case _:
                assert_never(self._window)
        centered = positions - positions.mean()

        # Stack the window into (samples, *other axes), so one product gives the
        # slope for every other axis (antenna, subcarrier, ...) at once.
        values = np.stack(tuple(self._recent_samples), axis=0)
        slope = np.tensordot(centered, values, axes=(0, 0)) / (
            np.sum(centered * centered) + _EPS
        )
        return as_signal_array(slope * self._scale)
