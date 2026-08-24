"""Rolling variance over a trailing time window."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, as_real_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError, StreamingError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.sampling import check_jitter, effective_rate_hz
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class WindowedVariance(Step):
    """Variance of each value over a trailing window of `win_size_s` seconds.

    A per-sample motion feature: high where the signal wobbles, low where it is
    steady. The window is a fixed number of samples (`win_size_s * rate`), stride
    one, keeping every time sample (short windows at the very start). Streaming
    carries the window state, so it matches batch exactly.
    """

    win_size_s: float = field(
        default=0.5, metadata={"doc": "trailing window length in seconds"}
    )
    strict: bool = field(
        default=False, metadata={"doc": "raise (not warn) on excessive jitter"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="windowed-variance",
        summary="compute a rolling variance over a trailing window",
        category=Category.TEMPORAL_FEATURES,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(note="per-sample rolling variance"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Shape and semantics are unchanged; each value becomes its variance."""

        self.require_inputs(layout)
        if self.win_size_s <= 0:
            raise LayoutError(f"win_size_s must be > 0, got {self.win_size_s}.")
        return layout

    def _window(self, rate_hz: float) -> int:
        """Trailing window length in samples at `rate_hz`."""

        return max(1, round(self.win_size_s * rate_hz))

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Run the rolling variance over the whole recording."""

        check_jitter(signal.times, step=self.name, strict=self.strict)
        rate_hz = effective_rate_hz(signal.times, profile.sampling_rate_hz)
        return _WindowedVarianceOperator(self._window(rate_hz), out_layout).push(signal)

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Carry the trailing window across chunks; needs a known rate to size it."""

        if profile.sampling_rate_hz is None:
            raise StreamingError(
                f"'{self.name}' needs a known sampling rate while streaming because "
                f"win_size_s={self.win_size_s} seconds must be converted into a number "
                "of samples. Set AcquisitionProfile.sampling_rate_hz or process the "
                "signal in batch, where the rate can be inferred from the timestamps."
            )
        return _WindowedVarianceOperator(
            self._window(profile.sampling_rate_hz), out_layout
        )


class _WindowedVarianceOperator(StreamOperator):
    """Keeps the trailing window's running sum and sum-of-squares."""

    def __init__(self, window: int, out_layout: Layout) -> None:
        """Bind the window length and output layout."""

        self._window = window
        self._out_layout = out_layout
        self._time_index = out_layout.dynamic_index
        self._samples: deque[RealArray] = deque()
        self._sum: RealArray | None = None
        self._sum_squares: RealArray | None = None

    def push(self, chunk: Signal) -> Signal:
        """Emit the variance of the trailing window at each new sample."""

        if self._time_index is None:
            raise ValueError("WindowedVariance requires a signal with a time axis.")

        # Work with time on the first axis, so each new sample is just data[index].
        data = as_real_array(np.moveaxis(chunk.values, self._time_index, 0))
        if data.shape[0] == 0:
            return empty_signal(self._out_layout)

        # Keep a running sum and sum of squares of the samples in the window. The
        # variance is mean(x^2) - mean(x)^2, so we can update it as samples enter
        # and leave, instead of re-summing the whole window at every step.
        if self._sum is None or self._sum_squares is None:
            self._sum = np.zeros_like(data[0])
            self._sum_squares = np.zeros_like(data[0])
        running_sum, running_squares = self._sum, self._sum_squares

        output = np.empty_like(data)
        for index in range(data.shape[0]):
            # Add the new sample to the running totals.
            sample = data[index]
            self._samples.append(sample)
            running_sum = running_sum + sample
            running_squares = running_squares + sample * sample

            # Once the window is full, take the oldest sample back out of them.
            if len(self._samples) > self._window:
                oldest = self._samples.popleft()
                running_sum = running_sum - oldest
                running_squares = running_squares - oldest * oldest

            # variance = mean of squares - square of mean. Clamp tiny negative
            # values (from rounding) up to zero.
            count = len(self._samples)
            mean = running_sum / count
            output[index] = np.maximum(0.0, running_squares / count - mean * mean)

        self._sum, self._sum_squares = running_sum, running_squares
        restored = as_real_array(np.moveaxis(output, 0, self._time_index))
        return chunk.with_values(restored, self._out_layout)

    def flush(self) -> Signal:
        """Nothing is emitted at the end; every sample was already produced."""

        return empty_signal(self._out_layout)
