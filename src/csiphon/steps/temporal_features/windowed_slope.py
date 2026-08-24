"""Local least-squares slope over a trailing time window (a smooth derivative)."""

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
from csiphon.core.sampling import effective_rate_hz
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming

_EPS = 1e-12


def _slope_fit(n: int) -> tuple[RealArray, float]:
    """Centered sample positions and their sum of squares for a window of `n`."""

    positions = as_real_array(np.arange(n) - (n - 1) / 2.0)
    return positions, float((positions * positions).sum())


@dataclass(frozen=True, slots=True)
class WindowedSlope(Step):
    """Per-sample slope of a least-squares line over the trailing window.

    A steadier derivative than `x[t] - x[t-1]`: constant signals give ~0, linear
    trends give a constant, and noise is averaged down. With `scale_by_dt` the
    slope is per second (using the sampling rate), otherwise per sample. Streaming
    carries the window, so it matches batch exactly.
    """

    window_size: int = field(
        default=10, metadata={"doc": "trailing window length in samples"}
    )
    scale_by_dt: bool = field(
        default=True, metadata={"doc": "divide by dt for a per-second slope"}
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

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Shape and semantics are unchanged; each value becomes a local slope."""

        self.require_inputs(layout)
        if self.window_size < 2:
            raise LayoutError("WindowedSlope needs window_size >= 2.")
        return layout

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Run the rolling slope over the whole recording."""

        inv_dt = 1.0
        if self.scale_by_dt:
            inv_dt = effective_rate_hz(signal.times, profile.sampling_rate_hz)
        return _WindowedSlopeOperator(self.window_size, inv_dt, out_layout).push(signal)

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Carry the window across chunks; a per-second slope needs a known rate."""

        inv_dt = 1.0
        if self.scale_by_dt:
            if profile.sampling_rate_hz is None:
                raise StreamingError(
                    f"'{self.name}' cannot compute a per-second slope while streaming "
                    "without a known sampling rate. Set "
                    "AcquisitionProfile.sampling_rate_hz, set scale_by_dt=False, "
                    "or process the signal in batch."
                )
            inv_dt = profile.sampling_rate_hz
        return _WindowedSlopeOperator(self.window_size, inv_dt, out_layout)


class _WindowedSlopeOperator(StreamOperator):
    """Fits a least-squares line to the trailing window and returns its slope."""

    def __init__(self, window: int, inv_dt: float, out_layout: Layout) -> None:
        """Precompute centered positions and denominators for every window fill."""

        self._window = window
        self._inv_dt = inv_dt
        self._out_layout = out_layout
        self._time_index = out_layout.dynamic_index
        self._samples: deque[RealArray] = deque()
        # For a window of `n` samples, slope = sum(centered_pos * value) / denom.
        self._fits = {n: _slope_fit(n) for n in range(2, window + 1)}

    def push(self, chunk: Signal) -> Signal:
        """Emit the trailing-window slope at each new sample (0 until two arrive)."""

        if self._time_index is None:
            raise ValueError("WindowedSlope requires a signal with a time axis.")

        # Work with time on the first axis, so each new sample is just data[index].
        data = as_real_array(np.moveaxis(chunk.values, self._time_index, 0))
        if data.shape[0] == 0:
            return empty_signal(self._out_layout)

        output = np.zeros_like(data)
        for index in range(data.shape[0]):
            # Add the new sample to the trailing window, dropping the oldest one
            # once the window is full.
            self._samples.append(data[index])
            if len(self._samples) > self._window:
                self._samples.popleft()

            # Two samples are the minimum for a slope; leave 0 until we have them.
            count = len(self._samples)
            if count < 2:
                continue

            # Fit a line to the window and keep its slope. With the positions
            # centered on zero, that slope is sum(position * value) / sum(position^2).
            positions, denom = self._fits[count]
            block = np.stack(tuple(self._samples), axis=0)
            slope = np.tensordot(positions, block, axes=(0, 0)) / (denom + _EPS)

            # inv_dt turns the per-sample slope into a per-second one (1.0 if off).
            output[index] = slope * self._inv_dt

        restored = as_real_array(np.moveaxis(output, 0, self._time_index))
        return chunk.with_values(restored, self._out_layout)

    def flush(self) -> Signal:
        """Nothing is emitted at the end; every sample was already produced."""

        return empty_signal(self._out_layout)
