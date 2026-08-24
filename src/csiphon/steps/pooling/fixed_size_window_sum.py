"""Sliding-window summation over the time axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import as_real_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError, StreamingError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.sampling import check_jitter, effective_rate_hz
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming
from csiphon.steps._support.windowing import WindowedOperator, WindowGeometry


@dataclass(frozen=True, slots=True)
class FixedSizeWindowSum(Step):
    """Sum values inside each sliding window.

    You give the window length in seconds, but it resolves to a fixed number of
    samples (window_s * sampling_rate); windows hop along by `hop_s`. Reduces the
    per-sample time axis to a coarser per-window one and leaves other axes alone.
    Streams via a ring buffer once the sampling rate is known.
    """

    window_s: float = field(default=0.128, metadata={"doc": "window length in seconds"})
    hop_s: float = field(
        default=0.10, metadata={"doc": "hop between windows in seconds"}
    )
    strict: bool = field(
        default=False, metadata={"doc": "raise (not warn) on excessive jitter"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="fixed-size-window-sum",
        summary="sum values inside each sliding fixed-size window",
        category=Category.POOLING,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(note="time samples grouped into windows"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Time stays the (dynamic) axis; every other axis is unchanged."""

        self.require_inputs(layout)
        if self.window_s <= 0:
            raise LayoutError(f"window_s must be > 0, got {self.window_s}.")
        if self.hop_s <= 0:
            raise LayoutError(f"hop_s must be > 0, got {self.hop_s}.")
        return layout

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Vectorized sliding-window sum via a cumulative sum."""

        time_index = signal.layout.dynamic_index
        if time_index is None:
            raise ValueError("FixedSizeWindowSum requires a time axis.")

        check_jitter(signal.times, step=self.name, strict=self.strict)
        rate_hz = effective_rate_hz(signal.times, profile.sampling_rate_hz)
        geometry = WindowGeometry.from_seconds(rate_hz, self.window_s, self.hop_s)
        values = as_real_array(np.moveaxis(signal.values, time_index, 0))
        starts, centers = geometry.indices(values.shape[0])

        # Sliding-window sum via a cumulative sum: prepend one zero row so every
        # window's sum is cumulative[start + w] - cumulative[start]; then keep only
        # the hop-spaced starts.
        cumulative = np.cumsum(values, axis=0)
        cumulative = np.concatenate([np.zeros_like(cumulative[:1]), cumulative], axis=0)
        window_size = geometry.window_size
        all_window_sums = cumulative[window_size:] - cumulative[:-window_size]
        windowed = all_window_sums[starts]

        restored = np.moveaxis(windowed, 0, time_index)
        return signal.with_values(
            restored, out_layout, times=signal.times[centers], coords=signal.coords
        )

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Ring-buffer operator that emits one window sum at a time."""

        if profile.sampling_rate_hz is None:
            raise StreamingError(
                f"'{self.name}' aggregates over a fixed number of samples per "
                "window (window_s * sampling_rate; the fixed size is an "
                "implementation detail). Streaming has to know that size up "
                "front, so set AcquisitionProfile.sampling_rate_hz or run in batch."
            )
        geometry = WindowGeometry.from_seconds(
            profile.sampling_rate_hz, self.window_s, self.hop_s
        )
        return WindowedOperator(
            geometry, lambda block: block.sum(axis=0), in_layout, out_layout
        )
