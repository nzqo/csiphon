"""Sliding-window summation over the time axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import as_real_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError, StreamingError
from csiphon.core.layout import Layout
from csiphon.core.sampling import (
    WindowDeterminationMode,
    check_jitter,
    divide_nominal_rate,
    effective_rate_hz,
)
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming
from csiphon.steps._support.time_windows import TimeWindowSumOperator
from csiphon.steps._support.windowing import SampleWindowOperator, WindowInSamples


@dataclass(frozen=True, slots=True)
class WindowSum(Step):
    """Sum values inside each sliding window.

    You give the window length and hop in seconds. Windows hop along by `hop_s`,
    and each output sample sits at the center of its window. The time axis
    becomes coarser (one sample per window) and other axes are untouched.

    `window_mode` decides which samples a window holds:

    - FROM_NOMINAL_RATE: a fixed number of samples (`window_s * sampling_rate`).
    - FROM_TIMESTAMPS: every sample whose timestamp falls inside the window. A
      window with no samples gives NaN (see NanScrub), so the output length
      depends only on the duration of the recording, not on lost samples. A
      window with fewer samples sums fewer values.

    Streams via a buffer in both modes.
    """

    window_s: float = field(default=0.128, metadata={"doc": "window length in seconds"})
    hop_s: float = field(
        default=0.10, metadata={"doc": "hop between windows in seconds"}
    )
    window_mode: WindowDeterminationMode = field(
        default=WindowDeterminationMode.FROM_NOMINAL_RATE,
        metadata={"doc": "windows by sample count (nominal rate) or by timestamp"},
    )
    strict: bool = field(
        default=False, metadata={"doc": "raise (not warn) on excessive jitter"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="window-sum",
        summary="sum values inside each sliding window",
        category=Category.POOLING,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(note="time samples grouped into windows"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout) -> Layout:
        """Time stays the (dynamic) axis; every other axis is unchanged."""

        self.require_inputs(layout)
        if self.window_s <= 0:
            raise LayoutError(f"window_s must be > 0, got {self.window_s}.")
        if self.hop_s <= 0:
            raise LayoutError(f"hop_s must be > 0, got {self.hop_s}.")

        # One output sample per hop. In timestamp mode that is exactly 1 / hop_s.
        if self.window_mode == WindowDeterminationMode.FROM_TIMESTAMPS:
            return layout.with_nominal_rate(1.0 / self.hop_s)

        # Without a known input rate the output has none.
        if layout.nominal_rate_hz is None:
            return layout
        hop_stride = WindowInSamples.from_seconds(
            layout.nominal_rate_hz, self.window_s, self.hop_s
        ).hop_stride
        return divide_nominal_rate(layout, hop_stride)

    def process(self, signal: Signal, out_layout: Layout) -> Signal:
        """Vectorized sliding-window sum via a cumulative sum."""

        time_index = signal.layout.dynamic_index
        if time_index is None:
            raise ValueError("WindowSum requires a time axis.")

        # By timestamp, the streaming operator already does the whole job: batch is
        # just a single chunk holding the entire recording.
        if self.window_mode == WindowDeterminationMode.FROM_TIMESTAMPS:
            operator = TimeWindowSumOperator(
                self.window_s, self.hop_s, signal.layout, out_layout
            )
            return operator.push(signal)

        # By sample count, a window is a fixed number of samples, which assumes an
        # evenly spaced signal. Strongly irregular timestamps break that, so we
        # warn about them (or raise, with `strict`).
        check_jitter(signal.times, step=self.name, strict=self.strict)

        # The window and hop are given in seconds, but here windows hold a fixed
        # number of samples. Converting takes the sampling rate: the nominal one,
        # or one estimated from the timestamps if the profile gave none.
        rate_hz = effective_rate_hz(signal.times, signal.layout.nominal_rate_hz)
        window = WindowInSamples.from_seconds(rate_hz, self.window_s, self.hop_s)

        # Put time first, and find the sample where each window starts and its
        # center sample (which gives the window its output time).
        values = as_real_array(np.moveaxis(signal.values, time_index, 0))
        starts, centers = window.indices(values.shape[0])

        # Sliding-window sum via a cumulative sum: prepend one zero row so every
        # window's sum is cumulative[start + w] - cumulative[start]; then keep only
        # the hop-spaced starts.
        cumulative = np.cumsum(values, axis=0)
        cumulative = np.concatenate([np.zeros_like(cumulative[:1]), cumulative], axis=0)
        window_size = window.window_size
        all_window_sums = cumulative[window_size:] - cumulative[:-window_size]
        windowed = all_window_sums[starts]

        # Put the window axis back where the time axis was. Each window takes the
        # timestamp of its center sample.
        restored = np.moveaxis(windowed, 0, time_index)
        return signal.with_values(
            restored, out_layout, times=signal.times[centers], coords=signal.coords
        )

    def stream(self, in_layout: Layout, out_layout: Layout) -> StreamOperator | None:
        """Return the operator that sums windows as chunks arrive.

        Windows can span chunks, so the operator keeps the samples that a window
        still needs. Which operator that is depends on `window_mode`.
        """

        # By timestamp, windows are placed by time and need no sampling rate.
        if self.window_mode == WindowDeterminationMode.FROM_TIMESTAMPS:
            return TimeWindowSumOperator(
                self.window_s, self.hop_s, in_layout, out_layout
            )

        # By sample count, we need the window in samples, which takes the sampling
        # rate. Batch can estimate it from the timestamps. Streaming cannot, since
        # the later timestamps have not arrived yet, so the profile must give it.
        rate_hz = in_layout.nominal_rate_hz
        if rate_hz is None:
            raise StreamingError(
                f"'{self.name}' needs the sampling rate to turn window_s into a "
                "number of samples, and streaming cannot estimate it from "
                "timestamps it has not seen yet. Set "
                "AcquisitionProfile.sampling_rate_hz, use "
                "window_mode=FROM_TIMESTAMPS, or run in batch."
            )
        window = WindowInSamples.from_seconds(rate_hz, self.window_s, self.hop_s)

        # We have a helper for stream operators that run a function on fixed
        # sliding windows. We tell it the window (its size and how far it moves)
        # and which function to run on each window. It keeps the chunks it has
        # seen so far as internal state, and slides the window over them as
        # specified. Here, the function sums each window over time.
        return SampleWindowOperator(
            window,
            lambda block: as_real_array(block.sum(axis=0)),
            in_layout,
            out_layout,
        )
