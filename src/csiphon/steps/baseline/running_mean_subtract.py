"""Causal running-mean subtraction (streaming alternative)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, as_real_array
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class RunningMeanSubtract(Step):
    """Subtract a causal exponential running mean.

    `mean_t = alpha * x_t + (1 - alpha) * mean_{t-1}` and the output is
    `x_t - mean_t`. Batch and streaming produce identical results because the
    recurrence is causal.
    """

    alpha: float = field(
        default=0.01, metadata={"doc": "EWMA smoothing factor in (0, 1]"}
    )

    # pylint: disable=duplicate-code  # boilerplate spec / method skeleton
    spec: ClassVar[StepSpec] = StepSpec(
        name="running-mean-subtract",
        summary="subtract a causal exponential running mean",
        category=Category.BASELINE,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Structure and semantics are unchanged."""

        self.require_inputs(layout)
        if not 0 < self.alpha <= 1:
            raise LayoutError(f"alpha must be in (0, 1], got {self.alpha}.")
        return layout

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Run the causal recurrence over the whole recording."""

        return _RunningMeanOperator(self.alpha, out_layout).push(signal)

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Return a stateful operator carrying the running mean across chunks."""

        return _RunningMeanOperator(self.alpha, out_layout)


class _RunningMeanOperator(StreamOperator):
    """Carries the exponential running mean across chunks."""

    def __init__(self, alpha: float, out_layout: Layout) -> None:
        """Bind the smoothing factor and output layout."""

        self._alpha = alpha
        self._out_layout = out_layout
        self._running_mean: RealArray | None = None
        self._time_index = out_layout.dynamic_index

    def push(self, chunk: Signal) -> Signal:
        """Subtract the running mean, updating state sample by sample."""

        if self._time_index is None or chunk.n_samples == 0:
            return empty_signal(self._out_layout)

        # Iterate over time (first axis) so the recurrence stays causal across
        # chunk boundaries; the running mean persists in self._running_mean.
        data = np.moveaxis(as_real_array(chunk.values), self._time_index, 0)
        output = np.empty_like(data)
        running_mean = (
            np.zeros_like(data[0]) if self._running_mean is None else self._running_mean
        )
        is_first_ever = self._running_mean is None
        for index in range(data.shape[0]):
            sample = data[index]
            if is_first_ever and index == 0:
                # Seed the running mean with the very first sample so the output
                # starts at zero instead of a large transient.
                running_mean = sample
            else:
                running_mean = self._alpha * sample + (1.0 - self._alpha) * running_mean
            output[index] = sample - running_mean
        self._running_mean = running_mean

        restored = np.moveaxis(output, 0, self._time_index)
        return chunk.with_values(restored, self._out_layout)

    def flush(self) -> Signal:
        """Nothing is buffered beyond the running mean."""

        return empty_signal(self._out_layout)
