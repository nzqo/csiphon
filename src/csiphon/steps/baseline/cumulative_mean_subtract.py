"""Causal subtraction of the mean of everything seen so far."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, as_real_array
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class CumulativeMeanSubtract(Step):
    """Subtract the mean of all samples so far, the current one included.

    Unlike RunningMeanSubtract, the mean never forgets anything. With `min_alpha`
    it does: after 1 / min_alpha samples, the mean turns exponential.
    """

    min_alpha: float = field(
        default=0.0,
        metadata={"doc": "smallest weight of a new sample in [0, 1); 0: never forget"},
    )

    # pylint: disable=duplicate-code  # boilerplate spec / method skeleton
    spec: ClassVar[StepSpec] = StepSpec(
        name="cumulative-mean-subtract",
        summary="subtract the mean of all samples so far",
        category=Category.BASELINE,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout) -> Layout:
        """Structure and semantics are unchanged."""

        self.require_inputs(layout)
        if not 0 <= self.min_alpha < 1:
            raise LayoutError(f"min_alpha must be in [0, 1), got {self.min_alpha}.")
        return layout

    def process(self, signal: Signal, out_layout: Layout) -> Signal:
        """Run over the whole recording as one chunk."""

        return _CumulativeMeanOperator(self.min_alpha, out_layout).push(signal)

    def stream(self, in_layout: Layout, out_layout: Layout) -> StreamOperator | None:
        """Return an operator that carries the mean across chunks."""

        return _CumulativeMeanOperator(self.min_alpha, out_layout)


class _CumulativeMeanOperator(StreamOperator):
    """Carries the mean and the number of samples behind it across chunks."""

    def __init__(self, min_alpha: float, out_layout: Layout) -> None:
        """Bind the smallest weight and the output layout."""

        self._min_alpha = min_alpha
        self._out_layout = out_layout
        self._time_index = out_layout.dynamic_index
        self._mean: RealArray | None = None
        self._count = 0

    def push(self, chunk: Signal) -> Signal:
        """Subtract the mean so far from every sample, updating it as we go."""

        if self._time_index is None or chunk.n_samples == 0:
            return empty_signal(self._out_layout)

        # The time axis can sit anywhere in the layout. Moving it to the front makes
        # data[index] the sample at that index.
        data = np.moveaxis(as_real_array(chunk.values), self._time_index, 0)

        # The mean after each sample, that sample included. Keep the last one (and
        # the count) for the next chunk.
        means = self._running_means(data)
        self._count += data.shape[0]
        self._mean = means[-1].copy()

        # Subtract in place: `means` becomes the output, so no further array is
        # allocated.
        np.subtract(data, means, out=means)

        # Move time back to where it was, so the output keeps the input's layout.
        restored = np.moveaxis(means, 0, self._time_index)
        return chunk.with_values(restored, self._out_layout)

    def _running_means(self, data: RealArray) -> RealArray:
        """Return the mean after each sample, continuing from the previous chunk."""

        n_samples = data.shape[0]
        counts = self._count + np.arange(1, n_samples + 1)
        previous_mean = np.zeros_like(data[0]) if self._mean is None else self._mean
        means = np.empty_like(data)

        # A new sample has the weight 1 / count while that is at least min_alpha,
        # which keeps the mean the exact average of all samples so far. From then
        # on the weight stays at min_alpha and old samples fade out exponentially.
        n_exact = int(np.count_nonzero(1.0 / counts >= self._min_alpha))

        # Exact part: (sum of all earlier samples + running sum of this chunk) / count.
        if n_exact > 0:
            counts_shape = (-1,) + (1,) * (data.ndim - 1)
            np.cumsum(data[:n_exact], axis=0, out=means[:n_exact])
            means[:n_exact] += self._count * previous_mean
            means[:n_exact] /= counts[:n_exact].reshape(counts_shape)

        # Exponential part: each mean depends on the one before, so this is a loop.
        mean = means[n_exact - 1] if n_exact > 0 else previous_mean
        for index in range(n_exact, n_samples):
            mean = mean + self._min_alpha * (data[index] - mean)
            means[index] = mean

        return means

    def flush(self) -> Signal:
        """Nothing is buffered beyond the mean."""

        return empty_signal(self._out_layout)
