"""Butterworth filtering along time (optional `[filters]` extra -> scipy).

Batch uses zero-phase `filtfilt` (the best-quality, non-causal choice when the
whole recording is available). Streaming uses a causal `lfilter` carrying its
state across chunks, a genuinely different (causal) variant, so batch and
streaming are intentionally not identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import ModuleType
from typing import ClassVar, cast

import numpy as np

from csiphon.core.arrays import RealArray, as_real_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError, MissingDependencyError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.sampling import effective_rate_hz
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


class Band(StrEnum):
    """Which frequencies a Butterworth filter keeps."""

    # fmt: off
    LOW  = "low"
    HIGH = "high"
    BAND = "band"
    # fmt: on


def _scipy_signal() -> ModuleType:
    """Import scipy.signal lazily with a helpful error if it is missing."""

    try:
        # Lazy so the base install stays numpy-only.
        from scipy import signal  # pylint: disable=import-outside-toplevel
    except ImportError as error:  # pragma: no cover - exercised via message
        raise MissingDependencyError("scipy", "filters") from error
    return cast(ModuleType, signal)


def _design(
    order: int, cutoff_hz: float | tuple[float, float], btype: Band, rate_hz: float
) -> tuple[RealArray, RealArray]:
    """Design Butterworth (numerator, denominator) coefficients."""

    scipy_signal = _scipy_signal()
    nyquist = rate_hz / 2.0
    if isinstance(cutoff_hz, tuple):
        normalized_cutoff: float | tuple[float, float] = (
            cutoff_hz[0] / nyquist,
            cutoff_hz[1] / nyquist,
        )
    else:
        normalized_cutoff = cutoff_hz / nyquist
    numerator, denominator = scipy_signal.butter(order, normalized_cutoff, btype=btype)
    return numerator, denominator


@dataclass(frozen=True, slots=True)
class ButterworthFilter(Step):
    """Butterworth filter along the time axis.

    `cutoff_hz` is a scalar for `low` / `high` and a `(low, high)` pair
    for `band`. Batch is zero-phase; streaming is causal.
    """

    cutoff_hz: float | tuple[float, float] = field(
        default=1.0, metadata={"doc": "cutoff in Hz (pair for band-pass)"}
    )
    order: int = field(default=2, metadata={"doc": "filter order"})
    btype: Band = field(
        default=Band.HIGH,
        metadata={"doc": "which band to keep: low, high, or band-pass"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="butterworth-filter",
        summary="apply a Butterworth filter along time (zero-phase batch, causal stream)",  # noqa: E501
        category=Category.FILTERING,
        admissible_values=(ValueKind.REAL, ValueKind.MAGNITUDE, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(),
        streaming=Streaming.BATCH_DIVERGENT,
        streaming_note="causal lfilter; batch is zero-phase filtfilt",
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Filtering keeps the structure and semantics unchanged."""

        self.require_inputs(layout)
        if self.order < 1:
            raise LayoutError(f"order must be >= 1, got {self.order}.")
        if self.btype == Band.BAND:
            if not (isinstance(self.cutoff_hz, tuple) and len(self.cutoff_hz) == 2):
                raise LayoutError(
                    "cutoff_hz must be a (low, high) pair for band-pass, "
                    f"got {self.cutoff_hz}."
                )
            low, high = self.cutoff_hz
            if not 0 < low < high:
                raise LayoutError(
                    "cutoff_hz must satisfy 0 < low < high for band-pass, "
                    f"got {self.cutoff_hz}."
                )
        else:
            if isinstance(self.cutoff_hz, tuple):
                raise LayoutError(
                    f"cutoff_hz must be a scalar for {self.btype} filtering, "
                    f"got {self.cutoff_hz}."
                )
            if self.cutoff_hz <= 0:
                raise LayoutError(f"cutoff_hz must be > 0, got {self.cutoff_hz}.")
        return layout

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Zero-phase filtfilt over the whole recording."""

        time_index = signal.layout.dynamic_index
        if time_index is None:
            raise LayoutError("ButterworthFilter requires a time axis.")

        scipy_signal = _scipy_signal()
        rate_hz = effective_rate_hz(signal.times, profile.sampling_rate_hz)
        numerator, denominator = _design(
            self.order, self.cutoff_hz, self.btype, rate_hz
        )
        filtered = scipy_signal.filtfilt(
            numerator, denominator, as_real_array(signal.values), axis=time_index
        )
        return signal.with_values(filtered, out_layout)

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Return a causal lfilter operator (needs a nominal rate)."""

        if profile.sampling_rate_hz is None:
            return None
        numerator, denominator = _design(
            self.order, self.cutoff_hz, self.btype, profile.sampling_rate_hz
        )
        return _CausalFilter(numerator, denominator, out_layout)


class _CausalFilter(StreamOperator):
    """Causal lfilter carrying its state across chunks, per channel."""

    def __init__(
        self, numerator: RealArray, denominator: RealArray, out_layout: Layout
    ) -> None:
        """Bind the filter coefficients; state is created on the first chunk."""

        self._numerator = numerator
        self._denominator = denominator
        self._out_layout = out_layout
        self._state: RealArray | None = None

    def push(self, chunk: Signal) -> Signal:
        """Filter this chunk causally, updating the filter state."""

        time_index = chunk.layout.dynamic_index
        if time_index is None or chunk.n_samples == 0:
            return empty_signal(self._out_layout)

        scipy_signal = _scipy_signal()
        data = np.moveaxis(as_real_array(chunk.values), time_index, 0)
        flat = data.reshape(data.shape[0], -1)

        if self._state is None:
            # Seed each channel's filter state from its first sample so the output
            # starts at steady state instead of a transient.
            steady = scipy_signal.lfilter_zi(self._numerator, self._denominator)
            self._state = np.outer(steady, flat[0])

        filtered, self._state = scipy_signal.lfilter(
            self._numerator, self._denominator, flat, axis=0, zi=self._state
        )
        restored = np.moveaxis(filtered.reshape(data.shape), 0, time_index)
        return chunk.with_values(restored, self._out_layout)

    def flush(self) -> Signal:
        """No buffered samples beyond the filter state."""

        return empty_signal(self._out_layout)
