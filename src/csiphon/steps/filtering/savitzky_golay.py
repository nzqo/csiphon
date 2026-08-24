"""Savitzky-Golay smoothing along time (optional `[filters]` extra -> scipy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import ClassVar, cast

from csiphon.core.arrays import as_signal_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import DataError, LayoutError, MissingDependencyError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


def _scipy_signal() -> ModuleType:
    """Import scipy.signal lazily with a helpful error if it is missing."""

    try:
        from scipy import signal  # pylint: disable=import-outside-toplevel
    except ImportError as error:  # pragma: no cover - exercised via message
        raise MissingDependencyError("scipy", "filters") from error
    return cast(ModuleType, signal)


@dataclass(frozen=True, slots=True)
class SavitzkyGolay(Step):
    """Savitzky-Golay polynomial smoothing along time.

    Fits a low-order polynomial in a sliding window and takes its center value:
    a denoiser that preserves peaks better than a moving average. The batch
    filter is centered (non-causal), so it is batch-only. Needs scipy (the
    `[filters]` extra).
    """

    window_length: int = field(
        default=11, metadata={"doc": "window length in samples (odd)"}
    )
    polyorder: int = field(
        default=2, metadata={"doc": "polynomial order (< window_length)"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="savitzky-golay",
        summary="smooth along time with a Savitzky-Golay filter",
        category=Category.FILTERING,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(note="smoothed along time"),
        streaming=Streaming.UNAVAILABLE,
        streaming_note="centered (non-causal) filter, so batch-only; needs scipy",
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Shape and semantics are unchanged; the signal is only smoothed."""

        self.require_inputs(layout)
        if self.window_length < 1:
            raise LayoutError(f"window_length must be >= 1, got {self.window_length}.")
        if self.polyorder < 0:
            raise LayoutError(f"polyorder must be >= 0, got {self.polyorder}.")
        if self.window_length % 2 == 0:
            raise LayoutError("SavitzkyGolay window_length must be odd.")
        if self.polyorder >= self.window_length:
            raise LayoutError("SavitzkyGolay needs polyorder < window_length.")
        return layout

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Apply the Savitzky-Golay filter along the time axis."""

        time_index = signal.layout.dynamic_index
        if time_index is None:
            raise ValueError("SavitzkyGolay requires a time axis.")

        # The filter fits its polynomial over a full window, so a recording shorter
        # than one window has nothing to fit. Fail clearly instead of letting scipy
        # raise a cryptic error.
        if signal.n_samples < self.window_length:
            raise DataError(
                f"SavitzkyGolay needs at least window_length={self.window_length} "
                f"samples, but got {signal.n_samples}."
            )

        smoothed = _scipy_signal().savgol_filter(
            as_signal_array(signal.values),
            window_length=self.window_length,
            polyorder=self.polyorder,
            axis=time_index,
            mode="interp",
        )
        return signal.with_values(as_signal_array(smoothed), out_layout)
