"""Whole-recording temporal-mean subtraction (batch-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from csiphon.core.arrays import as_real_array
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class TemporalMeanSubtract(Step):
    """Subtract each channel's mean over the whole recording (batch-only).

    This strips absolute cross-time energy (the SHARP recipe). Because it needs
    every time sample, it cannot stream, use
    RunningMeanSubtract
    for a causal approximation.
    """

    # pylint: disable=duplicate-code  # boilerplate spec / method skeleton
    spec: ClassVar[StepSpec] = StepSpec(
        name="temporal-mean-subtract",
        summary="subtract each channel's mean over the whole recording",
        category=Category.BASELINE,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(),
        streaming=Streaming.UNAVAILABLE,
        streaming_note="needs every sample; use RunningMeanSubtract to stream",
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Structure and semantics are unchanged."""

        self.require_inputs(layout)
        return layout

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Subtract the mean over the time axis."""

        time_index = signal.layout.dynamic_index
        if time_index is None or signal.n_samples == 0:
            return signal.with_values(signal.values, out_layout)

        data = as_real_array(signal.values)
        mean = data.mean(axis=time_index, keepdims=True)
        return signal.with_values(data - mean, out_layout)

    # stream() inherits the default (None): this step is batch-only.
