"""Whole-recording maximum normalization (batch-only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from csiphon.core.arrays import as_real_array
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming

EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class GlobalMaxNormalize(Step):
    """Divide the whole recording by its single largest value.

    Unlike `PerFrameMaxNormalize`, which rescales each time frame on its own, this
    divides every sample by one global maximum taken over the entire recording, so
    relative differences between frames are preserved.

    Because it needs every sample at once it cannot stream.
    """

    epsilon: float = field(
        default=EPSILON, metadata={"doc": "floor on the divisor to avoid dividing by 0"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="global-max-normalize",
        summary="divide the whole recording by its global maximum",
        category=Category.NORMALIZATION,
        admissible_values=(ValueKind.POWER, ValueKind.REAL, ValueKind.MAGNITUDE),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(),
        streaming=Streaming.UNAVAILABLE,
        streaming_note="needs every sample to find the global maximum",
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Structure and semantics are unchanged."""

        self.require_inputs(layout)
        return layout

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Divide by the global maximum over every axis."""

        data = as_real_array(signal.values)
        peak = float(data.max()) if data.size else 0.0
        return signal.with_values(data / max(peak, self.epsilon), out_layout)

    # stream() inherits the default (None): this step is batch-only.
