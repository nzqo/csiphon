"""Decibel compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_real_array
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming

EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class ToDecibels(PointwiseStep):
    """Decibel compression: `10 * log10(values / reference + epsilon)`."""

    reference: float = field(default=1.0, metadata={"doc": "reference level for 0 dB"})
    epsilon: float = field(
        default=EPSILON, metadata={"doc": "additive floor before log10"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="to-decibels",
        summary="compress to decibels, 10 log10(x / reference + epsilon)",
        category=Category.SCALING,
        admissible_values=(ValueKind.POWER, ValueKind.MAGNITUDE, ValueKind.REAL),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(value_kind=ValueKind.DECIBELS),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Power becomes decibels."""

        self.require_inputs(layout)
        if self.reference <= 0:
            raise LayoutError(f"reference must be > 0, got {self.reference}.")
        return layout.with_values(ValueKind.DECIBELS)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Return decibels relative to `reference`."""

        ratio = as_real_array(values) / self.reference + self.epsilon
        return as_real_array(10.0 * np.log10(ratio))
