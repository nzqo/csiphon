"""Clip values up to a noise floor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from csiphon.core.arrays import SignalArray, as_real_array
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class NoiseFloorClip(PointwiseStep):
    """Clip values below `10 ** level_log10` up to that floor."""

    level_log10: float = field(
        default=-4.5, metadata={"doc": "log10 of the noise floor"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="noise-floor-clip",
        summary="raise values below 10**level_log10 up to that floor",
        category=Category.CLEANING,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(note="values clipped to a floor"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Structure and semantics are unchanged."""

        self.require_inputs(layout)
        return layout

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Raise everything below the floor to the floor."""

        floor = 10.0**self.level_log10
        data = as_real_array(values).copy()
        data[data < floor] = floor
        return data
