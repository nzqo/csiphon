"""Element-wise phase angle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_real_array
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class Phase(PointwiseStep):
    """Take the element-wise phase angle in radians."""

    spec: ClassVar[StepSpec] = StepSpec(
        name="phase",
        summary="take the phase angle in radians",
        category=Category.COMPONENTS,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(value_kind=ValueKind.PHASE_RADIANS),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Complex input becomes a real phase in radians."""

        self.require_inputs(layout)
        return layout.with_values(ValueKind.PHASE_RADIANS)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Return `angle(values)`."""

        return as_real_array(np.angle(values))
