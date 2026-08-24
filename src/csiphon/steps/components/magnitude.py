"""Element-wise magnitude."""

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
class Magnitude(PointwiseStep):
    """Take the element-wise magnitude, discarding phase."""

    spec: ClassVar[StepSpec] = StepSpec(
        name="magnitude",
        summary="take the magnitude, discarding phase",
        category=Category.COMPONENTS,
        admissible_values=None,
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(value_kind=ValueKind.MAGNITUDE),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Complex or real input becomes a real magnitude."""

        self.require_inputs(layout)
        return layout.with_values(ValueKind.MAGNITUDE)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Return `abs(values)`."""

        return as_real_array(np.abs(values))
