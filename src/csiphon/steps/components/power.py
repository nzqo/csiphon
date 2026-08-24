"""Element-wise power (squared magnitude)."""

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
class Power(PointwiseStep):
    """Convert magnitude / complex values to power (squared magnitude)."""

    spec: ClassVar[StepSpec] = StepSpec(
        name="power",
        summary="square the magnitude to power",
        category=Category.COMPONENTS,
        admissible_values=(ValueKind.COMPLEX, ValueKind.MAGNITUDE, ValueKind.REAL),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(value_kind=ValueKind.POWER),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Any amplitude-like input becomes power."""

        self.require_inputs(layout)
        return layout.with_values(ValueKind.POWER)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Return `abs(values) ** 2`."""

        if np.iscomplexobj(values):
            return as_real_array(np.abs(values) ** 2)
        return as_real_array(values**2)
