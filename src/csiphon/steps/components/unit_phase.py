"""Unit-magnitude phase view of complex values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class UnitPhase(PointwiseStep):
    """Keep the phase as a unit-magnitude complex number: `exp(j*angle(z))`.

    Drops the amplitude but keeps a complex value (unlike Phase, which returns
    real radians). Pointwise, so it streams exactly.
    """

    spec: ClassVar[StepSpec] = StepSpec(
        name="unit-phase",
        summary="keep the phase as a unit-magnitude complex number",
        category=Category.COMPONENTS,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(note="unit-magnitude phase"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Shape and semantics are unchanged; only the magnitude is dropped."""

        self.require_inputs(layout)
        return layout

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Replace each value with its unit-magnitude phase."""

        return as_signal_array(np.exp(1j * np.angle(as_signal_array(values))))
