"""Natural-log compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_real_array
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming

EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class LogScale(PointwiseStep):
    """Natural-log scaling: `log(values + epsilon)`.

    For a decibel scale, use ToDecibels instead.
    """

    epsilon: float = field(
        default=EPSILON, metadata={"doc": "additive floor before log"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="log-compress",
        summary="scale by the natural log, log(x + epsilon)",
        category=Category.SCALING,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(value_kind=ValueKind.REAL),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Scaling keeps the representation but yields a plain real value."""

        self.require_inputs(layout)
        return layout.with_values(ValueKind.REAL)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Return `log(values + epsilon)`."""

        return as_real_array(np.log(as_real_array(values) + self.epsilon))
