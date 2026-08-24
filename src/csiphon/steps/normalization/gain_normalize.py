"""Per-sample gain normalization across an axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.core.axes import AxisName
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming

EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class GainNormalize(PointwiseStep):
    """
    Divide by the mean magnitude across an axis (default: subcarrier), per time sample.
    """

    axis: AxisName = field(
        default=AxisName.SUBCARRIER, metadata={"doc": "axis to average and divide by"}
    )
    epsilon: float = field(
        default=EPSILON, metadata={"doc": "additive floor to avoid divide-by-zero"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="gain-normalize",
        summary="divide by the mean magnitude over an axis, per time sample",
        category=Category.NORMALIZATION,
        admissible_values=(
            ValueKind.COMPLEX,
            ValueKind.MAGNITUDE,
            ValueKind.REAL,
            ValueKind.POWER,
        ),
        admissible_reprs=None,
        # The required axis is chosen at run time; see resolve_required_axes.
        requires_axes=(),
        layout_effect=LayoutEffect(),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require the configured axis to be present."""

        return (self.axis,)

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Require a static normalization axis; semantics are unchanged."""

        self.require_inputs(layout)
        layout.require_static_axis(self.axis)
        return layout

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Divide by the mean magnitude over `axis`, keeping dimensions and dtype."""

        position = signal.layout.axis_position(self.axis)
        data = as_signal_array(values)
        mean = np.mean(np.abs(data), axis=position, keepdims=True)
        return as_signal_array(data / (mean + self.epsilon))
