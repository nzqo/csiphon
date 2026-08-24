"""Stack complex values into a real [amplitude | phase] axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class StackAmpPhase(PointwiseStep):
    """Split complex values into real `[amplitude | phase]` along an axis.

    Doubles the axis: the first half is `|z|`, the second `angle(z)`, turning a
    complex channel into a real feature vector. Pointwise, so it streams exactly.
    (For standardization, follow with a normalization step.)
    """

    axis: AxisName = field(
        default=AxisName.SUBCARRIER, metadata={"doc": "axis to stack along"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="stack-amp-phase",
        summary="stack complex values into real [amplitude | phase] along an axis",
        category=Category.RESTRUCTURING,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        # The required axis is chosen at run time; see resolve_required_axes.
        requires_axes=(),
        layout_effect=LayoutEffect(
            value_kind=ValueKind.REAL, note="amplitude then phase; axis doubles"
        ),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require the configured axis to be present."""

        return (self.axis,)

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Double the axis (amplitude then phase) and make the values real."""

        self.require_inputs(layout)
        axis = layout.require_static_axis(self.axis)
        # a static axis always has a concrete size.
        assert axis.size is not None
        doubled = Axis.sized(self.axis, 2 * axis.size)
        return layout.replace_axis(self.axis, doubled).with_values(ValueKind.REAL)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Concatenate amplitude and phase along the chosen axis."""

        position = signal.layout.axis_position(self.axis)
        complex_values = as_signal_array(values)
        amplitude = np.abs(complex_values)
        phase = np.angle(complex_values)
        return as_signal_array(np.concatenate([amplitude, phase], axis=position))
