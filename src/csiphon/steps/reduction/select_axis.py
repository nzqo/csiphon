"""Keep a subset of positions along one axis."""
# The CONFIG_DEPENDENT spec + resolve_* shape is shared with the other run-time-axis
# steps, so pylint reads these small step bodies as clones.
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import CONFIG_DEPENDENT, Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class SelectAxis(PointwiseStep):
    """Keep a subset of positions along one axis (e.g. subcarrier selection).

    The axis keeps its name and meaning; only its coordinates shrink. Pointwise
    in time, so it streams exactly.
    """

    axis: AxisName = field(
        default=AxisName.SUBCARRIER, metadata={"doc": "axis to select positions from"}
    )
    indices: tuple[int, ...] = field(
        default=(), metadata={"doc": "positions to keep along the axis"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="select-axis",
        summary="keep a subset of positions along one axis",
        category=Category.REDUCTION,
        admissible_values=None,
        admissible_reprs=None,
        # The required axis is chosen at run time; see resolve_required_axes.
        requires_axes=(),
        layout_effect=CONFIG_DEPENDENT,  # resizes the chosen axis; see resolve below
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require the configured axis to be present."""

        return (self.axis,)

    def resolve_layout_effect(self) -> LayoutEffect:
        """Keeping a subset resizes the chosen axis; nothing else changes."""

        return LayoutEffect(resizes=(self.axis,), note="keeps the selected positions")

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Shrink the selected axis to the kept coordinates."""

        self.require_inputs(layout)
        if not self.indices:
            raise LayoutError("SelectAxis needs at least one index to keep.")

        axis = layout.require_static_axis(self.axis)
        if any(index < 0 for index in self.indices):
            raise LayoutError(f"SelectAxis indices must be >= 0, got {self.indices}.")
        if axis.size is not None and any(index >= axis.size for index in self.indices):
            raise LayoutError(
                f"SelectAxis indices {self.indices} exceed the "
                f"{self.axis} axis size {axis.size}."
            )
        if axis.coordinates is not None:
            kept = tuple(axis.coordinates[index] for index in self.indices)
            reduced = Axis.static(self.axis, kept, unit=axis.unit)
        else:
            reduced = Axis.sized(self.axis, len(self.indices), unit=axis.unit)
        return layout.replace_axis(self.axis, reduced)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Take the kept indices along the selected axis."""

        position = signal.layout.axis_position(self.axis)
        return as_signal_array(np.take(values, self.indices, axis=position))
