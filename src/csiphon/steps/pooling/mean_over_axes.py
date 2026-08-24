"""Average over one or more named axes, removing them."""
# The config-dependent spec shape (run-time axes, deferred layout effect) is shared
# verbatim with the other run-time-axis steps, so pylint reads the specs as clones.
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import (
    CONFIG_DEPENDENT,
    Category,
    LayoutEffect,
    StepSpec,
    Streaming,
)


@dataclass(frozen=True, slots=True)
class MeanOverAxes(PointwiseStep):
    """Average over one or more named axes, dropping them from the layout.

    For example, mean over `subcarrier` turns a `(time, rx, subcarrier)` signal
    into `(time, rx)`. The value kind and representation are unchanged (an average
    of magnitudes is still a magnitude). Each averaged axis must be static: the
    time axis cannot be averaged this way, because that is a whole-recording
    reduction rather than a per-sample one. Averaging keeps this pointwise in
    time, so it streams exactly.
    """

    axes: tuple[AxisName, ...] = field(
        default=(AxisName.SUBCARRIER,),
        metadata={"doc": "named axes to average over (and remove)"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="mean-over-axes",
        summary="average over named axes, removing them",
        category=Category.POOLING,
        admissible_values=None,
        admissible_reprs=None,
        # The required axes are chosen at run time; see resolve_required_axes.
        requires_axes=(),
        layout_effect=CONFIG_DEPENDENT,
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require every axis this step averages over to be present."""

        return self.axes

    def resolve_layout_effect(self) -> LayoutEffect:
        """Remove every averaged axis; the rest is unchanged."""

        return LayoutEffect(removes=self.axes, note="averaged away")

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Drop the averaged axes; keep everything else."""

        self.require_inputs(layout)
        if not self.axes:
            raise LayoutError("MeanOverAxes needs at least one axis to average over.")
        if len(set(self.axes)) != len(self.axes):
            raise LayoutError("MeanOverAxes was given a repeated axis.")
        for name in self.axes:
            # A static axis check also rejects the time axis (a per-sample step
            # must not reduce across time).
            layout.require_static_axis(name)
        return layout.remove_axes(self.axes)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Average the array over the chosen axes at once."""

        positions = tuple(signal.layout.axis_position(name) for name in self.axes)
        return as_signal_array(np.mean(values, axis=positions))
