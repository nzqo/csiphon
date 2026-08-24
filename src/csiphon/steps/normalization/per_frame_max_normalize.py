"""Per-frame maximum normalization."""

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
class PerFrameMaxNormalize(PointwiseStep):
    """Divide each time frame by its maximum over all other axes."""

    # pylint: disable=duplicate-code  # boilerplate spec / method skeleton
    spec: ClassVar[StepSpec] = StepSpec(
        name="per-frame-max-normalize",
        summary="divide each time frame by its maximum over the other axes",
        category=Category.NORMALIZATION,
        admissible_values=(ValueKind.POWER, ValueKind.REAL, ValueKind.MAGNITUDE),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(),
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
        """Divide by the per-frame max (0 where the max is 0)."""

        data = as_real_array(values)
        time_index = signal.layout.dynamic_index
        other_axes = tuple(axis for axis in range(data.ndim) if axis != time_index)
        peak = data.max(axis=other_axes, keepdims=True)
        return as_real_array(
            np.divide(data, peak, out=np.zeros_like(data), where=peak > 0)
        )
