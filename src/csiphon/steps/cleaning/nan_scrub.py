"""Replace non-finite values with finite numbers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class NanScrub(PointwiseStep):
    """Replace NaN / inf with finite numbers (`numpy.nan_to_num`)."""

    # pylint: disable=duplicate-code  # boilerplate spec / method skeleton
    spec: ClassVar[StepSpec] = StepSpec(
        name="nan-scrub",
        summary="replace NaN / inf with finite numbers",
        category=Category.CLEANING,
        admissible_values=None,
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(note="non-finite values scrubbed"),
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
        """Return `nan_to_num(values)`."""

        return as_signal_array(np.nan_to_num(values))
