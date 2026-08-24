"""Keep a window of delay taps and read them out as real features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_real_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class DelayTaps(PointwiseStep):
    """Truncate a delay response to a few taps and turn them real.

    Takes `num_taps` taps starting at `first_tap` from a complex delay axis.
    With `use_tap_abs` you get the tap magnitudes (`num_taps` values);
    otherwise you get the real and imaginary parts stacked (`2 * num_taps`
    values). This is the reduction half of the old delay-autocorrelation step:
    feed it the output of DelayAutocorrelation.
    """

    num_taps: int = field(default=3, metadata={"doc": "number of taps to keep"})
    first_tap: int = field(default=1, metadata={"doc": "index of the first tap kept"})
    use_tap_abs: bool = field(
        default=False, metadata={"doc": "keep |tap| instead of stacked re/im"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="delay-taps",
        summary="truncate a delay response to a few taps, as real features",
        category=Category.REDUCTION,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        requires_axes=(AxisName.DELAY,),
        layout_effect=LayoutEffect(
            value_kind=ValueKind.REAL,
            resizes=(AxisName.DELAY,),
            note="keeps num_taps taps",
        ),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Shrink the delay axis to the kept taps and drop to real values."""

        self.require_inputs(layout)
        if self.num_taps < 1:
            raise LayoutError(f"num_taps must be >= 1, got {self.num_taps}.")
        if self.first_tap < 0:
            raise LayoutError(f"first_tap must be >= 0, got {self.first_tap}.")
        delay = layout.require_static_axis(AxisName.DELAY)
        last_tap = self.first_tap + self.num_taps
        if delay.size is not None and last_tap > delay.size:
            raise LayoutError(
                f"taps {self.first_tap}..{last_tap - 1} exceed "
                f"delay axis size {delay.size}."
            )

        size = self.num_taps if self.use_tap_abs else 2 * self.num_taps
        return layout.replace_axis(
            AxisName.DELAY, Axis.sized(AxisName.DELAY, size, unit="tap")
        ).with_values(ValueKind.REAL)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Slice out the taps, then take magnitudes or stack re/im."""

        position = signal.layout.axis_position(AxisName.DELAY)
        taps = range(self.first_tap, self.first_tap + self.num_taps)
        kept = np.take(values, taps, axis=position)

        if self.use_tap_abs:
            return as_real_array(np.abs(kept))
        return as_real_array(np.concatenate([kept.real, kept.imag], axis=position))
