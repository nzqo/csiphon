"""Element-wise real part."""

# Same spec and method skeleton as ImagPart and the other one-component steps by
# design; only the part taken differs.
# pylint: disable=duplicate-code

from dataclasses import dataclass
from typing import ClassVar

from csiphon.core.arrays import SignalArray, as_real_array
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class RealPart(PointwiseStep):
    """Take the element-wise real part.

    With `ImagPart` and the `ComplexFromParts` merge, this runs a real-only step
    on complex CSI part by part.
    """

    spec: ClassVar[StepSpec] = StepSpec(
        name="real-part",
        summary="take the real part",
        category=Category.COMPONENTS,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(value_kind=ValueKind.REAL),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Complex input becomes real-valued."""

        self.require_inputs(layout)
        return layout.with_values(ValueKind.REAL)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Return `values.real`."""

        return as_real_array(values.real)
