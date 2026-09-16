"""Combine each position along an axis with a reference position."""
# The CONFIG_DEPENDENT spec + resolve_* shape is shared with the other run-time-axis
# steps, so pylint reads these small step bodies as clones.
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import CONFIG_DEPENDENT, Category, LayoutEffect, StepSpec, Streaming


class Combine(StrEnum):
    """How each value along the axis is combined with the reference."""

    # fmt: off
    DIVIDE    = "divide"
    CONJUGATE = "conjugate"
    # fmt: on


class Reference(StrEnum):
    """Which position along the axis acts as the reference."""

    # fmt: off
    FIXED    = "fixed"
    ADJACENT = "adjacent"
    # fmt: on


@dataclass(frozen=True, slots=True)
class AxisReference(PointwiseStep):
    """Combine each position along an axis with a reference, per time sample.

    This is the shared core of subcarrier ratios, antenna cross-spectra, and
    adjacent-tone products. Two settings control it:

    combine - how to combine each value with the reference:
              DIVIDE    -> value / reference   (a channel ratio)
              CONJUGATE -> value * conj(ref)   (a cross-spectrum)
    mode    - which position is the reference:
              FIXED    -> one chosen index, used for the whole axis
              ADJACENT -> the previous position, so the axis shrinks by one

    Works on any axis (subcarriers, antennas, ...). Pointwise in time, so it
    streams exactly.
    """

    axis: AxisName = field(
        default=AxisName.SUBCARRIER, metadata={"doc": "axis to combine along"}
    )
    combine: Combine = field(
        default=Combine.DIVIDE, metadata={"doc": "how to combine with the reference"}
    )
    mode: Reference = field(
        default=Reference.FIXED, metadata={"doc": "which position is the reference"}
    )
    index: int = field(default=0, metadata={"doc": "reference position (mode=FIXED)"})
    drop_reference: bool = field(
        default=False,
        metadata={"doc": "drop the reference position from the output (mode=FIXED)"},
    )
    epsilon: float = field(
        default=1e-12, metadata={"doc": "small floor on the divisor (combine=DIVIDE)"}
    )
    representation: Representation | None = field(
        default=None,
        metadata={"doc": "output label; defaults to ratio / cross-spectrum"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="axis-reference",
        summary="combine each position along an axis with a reference",
        category=Category.CALIBRATION,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        # The required axis and the layout effect (relabels, and resizes in some
        # modes) are chosen at run time; see the resolve_* methods below.
        requires_axes=(),
        layout_effect=CONFIG_DEPENDENT,
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require the configured axis to be present."""

        return (self.axis,)

    def resolve_layout_effect(self) -> LayoutEffect:
        """Relabel to a ratio / cross-spectrum; shrink the axis when a position drops.

        ADJACENT pairs each position with its predecessor (the first drops out), and
        FIXED with `drop_reference` removes the reference position; plain FIXED keeps
        the axis as it is.
        """

        drops = self.mode == Reference.ADJACENT or self.drop_reference
        resizes = (self.axis,) if drops else ()
        return LayoutEffect(
            resizes=resizes, note="ratio / cross-spectrum across the axis"
        )

    def _representation(self) -> Representation:
        """Output label: the explicit one, or the ratio/cross-spectrum default."""

        if self.representation is not None:
            return self.representation
        return (
            Representation.RATIO
            if self.combine == Combine.DIVIDE
            else Representation.CROSS_SPECTRUM
        )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Relabel to a ratio / cross-spectrum, shrinking the axis when it reduces."""

        self.require_inputs(layout)
        axis = layout.require_static_axis(self.axis)

        # require_static_axis guarantees a concrete size, so this never fails.
        size = axis.size
        assert size is not None

        relabeled = layout.with_representation(self._representation())

        # Adjacent pairs each position with its predecessor, so the first drops out.
        if self.mode == Reference.ADJACENT:
            return relabeled.replace_axis(
                self.axis, _shrunk_axis(axis, tuple(range(1, size)))
            )

        if not 0 <= self.index < size:
            raise LayoutError(
                f"Reference {self.index} is out of range for axis of size {size}."
            )

        # Fixed mode keeps the axis as it is. The only exception is drop_reference,
        # which removes the reference position itself from the output.
        if not self.drop_reference:
            return relabeled
        kept = tuple(position for position in range(size) if position != self.index)
        return relabeled.replace_axis(self.axis, _shrunk_axis(axis, kept))

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Divide by, or conjugate-multiply against, the reference positions."""

        position = signal.layout.axis_position(self.axis)

        # Adjacent: the reference is each value's predecessor along the axis.
        if self.mode == Reference.ADJACENT:
            size = values.shape[position]
            current = np.take(values, range(1, size), axis=position)
            previous = np.take(values, range(0, size - 1), axis=position)
            return self._apply(as_signal_array(current), as_signal_array(previous))

        # Fixed: one reference position, broadcast against every value on the axis.
        reference = np.take(values, [self.index], axis=position)
        combined = self._apply(as_signal_array(values), as_signal_array(reference))
        if not self.drop_reference:
            return combined

        # Drop the reference position itself from the result.
        size = values.shape[position]
        kept = [index for index in range(size) if index != self.index]
        return as_signal_array(np.take(combined, kept, axis=position))

    def _apply(self, values: SignalArray, reference: SignalArray) -> SignalArray:
        """Divide the values by the reference, or multiply them by its conjugate."""

        if self.combine == Combine.DIVIDE:
            return as_signal_array(values / (reference + self.epsilon))
        return as_signal_array(values * np.conj(reference))


def _shrunk_axis(axis: Axis, kept: tuple[int, ...]) -> Axis:
    """Rebuild an axis keeping only the positions in `kept`, coordinates and all."""

    if axis.coordinates is not None:
        coordinates = tuple(axis.coordinates[index] for index in kept)
        return Axis.static(axis.name, coordinates, unit=axis.unit)
    return Axis.sized(axis.name, len(kept), unit=axis.unit)
