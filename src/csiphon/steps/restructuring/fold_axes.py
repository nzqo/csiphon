"""Fold several named axes into one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import Representation
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
class FoldAxes(PointwiseStep):
    """Fold two or more named axes into a single axis.

    You list the axes to fold, most significant first: folding
    `(RX_ANTENNA, SUBCARRIER)` gives an index of
    `rx_antenna * n_subcarrier + subcarrier` (C-order). The named axes are
    replaced by one axis, `into`, sitting where the first named axis was; every
    other axis stays put. Pointwise in time, so it streams exactly.
    """

    axes: tuple[AxisName, ...] = field(
        default=(), metadata={"doc": "axes to fold, most significant first"}
    )
    into: AxisName = field(
        default=AxisName.FEATURE, metadata={"doc": "name of the folded axis"}
    )
    representation: Representation | None = field(
        default=None, metadata={"doc": "optional new representation label"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="fold-axes",
        summary="fold named axes into one (C-order)",
        category=Category.RESTRUCTURING,
        admissible_values=None,
        admissible_reprs=None,
        # The required axes are chosen at run time; see resolve_required_axes.
        requires_axes=(),
        layout_effect=CONFIG_DEPENDENT,
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require every axis being folded to be present."""

        return self.axes

    def resolve_layout_effect(self) -> LayoutEffect:
        """Fold the named axes into one axis (`into`), the product of their sizes."""

        # Folding into one of the folded axes keeps that name (resized); folding
        # into a fresh axis removes them all and adds the new one.
        if self.into in self.axes:
            removed = tuple(name for name in self.axes if name != self.into)
            return LayoutEffect(
                removes=removed, resizes=(self.into,), note="product of sizes"
            )
        return LayoutEffect(
            removes=self.axes, adds=(self.into,), note="product of sizes"
        )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Replace the named axes with one folded axis of their product size."""

        self.require_inputs(layout)
        if len(self.axes) < 2:
            raise LayoutError("FoldAxes needs at least two axes to fold.")
        if len(set(self.axes)) != len(self.axes):
            raise LayoutError("FoldAxes axes must be distinct.")

        folded = set(self.axes)
        size = 1
        for name in self.axes:
            axis = layout.require_static_axis(name)
            # require_static_axis guarantees this.
            assert axis.size is not None
            size *= axis.size

        # The folded name may be reused (it is one of the axes being consumed),
        # but it must not clash with an axis that survives the fold.
        if self.into not in folded and layout.has_axis(self.into):
            raise LayoutError(f"Axis '{self.into}' already exists.")

        merged = Axis.sized(self.into, size)
        new_axes = []
        for axis in layout.axes:
            if axis.name == self.axes[0]:
                # Put the merged axis where the first folded axis was.
                new_axes.append(merged)
            elif axis.name not in folded:
                # Keep untouched axes; the other folded axes are dropped.
                new_axes.append(axis)
        new_layout = layout.set_axes(new_axes)
        if self.representation is not None:
            new_layout = new_layout.with_representation(self.representation)
        return new_layout

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Gather the folded axes side by side (in order), then reshape to one."""

        # The output order, but with the folded axis expanded back into its parts.
        expanded: list[AxisName] = []
        for axis in out_layout.axes:
            expanded.extend(self.axes if axis.name == self.into else [axis.name])

        order = [signal.layout.axis_position(name) for name in expanded]
        gathered = np.transpose(values, order)

        # The folded parts are now contiguous starting at the folded axis' slot;
        # collapse that run into one dimension.
        start = out_layout.axis_names.index(self.into)
        stop = start + len(self.axes)
        shape = gathered.shape
        merged_size = int(np.prod(shape[start:stop]))
        new_shape = (*shape[:start], merged_size, *shape[stop:])
        return as_signal_array(gathered.reshape(new_shape))
