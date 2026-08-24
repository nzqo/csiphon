"""Local-PCA directional-bias statistics over sliding windows."""

# The windowed-reduction skeleton overlaps with the sibling statistics step by design.
# pylint: disable=duplicate-code
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_real_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import CONFIG_DEPENDENT, Category, LayoutEffect, StepSpec, Streaming
from csiphon.steps._support.broadcasting import (
    broadcast_channels,
    operated_axis_position,
)
from csiphon.steps._support.windowing import FrameFn, WindowedOperator, WindowGeometry

_EPS = 1e-12

# The fixed set of descriptors this step reports for each window. It is not a
# parameter: the step always produces exactly these three.
_STAT_NAMES = ("signed_bias", "consistency", "median_step")
_N_STATS = len(_STAT_NAMES)


def _window_bias(block: RealArray) -> RealArray:
    """Summarize one 2-D window with the three rotation-bias descriptors."""

    # Fit a 2-D PCA on the centered window and project the samples onto its top
    # two directions. That gives a 2-D path through the window.
    centered = block - block.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ right_vectors[:2].T

    # For each step along the path, the 2-D cross product says which way it turns
    # (its sign), and step_sizes says how far it moved.
    previous, following = scores[:-1], scores[1:]
    cross = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
    step_sizes = np.sqrt(np.sum((following - previous) ** 2, axis=1)) + _EPS

    # signed_bias: typical turn per unit of movement (does the path curl one way?).
    # consistency: how consistently it turns the same way (0..1).
    # median_step: the typical step size.
    signed_bias = float(np.median(cross / step_sizes))
    consistency = float(np.abs(np.mean(np.sign(cross))))
    median_step = float(np.median(step_sizes))
    return as_real_array(np.array([signed_bias, consistency, median_step], dtype=float))


def _bias_frame(feature_pos: int) -> FrameFn:
    """A frame function: one window to rotation-bias stats, per non-time channel."""

    def frame(block: SignalArray) -> RealArray:
        # The shared window signature is complex-capable; this stat is real-only.
        # Run the 2-D bias along the chosen axis, broadcasting the other axes.
        return broadcast_channels(
            as_real_array(block),
            feature_pos,
            lambda matrix: _window_bias(as_real_array(matrix)),
        )

    return frame


@dataclass(frozen=True, slots=True)
class LocalPcaBias(Step):
    """Rotation-bias statistics of a window-local 2-D PCA, over sliding windows.

    For each window it fits a 2-D PCA and projects the samples onto it, then
    summarizes how that 2-D trajectory turns: a signed rotation bias (does it
    curl one way?), a rotation consistency, and the median step size. Replaces the
    chosen `axis` with `feature[3]`, running once per any other axis present. Needs
    at least two positions along that axis. Streams exactly via a ring buffer.
    """

    axis: AxisName = field(
        default=AxisName.FEATURE, metadata={"doc": "axis to fit the local PCA over"}
    )
    window_size: int = field(default=100, metadata={"doc": "window length in samples"})
    hop_size: int = field(default=1, metadata={"doc": "hop between windows in samples"})

    spec: ClassVar[StepSpec] = StepSpec(
        name="local-pca-bias",
        summary="measure the rotation bias of a window-local 2-D PCA",
        category=Category.STATISTICS,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        # The axis to fit over is chosen at run time; see resolve_* below.
        requires_axes=(),
        layout_effect=CONFIG_DEPENDENT,  # the chosen axis becomes stats; see resolve
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require a time axis and the configured axis to fit the PCA over."""

        return (AxisName.TIME, self.axis)

    def resolve_layout_effect(self) -> LayoutEffect:
        """The chosen axis becomes a short feature vector of statistics."""

        stats = AxisName.FEATURE
        renamed = () if self.axis == stats else ((self.axis, stats),)
        resized = (stats,) if self.axis == stats else ()
        return LayoutEffect(
            value_kind=ValueKind.REAL,
            replaces=renamed,
            resizes=resized,
            note="pca-bias stats per window",
        )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Replace the chosen axis with a 3-vector of statistics, per window."""

        self.require_inputs(layout)
        feature = layout.require_static_axis(self.axis)
        # A rotation needs at least two path points, so a window under two samples
        # would divide by an empty step list and yield NaN.
        if self.window_size < 2:
            raise LayoutError(f"window_size must be >= 2, got {self.window_size}.")
        if self.hop_size < 1:
            raise LayoutError(f"hop_size must be >= 1, got {self.hop_size}.")
        if feature.size is not None and feature.size < 2:
            raise LayoutError("LocalPcaBias needs at least two positions on the axis.")
        stats = Axis.sized(AxisName.FEATURE, _N_STATS)
        return (
            layout.replace_axis(self.axis, stats)
            .with_values(ValueKind.REAL)
            .with_representation(Representation.FEATURE_VECTOR)
        )

    def _operator(self, in_layout: Layout, out_layout: Layout) -> WindowedOperator:
        """Build the shared windowing operator with the rotation-bias frame."""

        geometry = WindowGeometry(self.window_size, float(self.hop_size))
        feature_pos = operated_axis_position(in_layout, self.axis)
        return WindowedOperator(
            geometry, _bias_frame(feature_pos), in_layout, out_layout
        )

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Compute the stats for every window in the whole recording."""

        return self._operator(signal.layout, out_layout).push(signal)

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Emit one statistics row per completed window, carrying the buffer."""

        return self._operator(in_layout, out_layout)
