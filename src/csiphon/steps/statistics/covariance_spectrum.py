"""Covariance-eigenspectrum statistics over sliding windows."""

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

# The fixed set of descriptors this step reports for each window's eigenvalue
# spectrum. It is not a parameter: the step always produces exactly these five.
_STAT_NAMES = (
    "entropy",
    "effective_rank",
    "participation",
    "top_mass",
    "log_condition",
)
_N_STATS = len(_STAT_NAMES)

# How many of the largest eigenvalues the "top_mass" descriptor sums over.
_TOP_K = 5


def _spectrum_stats(eigenvalues: RealArray) -> RealArray:
    """Summarize an eigenvalue spectrum with the five descriptors in _STAT_NAMES."""

    # Start from a clean, non-negative spectrum and its total (the total variance).
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum()) + _EPS
    normalized = eigenvalues / total

    # Spectral entropy: high when the variance is spread evenly across many
    # directions, low when a single direction dominates.
    entropy = -float(np.sum(normalized * np.log(normalized + _EPS)))

    # Two "how many directions carry the variance" measures: the effective rank
    # (exp of the entropy) and the participation ratio.
    effective_rank = float(np.exp(entropy))
    participation = float((total * total) / (float(np.sum(eigenvalues**2)) + _EPS))

    # Share of the total variance held by the few largest directions.
    top = min(_TOP_K, eigenvalues.shape[0])
    top_mass = float(eigenvalues[:top].sum() / total)

    # Log condition number: log of the largest eigenvalue over the smallest.
    log_condition = float(np.log((eigenvalues[0] + _EPS) / (eigenvalues[-1] + _EPS)))

    return as_real_array(
        np.array(
            [entropy, effective_rank, participation, top_mass, log_condition],
            dtype=float,
        )
    )


def _window_stats(block: RealArray, shrinkage: float) -> RealArray:
    """Reduce one 2-D window `(window, feature)` to the five spectrum stats."""

    # Center the window, form the covariance across the feature axis, and add
    # a small ridge on the diagonal to keep the eigenvalues well-behaved.
    centered = block - block.mean(axis=0, keepdims=True)
    feature_dim = block.shape[1]
    covariance = (centered.T @ centered) / max(1, block.shape[0] - 1)
    covariance = covariance + shrinkage * np.eye(feature_dim)

    # Eigenvalues largest first, then reduce them to the descriptors.
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    return _spectrum_stats(as_real_array(eigenvalues))


def _covariance_frame(shrinkage: float, feature_pos: int) -> FrameFn:
    """A frame function: one window to spectrum stats, per non-time channel."""

    def frame(block: SignalArray) -> RealArray:
        # The shared window signature is complex-capable; this stat is real-only.
        # Run the 2-D stats along the chosen axis, broadcasting the other axes.
        return broadcast_channels(
            as_real_array(block),
            feature_pos,
            lambda matrix: _window_stats(as_real_array(matrix), shrinkage),
        )

    return frame


@dataclass(frozen=True, slots=True)
class CovarianceSpectrum(Step):
    """Statistics of the feature-covariance eigenspectrum, over sliding windows.

    For each window (window_size samples, stepping by hop_size), it builds the
    covariance across the feature axis, takes its eigenvalues, and summarizes
    them as five descriptors: spectral entropy, effective rank, participation
    ratio, top-5 eigenvalue mass, and log condition number. Replaces the chosen
    `axis` with `feature[5]`, running once per any other axis present. Streams
    exactly via a ring buffer.
    """

    axis: AxisName = field(
        default=AxisName.FEATURE, metadata={"doc": "axis to take the covariance over"}
    )
    window_size: int = field(default=100, metadata={"doc": "window length in samples"})
    hop_size: int = field(default=1, metadata={"doc": "hop between windows in samples"})
    shrinkage: float = field(
        default=1e-3, metadata={"doc": "covariance ridge for stability"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="covariance-spectrum",
        summary="summarize the windowed feature-covariance eigenspectrum",
        category=Category.STATISTICS,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        # The axis to summarize is chosen at run time; see resolve_* below.
        requires_axes=(),
        layout_effect=CONFIG_DEPENDENT,  # the chosen axis becomes stats; see resolve
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require a time axis and the configured axis to summarize."""

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
            note="covariance stats per window",
        )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Replace the chosen axis with a 5-vector of statistics, per window."""

        self.require_inputs(layout)
        layout.require_static_axis(self.axis)
        if self.window_size < 1:
            raise LayoutError(f"window_size must be >= 1, got {self.window_size}.")
        if self.hop_size < 1:
            raise LayoutError(f"hop_size must be >= 1, got {self.hop_size}.")
        stats = Axis.sized(AxisName.FEATURE, _N_STATS)
        return (
            layout.replace_axis(self.axis, stats)
            .with_values(ValueKind.REAL)
            .with_representation(Representation.FEATURE_VECTOR)
        )

    def _operator(self, in_layout: Layout, out_layout: Layout) -> WindowedOperator:
        """Build the shared windowing operator with the covariance frame."""

        geometry = WindowGeometry(self.window_size, float(self.hop_size))
        feature_pos = operated_axis_position(in_layout, self.axis)
        return WindowedOperator(
            geometry,
            _covariance_frame(self.shrinkage, feature_pos),
            in_layout,
            out_layout,
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
