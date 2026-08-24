"""Remove a best-fit linear phase ramp across an axis (e.g. subcarriers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_signal_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


class Method(StrEnum):
    """How the phase ramp's slope and offset are estimated."""

    # fmt: off
    CIRCULAR_DIFF = "circular-diff"
    WEIGHTED_LS   = "weighted-ls"
    # fmt: on


@dataclass(frozen=True, slots=True)
class LinearPhaseCorrection(PointwiseStep):
    """Remove a best-fit linear phase ramp across an axis, per time sample.

    A sampling-time offset (a small timing error at the receiver) makes the phase
    grow in a straight line across the subcarriers: phase = a*x + b. This step
    fits that line for each frame and multiplies it back out, so the phase comes
    out flat.

    Two ways to fit the line:

    CIRCULAR_DIFF - averages the phase change from each value to the next, without
                    unwrapping, so it stays reliable at low SNR. It ignores any
                    pair of values with a gap between them, like the DC gap.
    WEIGHTED_LS   - unwraps the phase first, then fits the line by weighted least
                    squares.

    In both cases x is the axis coordinates, and small values count less (they are
    noisy), so noise does not steer the fit. Pointwise in time, so it streams
    exactly.
    """

    axis: AxisName = field(
        default=AxisName.SUBCARRIER, metadata={"doc": "axis the ramp spans"}
    )
    method: Method = field(
        default=Method.CIRCULAR_DIFF, metadata={"doc": "how to estimate the ramp"}
    )
    weight_power: float = field(
        default=1.0, metadata={"doc": "value weights ~ |H|^weight_power"}
    )
    keep_quantile: float | None = field(
        default=0.7,
        metadata={
            "doc": "keep only the top-magnitude fraction of values (None = keep all)"
        },
    )
    epsilon: float = field(default=1e-8, metadata={"doc": "numerical floor"})

    spec: ClassVar[StepSpec] = StepSpec(
        name="linear-phase-correction",
        summary="remove a linear phase ramp across an axis",
        category=Category.CALIBRATION,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        # The required axis is chosen at run time; see resolve_required_axes.
        requires_axes=(),
        layout_effect=LayoutEffect(note="linear phase ramp removed"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require the configured axis to be present."""

        return (self.axis,)

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Shape and semantics are unchanged; only the phase ramp is removed."""

        self.require_inputs(layout)
        axis = layout.require_static_axis(self.axis)

        # A phase ramp is a slope, and a slope needs at least two points. With a
        # single position there is no ramp to fit, so fail with a clear message.
        if axis.size is not None and axis.size < 2:
            raise LayoutError(
                "LinearPhaseCorrection needs at least two positions on the axis."
            )

        if self.keep_quantile is not None and not 0 <= self.keep_quantile <= 1:
            raise LayoutError(
                f"keep_quantile must be in [0, 1], got {self.keep_quantile}."
            )
        return layout

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Fit and remove the ramp along the axis, one fit per time frame."""

        # Move the axis to the end and flatten everything else into rows, so each
        # row holds one frame's values along the axis (one line to fit per row).
        position = signal.layout.axis_position(self.axis)
        moved = as_signal_array(np.moveaxis(values, position, -1))
        shape = moved.shape
        length = shape[-1]

        # x is the axis coordinates when we have them (e.g. subcarrier indices),
        # otherwise just 0, 1, 2, ... along the axis.
        axis = signal.layout.axis(self.axis)
        if axis.coordinates is not None:
            x_values: RealArray = np.asarray(axis.coordinates, dtype=float)
        else:
            x_values = np.arange(length, dtype=float)

        corrected = self._remove_ramp(moved.reshape(-1, length), x_values)

        # Undo the moveaxis above, putting the axis back where it started.
        restored = np.moveaxis(corrected.reshape(shape), -1, position)
        return as_signal_array(restored)

    def _weights(self, magnitudes: RealArray) -> RealArray:
        """Give each value a weight from its magnitude, so strong values count more.

        The weight is magnitude ** weight_power. When keep_quantile is set, any
        value whose magnitude falls below that quantile gets weight zero and drops
        out, so noise on near-empty subcarriers cannot steer the fit.
        """

        weights = (magnitudes + self.epsilon) ** self.weight_power
        if self.keep_quantile is None:
            return weights

        # Keep only the strongest values: zero out every value whose magnitude is
        # below the keep_quantile threshold.
        threshold = np.quantile(magnitudes, self.keep_quantile, axis=-1, keepdims=True)
        return np.where(magnitudes >= threshold, weights, 0.0)

    def _remove_ramp(self, frames: SignalArray, x_values: RealArray) -> SignalArray:
        """Estimate slope + offset per frame and multiply out exp(-j(a*x + b))."""

        weights = self._weights(np.abs(frames))
        if self.method == Method.WEIGHTED_LS:
            slope, offset = _fit_wls_unwrap(frames, x_values, weights, self.epsilon)
        else:
            slope, offset = _fit_circ_diff(frames, x_values, weights, self.epsilon)
        correction = np.exp(
            -1j * (slope[:, None] * x_values[None, :] + offset[:, None])
        )
        return as_signal_array(frames * correction)


def _fit_wls_unwrap(
    frames: SignalArray, x_values: RealArray, weights: RealArray, epsilon: float
) -> tuple[RealArray, RealArray]:
    """Weighted least-squares fit of the unwrapped phase to a line a*x + b."""

    # Each row of `frames` is one time frame; the columns run along the axis.
    # Unwrap the phase so it is continuous along the axis before fitting.
    phase = np.unwrap(np.angle(frames), axis=-1)

    # Build the weighted sums a straight-line least-squares fit needs. Every sum
    # runs over the axis (axis=1), giving one value per frame:
    #   s_11 = sum of weights          s_x1 = sum of weight * x
    #   s_xx = sum of weight * x * x    s_y1 = sum of weight * phase
    #   s_yx = sum of weight * phase * x
    weighted_x = weights * x_values[None, :]
    s_xx = np.sum(weighted_x * x_values[None, :], axis=1)
    s_x1 = np.sum(weighted_x, axis=1)
    s_11 = np.sum(weights, axis=1)
    s_yx = np.sum((weights * phase) * x_values[None, :], axis=1)
    s_y1 = np.sum(weights * phase, axis=1)

    # Solve the 2x2 normal equations for [slope, offset] with Cramer's rule.
    # Floor the determinant so a degenerate frame cannot blow the result up.
    determinant = s_xx * s_11 - s_x1 * s_x1
    determinant = np.where(np.abs(determinant) < epsilon, epsilon, determinant)
    slope = (s_11 * s_yx - s_x1 * s_y1) / determinant
    offset = (-s_x1 * s_yx + s_xx * s_y1) / determinant
    return slope, offset


def _fit_circ_diff(  # pylint: disable=too-many-locals  # a short DSP estimator
    frames: SignalArray, x_values: RealArray, weights: RealArray, epsilon: float
) -> tuple[RealArray, RealArray]:
    """Robust slope from the phase change between adjacent values, then the offset."""

    # The x-values are usually evenly spaced. Find that spacing, then mark any
    # pair of values with a gap between them as bad, so it does not count.
    spacings = np.diff(x_values)
    unique, counts = np.unique(spacings, return_counts=True)
    step = unique[np.argmax(counts)]
    good_pair = np.isclose(spacings, step)

    # The slope is the average phase change per step. We average the changes as
    # complex unit vectors and take the angle of their sum, so the average never
    # wraps around the way a plain average of angles would.
    delta_phase = np.angle(frames[:, 1:] * np.conj(frames[:, :-1]))
    pair_weights = np.sqrt(weights[:, 1:] * weights[:, :-1]) * good_pair[None, :]
    resultant = np.sum(pair_weights * np.exp(1j * delta_phase), axis=1)
    slope = np.angle(resultant) / (step if abs(step) > epsilon else 1.0)

    # Remove the slope; whatever phase is left is the constant offset. Average it
    # the same way, as unit vectors.
    slope_removed = frames * np.exp(-1j * (slope[:, None] * x_values[None, :]))
    offset_resultant = np.sum(weights * np.exp(1j * np.angle(slope_removed)), axis=1)
    offset = np.angle(offset_resultant)
    return slope, offset
