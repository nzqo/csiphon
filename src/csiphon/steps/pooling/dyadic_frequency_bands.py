"""Sum frequency bins into dyadic (octave) bands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_real_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import DataError, LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming


@dataclass(frozen=True, slots=True)
class DyadicFrequencyBands(PointwiseStep):
    """Sum frequency bins into dyadic (octave) bands

    Band `k` covers `[nyquist / 2^k, nyquist / 2^(k-1))`.
    Needs a nominal sampling rate to place the band edges.
    Reduces frequency per time step, so it streams exactly.
    """

    num_bands: int = field(default=6, metadata={"doc": "number of octave bands"})
    first_band: int = field(
        default=1, metadata={"doc": "index of the first (highest) band"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="dyadic-frequency-bands",
        summary="sum frequency bins into dyadic (octave) bands",
        category=Category.POOLING,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(AxisName.FREQUENCY,),
        layout_effect=LayoutEffect(replaces=((AxisName.FREQUENCY, AxisName.BAND),)),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def _edges(
        self, profile: AcquisitionProfile
    ) -> tuple[RealArray, RealArray, RealArray]:
        """Return (lower, upper, centers) band edges from the nominal nyquist."""

        rate_hz = profile.require_sampling_rate(self.name)
        nyquist = rate_hz / 2.0
        levels = np.arange(self.first_band, self.first_band + self.num_bands)
        lower = as_real_array(nyquist / (2.0**levels))
        upper = as_real_array(nyquist / (2.0 ** (levels - 1)))
        return lower, upper, as_real_array(0.5 * (lower + upper))

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Replace the frequency axis with a band axis."""

        self.require_inputs(layout)
        if self.num_bands < 1:
            raise LayoutError(f"num_bands must be >= 1, got {self.num_bands}.")
        if self.first_band < 1:
            raise LayoutError(f"first_band must be >= 1, got {self.first_band}.")
        _, _, centers = self._edges(profile)
        band = Axis.static(AxisName.BAND, tuple(centers), unit="Hz")
        return layout.replace_axis(AxisName.FREQUENCY, band)

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Sum bins into bands using the runtime frequencies."""

        frequencies = _frequencies_of(signal)
        lower, upper, _ = self._edges(profile)

        # band_mask[b, f] == 1 when frequency f lies within band b's [lower, upper).
        band_mask = (
            (frequencies[None, :] >= lower[:, None])
            & (frequencies[None, :] < upper[:, None])
        ).astype(float)

        # Sum over the frequency axis: move it last, matmul by the mask, move back.
        position = signal.layout.axis_position(AxisName.FREQUENCY)
        moved = np.moveaxis(as_real_array(values), position, -1)
        banded = moved @ band_mask.T

        return as_real_array(np.moveaxis(banded, -1, position))


def _frequencies_of(signal: Signal) -> RealArray:
    """Return the frequency coordinates (runtime coords, else layout axis)."""

    frequencies = signal.coords.get(AxisName.FREQUENCY)
    if frequencies is not None:
        return frequencies

    axis = signal.layout.axis(AxisName.FREQUENCY)
    if axis.coordinates is None:
        raise DataError("DyadicFrequencyBands needs frequency coordinates.")
    return np.asarray(axis.coordinates, dtype=float)
