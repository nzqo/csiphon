"""Hann-windowed real-FFT power up to a Doppler band (STFT / SHARP front-end)."""
# Every time->time-frequency-power step declares the same spec shape (TIME_FREQUENCY,
# requires TIME, layout_effect adds FREQUENCY as POWER), so pylint reads this spec and
# the synchrosqueezed-power one as clones.
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from csiphon.core.arrays import RealArray, SignalArray, as_real_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.sampling import check_jitter
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming
from csiphon.steps._support.windowing import (
    FrameFn,
    WindowedOperator,
    WindowGeometry,
    rfft_band,
)


@dataclass(frozen=True, slots=True)
class WindowedFFTPower(Step):
    """Hann-windowed rFFT power on a sliding grid, kept up to a Doppler band.

    Needs a nominal sampling rate at compile time to size the frequency axis;
    actual timestamp jitter is tolerated (and warned about) at run time. Streams
    exactly via a ring buffer.
    """

    window_s: float = field(default=0.512, metadata={"doc": "window length in seconds"})
    hop_s: float = field(
        default=0.10, metadata={"doc": "hop between windows in seconds"}
    )
    band_hz: float = field(
        default=60.0, metadata={"doc": "keep frequencies ≤ this (Hz)"}
    )
    strict: bool = field(
        default=False, metadata={"doc": "raise (not warn) on excessive jitter"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="windowed-fft-power",
        summary="compute Hann-windowed rFFT power on a sliding grid, up to a Doppler band",  # noqa: E501
        category=Category.TIME_FREQUENCY,
        admissible_values=(ValueKind.REAL, ValueKind.MAGNITUDE, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(
            adds=(AxisName.FREQUENCY,),
            value_kind=ValueKind.POWER,
            note="time-frequency",
        ),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Insert a frequency axis after time; produce time-frequency power."""

        self.require_inputs(layout)
        if layout.dynamic_index != 0:
            raise LayoutError("WindowedFFTPower expects the time axis first.")
        if self.window_s <= 0:
            raise LayoutError(f"window_s must be > 0, got {self.window_s}.")
        if self.hop_s <= 0:
            raise LayoutError(f"hop_s must be > 0, got {self.hop_s}.")
        if self.band_hz <= 0:
            raise LayoutError(f"band_hz must be > 0, got {self.band_hz}.")

        rate_hz = profile.require_sampling_rate(self.name)
        geometry = WindowGeometry.from_seconds(rate_hz, self.window_s, self.hop_s)
        kept_frequencies, _ = rfft_band(geometry.window_size, rate_hz, self.band_hz)
        frequency = Axis.static(AxisName.FREQUENCY, tuple(kept_frequencies), unit="Hz")
        return (
            layout.insert_axis_after(AxisName.TIME, frequency)
            .with_representation(Representation.TIME_FREQUENCY)
            .with_values(ValueKind.POWER)
        )

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Gather sliding windows and take Hann-windowed FFT power."""

        check_jitter(signal.times, step=self.name, strict=self.strict)
        rate_hz = profile.require_sampling_rate(self.name)
        geometry = WindowGeometry.from_seconds(rate_hz, self.window_s, self.hop_s)
        kept_frequencies, keep = rfft_band(geometry.window_size, rate_hz, self.band_hz)

        # The time axis is first here (we check that at compile). Gather every
        # window's samples in one shot: for window w and offset s, take sample
        # start_w + s. This builds an (n_windows, window, *other_axes) array with
        # no Python loop over windows.
        samples = as_real_array(signal.values)
        starts, centers = geometry.indices(samples.shape[0])
        windows = samples[starts[:, None] + np.arange(geometry.window_size)[None, :]]

        # Apply a Hann taper along the window axis (axis 1) and take the real FFT
        # along it, then keep the power (squared magnitude) at the in-band bins.
        hann_shape = (1, geometry.window_size) + (1,) * (samples.ndim - 1)
        windows = windows * np.hanning(geometry.window_size).reshape(hann_shape)
        spectrum = np.fft.rfft(windows, n=geometry.window_size, axis=1)
        power = spectrum.real**2 + spectrum.imag**2
        power = power[:, keep]

        return signal.with_values(
            power,
            out_layout,
            times=signal.times[centers],
            coords={AxisName.FREQUENCY: kept_frequencies},
        )

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Ring-buffer operator that emits one spectrum per completed window."""

        rate_hz = profile.require_sampling_rate(self.name)
        geometry = WindowGeometry.from_seconds(rate_hz, self.window_s, self.hop_s)
        _, keep = rfft_band(geometry.window_size, rate_hz, self.band_hz)
        frame_fn = _hann_fft_power(geometry.window_size, keep)
        return WindowedOperator(geometry, frame_fn, in_layout, out_layout)


def _hann_fft_power(window_size: int, keep: npt.NDArray[np.bool_]) -> FrameFn:
    """Return a Hann-windowed rFFT-power function for one window block."""

    hann = np.hanning(window_size)

    def frame(block: SignalArray) -> RealArray:
        # A block is one window: axis 0 is time within the window, the rest are
        # the other axes carried along. Taper and rFFT along axis 0 (time).
        taper_shape = (window_size,) + (1,) * (block.ndim - 1)
        spectrum = np.fft.rfft(block * hann.reshape(taper_shape), n=window_size, axis=0)
        power = spectrum.real**2 + spectrum.imag**2
        return as_real_array(power[keep])

    return frame
