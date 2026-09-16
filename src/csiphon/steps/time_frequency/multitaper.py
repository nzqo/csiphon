"""Multitaper (DPSS) power spectrogram (optional `[filters]` extra -> scipy)."""

# Windowed-spectral spec/skeleton overlaps with WindowedFFTPower by design.
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from csiphon.core.arrays import RealArray, SignalArray, as_real_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError, MissingDependencyError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.sampling import check_jitter
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import CONFIG_DEPENDENT, Category, LayoutEffect, StepSpec, Streaming
from csiphon.steps._support.broadcasting import (
    broadcast_channels,
    operated_axis_position,
)
from csiphon.steps._support.windowing import (
    FrameFn,
    WindowedOperator,
    WindowGeometry,
    rfft_band,
)


def _dpss_tapers(
    window_size: int, time_bandwidth: float, num_tapers: int | None
) -> RealArray:
    """DPSS (Slepian) tapers, shape (num_tapers, window_size); needs scipy."""

    try:
        from scipy.signal.windows import dpss  # pylint: disable=import-outside-toplevel
    except ImportError as error:  # pragma: no cover - exercised via message
        raise MissingDependencyError("scipy", "filters") from error
    count = (
        num_tapers if num_tapers is not None else max(1, int(2 * time_bandwidth) - 1)
    )
    return as_real_array(dpss(window_size, NW=time_bandwidth, Kmax=count, sym=False))


def _window_power(
    block: RealArray, tapers: RealArray, keep: npt.NDArray[np.bool_]
) -> RealArray:
    """Reduce one 2-D window `(window, feature)` to in-band multitaper power.

    Axis 0 is time within the window, axis 1 is the feature axis.
    """

    window_size = block.shape[0]
    total: RealArray | None = None
    for taper in tapers:
        # Taper the window along time, then take its power spectrum along time
        # (axis 0). Averaging over several tapers is what cuts the variance.
        spectrum = np.fft.rfft(block * taper[:, None], n=window_size, axis=0)

        # Average the power over the feature axis (axis 1).
        power = (spectrum.real**2 + spectrum.imag**2).mean(axis=1)
        total = power if total is None else total + power

    # At least one taper always exists, so total is set by now.
    assert total is not None

    # Average over the tapers, then keep only the in-band frequencies.
    return as_real_array((total / tapers.shape[0])[keep])


def _multitaper_frame(
    tapers: RealArray, keep: npt.NDArray[np.bool_], feature_pos: int
) -> FrameFn:
    """A frame function: one window to in-band power, per non-time channel."""

    def frame(block: SignalArray) -> RealArray:
        """The multitaper spectrum of one window."""

        # The shared window signature is complex-capable; multitaper is real-only.
        # Take the spectrogram along the chosen axis, broadcasting the other axes.
        return broadcast_channels(
            as_real_array(block),
            feature_pos,
            lambda matrix: _window_power(as_real_array(matrix), tapers, keep),
        )

    return frame


@dataclass(frozen=True, slots=True)
class Multitaper(Step):
    """Multitaper (DPSS) power spectrogram, averaged over the feature axis.

    A variance-reduced alternative to WindowedFFTPower: each window is tapered by
    several Slepian sequences and their power spectra are averaged (and averaged
    over the chosen `axis`), then kept up to `band_hz`. Replaces that axis with a
    `frequency` axis, running once per any other axis present. Streams exactly via
    a ring buffer. Needs scipy (the `[filters]` extra).
    """

    axis: AxisName = field(
        default=AxisName.FEATURE, metadata={"doc": "axis to average the power over"}
    )
    window_s: float = field(default=0.512, metadata={"doc": "window length in seconds"})
    hop_s: float = field(
        default=0.10, metadata={"doc": "hop between windows in seconds"}
    )
    band_hz: float = field(
        default=60.0, metadata={"doc": "keep frequencies ≤ this (Hz)"}
    )
    time_bandwidth: float = field(
        default=3.5, metadata={"doc": "DPSS time-bandwidth product NW"}
    )
    num_tapers: int | None = field(
        default=None, metadata={"doc": "taper count (2*NW-1 if None)"}
    )
    strict: bool = field(
        default=False, metadata={"doc": "raise (not warn) on excessive jitter"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="multitaper-power",
        summary="compute a multitaper (DPSS) power spectrogram, averaged over an axis",
        category=Category.TIME_FREQUENCY,
        admissible_values=(ValueKind.REAL, ValueKind.MAGNITUDE, ValueKind.POWER),
        admissible_reprs=None,
        # The axis to average over, and so the layout effect (that axis becomes
        # frequency), is chosen at run time; see the resolve_* methods below.
        requires_axes=(),
        layout_effect=CONFIG_DEPENDENT,
        streaming=Streaming.BATCH_EQUIVALENT,
        streaming_note="needs scipy (the [filters] extra)",
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require a time axis and the configured axis to average over."""

        return (AxisName.TIME, self.axis)

    def resolve_layout_effect(self) -> LayoutEffect:
        """The chosen axis becomes the frequency axis of the spectrogram."""

        return LayoutEffect(
            replaces=((self.axis, AxisName.FREQUENCY),),
            value_kind=ValueKind.POWER,
            note="time-frequency",
        )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Replace the chosen axis with an in-band frequency axis of power."""

        self.require_inputs(layout)
        layout.require_static_axis(self.axis)
        if self.window_s <= 0:
            raise LayoutError(f"window_s must be > 0, got {self.window_s}.")
        if self.hop_s <= 0:
            raise LayoutError(f"hop_s must be > 0, got {self.hop_s}.")
        if self.band_hz <= 0:
            raise LayoutError(f"band_hz must be > 0, got {self.band_hz}.")
        if self.time_bandwidth <= 0:
            raise LayoutError(f"time_bandwidth must be > 0, got {self.time_bandwidth}.")
        if self.num_tapers is not None and self.num_tapers < 1:
            raise LayoutError(f"num_tapers must be >= 1, got {self.num_tapers}.")

        # Size the windows from the sampling rate, then work out which rFFT
        # frequency bins fall within band_hz. Those bins become the frequency axis.
        rate_hz = profile.require_sampling_rate(self.name)
        geometry = WindowGeometry.from_seconds(rate_hz, self.window_s, self.hop_s)
        kept_frequencies, _ = rfft_band(geometry.window_size, rate_hz, self.band_hz)
        frequency = Axis.static(AxisName.FREQUENCY, tuple(kept_frequencies), unit="Hz")

        return (
            layout.replace_axis(self.axis, frequency)
            .with_representation(Representation.TIME_FREQUENCY)
            .with_values(ValueKind.POWER)
        )

    def _operator(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> WindowedOperator:
        """Build the windowing operator with a DPSS-multitaper frame function."""

        rate_hz = profile.require_sampling_rate(self.name)
        geometry = WindowGeometry.from_seconds(rate_hz, self.window_s, self.hop_s)
        _, keep = rfft_band(geometry.window_size, rate_hz, self.band_hz)
        tapers = _dpss_tapers(
            geometry.window_size, self.time_bandwidth, self.num_tapers
        )
        feature_pos = operated_axis_position(in_layout, self.axis)

        return WindowedOperator(
            geometry,
            _multitaper_frame(tapers, keep, feature_pos),
            in_layout,
            out_layout,
        )

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Compute the multitaper spectrogram over the whole recording."""

        check_jitter(signal.times, step=self.name, strict=self.strict)
        return self._operator(signal.layout, out_layout, profile).push(signal)

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Emit one spectrum per completed window, carrying the buffer."""

        return self._operator(in_layout, out_layout, profile)
