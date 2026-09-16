"""Two-sided complex STFT magnitude (optional `[filters]` extra -> scipy).

`WindowedFFTPower` covers the common real-valued case: a one-sided rFFT power on
a sliding grid. This step is its complex sibling. It takes a *two-sided* STFT of
a complex signal, so positive and negative Doppler shifts stay distinct, and
keeps the magnitude of the low-frequency bins.

Batch matches `cpd`'s `stft_power_db` front-end exactly: frames centered on
`p * hop` with a zero-padded boundary (the first frame centered on sample 0), so
the frame grid runs from before the first sample to past the last. Streaming
uses *valid* windows instead: a frame is emitted only once its whole window has
arrived, timestamped at the window center, and no zero-padded edge frames are
fabricated. So the two agree on the shared interior but differ at the recording
ends: streaming is BATCH_DIVERGENT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_complex_array, as_real_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError, MissingDependencyError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming
from csiphon.steps._support.windowing import FrameFn, WindowedOperator, WindowGeometry


def _short_time_fft() -> tuple[Any, Any]:
    """Import scipy's ShortTimeFFT / get_window lazily, with a helpful error."""

    try:
        # Lazy so the base install stays numpy-only.
        from scipy.signal import (  # pylint: disable=import-outside-toplevel
            ShortTimeFFT,
            get_window,
        )
    except ImportError as error:  # pragma: no cover - exercised via message
        raise MissingDependencyError("scipy", "filters") from error
    return ShortTimeFFT, get_window


def _stft_frame(window_size: int, freq_bins: int, window: RealArray) -> FrameFn:
    """A per-window two-sided FFT magnitude, matching the batch scale (1/win.sum()).

    `window` is the periodic Hann taper; `scale_to="magnitude"` in the batch path
    is exactly a division by its sum, so a shared window yields the same magnitude
    in both paths.
    """

    scale = float(window.sum())

    def frame(block: SignalArray) -> RealArray:
        # A block is one window: axis 0 is time within it, the rest ride along.
        taper_shape = (window_size,) + (1,) * (block.ndim - 1)
        tapered = block * window.reshape(taper_shape)
        spectrum = np.fft.fft(tapered, n=window_size, axis=0)
        return as_real_array(np.abs(spectrum[:freq_bins]) / scale)

    return frame


@dataclass(frozen=True, slots=True)
class ComplexStftMagnitude(Step):
    """Two-sided complex STFT magnitude on a sliding grid, low bins kept.

    Runs along time (which must be the first axis) and inserts a frequency axis
    after it; every other axis is a channel carried along unchanged. So
    ``(time, feature)`` complex becomes ``(time, frequency, feature)`` magnitude,
    and ``(time, rx_antenna, subcarrier)`` becomes
    ``(time, frequency, rx_antenna, subcarrier)``, no need to fold first. It
    keeps the first ``freq_bins`` two-sided FFT bins (DC and the low positive
    Doppler frequencies) and needs a nominal sampling rate at compile time to size
    the frequency axis. Streams via a ring buffer (valid windows), which diverges
    from batch only at the recording ends (see the module docstring).
    """

    window_size: int = field(
        default=512, metadata={"doc": "STFT window length in samples"}
    )
    hop_size: int = field(default=10, metadata={"doc": "hop between frames in samples"})
    freq_bins: int = field(
        default=31, metadata={"doc": "number of low two-sided bins to keep"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="complex-stft-magnitude",
        summary="two-sided complex STFT magnitude on a sliding grid, keeping the low bins",  # noqa: E501
        category=Category.TIME_FREQUENCY,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(
            adds=(AxisName.FREQUENCY,),
            value_kind=ValueKind.MAGNITUDE,
            note="time-frequency",
        ),
        streaming=Streaming.BATCH_DIVERGENT,
        streaming_note="valid-window online STFT; batch zero-pads the edges",
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Insert a frequency axis after time; produce time-frequency magnitude."""

        self.require_inputs(layout)
        if layout.dynamic_index != 0:
            raise LayoutError("ComplexStftMagnitude expects the time axis first.")
        if self.window_size < 1:
            raise LayoutError(f"window_size must be >= 1, got {self.window_size}.")
        if self.hop_size < 1:
            raise LayoutError(f"hop_size must be >= 1, got {self.hop_size}.")
        if not 1 <= self.freq_bins <= self.window_size:
            raise LayoutError(
                f"freq_bins must be in [1, window_size={self.window_size}], "
                f"got {self.freq_bins}."
            )

        frequency = Axis.static(
            AxisName.FREQUENCY, tuple(self._bin_frequencies(profile)), unit="Hz"
        )
        return (
            layout.insert_axis_after(AxisName.TIME, frequency)
            .with_representation(Representation.TIME_FREQUENCY)
            .with_values(ValueKind.MAGNITUDE)
        )

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Run scipy's ShortTimeFFT per channel and keep the low-bin magnitudes."""

        # No samples means no frames; return early rather than index empty times.
        if signal.n_samples == 0:
            return empty_signal(out_layout)

        rate_hz = profile.require_sampling_rate(self.name)
        short_time_fft, get_window = _short_time_fft()

        # Set up scipy's STFT to match cpd's batch front-end. The frame grid
        # is zero-padded at the boundary so the first frame sits on sample 0
        # (p0=0 below) and the last runs just past the final sample.
        values = as_complex_array(signal.values)
        n_frames = int(np.ceil(values.shape[0] / self.hop_size)) + 1
        window = get_window("hann", self.window_size, fftbins=True)
        stft = short_time_fft(
            window,
            hop=self.hop_size,
            fs=rate_hz,
            fft_mode="twosided",
            mfft=self.window_size,
            scale_to="magnitude",
        )

        # Run the STFT along time (axis 0); every other axis is a channel carried
        # along. scipy replaces the time axis with frequency and appends the frame
        # axis last (freq, *channels, frame), so keep the low bins and move
        # frame to the front, frequency just after it: (frame, freq, *channels).
        spectrum = stft.stft(values, p0=0, p1=n_frames, padding="zeros", axis=0)
        magnitude = np.moveaxis(np.abs(spectrum[: self.freq_bins]), (0, -1), (1, 0))

        # Frame p is centered on input sample p * hop; reuse the nearest timestamp.
        center = np.minimum(np.arange(n_frames) * self.hop_size, signal.times.size - 1)
        return signal.with_values(
            as_real_array(magnitude),
            out_layout,
            times=signal.times[center],
            coords={AxisName.FREQUENCY: self._bin_frequencies(profile)},
        )

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Ring-buffer operator emitting one magnitude spectrum per valid window."""

        _, get_window = _short_time_fft()
        window = as_real_array(get_window("hann", self.window_size, fftbins=True))
        geometry = WindowGeometry(
            window_size=self.window_size, hop_stride=float(self.hop_size)
        )
        frame_fn = _stft_frame(self.window_size, self.freq_bins, window)
        return WindowedOperator(geometry, frame_fn, in_layout, out_layout)

    def _bin_frequencies(self, profile: AcquisitionProfile) -> np.ndarray:
        """The physical frequency of each kept two-sided bin, in Hz."""

        rate_hz = profile.require_sampling_rate(self.name)
        return as_real_array(np.arange(self.freq_bins) * rate_hz / self.window_size)
