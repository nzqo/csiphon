"""STFT magnitude of a complex signal (needs the `[filters]` extra, for scipy).

`WindowedFFTPower` handles real signals with a one-sided FFT. This step takes
complex signals and a two-sided FFT, so positive and negative Doppler
frequencies stay apart.

In batch, windows are centered on samples 0, hop, 2*hop, ..., with zeros beyond
both ends of the recording. Streaming emits a window only once all its samples
have arrived, stamped at its center, so it creates no zero-padded windows at the
ends. Batch and streaming agree in the interior and differ at the ends
(BATCH_DIVERGENT).

By default the FFT length is the window length and the kept bins start at DC.
If `nfft` is set, we zero-pad each window to that FFT length. If `signed_bins`
is True, we keep the bins in ascending frequency order around DC, negative
frequencies first. Windows are centered on their timestamp; for a window that
ends at its timestamp, shift the timestamps afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
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


def _stft_frame(
    window_size: int, nfft: int, kept: np.ndarray, window: RealArray
) -> FrameFn:
    """Return a function that computes the magnitude spectrum of one window.

    We divide by window.sum(). In batch, scipy's `scale_to="magnitude"` does the
    same division, so both paths give the same values.
    """

    scale = float(window.sum())

    def frame(block: SignalArray) -> RealArray:
        """The tapered FFT of one window."""

        # A block is one window: axis 0 is time within it, the rest ride along.
        taper_shape = (window_size,) + (1,) * (block.ndim - 1)
        tapered = block * window.reshape(taper_shape)
        # n=nfft zero-pads the tapered window up to the FFT length.
        spectrum = np.fft.fft(tapered, n=nfft, axis=0)
        return as_real_array(np.abs(spectrum[kept]) / scale)

    return frame


@dataclass(frozen=True, slots=True)
class ComplexStftMagnitude(Step):
    """STFT magnitude of a complex signal, computed along time.

    Time must be the first axis. We add a frequency axis after it and leave all
    other axes unchanged, so (time, rx_antenna, subcarrier) becomes
    (time, frequency, rx_antenna, subcarrier).

    For each window we apply a Hann taper, zero-pad to `nfft`, take the FFT, and
    keep `freq_bins` bins. By default these are the first bins (DC, then positive
    frequencies). If `signed_bins` is True, they are the bins nearest DC, in
    ascending frequency. The profile needs a sampling rate. Streaming differs
    from batch only at the recording ends (see the module docstring).
    """

    window_size: int = field(
        default=512, metadata={"doc": "STFT window length in samples"}
    )
    hop_size: int = field(default=10, metadata={"doc": "hop between frames in samples"})
    freq_bins: int = field(
        default=31, metadata={"doc": "number of low two-sided bins to keep"}
    )
    nfft: int | None = field(
        default=None,
        metadata={"doc": "FFT length, zero-padding the window (window_size if None)"},
    )
    signed_bins: bool = field(
        default=False,
        metadata={"doc": "keep the bins nearest DC in signed ascending order"},
    )
    symmetric_window: bool = field(
        default=False,
        metadata={"doc": "use a symmetric Hann taper instead of the periodic one"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="complex-stft-magnitude",
        summary="STFT magnitude of a complex signal, two-sided",
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
        if self.nfft is not None and self.nfft < self.window_size:
            raise LayoutError(
                f"nfft must be >= window_size={self.window_size}, got {self.nfft}."
            )
        if not 1 <= self.freq_bins <= self.fft_length:
            raise LayoutError(
                f"freq_bins must be in [1, nfft={self.fft_length}], "
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

        # Set up scipy's STFT with the batch framing from the module docstring. The grid
        # is zero-padded at the boundary so the first frame sits on sample 0
        # (p0=0 below) and the last runs just past the final sample.
        values = as_complex_array(signal.values)
        n_frames = int(np.ceil(values.shape[0] / self.hop_size)) + 1
        window = self._taper(get_window)
        stft = short_time_fft(
            window,
            hop=self.hop_size,
            fs=rate_hz,
            fft_mode="twosided",
            mfft=self.fft_length,
            scale_to="magnitude",
        )

        # Run the STFT along time (axis 0); every other axis is a channel carried
        # along. scipy replaces the time axis with frequency and appends the frame
        # axis last (freq, *channels, frame), so keep the low bins and move
        # frame to the front, frequency just after it: (frame, freq, *channels).
        spectrum = stft.stft(values, p0=0, p1=n_frames, padding="zeros", axis=0)
        magnitude = np.moveaxis(np.abs(spectrum[self._kept_bins()]), (0, -1), (1, 0))

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
        window = self._taper(get_window)
        geometry = WindowGeometry(
            window_size=self.window_size, hop_stride=float(self.hop_size)
        )
        frame_fn = _stft_frame(
            self.window_size, self.fft_length, self._kept_bins(), window
        )
        return WindowedOperator(geometry, frame_fn, in_layout, out_layout)

    @property
    def fft_length(self) -> int:
        """The FFT length: `nfft`, or `window_size` when nfft is None."""

        return self.window_size if self.nfft is None else self.nfft

    def _taper(self, get_window: Callable[..., RealArray]) -> RealArray:
        """The Hann taper; scipy's `fftbins=True` is the periodic form."""

        return as_real_array(
            get_window("hann", self.window_size, fftbins=not self.symmetric_window)
        )

    def _kept_bins(self) -> np.ndarray:
        """Return the indices of the FFT bins to keep, in output order.

        If `signed_bins` is False: the first `freq_bins` bins in raw FFT order (DC,
        then positive frequencies, then negative ones if `freq_bins` exceeds
        nfft / 2).
        If `signed_bins` is True: the `freq_bins` bins nearest DC, from negative to
        positive.
        """

        if not self.signed_bins:
            return np.arange(self.freq_bins)

        # fftshift puts DC at index nfft // 2; take freq_bins centered on it.
        shifted = np.fft.fftshift(np.arange(self.fft_length))
        start = self.fft_length // 2 - self.freq_bins // 2
        return shifted[start : start + self.freq_bins]

    def _bin_frequencies(self, profile: AcquisitionProfile) -> np.ndarray:
        """The physical frequency of each kept two-sided bin, in Hz."""

        rate_hz = profile.require_sampling_rate(self.name)

        # fftfreq follows raw FFT order, so bins past nfft / 2 come out negative.
        frequencies = np.fft.fftfreq(self.fft_length, 1.0 / rate_hz)
        return as_real_array(frequencies[self._kept_bins()])
