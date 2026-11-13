"""STFT magnitude of a complex signal (needs the `[filters]` extra, for scipy).

`WindowedFFTPower` handles real signals with a one-sided FFT. This step takes
complex signals and a two-sided FFT, so positive and negative Doppler
frequencies stay apart.

By default (`edges="pad"`), batch windows are centered on samples 0, hop, 2*hop,
..., with zeros beyond both ends of the recording. Streaming emits a window only
once all its samples have arrived, stamped at its center, so it creates no
zero-padded windows at the ends. Batch and streaming then agree in the interior
and differ at the ends (BATCH_DIVERGENT). With `edges="valid"`, batch keeps only
full windows as well, so both agree exactly (BATCH_EQUIVALENT).

By default the FFT length is the window length and the kept bins start at DC.
If `nfft` is set, we zero-pad each window to that FFT length. If `signed_bins`
is True, we keep the bins in ascending frequency order around DC, negative
frequencies first. Windows are centered on their timestamp; for a window that
ends at its timestamp, shift the timestamps afterwards.

If `power` is True, we return the squared magnitude. If `mean_over` is set, we
average over those axes while computing, one block of windows at a time, so the
full spectrum of every channel never exists at once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, cast

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from csiphon.core.arrays import (
    RealArray,
    SignalArray,
    as_complex_array,
    as_real_array,
    as_signal_array,
)
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError, MissingDependencyError
from csiphon.core.layout import Layout
from csiphon.core.sampling import divide_nominal_rate, require_nominal_rate
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import (
    CONFIG_DEPENDENT,
    Category,
    LayoutEffect,
    StepSpec,
    Streaming,
)
from csiphon.steps._support.windowing import (
    IndexArray,
    SampleWindowOperator,
    WindowInSamples,
    sliding_window_indices,
)

# Batch computes this many windows at a time, so its memory use does not grow with
# the recording: a block's complex spectrum is the largest array that exists.
_WINDOWS_PER_BLOCK = 128


def _get_window() -> Callable[..., RealArray]:
    """Import scipy's get_window lazily, with a helpful error."""

    try:
        # Lazy so the base install stays numpy-only.
        from scipy.signal import get_window  # pylint: disable=import-outside-toplevel
    except ImportError as error:  # pragma: no cover - exercised via message
        raise MissingDependencyError("scipy", "filters") from error
    return get_window  # type: ignore[no-any-return]


# One dataclass field per option, so the attribute count follows the options.
@dataclass(frozen=True, slots=True)
class ComplexStftMagnitude(Step):  # pylint: disable=too-many-instance-attributes
    """STFT magnitude of a complex signal, computed along time.

    Time must be the first axis. We add a frequency axis after it and leave all
    other axes unchanged, so (time, rx_antenna, subcarrier) becomes
    (time, frequency, rx_antenna, subcarrier).

    For each window we apply a Hann taper, zero-pad to `nfft`, take the FFT, and
    keep `freq_bins` bins. By default these are the first bins (DC, then positive
    frequencies). If `signed_bins` is True, they are the bins nearest DC, in
    ascending frequency. The time axis needs a nominal sampling rate.

    Streaming differs from batch only at the recording ends, unless
    `edges="valid"` (see the module docstring).
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
    edges: str = field(
        default="pad",
        metadata={
            "doc": '"pad": also center windows near the ends, padding with zeros; '
            '"valid": full windows only (as in streaming)'
        },
    )
    power: bool = field(
        default=False,
        metadata={"doc": "output the squared magnitude instead of the magnitude"},
    )
    mean_over: tuple[AxisName, ...] = field(
        default=(),
        metadata={"doc": "axes to average over and remove"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="complex-stft-magnitude",
        summary="STFT magnitude of a complex signal, two-sided",
        category=Category.TIME_FREQUENCY,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=CONFIG_DEPENDENT,
        streaming=CONFIG_DEPENDENT,
        streaming_note='edges="pad": batch zero-pads the edges; "valid": same as batch',
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require time, and every axis we average over."""

        return (AxisName.TIME, *self.mean_over)

    def resolve_streaming(self) -> Streaming:
        """Batch matches streaming exactly only when it keeps full windows."""

        if self.edges == "valid":
            return Streaming.BATCH_EQUIVALENT
        return Streaming.BATCH_DIVERGENT

    def resolve_layout_effect(self) -> LayoutEffect:
        """Add a frequency axis, remove the averaged axes, set the value kind."""

        return LayoutEffect(
            adds=(AxisName.FREQUENCY,),
            removes=self.mean_over,
            value_kind=ValueKind.POWER if self.power else ValueKind.MAGNITUDE,
            note="time-frequency",
        )

    def output_layout(self, layout: Layout) -> Layout:
        """Insert a frequency axis after time; drop the averaged axes."""

        self.require_inputs(layout)

        # The windows slide along time, so time must be the first axis.
        if layout.dynamic_index != 0:
            raise LayoutError("ComplexStftMagnitude expects the time axis first.")

        # A window needs at least one sample, and the hop must move forward. Without
        # this the window grid is empty or never advances.
        if self.window_size < 1:
            raise LayoutError(f"window_size must be >= 1, got {self.window_size}.")
        if self.hop_size < 1:
            raise LayoutError(f"hop_size must be >= 1, got {self.hop_size}.")

        # Zero-padding can only make the FFT longer than the window, never shorter.
        if self.nfft is not None and self.nfft < self.window_size:
            raise LayoutError(
                f"nfft must be >= window_size={self.window_size}, got {self.nfft}."
            )

        # The FFT has nfft bins, so we cannot keep more than that.
        if not 1 <= self.freq_bins <= self.fft_length:
            raise LayoutError(
                f"freq_bins must be in [1, nfft={self.fft_length}], "
                f"got {self.freq_bins}."
            )

        if self.edges not in ("pad", "valid"):
            raise LayoutError(f'edges must be "pad" or "valid", got {self.edges!r}.')

        # Each averaged axis must exist once and have a fixed size. The fixed-size
        # check also rejects the time axis, which we cannot average away.
        if len(set(self.mean_over)) != len(self.mean_over):
            raise LayoutError("mean_over was given a repeated axis.")
        for name in self.mean_over:
            layout.require_static_axis(name)

        # The frequency axis carries the Hz value of every kept bin.
        frequency = Axis.static(
            AxisName.FREQUENCY, tuple(self._bin_frequencies(layout)), unit="Hz"
        )

        # Drop the averaged axes, put the frequency axis right after time, and
        # relabel the values: power if we square them, magnitude otherwise.
        output_values = ValueKind.POWER if self.power else ValueKind.MAGNITUDE
        # One output window per hop, so the output rate is the input rate over it.
        return (
            divide_nominal_rate(layout.remove_axes(self.mean_over), self.hop_size)
            .insert_axis_after(AxisName.TIME, frequency)
            .with_representation(Representation.TIME_FREQUENCY)
            .with_values(output_values)
        )

    def process(self, signal: Signal, out_layout: Layout) -> Signal:
        """Compute the output of every window, and stamp each at its center."""

        # No samples means no windows; return early rather than index empty times.
        if signal.n_samples == 0:
            return empty_signal(out_layout)

        values, starts, centers = self._place_windows(as_complex_array(signal.values))

        # A recording shorter than one window has no full window (edges="valid").
        if starts.size == 0:
            return empty_signal(out_layout)

        output = self._spectra_of_all_windows(values, starts, signal.layout, out_layout)
        return signal.with_values(
            output,
            out_layout,
            times=signal.times[centers],
            coords={AxisName.FREQUENCY: self._bin_frequencies(signal.layout)},
        )

    def stream(self, in_layout: Layout, out_layout: Layout) -> StreamOperator | None:
        """Ring-buffer operator emitting one spectrum per valid window."""

        taper = self._taper(_get_window())
        mean_axes = self._mean_axes(in_layout)

        def frame(block: SignalArray) -> RealArray:
            """The output for one window, (window, *channels) in."""

            return as_real_array(self._spectra(block[None], taper, mean_axes)[0])

        window = WindowInSamples(
            window_size=self.window_size, hop_stride=float(self.hop_size)
        )
        # We have a helper for stream operators that run a function on fixed
        # sliding windows. We tell it the window (its size and how far it moves)
        # and which function to run on each window. It keeps the chunks it has
        # seen so far as internal state, and slides the window over them as
        # specified. Here, the function computes the complex STFT of one window.
        return SampleWindowOperator(window, frame, in_layout, out_layout)

    def _place_windows(
        self, values: SignalArray
    ) -> tuple[SignalArray, IndexArray, IndexArray]:
        """Decide where the windows go.

        Returns the values to cut the windows from, the index of each window's
        first sample in them, and the index of each window's center sample in the
        original recording (used for its timestamp).
        """

        n_samples = values.shape[0]

        if self.edges == "valid":
            # Only windows that lie fully inside the recording.
            starts, centers = sliding_window_indices(
                n_samples, self.window_size, float(self.hop_size)
            )
            return values, starts, centers

        # edges="pad": window k is centered on sample k * hop_size, so the first
        # window is centered on sample 0 and the last one just past the final
        # sample. Zeros fill the parts of these windows outside the recording.
        # Padding first lets us cut them like any other window.
        n_windows = int(np.ceil(n_samples / self.hop_size)) + 1
        centers = np.arange(n_windows) * self.hop_size
        pad_before = self.window_size // 2
        pad_after = max(0, centers[-1] + self.window_size - pad_before - n_samples)
        no_padding = [(0, 0)] * (values.ndim - 1)
        padded = np.pad(values, [(pad_before, pad_after), *no_padding])

        # After padding in front, a window centered on sample c starts at index c.
        starts = centers
        timestamp_samples = np.minimum(centers, n_samples - 1)
        return as_signal_array(padded), starts, timestamp_samples

    def _spectra_of_all_windows(
        self,
        values: SignalArray,
        starts: IndexArray,
        in_layout: Layout,
        out_layout: Layout,
    ) -> RealArray:
        """Return the output of every window, a block of windows at a time.

        Output axes: (window, frequency, *channels left after `mean_over`).
        """

        taper = self._taper(_get_window())
        mean_axes = self._mean_axes(in_layout)

        # Every possible window as (window index, sample in window, *channels).
        # This is only a view of `values`: nothing is copied until we pick windows.
        all_windows = np.moveaxis(
            sliding_window_view(values, self.window_size, axis=0), -1, 1
        )

        # Every axis after time has a fixed size, so the output shape is known now.
        channel_shape = cast(tuple[int, ...], out_layout.shape[1:])
        output = np.empty((starts.size, *channel_shape))

        # One block of windows at a time. A block's complex spectrum is the largest
        # array that exists, so memory does not grow with the recording length.
        for first in range(0, starts.size, _WINDOWS_PER_BLOCK):
            block_starts = starts[first : first + _WINDOWS_PER_BLOCK]
            block_windows = as_signal_array(all_windows[block_starts])
            last = first + block_starts.size
            output[first:last] = self._spectra(block_windows, taper, mean_axes)

        return output

    @property
    def fft_length(self) -> int:
        """The FFT length: `nfft`, or `window_size` when nfft is None."""

        return self.window_size if self.nfft is None else self.nfft

    def _taper(self, get_window: Callable[..., RealArray]) -> RealArray:
        """The Hann taper; scipy's `fftbins=True` is the periodic form."""

        return as_real_array(
            get_window("hann", self.window_size, fftbins=not self.symmetric_window)
        )

    def _mean_axes(self, in_layout: Layout) -> tuple[int, ...]:
        """Array axes of `mean_over` in a (window index, window, *channels) array."""

        # The time axis is first, so a channel axis sits one place later than in the
        # layout, after the added window index.
        return tuple(in_layout.axis_position(name) + 1 for name in self.mean_over)

    def _spectra(
        self, windows: SignalArray, taper: RealArray, mean_axes: tuple[int, ...]
    ) -> RealArray:
        """Return the output of a few windows at once.

        Input axes: (window, sample in window, *channels).
        Output axes: (window, frequency, *channels), minus the `mean_axes`.
        """

        # Multiply every window by the taper (broadcast over windows and channels).
        taper_shape = (1, self.window_size) + (1,) * (windows.ndim - 2)
        tapered = windows * taper.reshape(taper_shape)

        # n=nfft zero-pads each tapered window up to the FFT length.
        spectrum = np.fft.fft(tapered, n=self.fft_length, axis=1)

        # Keep the wanted bins, scaled so a unit-amplitude tone has magnitude 1.
        magnitude = np.abs(spectrum[:, self._kept_bins()]) / float(taper.sum())

        result = magnitude**2 if self.power else magnitude
        if mean_axes:
            result = result.mean(axis=mean_axes)
        return as_real_array(result)

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

    def _bin_frequencies(self, layout: Layout) -> np.ndarray:
        """The physical frequency of each kept two-sided bin, in Hz."""

        rate_hz = require_nominal_rate(layout, self.name)

        # fftfreq follows raw FFT order, so bins past nfft / 2 come out negative.
        frequencies = np.fft.fftfreq(self.fft_length, 1.0 / rate_hz)
        return as_real_array(frequencies[self._kept_bins()])
