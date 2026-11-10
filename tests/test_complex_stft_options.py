"""ComplexStftMagnitude: zero-padding (nfft), signed bins, and the taper option."""
# Each test compiles a small pipeline the same way, by design.
# pylint: disable=duplicate-code

import numpy as np
import pytest
from conftest import fold_channels_into_feature, stream_in_chunks

from csiphon import AcquisitionProfile, AxisName, Pipeline, Signal, Siphon
from csiphon.core.errors import CompileError
from csiphon.steps import ComplexStftMagnitude

pytest.importorskip("scipy")

_RATE_HZ = 1000.0
_WINDOW = 31
_NFFT = 100


def _profile() -> AcquisitionProfile:
    return AcquisitionProfile(
        n_rx_antennas=1, subcarrier_indices=(0, 1), sampling_rate_hz=_RATE_HZ
    )


def _tone(frequency_hz: float, n_samples: int = 300) -> Signal:
    """A complex tone on every subcarrier, as a raw signal."""

    times = np.arange(n_samples) / _RATE_HZ
    csi = np.exp(2j * np.pi * frequency_hz * times)[:, None, None] * np.ones((1, 1, 2))
    return _profile().raw_signal(csi, times)


def _noise(n_samples: int = 300) -> Signal:
    rng = np.random.default_rng(7)
    csi = rng.standard_normal((n_samples, 1, 2)) + 1j * rng.standard_normal(
        (n_samples, 1, 2)
    )
    return _profile().raw_signal(csi, np.arange(n_samples) / _RATE_HZ)


def _compile(step: ComplexStftMagnitude) -> Siphon:
    return Pipeline().then(fold_channels_into_feature()).then(step).compile(_profile())


def _sharp_like(**overrides) -> ComplexStftMagnitude:
    options = {
        "window_size": _WINDOW,
        "hop_size": 1,
        "nfft": _NFFT,
        "freq_bins": _NFFT,
        "signed_bins": True,
    }
    return ComplexStftMagnitude(**(options | overrides))


@pytest.mark.parametrize("frequency_hz", [100.0, -100.0])
def test_signed_bins_peak_at_the_tone_frequency_with_its_sign(frequency_hz) -> None:
    """A complex tone at +f / -f peaks at the bin labelled +f / -f, which needs
    zero-padding (nfft > window_size) and signed ascending bins."""

    output = _compile(_sharp_like()).pour(_tone(frequency_hz)).single()
    frequencies = output.coords[AxisName.FREQUENCY]
    interior = output.values[len(output.values) // 2]

    peak_bin = np.argmax(interior[:, 0])
    assert frequencies[peak_bin] == pytest.approx(frequency_hz)
    assert np.all(np.diff(frequencies) > 0)


def test_signed_bins_are_centred_on_dc() -> None:
    """With nfft=100 the signed axis runs -500 .. +490 Hz, DC at the middle bin."""

    layout = _compile(_sharp_like()).outlet_layout
    frequencies = np.asarray(layout.axis(AxisName.FREQUENCY).coordinates)

    assert frequencies.shape == (_NFFT,)
    assert frequencies[0] == pytest.approx(-500.0)
    assert frequencies[_NFFT // 2] == pytest.approx(0.0)
    assert frequencies[-1] == pytest.approx(490.0)


def test_signed_bins_keep_the_freq_bins_nearest_dc() -> None:
    """Fewer signed bins are taken symmetrically around DC (31 -> -15 .. +15)."""

    step = _sharp_like(freq_bins=31)
    layout = _compile(step).outlet_layout
    frequencies = np.asarray(layout.axis(AxisName.FREQUENCY).coordinates)

    assert frequencies.shape == (31,)
    assert frequencies[15] == pytest.approx(0.0)
    assert frequencies[0] == pytest.approx(-150.0)
    assert frequencies[-1] == pytest.approx(150.0)


def test_nfft_zero_pads_without_changing_the_amplitude_scale() -> None:
    """A tone's peak magnitude is unchanged by zero-padding (scale is 1/window.sum)."""

    on_grid = 1000.0 / _NFFT * 10
    padded = _compile(_sharp_like()).pour(_tone(on_grid)).single()
    peak = padded.values[len(padded.values) // 2].max()

    # A unit-amplitude tone on an FFT bin gives a magnitude of exactly 1.
    assert peak == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("signed_bins", [False, True])
def test_streaming_matches_batch_on_the_interior_frames(signed_bins) -> None:
    """Streaming emits valid windows only; these equal the batch frames at the
    same window centres, with zero-padding and in either bin order."""

    signal = _noise()
    compiled = _compile(_sharp_like(hop_size=5, signed_bins=signed_bins))

    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, 17)

    # Streamed frame i spans samples [5i, 5i + window); batch frame p is centred
    # on sample 5p, so streamed frame i is batch frame i + (window // 2) / 5.
    offset = (_WINDOW // 2) // 5
    count = streamed.values.shape[0]
    np.testing.assert_allclose(
        streamed.values, batch.values[offset : offset + count], atol=1e-10
    )


def test_default_output_is_unchanged() -> None:
    """The defaults must not change: centred frames, zero-padded
    edges, periodic Hann, FFT bins from DC upward, scaled by 1/window.sum()."""

    from scipy.signal import get_window  # pylint: disable=import-outside-toplevel

    window_size, hop, bins = 64, 8, 10
    signal = _noise(200)
    step = ComplexStftMagnitude(window_size=window_size, hop_size=hop, freq_bins=bins)
    output = _compile(step).pour(signal).single()

    # Independent reference on the folded (time, feature) data.
    data = np.asarray(signal.values).reshape(200, 2)
    window = get_window("hann", window_size, fftbins=True)
    padded = np.pad(data, ((window_size, window_size), (0, 0)))
    n_frames = int(np.ceil(200 / hop)) + 1
    expected = np.empty((n_frames, bins, 2))
    for frame in range(n_frames):
        # Frame `frame` is centred on sample frame * hop (padded: + window_size).
        start = window_size + frame * hop - window_size // 2
        block = padded[start : start + window_size] * window[:, None]
        expected[frame] = np.abs(np.fft.fft(block, axis=0)[:bins]) / window.sum()

    np.testing.assert_allclose(output.values, expected, atol=1e-10)
    frequencies = output.coords[AxisName.FREQUENCY]
    np.testing.assert_allclose(frequencies, np.arange(bins) * _RATE_HZ / window_size)


def test_symmetric_window_matches_numpy_hanning() -> None:
    """symmetric_window=True tapers with np.hanning, as SHARP does."""

    signal = _noise()
    compiled = _compile(_sharp_like(symmetric_window=True))
    batch = compiled.pour(signal).single()

    data = np.asarray(signal.values).reshape(-1, 2)
    window = np.hanning(_WINDOW)
    # Batch frame p is centred on sample p, so the centre frame 100 spans 85..115.
    block = data[100 - _WINDOW // 2 : 100 + _WINDOW // 2 + 1] * window[:, None]
    expected = np.fft.fftshift(np.abs(np.fft.fft(block, n=_NFFT, axis=0)), axes=0)
    np.testing.assert_allclose(batch.values[100], expected / window.sum(), atol=1e-10)


def test_nfft_below_window_size_is_rejected() -> None:
    """Zero-padding cannot shorten the window, so nfft < window_size fails."""

    with pytest.raises(CompileError, match="nfft"):
        _compile(ComplexStftMagnitude(window_size=64, nfft=32, freq_bins=8))


def test_freq_bins_above_nfft_is_rejected_and_below_it_is_accepted() -> None:
    """freq_bins is bounded by the FFT length, not the window length."""

    with pytest.raises(CompileError, match="freq_bins"):
        _compile(ComplexStftMagnitude(window_size=8, nfft=16, freq_bins=17))

    _compile(ComplexStftMagnitude(window_size=8, nfft=16, freq_bins=16))


@pytest.mark.parametrize("frequency_hz", [100.0, -100.0])
def test_unsigned_bins_beyond_half_the_fft_are_labelled_as_negative(
    frequency_hz: float,
) -> None:
    """Without signed_bins, bins past nfft / 2 are negative frequencies in raw FFT
    order, so a tone at -f must peak at a bin labelled -f (not at fs - f).
    Guards against labelling every kept bin as a frequency from DC upward."""

    step = _sharp_like(signed_bins=False)
    output = _compile(step).pour(_tone(frequency_hz)).single()
    frequencies = output.coords[AxisName.FREQUENCY]
    interior = output.values[len(output.values) // 2]

    assert frequencies[np.argmax(interior[:, 0])] == pytest.approx(frequency_hz)
