"""ComplexStftMagnitude on raw 5-D CSI: batch padding, edges, power, bounded memory."""
# Each test compiles a small pipeline the same way, by design.
# pylint: disable=duplicate-code

import tracemalloc

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon import AcquisitionProfile, AxisName, Pipeline, Signal
from csiphon.core.errors import CompileError
from csiphon.steps import ComplexStftMagnitude, MeanOverAxes, Power

pytest.importorskip("scipy")

_RATE_HZ = 173.0
_WINDOW = 31
_NFFT = 100


def _raw_signal(
    n_samples: int, n_subcarriers: int
) -> tuple[AcquisitionProfile, Signal]:
    """Complex noise in the raw layout (time, receiver, tx, rx, subcarrier)."""

    indices = tuple(range(n_subcarriers))
    profile = AcquisitionProfile(
        subcarrier_indices=indices, n_rx_antennas=1, sampling_rate_hz=_RATE_HZ
    )
    rng = np.random.default_rng(0)
    shape = (n_samples, 1, n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return profile, profile.raw_signal(csi, np.arange(n_samples) / _RATE_HZ)


def _sharp_like(**overrides: object) -> ComplexStftMagnitude:
    options: dict[str, object] = {
        "window_size": _WINDOW,
        "hop_size": 1,
        "freq_bins": _NFFT,
        "nfft": _NFFT,
        "signed_bins": True,
        "symmetric_window": True,
    }
    return ComplexStftMagnitude(**(options | overrides))  # type: ignore[arg-type]


def test_zero_padding_runs_on_raw_layout_and_streams_alike() -> None:
    """Zero-padding (nfft > window_size) on a raw 5-D signal runs in batch, and the
    interior windows equal the streaming output. Guards against scipy's padding,
    which only handles 1-D and 2-D input."""

    profile, signal = _raw_signal(300, 4)
    compiled = Pipeline().then(_sharp_like()).compile(profile)

    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, 40)

    # Batch frame p is centred on sample p, streamed frame i on sample i + 15.
    offset = _WINDOW // 2
    count = streamed.values.shape[0]
    assert batch.values.shape[1:] == (_NFFT, 1, 1, 1, 4)
    np.testing.assert_allclose(
        streamed.values, batch.values[offset : offset + count], atol=1e-10
    )


@pytest.mark.parametrize("hop_size", [1, 5])
def test_valid_edges_batch_equals_streaming(hop_size: int) -> None:
    """edges="valid" keeps only full windows, so batch and streaming give the same
    values, the same number of windows (floor((N - window) / hop) + 1) and the same
    timestamps (each window's centre sample)."""

    profile, signal = _raw_signal(300, 4)
    compiled = (
        Pipeline().then(_sharp_like(hop_size=hop_size, edges="valid")).compile(profile)
    )

    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, 37)

    assert batch.values.shape[0] == (300 - _WINDOW) // hop_size + 1
    assert streamed.values.shape == batch.values.shape
    np.testing.assert_allclose(streamed.values, batch.values, atol=1e-10)
    np.testing.assert_array_equal(streamed.times, batch.times)
    first_centre = _WINDOW // 2
    np.testing.assert_allclose(
        batch.times[:3], (first_centre + hop_size * np.arange(3)) / _RATE_HZ
    )


def test_valid_edges_on_a_recording_shorter_than_the_window_is_empty() -> None:
    """No full window fits, so edges="valid" gives zero windows, as streaming does."""

    profile, signal = _raw_signal(_WINDOW - 1, 2)
    compiled = Pipeline().then(_sharp_like(edges="valid")).compile(profile)

    assert compiled.pour(signal).single().values.shape[0] == 0


def test_unknown_edges_value_is_rejected() -> None:
    """edges must be "pad" or "valid"."""

    profile, _ = _raw_signal(50, 2)
    with pytest.raises(CompileError, match="edges"):
        Pipeline().then(_sharp_like(edges="full")).compile(profile)


@pytest.mark.parametrize("edges", ["pad", "valid"])
@pytest.mark.parametrize("power", [True, False])
def test_power_and_mean_over_equal_the_separate_steps(edges: str, power: bool) -> None:
    """power=True with mean_over=(SUBCARRIER,) equals the STFT magnitude, then Power,
    then MeanOverAxes. Without power, it is the mean of the magnitude."""

    profile, signal = _raw_signal(200, 5)
    one_step = _sharp_like(edges=edges, power=power, mean_over=(AxisName.SUBCARRIER,))
    chain = Pipeline().then(_sharp_like(edges=edges))
    if power:
        chain = chain.then(Power())
    chain = chain.then(MeanOverAxes((AxisName.SUBCARRIER,)))

    fused = Pipeline().then(one_step).compile(profile).pour(signal).single()
    separate = chain.compile(profile).pour(signal).single()

    assert fused.layout == separate.layout
    np.testing.assert_allclose(fused.values, separate.values, atol=1e-10)
    np.testing.assert_array_equal(fused.times, separate.times)


def test_power_and_mean_over_stream_like_the_separate_steps() -> None:
    """The streaming path applies power and the mean per window, with the same
    result as the separate steps."""

    profile, signal = _raw_signal(200, 5)
    fused = Pipeline().then(
        _sharp_like(edges="valid", power=True, mean_over=(AxisName.SUBCARRIER,))
    )
    separate = (
        Pipeline()
        .then(_sharp_like(edges="valid"))
        .then(Power())
        .then(MeanOverAxes((AxisName.SUBCARRIER,)))
    )

    streamed = stream_in_chunks(fused.compile(profile), signal, 29)
    expected = separate.compile(profile).pour(signal).single()

    np.testing.assert_allclose(streamed.values, expected.values, atol=1e-10)


def test_mean_over_rejects_the_time_axis_and_repeats() -> None:
    """mean_over takes static axes only, each once."""

    profile, _ = _raw_signal(50, 2)
    for axes in ((AxisName.TIME,), (AxisName.SUBCARRIER, AxisName.SUBCARRIER)):
        with pytest.raises(CompileError, match="complex-stft-magnitude"):
            Pipeline().then(_sharp_like(mean_over=axes)).compile(profile)


def test_power_and_mean_over_keep_memory_bounded() -> None:
    """One antenna of one SHARP file (18 750 x 242) with power and a subcarrier mean
    stays far below 1 GB. The full spectrum alone would be about 7 GB."""

    profile, signal = _raw_signal(18_750, 242)
    compiled = (
        Pipeline()
        .then(_sharp_like(edges="valid", power=True, mean_over=(AxisName.SUBCARRIER,)))
        .compile(profile)
    )

    tracemalloc.start()
    try:
        output = compiled.pour(signal).single()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert output.values.shape == (18_750 - _WINDOW + 1, _NFFT, 1, 1, 1)
    assert peak_bytes < 1e9, f"peak {peak_bytes / 1e6:.0f} MB"


def test_defaults_on_raw_layout_equal_the_reference() -> None:
    """With no new option set, the raw 5-D output is the centred, zero-edged,
    periodic-Hann STFT with bins from DC upward (what 0.5.0 produced)."""

    from scipy.signal import get_window  # pylint: disable=import-outside-toplevel

    window_size, hop, bins = 64, 8, 10
    profile, signal = _raw_signal(200, 3)
    step = ComplexStftMagnitude(window_size=window_size, hop_size=hop, freq_bins=bins)
    output = Pipeline().then(step).compile(profile).pour(signal).single()

    data = np.asarray(signal.values).reshape(200, 3)
    window = get_window("hann", window_size, fftbins=True)
    padded = np.pad(data, ((window_size, window_size), (0, 0)))
    n_frames = int(np.ceil(200 / hop)) + 1
    expected = np.empty((n_frames, bins, 3))
    for frame in range(n_frames):
        start = window_size + frame * hop - window_size // 2
        block = padded[start : start + window_size] * window[:, None]
        expected[frame] = np.abs(np.fft.fft(block, axis=0)[:bins]) / window.sum()

    np.testing.assert_allclose(
        output.values.reshape(n_frames, bins, 3), expected, atol=1e-10
    )
    np.testing.assert_allclose(
        output.coords[AxisName.FREQUENCY], np.arange(bins) * _RATE_HZ / window_size
    )
