"""Ported spectral steps: channel impulse response and the multitaper spectrogram."""

# Feature-signal setup overlaps with other test modules by design.
# pylint: disable=duplicate-code
from __future__ import annotations

import numpy as np
import pytest
from conftest import fold_channels_into_feature, stream_in_chunks

from csiphon import (
    AcquisitionProfile,
    Axis,
    AxisName,
    Layout,
    Pipeline,
    Representation,
    Signal,
    ValueKind,
    create_signal,
)
from csiphon.core import CompileError
from csiphon.steps import (
    ChannelImpulseResponse,
    ComplexStftMagnitude,
    Magnitude,
    Multitaper,
    SynchrosqueezedPower,
    WindowedFFTPower,
)
from csiphon.steps.delay.delay_autocorrelation import resolve_nfft


def test_cir_recovers_a_known_impulse_response(  # pylint: disable=too-many-locals
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """
    CSI built from a sparse CIR inverts to that CIR in the leading taps, zeros after.
    """

    num_taps = 4
    nfft = resolve_nfft(profile.n_subcarriers)
    subcarriers = np.asarray(profile.subcarrier_indices, dtype=float)
    taps = np.arange(num_taps, dtype=float)
    forward = np.exp(-1j * 2.0 * np.pi * np.outer(subcarriers, taps) / nfft) / np.sqrt(
        nfft
    )

    rng = np.random.default_rng(4)
    impulse = rng.standard_normal(num_taps) + 1j * rng.standard_normal(num_taps)
    frame = forward @ impulse  # (n_subcarriers,)
    values = np.broadcast_to(frame, raw_signal.values.shape).copy()
    signal = raw_signal.with_values(values, raw_signal.layout)

    # A model order of num_taps fits exactly (well-conditioned over-determined LS).
    compiled = (
        Pipeline().then(ChannelImpulseResponse(num_taps=num_taps)).compile(profile)
    )
    out = compiled.pour(signal).single()
    assert out.layout.axis(AxisName.DELAY).size == num_taps
    position = out.layout.axis_position(AxisName.DELAY)
    recovered = np.moveaxis(out.values, position, -1)
    assert np.allclose(recovered, impulse, atol=1e-8)


def test_default_cir_is_the_full_length_min_norm_inverse(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """
    Default num_taps=None gives all nfft taps:
    the plain inverse DFT of the observed tones.
    """

    nfft = resolve_nfft(profile.n_subcarriers)
    subcarriers = np.asarray(profile.subcarrier_indices, dtype=float)

    rng = np.random.default_rng(1)
    frame = rng.standard_normal(profile.n_subcarriers) + 1j * rng.standard_normal(
        profile.n_subcarriers
    )
    values = np.broadcast_to(frame, raw_signal.values.shape).copy()
    signal = raw_signal.with_values(values, raw_signal.layout)

    out = (
        Pipeline().then(ChannelImpulseResponse()).compile(profile).pour(signal).single()
    )
    assert out.layout.axis(AxisName.DELAY).size == nfft

    # Min-norm inverse == the adjoint (sum over observed subcarriers only).
    taps = np.arange(nfft, dtype=float)
    adjoint = np.exp(1j * 2.0 * np.pi * np.outer(taps, subcarriers) / nfft) / np.sqrt(
        nfft
    )
    expected = adjoint @ frame  # (nfft,)
    position = out.layout.axis_position(AxisName.DELAY)
    recovered = np.moveaxis(out.values, position, -1)
    assert np.allclose(recovered, expected, atol=1e-10)


def test_windowed_fft_power_frequency_axis_is_physical_and_in_band(
    profile: AcquisitionProfile,
) -> None:
    """WindowedFFTPower adds a frequency axis in physical Hz.

    The bins start at 0 Hz, are evenly spaced, and never exceed the requested
    band or the Nyquist frequency -- they are real Doppler frequencies, not bin
    indices.
    """

    signal = _feature_signal()
    band_hz = 60.0
    out = (
        Pipeline()
        .then(WindowedFFTPower(window_s=0.128, hop_s=0.05, band_hz=band_hz))
        .compile(profile, inlet=signal.layout)
        .pour(signal)
        .single()
    )

    assert out.layout.axis_names[:2] == (AxisName.TIME, AxisName.FREQUENCY)
    freq_axis = out.layout.axis(AxisName.FREQUENCY)
    assert freq_axis.unit == "Hz"
    assert profile.sampling_rate_hz is not None
    freqs = np.asarray(freq_axis.coordinates)
    nyquist = profile.sampling_rate_hz / 2.0

    assert freqs[0] == 0.0
    spacing = np.diff(freqs)
    assert np.all(spacing > 0.0)  # strictly increasing
    assert np.allclose(spacing, spacing[0])  # evenly spaced FFT bins
    assert freqs[-1] <= band_hz + 1e-9  # stays inside the requested band
    assert freqs[-1] <= nyquist  # and below Nyquist


def test_synchrosqueezed_frequency_axis_is_physical_up_to_nyquist(
    profile: AcquisitionProfile,
) -> None:
    """SST adds a runtime-sized frequency axis in physical Hz, reaching Nyquist.

    Its coordinates are positive real frequencies that go up to -- but never past
    -- the Nyquist frequency of the recording.
    """

    pytest.importorskip("ssqueezepy")  # SST's backend, not scipy
    signal = _feature_signal()
    out = (
        Pipeline()
        .then(SynchrosqueezedPower())
        .compile(profile, inlet=signal.layout)
        .pour(signal)
        .single()
    )

    # The frequency axis is inserted right after time.
    assert out.layout.axis_names.index(AxisName.FREQUENCY) == 1
    assert profile.sampling_rate_hz is not None
    freqs = np.asarray(out.coords[AxisName.FREQUENCY])
    nyquist = profile.sampling_rate_hz / 2.0

    assert np.all(freqs > 0.0)
    assert np.all(freqs <= nyquist + 1e-9)
    assert np.isclose(freqs.max(), nyquist)  # the top bin is the Nyquist frequency


def test_synchrosqueezed_carries_extra_channel_axes() -> None:
    """SST broadcasts the per-channel transform over every channel axis.

    A raw multi-channel recording transforms directly -- no fold needed -- and gives
    the same numbers as folding the channels into one first, in one clean reshape.
    """

    pytest.importorskip("ssqueezepy")
    profile = AcquisitionProfile(
        n_rx_antennas=2, subcarrier_indices=tuple(range(3)), sampling_rate_hz=1000.0
    )
    rng = np.random.default_rng(0)
    shape = (256, profile.n_rx_antennas, profile.n_subcarriers)
    signal = profile.raw_signal(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        np.arange(256) / 1000.0,
    )

    # Magnitude puts the signal in SST's admissible (real) domain, keeping the axes.
    nd = (
        (Pipeline().then(Magnitude()).then(SynchrosqueezedPower()).compile(profile))
        .pour(signal)
        .single()
    )
    channel_axes = signal.layout.axis_names[1:]
    assert nd.layout.axis_names == (AxisName.TIME, AxisName.FREQUENCY, *channel_axes)

    folded = (
        Pipeline()
        .then(Magnitude())
        .then(fold_channels_into_feature())
        .then(SynchrosqueezedPower())
        .compile(profile)
        .pour(signal)
        .single()
    )
    assert np.allclose(nd.values.reshape(folded.values.shape), folded.values)


def _feature_signal(n: int = 1500, d: int = 6) -> Signal:
    """A real `(time, feature)` signal for the multitaper spectrogram."""

    rng = np.random.default_rng(0)
    layout = Layout(
        axes=(Axis.dynamic(AxisName.TIME, unit="s"), Axis.sized(AxisName.FEATURE, d)),
        representation=Representation.FEATURE_VECTOR,
        values=ValueKind.REAL,
    )
    return create_signal(rng.standard_normal((n, d)), np.arange(n) / 1000.0, layout)


def test_multitaper_reduces_to_time_frequency(profile: AcquisitionProfile) -> None:
    """Multitaper collapses the feature axis and adds an in-band frequency axis."""

    pytest.importorskip("scipy")
    signal = _feature_signal()
    compiled = (
        Pipeline()
        .then(Multitaper(window_s=0.128, hop_s=0.05, band_hz=60.0))
        .compile(profile, inlet=signal.layout)
    )
    out = compiled.pour(signal).single()
    assert out.layout.axis_names == (AxisName.TIME, AxisName.FREQUENCY)
    assert out.layout.values == ValueKind.POWER
    assert np.all(out.values >= 0.0)


@pytest.mark.parametrize("chunk", [37, 128])
def test_multitaper_stream_equals_batch(
    profile: AcquisitionProfile, chunk: int
) -> None:
    """The buffered multitaper spectrogram streams identically to batch."""

    pytest.importorskip("scipy")
    signal = _feature_signal()
    compiled = (
        Pipeline()
        .then(Multitaper(window_s=0.128, hop_s=0.05, band_hz=60.0))
        .compile(profile, inlet=signal.layout)
    )
    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, chunk)
    assert batch.values.shape == streamed.values.shape
    assert np.allclose(batch.values, streamed.values)
    assert np.allclose(batch.times, streamed.times)


def _complex_feature_signal(n: int = 2000, d: int = 12) -> Signal:
    """A complex `(time, feature)` signal for the complex STFT."""

    rng = np.random.default_rng(3)
    layout = Layout(
        axes=(Axis.dynamic(AxisName.TIME, unit="s"), Axis.sized(AxisName.FEATURE, d)),
        representation=Representation.CROSS_SPECTRUM,
        values=ValueKind.COMPLEX,
    )
    data = rng.standard_normal((n, d)) + 1j * rng.standard_normal((n, d))
    return create_signal(data, np.arange(n) / 1000.0, layout)


@pytest.mark.parametrize("chunk", [1, 13, 64, 500])
def test_complex_stft_streams_valid_windows(
    profile: AcquisitionProfile, chunk: int
) -> None:
    """Each streamed frame is the two-sided FFT magnitude of the window it covers.

    Streaming uses valid windows (no fabricated edge padding), so the result is
    independent of the chunk size and every frame equals a direct per-window
    transform of the exact samples it spans, timestamped at the window center.
    """

    scipy_signal = pytest.importorskip("scipy.signal")
    signal = _complex_feature_signal()
    win_size, hop, bins = 256, 10, 16
    compiled = (
        Pipeline()
        .then(ComplexStftMagnitude(window_size=win_size, hop_size=hop, freq_bins=bins))
        .compile(profile, inlet=signal.layout)
    )

    streamed = stream_in_chunks(compiled, signal, chunk)

    # Independent reference: the same periodic-Hann, two-sided, magnitude-scaled
    # FFT applied directly to each valid window of the original samples.
    data = np.asarray(signal.values)
    window = scipy_signal.get_window("hann", win_size, fftbins=True)
    scale = window.sum()
    n_frames = streamed.values.shape[0]
    expected = np.stack(
        [
            np.abs(
                np.fft.fft(
                    data[i * hop : i * hop + win_size] * window[:, None],
                    n=win_size,
                    axis=0,
                )[:bins]
            )
            / scale
            for i in range(n_frames)
        ]
    )

    assert streamed.values.shape == (n_frames, bins, data.shape[1])
    assert np.allclose(streamed.values, expected)

    # Frames are timestamped at the window center, and no window runs off the end.
    centers = (np.arange(n_frames) * hop + win_size // 2) / 1000.0
    assert np.allclose(streamed.times, centers)
    assert (n_frames - 1) * hop + win_size <= data.shape[0]


def _stft_profile(fs: float = 1000.0, n_features: int = 4) -> AcquisitionProfile:
    """A single-antenna profile whose subcarrier axis stands in for the features."""

    return AcquisitionProfile(
        n_rx_antennas=1,
        subcarrier_indices=tuple(range(n_features)),
        sampling_rate_hz=fs,
    )


def _complex_tone(bin_index: int, n: int, win: int, features: int) -> np.ndarray:
    """A complex exponential at exactly FFT `bin_index`, repeated across features."""

    phase = np.exp(2j * np.pi * bin_index * np.arange(n) / win)
    return np.broadcast_to(phase[:, None], (n, features)).copy()


def test_complex_stft_frequency_axis_is_physical() -> None:
    """The kept two-sided bins carry real Hz coordinates, evenly spaced from 0."""

    pytest.importorskip("scipy")
    fs, win, bins = 1000.0, 256, 16
    profile = _stft_profile(fs)
    signal = profile.raw_signal(
        _complex_tone(3, 2000, win, profile.n_subcarriers), np.arange(2000) / fs
    )
    out = (
        Pipeline()
        .then(fold_channels_into_feature())
        .then(ComplexStftMagnitude(window_size=win, hop_size=10, freq_bins=bins))
        .compile(profile)
        .pour(signal)
        .single()
    )

    freq_axis = out.layout.axis(AxisName.FREQUENCY)
    assert out.layout.values == ValueKind.MAGNITUDE
    assert out.layout.representation == Representation.TIME_FREQUENCY
    assert freq_axis.unit == "Hz"
    assert freq_axis.size == bins
    coords = np.asarray(freq_axis.coordinates)
    assert np.allclose(coords, np.arange(bins) * fs / win)  # DC + low positive bins
    assert coords[0] == 0.0
    assert np.allclose(np.diff(coords), fs / win)  # evenly spaced


def test_complex_stft_isolates_positive_doppler() -> None:
    """Two-sided: a +f tone lands in its low bin; the mirror -f tone does not.

    A real (one-sided) transform could not tell the two apart -- this is the whole
    point of keeping the STFT complex.
    """

    pytest.importorskip("scipy")
    fs, win, hop, bins, k = 1000.0, 256, 10, 16, 5
    profile = _stft_profile(fs)
    times = np.arange(2000) / fs

    def run(tone: np.ndarray) -> np.ndarray:
        signal = profile.raw_signal(tone, times)
        out = (
            Pipeline()
            .then(fold_channels_into_feature())
            .then(ComplexStftMagnitude(window_size=win, hop_size=hop, freq_bins=bins))
            .compile(profile)
            .pour(signal)
            .single()
        )
        return np.asarray(out.values)

    positive = run(_complex_tone(k, 2000, win, profile.n_subcarriers))
    negative = run(_complex_tone(-k, 2000, win, profile.n_subcarriers))

    mid = positive.shape[0] // 2  # an interior frame, away from the padded edges
    assert int(np.argmax(positive[mid, :, 0])) == k
    assert np.isclose(positive[mid, k, 0], 1.0, atol=1e-6)  # magnitude scaling
    # The -f tone's energy sits at bin win-k, which is not among the kept low bins.
    assert negative[mid, :, 0].max() < 1e-6


def test_complex_stft_batch_shape_and_timestamps() -> None:
    """Batch keeps the feature axis and emits ceil(N/hop)+1 center-clamped frames."""

    pytest.importorskip("scipy")
    fs, win, hop, bins, n = 1000.0, 128, 10, 8, 1000
    profile = _stft_profile(fs, n_features=6)
    times = np.arange(n) / fs
    signal = profile.raw_signal(_complex_tone(2, n, win, profile.n_subcarriers), times)
    out = (
        Pipeline()
        .then(fold_channels_into_feature())
        .then(ComplexStftMagnitude(window_size=win, hop_size=hop, freq_bins=bins))
        .compile(profile)
        .pour(signal)
        .single()
    )

    n_frames = int(np.ceil(n / hop)) + 1
    # Folding the singleton structural axes with the subcarrier axis leaves one
    # channel (feature) axis, which the STFT keeps after the frequency axis.
    assert out.layout.axis_names == (
        AxisName.TIME,
        AxisName.FREQUENCY,
        AxisName.FEATURE,
    )
    assert out.values.shape == (n_frames, bins, profile.n_subcarriers)
    centers = np.minimum(np.arange(n_frames) * hop, n - 1)
    assert np.allclose(out.times, times[centers])
    assert np.all(out.values >= 0.0)


def test_complex_stft_carries_extra_axes(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """The STFT runs along time and carries every other axis through unchanged.

    A full `(time, receiver, tx, rx, subcarrier)` recording becomes
    `(time, frequency, receiver, tx, rx, subcarrier)` with no folding needed.
    Because the transform is per-channel, folding those axes into one first gives
    identical numbers -- in batch and while streaming.
    """

    pytest.importorskip("scipy")
    step = ComplexStftMagnitude(window_size=256, hop_size=10, freq_bins=16)
    compiled = Pipeline().then(step).compile(profile)

    nd = compiled.pour(raw_signal).single()
    # Frequency is inserted right after time; every input channel axis is carried.
    channel_axes = raw_signal.layout.axis_names[1:]
    assert nd.layout.axis_names == (AxisName.TIME, AxisName.FREQUENCY, *channel_axes)

    folded = (
        Pipeline()
        .then(fold_channels_into_feature())
        .then(step)
        .compile(profile)
        .pour(raw_signal)
        .single()
    )
    assert np.allclose(nd.values.reshape(folded.values.shape), folded.values)

    # Streaming carries the extra axes the same way.
    nd_stream = stream_in_chunks(compiled, raw_signal, 128)
    folded_stream = stream_in_chunks(
        Pipeline().then(fold_channels_into_feature()).then(step).compile(profile),
        raw_signal,
        128,
    )
    assert np.allclose(
        nd_stream.values.reshape(folded_stream.values.shape), folded_stream.values
    )


def test_complex_stft_rejects_more_bins_than_window() -> None:
    """Keeping more bins than the window has is a compile-time error."""

    pytest.importorskip("scipy")
    profile = _stft_profile()
    with pytest.raises(CompileError, match="complex-stft-magnitude"):
        Pipeline().then(fold_channels_into_feature()).then(
            ComplexStftMagnitude(window_size=8, freq_bins=16)
        ).compile(profile)


def test_complex_stft_needs_a_sampling_rate() -> None:
    """Sizing the frequency axis needs a nominal rate, so a rate-less profile fails."""

    pytest.importorskip("scipy")
    profile = AcquisitionProfile(
        n_rx_antennas=1, subcarrier_indices=tuple(range(4)), sampling_rate_hz=None
    )
    with pytest.raises(CompileError, match="complex-stft-magnitude"):
        Pipeline().then(fold_channels_into_feature()).then(
            ComplexStftMagnitude()
        ).compile(profile)


def _complex_noise(n: int, features: int, seed: int) -> np.ndarray:
    """A complex `(n, features)` white-noise block."""

    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, features)) + 1j * rng.standard_normal((n, features))


@pytest.mark.parametrize("window,hop", [(256, 8), (256, 10), (128, 20)])
def test_complex_stft_hop_and_window_set_the_frame_grid(window: int, hop: int) -> None:
    """Input length, hop, and window fully determine the output frame grid.

    Batch emits ceil(N/hop)+1 frames centered on p*hop (clamped to the last
    sample); streaming emits the valid-window count with centers p*hop+window//2.
    """

    pytest.importorskip("scipy")
    fs, n = 1000.0, 1000
    profile = _stft_profile(fs)
    noise = _complex_noise(n, profile.n_subcarriers, 0)
    signal = profile.raw_signal(noise, np.arange(n) / fs)
    compiled = (
        Pipeline()
        .then(fold_channels_into_feature())
        .then(ComplexStftMagnitude(window_size=window, hop_size=hop, freq_bins=8))
        .compile(profile)
    )

    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, 64)

    # Batch: length and hop spacing (centers at p*hop, clamped at the end).
    n_batch = int(np.ceil(n / hop)) + 1
    assert batch.values.shape[0] == n_batch
    batch_centers = np.minimum(np.arange(n_batch) * hop, n - 1)
    assert np.allclose(batch.times * fs, batch_centers)

    # Streaming: valid-window count and centers at p*hop + window//2.
    n_stream = (n - window) // hop + 1
    assert streamed.values.shape[0] == n_stream
    stream_centers = np.arange(n_stream) * hop + window // 2
    assert np.allclose(streamed.times * fs, stream_centers)


@pytest.mark.parametrize("window", [128, 256, 512])
def test_complex_stft_window_size_sets_frequency_resolution(window: int) -> None:
    """A larger window gives a finer frequency axis: bin spacing is fs/window."""

    pytest.importorskip("scipy")
    fs = 1000.0
    profile = _stft_profile(fs)
    signal = profile.raw_signal(
        _complex_tone(3, 2000, window, profile.n_subcarriers), np.arange(2000) / fs
    )
    out = (
        Pipeline()
        .then(fold_channels_into_feature())
        .then(ComplexStftMagnitude(window_size=window, hop_size=10, freq_bins=8))
        .compile(profile)
        .pour(signal)
        .single()
    )
    coords = np.asarray(out.layout.axis(AxisName.FREQUENCY).coordinates)
    assert np.allclose(np.diff(coords), fs / window)


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"window_size": 0}, "window_size"),
        ({"window_size": -4}, "window_size"),
        ({"hop_size": 0}, "hop_size"),
        ({"hop_size": -3}, "hop_size"),
    ],
)
def test_complex_stft_rejects_nonpositive_geometry(
    kwargs: dict[str, int], needle: str
) -> None:
    """A zero or negative window/hop fails at compile with a message naming it."""

    pytest.importorskip("scipy")
    profile = _stft_profile()
    with pytest.raises(CompileError, match=needle):
        Pipeline().then(fold_channels_into_feature()).then(
            ComplexStftMagnitude(**kwargs)
        ).compile(profile)


def test_complex_stft_batch_and_stream_agree_on_shared_interior() -> None:
    """Where the grids align (hop divides window//2), interior frames match exactly.

    Batch centers a frame on p*hop, streaming on k*hop + window//2. With
    hop | window//2 the two grids share windows, so a fully-interior batch frame
    (no zero-padding) equals its streamed counterpart to machine precision -- the
    two paths run the identical per-window transform.
    """

    pytest.importorskip("scipy")
    fs, n, window, hop = 1000.0, 1000, 256, 8
    profile = _stft_profile(fs)
    noise = _complex_noise(n, profile.n_subcarriers, 2)
    signal = profile.raw_signal(noise, np.arange(n) / fs)
    compiled = (
        Pipeline()
        .then(fold_channels_into_feature())
        .then(ComplexStftMagnitude(window_size=window, hop_size=hop, freq_bins=8))
        .compile(profile)
    )

    batch = compiled.pour(signal).single().values
    streamed = stream_in_chunks(compiled, signal, 64).values

    # batch frame (offset + k) shares a window with stream frame k
    offset = (window // 2) // hop
    assert np.allclose(batch[offset : offset + 20], streamed[:20], atol=1e-12)


def test_synchrosqueezed_block_local_streaming_is_chunk_invariant(
    profile: AcquisitionProfile,
) -> None:
    """Block-local SST buffers whole blocks, so streaming is the same however it's cut.

    It is BATCH_DIVERGENT, so we compare two streamings rather than stream vs batch:
    feeding one 256-sample block per push must equal feeding each block split across
    several pushes. This drives the block-buffer path no other test feeds with data.
    """

    pytest.importorskip("ssqueezepy")
    signal = _feature_signal(n=768)  # three whole 256-sample blocks
    compiled = (
        Pipeline()
        .then(SynchrosqueezedPower(streaming_window=256))
        .compile(profile, inlet=signal.layout)
    )
    whole_blocks = stream_in_chunks(compiled, signal, 256)
    split_blocks = stream_in_chunks(compiled, signal, 64)

    assert np.array_equal(whole_blocks.times, split_blocks.times)
    assert np.allclose(whole_blocks.values, split_blocks.values)


def test_synchrosqueezed_concentrates_power_at_the_tone_frequency(
    profile: AcquisitionProfile,
) -> None:
    """SST power piles up near a pure tone's frequency -- an independent numeric check.

    The in-suite SST tests otherwise only compare the transform to itself; this pins its
    numbers against a signal whose answer is known (a 60 Hz cosine).
    """

    pytest.importorskip("ssqueezepy")
    assert profile.sampling_rate_hz is not None
    tone_hz = 60.0
    times = np.arange(1024) / profile.sampling_rate_hz
    layout = Layout(
        (Axis.dynamic(AxisName.TIME, unit="s"), Axis.sized(AxisName.FEATURE, 1)),
        Representation.FEATURE_VECTOR,
        ValueKind.REAL,
    )
    tone = np.cos(2.0 * np.pi * tone_hz * times)[:, None]
    signal = create_signal(tone, times, layout)

    out = (
        Pipeline()
        .then(SynchrosqueezedPower())
        .compile(profile, inlet=layout)
        .pour(signal)
    ).single()
    freqs = np.asarray(out.coords[AxisName.FREQUENCY])
    power = np.asarray(out.values).mean(
        axis=(0, 2)
    )  # average over time and the feature
    peak_hz = float(freqs[power.argmax()])
    assert abs(peak_hz - tone_hz) < 0.2 * tone_hz  # within 20 % (log-spaced bins)
