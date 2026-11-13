"""Window steps place windows by timestamp, not by sample count, on request."""

import json

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon import AcquisitionProfile, Pipeline, Signal, Siphon, describe
from csiphon.core.errors import CompileError, StreamingError
from csiphon.steps import (
    FromNominalRate,
    FromTimestamps,
    Magnitude,
    WindowDeterminationMode,
    WindowedSlope,
    WindowedVariance,
    WindowSum,
)


def _profile(rate_hz: float | None) -> AcquisitionProfile:
    return AcquisitionProfile(subcarrier_indices=(0,), sampling_rate_hz=rate_hz)


def _signal(profile: AcquisitionProfile, values, times) -> Signal:
    """One-antenna, one-subcarrier CSI whose magnitude is `values`."""

    csi = np.asarray(values, dtype=complex)[:, None, None]
    return profile.raw_signal(csi, np.asarray(times, dtype=float))


def _compiled(profile: AcquisitionProfile, mode, window_s=0.5, hop_s=0.5) -> Siphon:
    step = WindowSum(window_s=window_s, hop_s=hop_s, window_mode=mode)
    return Pipeline().then(Magnitude()).then(step).compile(profile)


_BY_TIMESTAMP = WindowDeterminationMode.FROM_TIMESTAMPS
_BY_RATE = WindowDeterminationMode.FROM_NOMINAL_RATE


def test_each_window_sums_the_samples_inside_its_time_span() -> None:
    """Window k covers [k * hop, k * hop + window) from the first timestamp, however
    the samples are spaced. Here [0, 0.5) holds 1+2+3 and [0.5, 1.0) holds 4+5. The
    sample at 1.1 only proves the second window is complete."""

    profile = _profile(100.0)
    signal = _signal(profile, [1, 2, 3, 4, 5, 6], [0.0, 0.1, 0.2, 0.5, 0.6, 1.1])

    out = _compiled(profile, _BY_TIMESTAMP).pour(signal).single()

    assert out.values.ravel() == pytest.approx([6.0, 9.0])
    # Windows are on a regular grid, so their centers are too.
    assert out.times == pytest.approx([0.25, 0.75])


def test_a_gap_gives_an_empty_window_as_nan_not_a_shorter_output() -> None:
    """The output length follows the recording's duration, so a lost stretch of
    samples shows up as a NaN window instead of removing it from the output."""

    profile = _profile(100.0)
    signal = _signal(profile, [1, 2, 3, 4, 5, 6], [0.0, 0.1, 0.2, 1.0, 1.1, 1.6])

    out = _compiled(profile, _BY_TIMESTAMP).pour(signal).single()

    values = out.values.ravel()
    assert values.shape == (3,)
    assert values[0] == 6.0
    assert np.isnan(values[1])
    assert values[2] == 9.0


def test_regular_data_matches_the_nominal_rate_mode() -> None:
    """On a perfectly regular capture both modes agree. Timestamp mode waits for a
    sample past the end of a window, so it may emit one window fewer at the end."""

    profile = _profile(100.0)
    rng = np.random.default_rng(0)
    signal = _signal(profile, rng.random(100), np.arange(100) / 100.0)

    by_timestamp = _compiled(profile, _BY_TIMESTAMP, 0.1, 0.1).pour(signal).single()
    by_rate = _compiled(profile, _BY_RATE, 0.1, 0.1).pour(signal).single()

    count = by_timestamp.n_samples
    assert count >= by_rate.n_samples - 1
    np.testing.assert_allclose(
        by_timestamp.values[:count], by_rate.values[:count], atol=1e-12
    )


@pytest.mark.parametrize("chunk", [1, 7, 50])
def test_streaming_matches_batch_on_irregular_timestamps(chunk: int) -> None:
    """The streaming operator is the batch operator fed in pieces, so chunk sizes
    cannot change the windows. The recording has random loss, so some are empty."""

    profile = _profile(100.0)
    rng = np.random.default_rng(3)
    kept = np.sort(rng.choice(1000, 500, replace=False))
    signal = _signal(profile, rng.random(500), kept / 100.0)
    compiled = _compiled(profile, _BY_TIMESTAMP, 0.4, 0.25)

    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, chunk)

    assert streamed.times == pytest.approx(batch.times)
    np.testing.assert_allclose(streamed.values, batch.values, atol=1e-10)


def test_the_output_rate_is_exactly_one_over_the_hop() -> None:
    """Output samples are one hop apart, so the nominal rate of the output is
    1 / hop_s, whatever the input rate was."""

    compiled = _compiled(_profile(100.0), _BY_TIMESTAMP, 0.4, 0.25)

    assert compiled.outlet_layout.nominal_rate_hz == pytest.approx(4.0)


def test_no_sampling_rate_is_needed_when_streaming_by_timestamp() -> None:
    """Window sizes come from the timestamps, so a profile without a rate streams.
    The nominal-rate mode needs the rate up front and rejects this."""

    profile = _profile(None)
    times = np.arange(40) / 10.0
    signal = _signal(profile, np.ones(40), times)

    streamed = stream_in_chunks(_compiled(profile, _BY_TIMESTAMP), signal, 6)

    assert streamed.values.ravel() == pytest.approx([5.0] * 7)


def _variance_pipeline(profile: AcquisitionProfile, mode, win_size_s: float) -> Siphon:
    step = WindowedVariance(win_size_s=win_size_s, window_mode=mode)
    return Pipeline().then(Magnitude()).then(step).compile(profile)


def test_variance_window_covers_the_last_seconds_not_the_last_samples() -> None:
    """With timestamps, the trailing 0.3 s window at t=1.0 holds only the samples
    in (0.7, 1.0]. Here that is the values 2 and 6, whose variance is 4. The two
    early samples are older than the window and must not count."""

    profile = _profile(100.0)
    signal = _signal(profile, [10, 20, 2, 6], [0.0, 0.1, 0.9, 1.0])

    out = _variance_pipeline(profile, _BY_TIMESTAMP, 0.3).pour(signal).single()

    assert out.values.ravel()[-1] == pytest.approx(4.0)
    assert out.n_samples == 4


@pytest.mark.parametrize("chunk", [1, 5, 40])
def test_variance_by_timestamp_streams_like_batch(chunk: int) -> None:
    """Streaming carries the window by time, so chunking cannot change the result."""

    profile = _profile(100.0)
    rng = np.random.default_rng(5)
    kept = np.sort(rng.choice(600, 300, replace=False))
    signal = _signal(profile, rng.random(300), kept / 100.0)
    compiled = _variance_pipeline(profile, _BY_TIMESTAMP, 0.4)

    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, chunk)

    np.testing.assert_allclose(streamed.values, batch.values, atol=1e-9)


def _slope_pipeline(profile: AcquisitionProfile, window_s: float) -> Siphon:
    step = WindowedSlope(window=FromTimestamps(window_s=window_s))
    return Pipeline().then(Magnitude()).then(step).compile(profile)


def test_slope_fits_against_the_real_timestamps() -> None:
    """Values that grow 5 per second on unevenly spaced times give a slope of 5.
    A fit that assumed even spacing would not."""

    profile = _profile(100.0)
    times = np.array([0.0, 0.05, 0.2, 0.21, 0.4, 0.9])
    signal = _signal(profile, 5.0 * times, times)

    out = _slope_pipeline(profile, 10.0).pour(signal).single()

    assert out.values.ravel()[1:] == pytest.approx([5.0] * 5)
    assert out.values.ravel()[0] == 0.0


def test_slope_window_covers_the_last_seconds() -> None:
    """The window at t=1.0 holds only (0.7, 1.0], so the old samples with a very
    different trend do not count. Here the kept values rise 10 per second."""

    profile = _profile(100.0)
    signal = _signal(profile, [100, 0, 1, 2], [0.0, 0.8, 0.9, 1.0])

    out = _slope_pipeline(profile, 0.3).pour(signal).single()

    assert out.values.ravel()[-1] == pytest.approx(10.0)


@pytest.mark.parametrize("chunk", [1, 5, 40])
def test_slope_by_timestamp_streams_like_batch(chunk: int) -> None:
    """Streaming carries the window by time, so chunking cannot change the slopes."""

    profile = _profile(100.0)
    rng = np.random.default_rng(7)
    kept = np.sort(rng.choice(600, 300, replace=False))
    signal = _signal(profile, rng.random(300), kept / 100.0)
    compiled = _slope_pipeline(profile, 0.4)

    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, chunk)

    np.testing.assert_allclose(streamed.values, batch.values, atol=1e-9)


def test_slope_variants_only_accept_their_own_parameters() -> None:
    """A timestamp window has no per-sample option and a rate window has no
    seconds, so a mixed-up combination fails when it is written down."""

    with pytest.raises(TypeError):
        FromTimestamps(scale_by_dt=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        FromNominalRate(window_s=0.1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("window", "message"),
    [
        (FromTimestamps(window_s=0.0), "window_s"),
        (FromNominalRate(window_size=1), "window_size"),
    ],
)
def test_slope_rejects_an_unusable_window(window, message: str) -> None:
    """A window that cannot hold two samples is rejected when compiling."""

    with pytest.raises(CompileError, match=message):
        Pipeline().then(Magnitude()).then(WindowedSlope(window=window)).compile(
            _profile(100.0)
        )


def test_samples_on_a_window_edge_are_not_moved_by_rounding() -> None:
    """Samples every 0.1 s with a window of 3 * 0.1 s hold exactly 3 samples each,
    although 0.1 * 3 and the sums of the timestamps differ in the last bits. Guards
    against a rounding error moving an edge sample into the neighboring window."""

    profile = _profile(10.0)
    window_s = 3 * 0.1
    signal = _signal(profile, np.ones(400), np.arange(400) * 0.1)
    compiled = _compiled(profile, _BY_TIMESTAMP, window_s, window_s)

    out = compiled.pour(signal).single()

    assert out.n_samples > 100
    assert set(out.values.ravel()) == {3.0}


def test_overlapping_windows_share_samples() -> None:
    """With a window of 0.4 s and a hop of 0.2 s, every sample lands in two windows.
    Here the windows [0, 0.4), [0.2, 0.6) and [0.4, 0.8) hold 4 of the 8 samples
    spaced 0.1 s apart, and sample 0.8 completes the last one."""

    profile = _profile(10.0)
    signal = _signal(profile, np.arange(1, 10), np.arange(9) * 0.1)

    out = _compiled(profile, _BY_TIMESTAMP, 0.4, 0.2).pour(signal).single()

    # Window 0 holds 1+2+3+4, window 1 holds 3+4+5+6, window 2 holds 5+6+7+8.
    assert out.values.ravel() == pytest.approx([10.0, 18.0, 26.0])


@pytest.mark.parametrize(
    "step",
    [
        WindowSum(window_s=0.1, hop_s=0.1),
        WindowedVariance(win_size_s=0.1),
        WindowedSlope(),
    ],
    ids=["window-sum", "windowed-variance", "windowed-slope"],
)
def test_nominal_rate_windows_need_the_rate_when_streaming(step) -> None:
    """A window of N samples needs the sampling rate, and a stream cannot estimate
    it from timestamps it has not seen yet. Compiling is fine, streaming says why."""

    compiled = Pipeline().then(Magnitude()).then(step).compile(_profile(None))

    with pytest.raises(StreamingError, match="sampling rate"):
        compiled.stream()


@pytest.mark.parametrize(
    "step",
    [
        WindowSum(window_s=0.1, hop_s=0.1, window_mode=_BY_TIMESTAMP),
        WindowedVariance(win_size_s=0.1, window_mode=_BY_TIMESTAMP),
        WindowedSlope(window=FromTimestamps(window_s=0.1)),
        WindowedSlope(window=FromNominalRate(scale_by_dt=False)),
    ],
    ids=["window-sum", "windowed-variance", "slope-timestamps", "slope-per-sample"],
)
def test_windows_that_need_no_rate_stream_without_one(step) -> None:
    """By timestamp, or for a slope per sample, no sampling rate is needed."""

    compiled = Pipeline().then(Magnitude()).then(step).compile(_profile(None))

    assert compiled.stream() is not None


def test_a_window_choice_is_listed_as_data_in_the_description() -> None:
    """describe() shows the window choice with its name and parameters, so the
    saved JSON says which window a pipeline used."""

    step = WindowedSlope(window=FromTimestamps(window_s=0.05))

    parameters = json.loads(describe(step).to_json())["params"]
    window = next(p for p in parameters if p["name"] == "window")

    assert window["value"] == {"kind": "FromTimestamps", "window_s": 0.05}
