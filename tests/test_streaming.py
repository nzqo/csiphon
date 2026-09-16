"""Batch/stream equivalence and streaming-refusal tests."""
# Tests repeat the batch/stream compile-and-compare pattern by design.
# pylint: disable=duplicate-code

import numpy as np
import pytest
from conftest import fold_channels_into_feature, stream_in_chunks

from csiphon import AxisName, Pipeline
from csiphon.core import StreamingError
from csiphon.steps import (
    DelayAutocorrelation,
    DelayTaps,
    FixedSizeWindowSum,
    FoldAxes,
    GainNormalize,
    Magnitude,
    PolarLinear,
    Resample,
    RobustPca,
    RunningMeanSubtract,
    SavitzkyGolay,
    TemporalMeanSubtract,
    WindowedFFTPower,
)
from csiphon.steps._support.windowing import sliding_window_indices
from csiphon.steps.filtering import Band, ButterworthFilter


@pytest.mark.parametrize("chunk", [1, 7, 50, 333])
def test_pointwise_stream_equals_batch(profile, raw_signal, chunk) -> None:
    """Pointwise chain: streaming any chunk size equals batch exactly."""

    pipeline = (
        Pipeline()
        .then(Magnitude())
        .then(GainNormalize())
        .then(DelayAutocorrelation())
        .then(DelayTaps())
        .then(FoldAxes(axes=(AxisName.RX_ANTENNA, AxisName.DELAY)))
    )
    compiled = pipeline.compile(profile)
    batch = compiled.pour(raw_signal).single()
    streamed = stream_in_chunks(compiled, raw_signal, chunk)
    assert np.allclose(batch.values, streamed.values)
    assert np.allclose(batch.times, streamed.times)


@pytest.mark.parametrize("chunk", [13, 64, 200])
def test_windowed_fft_stream_equals_batch(profile, raw_signal, chunk) -> None:
    """Windowed FFT power: buffered streaming equals batch exactly."""

    pipeline = (
        Pipeline()
        .then(Magnitude())
        .then(GainNormalize())
        .then(FoldAxes(axes=(AxisName.RX_ANTENNA, AxisName.SUBCARRIER)))
        .then(WindowedFFTPower(window_s=0.128, hop_s=0.05, band_hz=60.0))
    )
    compiled = pipeline.compile(profile)
    batch = compiled.pour(raw_signal).single()
    streamed = stream_in_chunks(compiled, raw_signal, chunk)
    assert batch.values.shape == streamed.values.shape
    assert np.allclose(batch.values, streamed.values)
    assert np.allclose(batch.times, streamed.times)


@pytest.mark.parametrize("chunk", [9, 100])
def test_time_window_stream_equals_batch(profile, raw_signal, chunk) -> None:
    """Sliding-window sum: buffered streaming equals batch exactly."""

    pipeline = (
        Pipeline()
        .then(Magnitude())
        .then(GainNormalize())
        .then(FoldAxes(axes=(AxisName.RX_ANTENNA, AxisName.SUBCARRIER)))
        .then(FixedSizeWindowSum(window_s=0.128, hop_s=0.05))
    )
    compiled = pipeline.compile(profile)
    batch = compiled.pour(raw_signal).single()
    streamed = stream_in_chunks(compiled, raw_signal, chunk)
    assert np.allclose(batch.values, streamed.values)


@pytest.mark.parametrize("chunk", [1, 17, 250])
def test_running_mean_stream_equals_batch(profile, raw_signal, chunk) -> None:
    """Causal running-mean subtraction is identical batch vs streaming."""

    pipeline = Pipeline().then(Magnitude()).then(RunningMeanSubtract(alpha=0.02))
    compiled = pipeline.compile(profile)
    batch = compiled.pour(raw_signal).single()
    streamed = stream_in_chunks(compiled, raw_signal, chunk)
    assert np.allclose(batch.values, streamed.values)


def _folded(*steps) -> Pipeline:
    """Magnitude, fold every channel axis into one feature axis, then the given steps.

    A step like RobustPca wants a plain `(time, feature)` matrix, so fold all the
    structural axes (receiver / tx / rx) together with subcarrier -- not just rx and
    subcarrier -- otherwise the singleton receiver / tx axes would be left dangling.
    """

    pipeline = Pipeline().then(Magnitude()).then(fold_channels_into_feature())
    for step in steps:
        pipeline = pipeline.then(step)
    return pipeline


# Every step that declares itself batch-only (Streaming.UNAVAILABLE) must refuse to
# stream, naming itself in the error. The name is the spec name of that step.
_BATCH_ONLY = [
    (
        "temporal-mean-subtract",
        Pipeline().then(Magnitude()).then(TemporalMeanSubtract()),
    ),
    # PolarLinear unwraps phase over the whole recording, so it cannot stream.
    ("resample", Pipeline().then(Resample(rate_hz=500.0, fill=PolarLinear()))),
    ("savitzky-golay", Pipeline().then(Magnitude()).then(SavitzkyGolay())),
]


@pytest.mark.parametrize("name, pipeline", _BATCH_ONLY, ids=[n for n, _ in _BATCH_ONLY])
def test_batch_only_step_refuses_to_stream(profile, name, pipeline) -> None:
    """A step that needs the whole recording refuses streaming, naming itself."""

    compiled = pipeline.compile(profile)
    with pytest.raises(StreamingError, match=name):
        compiled.stream()


def test_robust_pca_refuses_to_stream(profile) -> None:
    """RobustPca fits on the whole recording, so streaming refuses and names it."""

    compiled = _folded(RobustPca(num_components=2)).compile(profile)
    with pytest.raises(StreamingError, match="robust-pca"):
        compiled.stream()


@pytest.mark.parametrize("hop", [1.5, 2.5, 1.25, 7 / 3])
def test_batch_window_grid_matches_the_streaming_grid(hop: float) -> None:
    """Batch window starts equal the ones the streaming operator emits.

    Regression: for a fractional hop the batch count used floor(last/hop), dropping the
    final window that the stream (a window per k with round(k*hop) <= last_start) emits.
    So a windowed step declaring BATCH_EQUIVALENT actually diverged.
    """

    for n in range(5, 30):
        starts, _ = sliding_window_indices(n, 3, hop)
        last_start = n - 3
        expected = sorted(
            {round(k * hop) for k in range(n) if round(k * hop) <= last_start}
        )
        assert list(starts) == expected, (hop, n)


def test_butterworth_streaming_is_chunk_invariant(profile, raw_signal) -> None:
    """A causal Butterworth filter streams the same output however the stream is cut.

    It is BATCH_DIVERGENT (its start-up transient differs from batch's zero-phase pass),
    so instead of a batch comparison we assert chunk-invariance: feeding one sample at a
    time must match feeding big chunks. This exercises the filter state carried across
    chunk boundaries -- the block-divergent path that no other test drives with data.
    """

    pytest.importorskip("scipy")
    compiled = (
        Pipeline()
        .then(Magnitude())
        .then(ButterworthFilter(cutoff_hz=100.0, btype=Band.LOW))
        .compile(profile)
    )
    fine = stream_in_chunks(compiled, raw_signal, 1)
    coarse = stream_in_chunks(compiled, raw_signal, 333)

    assert np.array_equal(fine.times, coarse.times)
    assert np.allclose(fine.values, coarse.values)
