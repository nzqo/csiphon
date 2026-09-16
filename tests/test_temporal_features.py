"""Temporal-feature steps: batch correctness and batch/stream equivalence."""

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon import AcquisitionProfile, Pipeline, Signal, create_signal
from csiphon.steps import Magnitude, TimeDifference, WindowedSlope, WindowedVariance
from csiphon.steps.temporal_features import Mode


def _magnitude_signal(profile: AcquisitionProfile, n: int = 400) -> Signal:
    """A real magnitude signal to feed the real-valued temporal-feature steps."""

    compiled = Pipeline().then(Magnitude()).compile(profile)
    rng = np.random.default_rng(5)
    shape = (n, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    raw = profile.raw_signal(csi, np.arange(n) / 1000.0)
    return compiled.pour(raw).single()


def test_time_difference_conjugate_matches_reference(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """
    x[t] * conj(x[t-1]) with one fewer sample, matching a direct numpy computation.
    """

    compiled = Pipeline().then(TimeDifference(mode=Mode.CONJUGATE)).compile(profile)
    out = compiled.pour(raw_signal).single()
    expected = raw_signal.values[1:] * np.conj(raw_signal.values[:-1])
    assert out.values.shape[0] == raw_signal.n_samples - 1
    assert np.allclose(out.values, expected)
    assert np.allclose(out.times, raw_signal.times[1:])


@pytest.mark.parametrize("mode", [Mode.DIFFERENCE, Mode.CONJUGATE, Mode.PHASE_ONLY])
@pytest.mark.parametrize("chunk", [7, 64, 300])
def test_time_difference_stream_equals_batch(
    profile: AcquisitionProfile, raw_signal: Signal, mode: Mode, chunk: int
) -> None:
    """Carrying one sample makes streamed differences identical to batch."""

    compiled = Pipeline().then(TimeDifference(mode=mode)).compile(profile)
    batch = compiled.pour(raw_signal).single()
    streamed = stream_in_chunks(compiled, raw_signal, chunk)
    assert np.allclose(batch.values, streamed.values)
    assert np.allclose(batch.times, streamed.times)


def test_windowed_variance_matches_cumsum(profile: AcquisitionProfile) -> None:
    """Rolling variance equals the trailing-window variance computed directly."""

    signal = _magnitude_signal(profile)
    compiled = (
        Pipeline()
        .then(WindowedVariance(win_size_s=0.05))
        .compile(profile, inlet=signal.layout)
    )
    out = compiled.pour(signal).single()

    window = round(0.05 * 1000.0)
    values = signal.values
    expected = np.empty_like(values)
    for t in range(values.shape[0]):
        block = values[max(0, t - window + 1) : t + 1]
        expected[t] = block.var(axis=0)
    assert out.values.shape == values.shape
    assert np.allclose(out.values, expected)


@pytest.mark.parametrize("chunk", [11, 128])
def test_windowed_variance_stream_equals_batch(
    profile: AcquisitionProfile, chunk: int
) -> None:
    """The trailing-window state carries across chunks exactly."""

    signal = _magnitude_signal(profile)
    compiled = (
        Pipeline()
        .then(WindowedVariance(win_size_s=0.05))
        .compile(profile, inlet=signal.layout)
    )
    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, chunk)
    assert np.allclose(batch.values, streamed.values)


def test_windowed_slope_constant_is_zero_linear_is_constant(
    profile: AcquisitionProfile,
) -> None:
    """Slope of a flat signal is ~0; slope of a linear ramp is the ramp rate."""

    n = 200
    # The magnitude outlet keeps the singleton receiver/tx axes, so the ramp is
    # built on the full (time, receiver, tx_antenna, rx_antenna, subcarrier) shape.
    base = np.ones((1, 1, 1, profile.n_rx_antennas, profile.n_subcarriers))
    ramp = np.linspace(0.0, 4.0, n)[:, None, None, None, None] * base
    times = np.arange(n) / 1000.0
    layout = Pipeline().then(Magnitude()).compile(profile).outlet_layout
    signal = create_signal(ramp, times, layout)

    compiled = (
        Pipeline().then(WindowedSlope(scale_by_dt=False)).compile(profile, inlet=layout)
    )
    out = compiled.pour(signal).single()
    per_sample_rate = 4.0 / (n - 1)
    # Once the window is full, every slope equals the per-sample ramp rate.
    assert np.allclose(out.values[20:], per_sample_rate)


@pytest.mark.parametrize("chunk", [9, 100])
def test_windowed_slope_stream_equals_batch(
    profile: AcquisitionProfile, chunk: int
) -> None:
    """The trailing window carries across chunks exactly."""

    signal = _magnitude_signal(profile)
    compiled = Pipeline().then(WindowedSlope()).compile(profile, inlet=signal.layout)
    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, chunk)
    assert np.allclose(batch.values, streamed.values)
