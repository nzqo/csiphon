"""Cross-cutting invariants every step should respect.

These are the "no matter the step" expectations: shapes and dtypes are sensible,
axes are addressed by name (not position), and awkward inputs -- negative values,
zeros, and an empty recording -- are handled without surprises. They read as the
contract a new step must not break.
"""

from __future__ import annotations

import numpy as np

from csiphon import (
    Axis,
    AxisName,
    Layout,
    Pipeline,
    Representation,
    Signal,
    ValueKind,
    create_signal,
)
from csiphon.steps import (
    ComplexStftMagnitude,
    GainNormalize,
    LogScale,
    Magnitude,
    NanScrub,
    PerFrameMaxNormalize,
    Phase,
    Power,
    TemporalMeanSubtract,
    ToDecibels,
    WindowedVariance,
)


def _real_signal(
    axes: tuple[Axis, ...], values: np.ndarray, rate: float = 1000.0
) -> Signal:
    """A real-valued signal on an explicit layout (for feeding steps directly)."""

    layout = Layout(
        axes, Representation.CHANNEL_FREQUENCY_RESPONSE, ValueKind.MAGNITUDE
    )
    times = np.arange(values.shape[0]) / rate
    return create_signal(values, times, layout)


# --- empty (zero-length) recordings ---


def test_pointwise_step_on_an_empty_recording_stays_empty(profile) -> None:
    """A recording with no samples flows through a pointwise step to an empty output."""

    shape = (0, profile.n_rx_antennas, profile.n_subcarriers)
    empty = profile.raw_signal(np.zeros(shape, complex), np.zeros(0))
    out = Pipeline().then(Magnitude()).compile(profile).pour(empty).single()
    assert out.n_samples == 0
    assert out.values.shape == empty.values.shape


def test_windowed_step_on_an_empty_recording_yields_no_windows(profile) -> None:
    """A window step given no samples produces no windows rather than erroring."""

    shape = (0, profile.n_rx_antennas, profile.n_subcarriers)
    empty = profile.raw_signal(np.zeros(shape, complex), np.zeros(0))
    out = (
        Pipeline()
        .then(Magnitude())
        .then(WindowedVariance(win_size_s=0.05))
        .compile(profile)
        .pour(empty)
        .single()
    )
    assert out.n_samples == 0


def test_complex_stft_on_an_empty_recording_stays_empty(profile) -> None:
    """The STFT returns empty for an empty recording instead of indexing empty times.

    The early return fires before any scipy call, so this holds even without scipy.
    """

    shape = (0, profile.n_rx_antennas, profile.n_subcarriers)
    empty = profile.raw_signal(np.zeros(shape, complex), np.zeros(0))
    out = (
        Pipeline()
        .then(ComplexStftMagnitude(window_size=8, hop_size=4, freq_bins=3))
        .compile(profile)
        .pour(empty)
        .single()
    )
    assert out.n_samples == 0


# --- value handling: negatives and zeros ---


def test_magnitude_of_complex_is_real_and_matches_abs(profile, raw_signal) -> None:
    """Magnitude returns non-negative real floats equal to numpy's abs, same shape."""

    out = Pipeline().then(Magnitude()).compile(profile).pour(raw_signal).single()
    assert out.values.dtype == np.float64
    assert out.values.shape == raw_signal.values.shape
    assert np.all(out.values >= 0.0)
    assert np.allclose(out.values, np.abs(raw_signal.values))


def test_power_squares_real_input_including_negatives(profile) -> None:
    """Power is the squared value: it turns negatives non-negative, not clipped."""

    axes = (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 4))
    signal = _real_signal(axes, np.array([[-2.0, -1.0, 0.0, 3.0]]))
    out = Pipeline().then(Power()).compile(profile, inlet=signal.layout).pour(signal)
    values = out.single().values
    assert np.all(values >= 0.0)
    assert np.allclose(values, [[4.0, 1.0, 0.0, 9.0]])


def test_log_scale_maps_zero_to_a_finite_floor(profile, raw_signal) -> None:
    """LogScale adds a small epsilon, so exact zeros map to a finite value, not -inf."""

    zeros = create_signal(
        np.zeros_like(raw_signal.values), raw_signal.times, raw_signal.layout
    )
    out = (
        Pipeline()
        .then(Magnitude())
        .then(Power())
        .then(LogScale())
        .compile(profile)
        .pour(zeros)
        .single()
    )
    assert np.all(np.isfinite(out.values))
    assert np.allclose(out.values, np.log(1e-12))


# --- axis addressing by name, not position ---


def test_gain_normalize_finds_subcarrier_wherever_it_sits(profile) -> None:
    """Steps address axes by name: GainNormalize normalizes subcarriers even when
    that axis is not in its usual position."""

    # subcarrier deliberately placed before rx_antenna (the reverse of raw layout).
    axes = (
        Axis.dynamic(AxisName.TIME),
        Axis.sized(AxisName.SUBCARRIER, 4),
        Axis.sized(AxisName.RX_ANTENNA, 3),
    )
    rng = np.random.default_rng(0)
    signal = _real_signal(axes, np.abs(rng.standard_normal((20, 4, 3))) + 0.1)
    out = (
        Pipeline()
        .then(GainNormalize())
        .compile(profile, inlet=signal.layout)
        .pour(signal)
    )
    result = out.single()
    subcarrier_mean = result.values.mean(
        axis=result.layout.axis_position(AxisName.SUBCARRIER)
    )
    assert np.allclose(subcarrier_mean, 1.0, atol=1e-6)


# --- specific value / range checks ---


def test_phase_is_real_and_within_pi(profile, raw_signal) -> None:
    """Phase returns the angle: real values in (-pi, pi], matching numpy."""

    out = Pipeline().then(Phase()).compile(profile).pour(raw_signal).single()
    assert out.values.dtype == np.float64
    assert np.all(np.abs(out.values) <= np.pi)
    assert np.allclose(out.values, np.angle(raw_signal.values))


def test_to_decibels_is_ten_log10_of_the_ratio(profile) -> None:
    """ToDecibels is 10*log10(value/reference): 1x -> 0 dB, 10x -> 10, 100x -> 20."""

    axes = (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 3))
    signal = _real_signal(axes, np.array([[1.0, 10.0, 100.0]]))
    out = ToDecibels(reference=1.0)
    result = Pipeline().then(out).compile(profile, inlet=signal.layout).pour(signal)
    assert np.allclose(result.single().values, [[0.0, 10.0, 20.0]], atol=1e-6)


def test_temporal_mean_subtract_removes_the_mean_over_time(profile) -> None:
    """TemporalMeanSubtract centres each series in time, leaving a ~zero time-mean."""

    axes = (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 3))
    rng = np.random.default_rng(0)
    signal = _real_signal(axes, rng.standard_normal((50, 3)))
    result = (
        Pipeline()
        .then(TemporalMeanSubtract())
        .compile(profile, inlet=signal.layout)
        .pour(signal)
        .single()
    )
    assert np.allclose(result.values.mean(axis=0), 0.0, atol=1e-12)


def test_per_frame_max_normalize_scales_each_frame_to_one(profile) -> None:
    """PerFrameMaxNormalize divides each time frame by its own max, so the peak is 1."""

    axes = (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 3))
    signal = _real_signal(axes, np.array([[2.0, 4.0, 8.0]]))
    result = (
        Pipeline()
        .then(PerFrameMaxNormalize())
        .compile(profile, inlet=signal.layout)
        .pour(signal)
        .single()
    )
    assert np.allclose(result.values, [[0.25, 0.5, 1.0]])


def test_nan_scrub_replaces_non_finite_values(profile) -> None:
    """NanScrub replaces NaN with a finite value, so later steps see clean data."""

    axes = (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 3))
    signal = _real_signal(axes, np.array([[1.0, np.nan, 3.0]]))
    result = (
        Pipeline()
        .then(NanScrub())
        .compile(profile, inlet=signal.layout)
        .pour(signal)
        .single()
    )
    assert np.all(np.isfinite(result.values))
    assert np.allclose(result.values, [[1.0, 0.0, 3.0]])
