"""Resample: put a stream onto a uniform grid, filled by the chosen method.

The fill choices are checked on a small irregular recording whose every value equals
its own timestamp, so a held / nearest / interpolated choice is visible directly in
the output values. Complex-specific behaviour (PolarLinear) and streaming equivalence
get their own cases.
"""

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon.core import CompileError, StreamingError
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.signal import Signal
from csiphon.pipeline import Pipeline
from csiphon.steps import (
    CubicSpline,
    FillMethod,
    Hold,
    Linear,
    Magnitude,
    Nearest,
    PolarLinear,
    Resample,
)

# Times [0,1,2,5,6,10] have gaps, so the fills disagree in a legible way; each value
# equals its own timestamp, so the output values read back as "which source time won".
_IRREGULAR_TIMES = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 10.0])


def _timed_signal(
    times: np.ndarray, n_subcarriers: int = 3
) -> tuple[AcquisitionProfile, Signal]:
    """A single-antenna signal whose every value equals its own timestamp."""

    profile = AcquisitionProfile(subcarrier_indices=tuple(range(n_subcarriers)))
    values = np.tile(np.asarray(times, float)[:, None], (1, n_subcarriers))
    return profile, profile.raw_signal(values.astype(np.complex128), times)


def _values_at_grid(signal: Signal) -> np.ndarray:
    """The per-grid-point value (all channels equal), flattened to one column."""

    return signal.values.reshape(signal.n_samples, -1)[:, 0]


def _channels_at_grid(signal: Signal) -> np.ndarray:
    """Every grid point's value per channel, shaped (grid, channels)."""

    return signal.values.reshape(signal.n_samples, -1)


def _ramped_signal(times: np.ndarray) -> tuple[AcquisitionProfile, Signal]:
    """A 4-subcarrier signal where channel c carries (t + 100c) + j(2t + 10c).

    Distinct per channel and distinct real-vs-imaginary, so a fill that scrambled the
    channels or dropped the imaginary part is caught -- an all-identical signal cannot
    see either mistake.
    """

    profile = AcquisitionProfile(subcarrier_indices=tuple(range(4)))
    time = np.asarray(times, float)[:, None]
    channel = np.arange(4)[None, :]
    values = (time + 100.0 * channel) + 1j * (2.0 * time + 10.0 * channel)
    return profile, profile.raw_signal(values, np.asarray(times, float))


# --- the grid ----------------------------------------------------------------


def test_output_lands_on_a_precise_grid(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """500 Hz on a 1 kHz stream gives output timestamps exactly 2 ms apart."""

    out = (
        Pipeline()
        .then(Resample(rate_hz=500.0, fill=Hold()))
        .compile(profile)
        .pour(raw_signal)
    ).single()

    assert out.times[0] == raw_signal.times[0]
    assert np.allclose(np.diff(out.times), 1 / 500.0)


def test_rate_hz_and_step_s_are_equivalent(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """`rate_hz=500` and `step_s=0.002` describe the same grid."""

    by_rate = Resample(rate_hz=500.0, fill=Hold())
    by_step = Resample(step_s=0.002, fill=Hold())
    out_rate = Pipeline().then(by_rate).compile(profile).pour(raw_signal).single()
    out_step = Pipeline().then(by_step).compile(profile).pour(raw_signal).single()

    assert np.array_equal(out_rate.times, out_step.times)
    assert np.array_equal(out_rate.values, out_step.values)


# --- the fills ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("fill", "expected"),
    [
        # grid                          [0, 2,  4,  6,  8, 10]
        (Hold(), [0.0, 2.0, 2.0, 6.0, 6.0, 10.0]),  # last value at or before
        (Nearest(), [0.0, 2.0, 5.0, 6.0, 6.0, 10.0]),  # closest (ties -> earlier)
        (Linear(), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]),  # value==time, so line -> grid
    ],
)
def test_fill_chooses_the_right_source(fill: FillMethod, expected: list[float]) -> None:
    """Each fill resolves the grid points the way its contract says it should."""

    profile, signal = _timed_signal(_IRREGULAR_TIMES)
    out = (
        Pipeline().then(Resample(step_s=2.0, fill=fill)).compile(profile).pour(signal)
    ).single()

    assert np.array_equal(out.times, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert np.allclose(_values_at_grid(out).real, expected)


def test_polar_linear_keeps_magnitude_where_cartesian_linear_shrinks_it() -> None:
    """Across a phase rotation, PolarLinear holds |x|; cartesian Linear shrinks it.

    Two unit samples a quarter-turn apart (1+0j at t=0, 0+1j at t=1). At the midpoint
    the true magnitude is still 1; interpolating real/imag gives 0.5+0.5j (|.|~0.707),
    while interpolating magnitude and phase gives a unit vector at 45 degrees.
    """

    profile = AcquisitionProfile(subcarrier_indices=(0,))
    signal = profile.raw_signal(np.array([[1 + 0j], [0 + 1j]]), np.array([0.0, 1.0]))

    def midpoint(fill: FillMethod) -> complex:
        """The value `fill` produces halfway between the two samples."""

        out = (
            Pipeline()
            .then(Resample(step_s=0.5, fill=fill))
            .compile(profile)
            .pour(signal)
        ).single()
        return complex(_values_at_grid(out)[1])  # grid is [0.0, 0.5, 1.0]

    assert np.isclose(abs(midpoint(PolarLinear())), 1.0)
    assert np.isclose(abs(midpoint(Linear())), np.sqrt(0.5))


def test_cubic_spline_follows_a_straight_line() -> None:
    """A cubic spline through collinear points reproduces the line at the grid."""

    pytest.importorskip("scipy")
    times = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 10.0])
    profile, signal = _timed_signal(times)  # value == time, a straight line
    out = (
        Pipeline()
        .then(Resample(step_s=2.0, fill=CubicSpline()))
        .compile(profile)
        .pour(signal)
    ).single()

    assert np.allclose(_values_at_grid(out).real, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


def test_cubic_spline_falls_back_to_linear_below_four_points() -> None:
    """Fewer than four samples cannot fix a spline, so it falls back to linear."""

    pytest.importorskip("scipy")
    profile, signal = _timed_signal(np.array([0.0, 1.0, 4.0]))  # only three points
    out = (
        Pipeline()
        .then(Resample(step_s=1.0, fill=CubicSpline()))
        .compile(profile)
        .pour(signal)
    ).single()

    # value == time, so a linear fill returns the grid times exactly.
    assert np.allclose(_values_at_grid(out).real, out.times)


# --- streaming ---------------------------------------------------------------

_STREAMING_FILLS = [Hold(), Nearest(), Linear()]
_STREAMING_IDS = ["hold", "nearest", "linear"]


@pytest.mark.parametrize("fill", _STREAMING_FILLS, ids=_STREAMING_IDS)
@pytest.mark.parametrize("chunk", [1, 13, 128, 800])
def test_causal_fills_stream_like_batch(
    profile: AcquisitionProfile, raw_signal: Signal, fill: FillMethod, chunk: int
) -> None:
    """Hold / Nearest / Linear only bracket a grid point, so streaming is exact."""

    compiled = Pipeline().then(Resample(step_s=0.017, fill=fill)).compile(profile)
    batch = compiled.pour(raw_signal).single()
    streamed = stream_in_chunks(compiled, raw_signal, chunk)

    assert np.array_equal(batch.times, streamed.times)
    assert np.allclose(batch.values, streamed.values)


@pytest.mark.parametrize("fill", _STREAMING_FILLS, ids=_STREAMING_IDS)
@pytest.mark.parametrize("chunk", [1, 2, 3])
def test_causal_fills_stream_like_batch_across_gaps(
    fill: FillMethod, chunk: int
) -> None:
    """The one-sample carry reproduces batch even when a grid gap straddles a chunk."""

    times = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 10.0, 11.0, 12.0, 15.0, 20.0])
    profile, signal = _timed_signal(times)
    compiled = Pipeline().then(Resample(step_s=2.0, fill=fill)).compile(profile)
    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, chunk)

    assert np.array_equal(batch.times, streamed.times)
    assert np.allclose(batch.values, streamed.values)


@pytest.mark.parametrize("fill", [PolarLinear(), CubicSpline()])
def test_whole_recording_fills_refuse_to_stream(
    profile: AcquisitionProfile, fill: FillMethod
) -> None:
    """PolarLinear and CubicSpline read the whole recording, so they cannot stream."""

    compiled = Pipeline().then(Resample(step_s=0.01, fill=fill)).compile(profile)
    with pytest.raises(StreamingError, match="resample"):
        compiled.stream()


# --- validation --------------------------------------------------------------


def test_rejects_neither_or_both_grid_parameters(profile: AcquisitionProfile) -> None:
    """Exactly one of `rate_hz` / `step_s` must pin the grid."""

    with pytest.raises(CompileError, match="exactly one"):
        Pipeline().then(Resample(fill=Hold())).compile(profile)
    with pytest.raises(CompileError, match="exactly one"):
        Pipeline().then(Resample(rate_hz=500.0, step_s=0.002, fill=Hold())).compile(
            profile
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"rate_hz": 0.0}, "rate_hz"), ({"step_s": -1.0}, "step_s")],
)
def test_rejects_nonpositive_grid(
    profile: AcquisitionProfile, kwargs: dict[str, float], message: str
) -> None:
    """A zero or negative rate / step is caught at compile."""

    with pytest.raises(CompileError, match=message):
        Pipeline().then(Resample(fill=Hold(), **kwargs)).compile(profile)


# --- per-channel + complex correctness (an all-identical signal cannot see these) ---


@pytest.mark.parametrize("fill", [Linear(), CubicSpline()], ids=["linear", "cubic"])
def test_fills_interpolate_each_channel_and_part_independently(
    fill: FillMethod,
) -> None:
    """Each channel's real and imaginary ramps are interpolated on their own.

    Every subcarrier carries a different complex ramp; a linear fill (and a cubic
    spline through collinear points) must reproduce each one exactly. This catches a
    channel scramble or a dropped/copied imaginary part -- both invisible when every
    channel holds the same value.
    """

    if isinstance(fill, CubicSpline):
        pytest.importorskip("scipy")
    profile, signal = _ramped_signal(
        _IRREGULAR_TIMES
    )  # six points, enough for a spline
    out = (
        Pipeline().then(Resample(step_s=2.0, fill=fill)).compile(profile).pour(signal)
    ).single()

    grid = out.times
    channels = _channels_at_grid(out)
    for channel in range(4):
        assert np.allclose(channels[:, channel].real, grid + 100.0 * channel)
        assert np.allclose(channels[:, channel].imag, 2.0 * grid + 10.0 * channel)


def test_polar_linear_preserves_each_channels_magnitude() -> None:
    """PolarLinear keeps every channel's own magnitude across a phase rotation."""

    profile = AcquisitionProfile(subcarrier_indices=tuple(range(3)))
    magnitude = 1.0 + np.arange(3)  # a distinct magnitude per channel
    start = magnitude * np.exp(1j * 0.0)
    end = magnitude * np.exp(1j * np.pi / 2)  # a quarter turn later
    signal = profile.raw_signal(np.stack([start, end]), np.array([0.0, 1.0]))

    out = (
        Pipeline()
        .then(Resample(step_s=0.5, fill=PolarLinear()))
        .compile(profile)
        .pour(signal)
    ).single()

    midpoint = _channels_at_grid(out)[1]  # grid is [0.0, 0.5, 1.0]
    assert np.allclose(np.abs(midpoint), magnitude)


# --- edge cases -------------------------------------------------------------


def test_grid_includes_the_final_point_on_an_exact_multiple() -> None:
    """A span that is an exact multiple of the step still emits the last grid point.

    Regression: 1.0 / 0.1 evaluates to just under 10 in float, which used to drop the
    grid point at 1.0 s.
    """

    profile, signal = _timed_signal(np.linspace(0.0, 1.0, 11))
    out = (
        Pipeline().then(Resample(step_s=0.1, fill=Hold())).compile(profile).pour(signal)
    ).single()

    assert out.n_samples == 11
    assert np.isclose(out.times[-1], 1.0)


def test_grid_coarser_than_the_recording_keeps_only_the_first_point() -> None:
    """A step wider than the whole span leaves a single grid point at t0."""

    profile, signal = _timed_signal(np.array([0.0, 0.3, 0.7]))
    out = (
        Pipeline().then(Resample(step_s=5.0, fill=Hold())).compile(profile).pour(signal)
    ).single()

    assert out.n_samples == 1
    assert out.times[0] == 0.0


def test_single_sample_recording_resamples_to_that_sample() -> None:
    """One input sample gives a one-point grid holding that value."""

    profile = AcquisitionProfile(subcarrier_indices=(0,))
    signal = profile.raw_signal(np.array([[3.0 + 1j]]), np.array([0.5]))
    out = (
        Pipeline().then(Resample(step_s=0.1, fill=Hold())).compile(profile).pour(signal)
    ).single()

    assert out.n_samples == 1
    assert out.times[0] == 0.5
    assert _values_at_grid(out)[0] == 3.0 + 1j


def test_empty_recording_stays_empty() -> None:
    """Resampling nothing yields nothing, rather than crashing on times[-1]."""

    profile = AcquisitionProfile(subcarrier_indices=(0,))
    empty = profile.raw_signal(np.zeros((0, 1), dtype=np.complex128), np.zeros(0))
    out = (
        Pipeline()
        .then(Resample(rate_hz=100.0, fill=Hold()))
        .compile(profile)
        .pour(empty)
    ).single()

    assert out.n_samples == 0


def test_polar_linear_rejects_a_real_signal() -> None:
    """PolarLinear is complex-only; a real input is rejected at compile-time."""

    profile, _ = _timed_signal(_IRREGULAR_TIMES)
    with pytest.raises(CompileError, match="complex"):
        (
            Pipeline()
            .then(Magnitude())  # turns the signal real (magnitude)
            .then(Resample(step_s=2.0, fill=PolarLinear()))
            .compile(profile)
        )
