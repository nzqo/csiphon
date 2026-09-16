"""MeanOverAxes: average named axes away, keeping the rest and the value kind."""
# A test requesting a fixture by its name (raw_signal(profile)) is the pytest idiom,
# which pylint reads as shadowing.
# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from conftest import stream_in_chunks

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
from csiphon.steps import DyadicFrequencyBands, FixedSizeWindowSum, Magnitude
from csiphon.steps.pooling import MeanOverAxes


@pytest.fixture
def profile() -> AcquisitionProfile:
    """A small 3-antenna / 8-subcarrier profile at 1 kHz."""

    return AcquisitionProfile(
        n_rx_antennas=3, subcarrier_indices=tuple(range(8)), sampling_rate_hz=1000.0
    )


@pytest.fixture
def raw_signal(profile: AcquisitionProfile) -> Signal:
    """A short random complex recording."""

    rng = np.random.default_rng(0)
    shape = (40, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return profile.raw_signal(csi, np.arange(shape[0]) / 1000.0)


def test_mean_over_subcarrier_drops_the_axis(profile, raw_signal) -> None:
    """Averaging over subcarrier removes it and equals a manual mean."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .then(MeanOverAxes(axes=(AxisName.SUBCARRIER,)))
        .compile(profile)
    )
    out = siphon.pour(raw_signal).single()

    assert out.layout.axis_names == (
        AxisName.TIME,
        AxisName.RECEIVER,
        AxisName.TX_ANTENNA,
        AxisName.RX_ANTENNA,
    )
    assert out.layout.values.value == "magnitude"  # value kind unchanged
    subcarrier = raw_signal.layout.axis_position(AxisName.SUBCARRIER)
    expected = np.abs(raw_signal.values).mean(axis=subcarrier)
    assert np.allclose(out.values, expected)


def test_mean_over_several_axes_at_once(profile, raw_signal) -> None:
    """Averaging over two axes removes both."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .then(MeanOverAxes(axes=(AxisName.RX_ANTENNA, AxisName.SUBCARRIER)))
        .compile(profile)
    )
    out = siphon.pour(raw_signal).single()

    assert out.layout.axis_names == (
        AxisName.TIME,
        AxisName.RECEIVER,
        AxisName.TX_ANTENNA,
    )
    rx_antenna = raw_signal.layout.axis_position(AxisName.RX_ANTENNA)
    subcarrier = raw_signal.layout.axis_position(AxisName.SUBCARRIER)
    expected = np.abs(raw_signal.values).mean(axis=(rx_antenna, subcarrier))
    assert np.allclose(out.values, expected)


def test_mean_over_time_is_rejected(profile) -> None:
    """The time axis is dynamic, so it cannot be averaged by this per-sample step."""

    with pytest.raises(CompileError, match="static"):
        Pipeline().then(Magnitude()).then(MeanOverAxes(axes=(AxisName.TIME,))).compile(
            profile
        )


# --- edge cases ---


def test_mean_needs_at_least_one_axis(profile) -> None:
    """Averaging over no axes is meaningless, so an empty list is refused."""

    with pytest.raises(CompileError, match="at least one axis"):
        Pipeline().then(Magnitude()).then(MeanOverAxes(axes=())).compile(profile)


def test_mean_rejects_a_repeated_axis(profile) -> None:
    """An axis listed twice is a mistake, not a reason to average it twice."""

    repeated = (AxisName.SUBCARRIER, AxisName.SUBCARRIER)
    with pytest.raises(CompileError, match="repeated"):
        Pipeline().then(Magnitude()).then(MeanOverAxes(axes=repeated)).compile(profile)


def test_mean_over_a_missing_axis_is_rejected(profile) -> None:
    """You can only average an axis the signal actually has."""

    with pytest.raises(CompileError, match="delay"):
        Pipeline().then(Magnitude()).then(MeanOverAxes(axes=(AxisName.DELAY,))).compile(
            profile
        )


def test_mean_streams_like_batch(profile, raw_signal) -> None:
    """MeanOverAxes is per-sample, so streaming it gives the same result as batch."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .then(MeanOverAxes(axes=(AxisName.SUBCARRIER,)))
        .compile(profile)
    )
    batch = siphon.pour(raw_signal).single()
    streamed = stream_in_chunks(siphon, raw_signal, chunk=7)
    assert np.allclose(batch.values, streamed.values)


# --- FixedSizeWindowSum: our own sliding-window sum, so pin its exact values ---


def test_fixed_size_window_sum_adds_up_each_full_window(profile) -> None:
    """Each output value is the sum of one window of samples, placed at its centre.

    Eight samples valued 1..8, a window and hop of three samples: the first window
    is 1+2+3, the second is 4+5+6. The trailing 7,8 is an incomplete window and is
    dropped. Window centres fall on sample indices 1 and 4 (times 0.001, 0.004).
    """

    layout = Layout(
        (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 1)),
        Representation.CHANNEL_FREQUENCY_RESPONSE,
        ValueKind.MAGNITUDE,
    )
    signal = create_signal(
        np.arange(1, 9, dtype=float)[:, None], np.arange(8) / 1000.0, layout
    )
    out = (
        Pipeline()
        .then(FixedSizeWindowSum(window_s=0.003, hop_s=0.003))
        .compile(profile, inlet=layout)
        .pour(signal)
        .single()
    )
    assert np.allclose(out.values.ravel(), [6.0, 15.0])  # 1+2+3, 4+5+6
    assert np.allclose(out.times, [0.001, 0.004])  # window centres


# --- DyadicFrequencyBands: our own octave banding, so pin its exact values ---


def test_dyadic_frequency_bands_sum_bins_per_octave() -> None:
    """Each band sums the frequency bins inside one octave [nyq/2^k, nyq/2^(k-1)).

    At 16 Hz the Nyquist is 8 Hz. With three bands starting at level 1 the edges
    are [4,8), [2,4), [1,2). Frequencies 0..7 all valued 1 therefore contribute
    four bins (4,5,6,7) to the top band, two (2,3) to the next, and one (1) to the
    last -- so the band sums are 4, 2, 1.
    """

    profile = AcquisitionProfile(
        subcarrier_indices=tuple(range(4)), sampling_rate_hz=16.0
    )
    layout = Layout(
        (
            Axis.dynamic(AxisName.TIME),
            Axis.static(
                AxisName.FREQUENCY, tuple(float(i) for i in range(8)), unit="Hz"
            ),
        ),
        Representation.TIME_FREQUENCY,
        ValueKind.POWER,
    )
    signal = create_signal(np.ones((2, 8)), np.arange(2) / 1000.0, layout)
    out = (
        Pipeline()
        .then(DyadicFrequencyBands(num_bands=3, first_band=1))
        .compile(profile, inlet=layout)
        .pour(signal)
        .single()
    )
    assert out.layout.axis(AxisName.BAND).size == 3
    assert np.allclose(out.values[0], [4.0, 2.0, 1.0])  # bins in [4,8), [2,4), [1,2)


def test_dyadic_band_centres_are_octave_midpoints() -> None:
    """The band coordinates are the midpoints of the octave edges.

    Band k spans [nyquist/2^k, nyquist/2^(k-1)); its coordinate is the midpoint.
    At 16 Hz (Nyquist 8) the three bands are centred at 6, 3, and 1.5 Hz.
    """

    profile = AcquisitionProfile(
        subcarrier_indices=tuple(range(4)), sampling_rate_hz=16.0
    )
    layout = Layout(
        (
            Axis.dynamic(AxisName.TIME),
            Axis.static(
                AxisName.FREQUENCY, tuple(float(i) for i in range(8)), unit="Hz"
            ),
        ),
        Representation.TIME_FREQUENCY,
        ValueKind.POWER,
    )
    out = (
        Pipeline()
        .then(DyadicFrequencyBands(num_bands=3, first_band=1))
        .compile(profile, inlet=layout)
        .pour(create_signal(np.ones((1, 8)), np.zeros(1), layout))
        .single()
    )

    nyquist = 8.0
    expected = [0.5 * (nyquist / 2**k + nyquist / 2 ** (k - 1)) for k in (1, 2, 3)]
    assert np.allclose(out.layout.axis(AxisName.BAND).coordinates, expected)  # 6,3,1.5


def test_dyadic_bands_are_zero_when_all_frequencies_fall_below_them() -> None:
    """Frequencies entirely below the lowest band contribute to nothing.

    This is a legitimately empty result -- correct, not a bug -- so it is pinned
    here so it cannot silently change: bands reach down to 1 Hz, every input
    frequency is below 1 Hz, so every band sums to zero.
    """

    profile = AcquisitionProfile(
        subcarrier_indices=tuple(range(4)), sampling_rate_hz=16.0
    )
    layout = Layout(
        (
            Axis.dynamic(AxisName.TIME),
            Axis.static(
                AxisName.FREQUENCY,
                tuple(np.linspace(0.0, 0.9, 8).tolist()),  # all below 1 Hz
                unit="Hz",
            ),
        ),
        Representation.TIME_FREQUENCY,
        ValueKind.POWER,
    )
    out = (
        Pipeline()
        .then(DyadicFrequencyBands(num_bands=3, first_band=1))
        .compile(profile, inlet=layout)
        .pour(create_signal(np.ones((2, 8)), np.arange(2) / 1000.0, layout))
        .single()
    )
    assert np.all(out.values == 0.0)
