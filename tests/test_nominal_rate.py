"""The nominal sampling rate lives on the time axis and follows the sampling grid."""

import warnings

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon import (
    AcquisitionProfile,
    Axis,
    AxisName,
    Fuse,
    Hold,
    Layout,
    Pipeline,
    Representation,
    Signal,
    ValueKind,
)
from csiphon.core.errors import CompileError, LayoutError
from csiphon.core.sampling import RateMismatchWarning, divide_nominal_rate
from csiphon.steps import (
    Hold as HoldFill,
)
from csiphon.steps import (
    Magnitude,
    MeanOverAxes,
    Resample,
    SubsampleEvery,
    WindowedFFTPower,
    WindowedVariance,
    WindowSum,
)


def _profile(rate_hz: float | None = 1000.0) -> AcquisitionProfile:
    return AcquisitionProfile(
        subcarrier_indices=(0, 1), n_rx_antennas=1, sampling_rate_hz=rate_hz
    )


def _ones(n_samples: int, rate_hz: float = 1000.0) -> Signal:
    """A raw signal with magnitude one everywhere, on a regular grid."""

    csi = np.ones((n_samples, 1, 2), dtype=complex)
    return _profile(rate_hz).raw_signal(csi, np.arange(n_samples) / rate_hz)


def test_only_the_time_axis_has_a_nominal_rate() -> None:
    """A nominal rate on any other axis is rejected."""

    with pytest.raises(LayoutError, match="time axis"):
        Axis(AxisName.SUBCARRIER, 4, nominal_rate_hz=10.0)


def test_the_nominal_rate_must_be_positive() -> None:
    """A rate of zero or less makes no sense."""

    with pytest.raises(LayoutError, match="> 0"):
        Axis.dynamic(AxisName.TIME, nominal_rate_hz=0.0)


def test_the_raw_layout_carries_the_profile_rate() -> None:
    """The profile specifies the input, so its rate starts on the time axis."""

    assert _profile(250.0).raw_csi_layout().nominal_rate_hz == 250.0
    assert _profile(None).raw_csi_layout().nominal_rate_hz is None


def test_an_unknown_rate_stays_unknown_when_divided() -> None:
    """Dividing a missing rate gives a missing rate, not an error."""

    layout = _profile(None).raw_csi_layout()

    assert divide_nominal_rate(layout, 8).nominal_rate_hz is None
    assert (
        divide_nominal_rate(_profile(800.0).raw_csi_layout(), 8).nominal_rate_hz == 100
    )


def test_a_step_that_needs_the_rate_says_so_when_there_is_none() -> None:
    """Without a nominal rate, a step that needs one fails at compile time."""

    with pytest.raises(CompileError, match="nominal sampling rate"):
        Pipeline().then(Magnitude()).then(WindowedFFTPower()).compile(_profile(None))


@pytest.mark.parametrize("streamed", [False, True], ids=["batch", "stream"])
def test_time_based_steps_use_the_rate_after_subsampling(streamed: bool) -> None:
    """After SubsampleEvery(8) a 1 kHz capture runs at 125 Hz, so a window of
    0.128 s is 16 samples and the outputs are 0.1 s apart. Guards against later
    steps still using the rate of the original capture (128 samples, 0.8 s)."""

    signal = _ones(4000)
    compiled = (
        Pipeline()
        .then(Magnitude())
        .then(SubsampleEvery(8))
        .then(WindowSum(window_s=0.128, hop_s=0.1))
        .compile(_profile())
    )

    output = (
        stream_in_chunks(compiled, signal, 100)
        if streamed
        else compiled.pour(signal).single()
    )

    # Every value is one, so a window sum is the number of samples in the window.
    assert output.values[0, 0, 0, 0, 0] == 16
    assert np.diff(output.times).mean() == pytest.approx(0.1, rel=0.02)


def test_resample_sets_the_rate_of_its_grid() -> None:
    """After Resample, later steps see the grid's rate, not the capture's."""

    compiled = (
        Pipeline()
        .then(Resample(rate_hz=250.0, fill=HoldFill()))
        .compile(_profile(1000.0))
    )

    assert compiled.outlet_layout.nominal_rate_hz == 250.0


def test_the_rate_follows_a_window_hop() -> None:
    """One output sample per hop: a 0.1 s hop gives 10 Hz."""

    compiled = Pipeline().then(Magnitude()).then(WindowedFFTPower()).compile(_profile())

    assert compiled.outlet_layout.nominal_rate_hz == pytest.approx(10.0)


def test_an_inlet_layout_without_a_rate_takes_the_profile_rate() -> None:
    """compile(profile, inlet=layout): the profile gives a rate the layout lacks."""

    layout = Layout(
        (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 2)),
        Representation.CHANNEL_FREQUENCY_RESPONSE,
        ValueKind.MAGNITUDE,
    )

    compiled = (
        Pipeline().then(SubsampleEvery(4)).compile(_profile(1000.0), inlet=layout)
    )

    assert compiled.outlet_layout.nominal_rate_hz == 250.0


@pytest.mark.parametrize(("reference", "rate_hz"), [(0, 1000.0), (1, 10.0)])
def test_a_merge_takes_the_rate_of_its_reference_branch(
    reference: int, rate_hz: float
) -> None:
    """The merged signal lies on the reference branch's timeline, so it has that
    branch's rate (here a per-sample branch and a 10 Hz window branch)."""

    signal = _ones(1500)
    compiled = (
        Pipeline()
        .then(Magnitude())
        .branch(
            variance=Pipeline().then(WindowedVariance(win_size_s=0.05)),
            spectral=Pipeline()
            .then(WindowedFFTPower())
            .then(MeanOverAxes(axes=(AxisName.SUBCARRIER,))),
        )
        .merge(using=Fuse(), align=Hold(reference=reference))
        .compile(_profile())
    )

    output = compiled.pour(signal).single()

    declared = compiled.outlet_layout.nominal_rate_hz
    assert declared == pytest.approx(rate_hz)
    assert declared == pytest.approx(1.0 / np.median(np.diff(output.times)), rel=0.1)


def _timestamps_at(rate_hz: float, count: int = 100) -> Signal:
    return _profile(1000.0).raw_signal(
        np.ones((count, 1, 2), dtype=complex), np.arange(count) / rate_hz
    )


@pytest.mark.parametrize("streamed", [False, True], ids=["batch", "stream"])
@pytest.mark.parametrize("actual_hz", [125.0, 4000.0], ids=["too-slow", "too-fast"])
def test_a_wrong_nominal_rate_warns_once(streamed: bool, actual_hz: float) -> None:
    """Timestamps at 125 Hz against a profile saying 1000 Hz is the stale-rate
    mistake this guards against, and a rate that is too high is as wrong. The
    warning appears once per run, not once per chunk."""

    compiled = Pipeline().then(Magnitude()).compile(_profile(1000.0))
    signal = _timestamps_at(actual_hz)

    with pytest.warns(RateMismatchWarning) as caught:
        if streamed:
            stream_in_chunks(compiled, signal, 10)
        else:
            compiled.pour(signal)

    assert len(caught) == 1


def test_lost_samples_do_not_trigger_the_rate_warning() -> None:
    """Random loss leaves the median interval at the nominal one, so a lossy but
    correctly labeled capture stays quiet."""

    rng = np.random.default_rng(0)
    kept = np.sort(rng.choice(1000, 700, replace=False))
    signal = _profile(1000.0).raw_signal(
        np.ones((700, 1, 2), dtype=complex), kept / 1000.0
    )
    compiled = Pipeline().then(Magnitude()).compile(_profile(1000.0))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compiled.pour(signal)


def test_a_short_recording_is_not_checked_for_the_rate() -> None:
    """A handful of timestamps gives no steady median interval, so even a wrong
    rate stays quiet instead of risking a false alarm."""

    compiled = Pipeline().then(Magnitude()).compile(_profile(1000.0))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compiled.pour(_timestamps_at(125.0, count=10))
