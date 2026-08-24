"""Negative tests for compile-time input-validation guards.

Each case builds a minimal valid input for a step, sets one bad parameter, and
asserts the pipeline fails to compile with a CompileError naming the step. The
guards live in each step's ``output_layout``, so they fire at compile time (no
signal or scipy needed to trigger them).
"""

from __future__ import annotations

import pytest

from csiphon import AcquisitionProfile, Pipeline, Siphon
from csiphon.core import CompileError
from csiphon.pipeline.step import Step
from csiphon.steps import (
    ButterworthFilter,
    ChannelImpulseResponse,
    DelayAutocorrelation,
    Magnitude,
    SavitzkyGolay,
    WindowedSlope,
    WindowedVariance,
)
from csiphon.steps.filtering.butterworth_filter import Band


def _compile(profile: AcquisitionProfile, *steps: Step) -> Siphon:
    """Compile a pipeline of steps against the profile's raw-CSI inlet."""

    pipeline = Pipeline()
    for step in steps:
        pipeline = pipeline.then(step)
    return pipeline.compile(profile)


# --- butterworth-filter -------------------------------------------------------


def test_butterworth_rejects_nonpositive_order(profile: AcquisitionProfile) -> None:
    """order < 1 fails at compile with a message naming the step."""

    with pytest.raises(CompileError, match="butterworth-filter"):
        _compile(profile, Magnitude(), ButterworthFilter(order=0))


def test_butterworth_band_rejects_scalar_cutoff(
    profile: AcquisitionProfile,
) -> None:
    """A band-pass filter needs a (low, high) pair, not a scalar cutoff."""

    with pytest.raises(CompileError, match="butterworth-filter"):
        _compile(
            profile, Magnitude(), ButterworthFilter(cutoff_hz=1.0, btype=Band.BAND)
        )


def test_butterworth_band_rejects_unordered_cutoff(
    profile: AcquisitionProfile,
) -> None:
    """A band-pass filter needs 0 < low < high."""

    with pytest.raises(CompileError, match="butterworth-filter"):
        _compile(
            profile,
            Magnitude(),
            ButterworthFilter(cutoff_hz=(5.0, 2.0), btype=Band.BAND),
        )


def test_butterworth_low_rejects_negative_cutoff(
    profile: AcquisitionProfile,
) -> None:
    """A low-pass filter needs a positive scalar cutoff."""

    with pytest.raises(CompileError, match="butterworth-filter"):
        _compile(
            profile, Magnitude(), ButterworthFilter(cutoff_hz=-1.0, btype=Band.LOW)
        )


def test_butterworth_low_rejects_tuple_cutoff(profile: AcquisitionProfile) -> None:
    """A low-pass filter needs a scalar cutoff, not a pair."""

    with pytest.raises(CompileError, match="butterworth-filter"):
        _compile(
            profile,
            Magnitude(),
            ButterworthFilter(cutoff_hz=(1.0, 2.0), btype=Band.LOW),
        )


# --- savitzky-golay -----------------------------------------------------------


def test_savitzky_golay_rejects_negative_window_length(
    profile: AcquisitionProfile,
) -> None:
    """A negative window_length (with a negative polyorder) fails at compile."""

    with pytest.raises(CompileError, match="savitzky-golay"):
        _compile(profile, Magnitude(), SavitzkyGolay(window_length=-3, polyorder=-5))


def test_savitzky_golay_rejects_negative_polyorder(
    profile: AcquisitionProfile,
) -> None:
    """A negative polyorder fails at compile."""

    with pytest.raises(CompileError, match="savitzky-golay"):
        _compile(profile, Magnitude(), SavitzkyGolay(polyorder=-1))


def test_savitzky_golay_rejects_even_window_length(
    profile: AcquisitionProfile,
) -> None:
    """The pre-existing odd-length guard rejects an even window_length."""

    with pytest.raises(CompileError, match="savitzky-golay"):
        _compile(profile, Magnitude(), SavitzkyGolay(window_length=10))


def test_savitzky_golay_rejects_polyorder_ge_window(
    profile: AcquisitionProfile,
) -> None:
    """The pre-existing guard rejects polyorder >= window_length."""

    with pytest.raises(CompileError, match="savitzky-golay"):
        _compile(profile, Magnitude(), SavitzkyGolay(window_length=5, polyorder=7))


# --- windowed-variance --------------------------------------------------------


@pytest.mark.parametrize("win_size_s", [0.0, -1.0])
def test_windowed_variance_rejects_nonpositive_window(
    profile: AcquisitionProfile, win_size_s: float
) -> None:
    """win_size_s <= 0 fails at compile with a message naming the step."""

    with pytest.raises(CompileError, match="windowed-variance"):
        _compile(profile, Magnitude(), WindowedVariance(win_size_s=win_size_s))


# --- windowed-slope (pre-existing guard) --------------------------------------


@pytest.mark.parametrize("window_size", [0, 1])
def test_windowed_slope_rejects_tiny_window(
    profile: AcquisitionProfile, window_size: int
) -> None:
    """The pre-existing guard rejects window_size < 2."""

    with pytest.raises(CompileError, match="windowed-slope"):
        _compile(profile, Magnitude(), WindowedSlope(window_size=window_size))


# --- delay-autocorrelation ----------------------------------------------------


@pytest.mark.parametrize("nfft", [0, -4])
def test_delay_autocorrelation_rejects_nonpositive_nfft(
    profile: AcquisitionProfile, nfft: int
) -> None:
    """An explicit nfft < 1 fails at compile with a message naming the step."""

    with pytest.raises(CompileError, match="delay-autocorrelation"):
        _compile(profile, Magnitude(), DelayAutocorrelation(nfft=nfft))


def test_delay_autocorrelation_rejects_uninferrable_nfft() -> None:
    """More than 512 subcarriers cannot infer an nfft, so it fails at compile."""

    wide = AcquisitionProfile(
        n_rx_antennas=1,
        subcarrier_indices=tuple(range(600)),
        sampling_rate_hz=1000.0,
    )
    with pytest.raises(CompileError, match="delay-autocorrelation"):
        _compile(wide, Magnitude(), DelayAutocorrelation())


# --- channel-impulse-response -------------------------------------------------


@pytest.mark.parametrize("nfft", [0, -4])
def test_channel_impulse_response_rejects_nonpositive_nfft(
    profile: AcquisitionProfile, nfft: int
) -> None:
    """CIR shares resolve_nfft, so an explicit nfft < 1 fails at compile too."""

    with pytest.raises(CompileError, match="channel-impulse-response"):
        _compile(profile, ChannelImpulseResponse(nfft=nfft))


def test_channel_impulse_response_rejects_zero_taps(
    profile: AcquisitionProfile,
) -> None:
    """num_taps must be at least 1."""

    with pytest.raises(CompileError, match="channel-impulse-response"):
        _compile(profile, ChannelImpulseResponse(num_taps=0))


def test_channel_impulse_response_rejects_out_of_range_taps(
    profile: AcquisitionProfile,
) -> None:
    """num_taps above nfft is out of range and fails at compile."""

    with pytest.raises(CompileError, match="channel-impulse-response"):
        _compile(profile, ChannelImpulseResponse(num_taps=10_000))
