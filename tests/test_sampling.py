"""Sampling-rate and jitter behaviour (the WiFi-CSI reality: never uniform)."""

from __future__ import annotations

import numpy as np
import pytest

from csiphon import AcquisitionProfile, AxisName, Pipeline, Signal
from csiphon.core import CompileError, DataError, JitterWarning
from csiphon.core.sampling import effective_rate_hz, estimate_rate_hz, jitter_ratio
from csiphon.steps import (
    FoldAxes,
    GainNormalize,
    Linear,
    Magnitude,
    Resample,
    WindowedFFTPower,
)


def _jittered_raw(profile: AcquisitionProfile, jitter: float) -> Signal:
    """Build a raw signal whose timestamps are irregular by `jitter` fraction."""

    assert profile.sampling_rate_hz is not None
    rng = np.random.default_rng(7)
    n = 600
    shape = (n, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    step = 1.0 / profile.sampling_rate_hz
    noise = rng.normal(0.0, jitter * step, size=n)
    times = np.cumsum(np.abs(step + noise))
    return profile.raw_signal(csi, times)


def _fft_pipeline(**kwargs) -> Pipeline:
    """Magnitude -> gain-norm -> fold -> windowed FFT power."""

    return (
        Pipeline()
        .then(Magnitude())
        .then(GainNormalize())
        .then(FoldAxes(axes=(AxisName.RX_ANTENNA, AxisName.SUBCARRIER)))
        .then(WindowedFFTPower(window_s=0.128, hop_s=0.05, band_hz=60.0, **kwargs))
    )


def test_fft_runs_on_mild_jitter(profile) -> None:
    """A realistic small jitter is tolerated silently and produces output."""

    signal = _jittered_raw(profile, jitter=0.05)
    compiled = _fft_pipeline().compile(profile)
    out = compiled.pour(signal).single()
    assert out.n_samples > 0


def test_fft_warns_on_excessive_jitter(profile) -> None:
    """Heavily irregular timestamps trigger a jitter warning (not an error)."""

    signal = _jittered_raw(profile, jitter=2.0)
    compiled = _fft_pipeline().compile(profile)
    with pytest.warns(JitterWarning):
        compiled.pour(signal).single()


def test_strict_mode_raises_on_excessive_jitter(profile) -> None:
    """Strict mode upgrades the jitter warning to an error."""

    signal = _jittered_raw(profile, jitter=2.0)
    compiled = _fft_pipeline(strict=True).compile(profile)
    with pytest.raises(DataError):
        compiled.pour(signal).single()


def test_fft_requires_nominal_rate_for_compile() -> None:
    """Without a nominal rate the frequency axis cannot be sized -> compile error."""

    profile = AcquisitionProfile(n_rx_antennas=2, subcarrier_indices=tuple(range(52)))
    with pytest.raises(CompileError):
        _fft_pipeline().compile(profile)


def test_resample_makes_grid_uniform(profile) -> None:
    """Resampling irregular timestamps onto a uniform grid gives even spacing."""

    signal = _jittered_raw(profile, jitter=0.5)
    pipeline = Pipeline().then(Resample(rate_hz=1000.0, fill=Linear()))
    out = pipeline.compile(profile).pour(signal).single()
    steps = np.diff(out.times)
    assert np.allclose(steps, steps[0])


# --- the rate/jitter estimators, unit-tested directly ---


def test_estimate_rate_hz_reads_a_uniform_grid() -> None:
    """A 1 kHz grid estimates as 1000 Hz (the reciprocal of the interval)."""

    times = np.arange(100) / 1000.0
    assert estimate_rate_hz(times) == pytest.approx(1000.0)


def test_estimate_rate_hz_uses_the_median_so_one_gap_does_not_skew_it() -> None:
    """The estimate is the median interval, so a single long gap is ignored."""

    times = np.array([0.0, 1e-3, 2e-3, 3e-3, 10e-3])  # last interval is 7x
    assert estimate_rate_hz(times) == pytest.approx(1000.0)


def test_estimate_rate_hz_needs_at_least_two_timestamps() -> None:
    """One timestamp has no interval to measure, so it fails loudly."""

    with pytest.raises(DataError, match="two timestamps"):
        estimate_rate_hz(np.zeros(1))


def test_estimate_rate_hz_rejects_non_increasing_timestamps() -> None:
    """Decreasing timestamps give a non-positive median interval, which is an error."""

    with pytest.raises(DataError, match="strictly increasing"):
        estimate_rate_hz(np.array([3.0, 2.0, 1.0]))


def test_jitter_ratio_is_zero_for_a_uniform_grid_or_too_few_samples() -> None:
    """A perfectly even grid has no jitter; under three samples reports zero too."""

    assert jitter_ratio(np.arange(50) / 1000.0) == pytest.approx(0.0)
    assert jitter_ratio(np.array([0.0, 1e-3])) == 0.0


def test_jitter_ratio_grows_with_irregularity() -> None:
    """More irregular intervals report a higher std/median ratio."""

    rng = np.random.default_rng(0)
    step = 1e-3
    mild = np.cumsum(np.abs(step + rng.normal(0, 0.02 * step, 200)))
    wild = np.cumsum(np.abs(step + rng.normal(0, 1.0 * step, 200)))
    assert jitter_ratio(mild) < jitter_ratio(wild)


def test_effective_rate_prefers_the_nominal_rate_over_the_timestamps() -> None:
    """A declared nominal rate is returned verbatim; only None estimates."""

    jittery = np.array([0.0, 5e-3, 6e-3])  # would estimate very differently
    assert effective_rate_hz(jittery, nominal_hz=1000.0) == 1000.0
    assert effective_rate_hz(np.arange(10) / 500.0, nominal_hz=None) == pytest.approx(
        500.0
    )
