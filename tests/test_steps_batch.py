"""Unit tests for the batch behaviour of individual steps."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fold_channels_into_feature

from csiphon import AcquisitionProfile, Pipeline, Signal
from csiphon.core import AxisName, CompileError, Representation, ValueKind
from csiphon.steps import (
    AxisReference,
    ComplexStftMagnitude,
    DelayAutocorrelation,
    DelayTaps,
    DyadicFrequencyBands,
    FoldAxes,
    GainNormalize,
    GlobalMaxNormalize,
    LogScale,
    Magnitude,
    Power,
)
from csiphon.steps.calibration import Combine, Reference


def _run(profile: AcquisitionProfile, signal: Signal, *steps) -> Signal:
    """Compile a pipeline of steps and run it in batch."""

    pipeline = Pipeline()
    for step in steps:
        pipeline = pipeline.then(step)
    return pipeline.compile(profile).pour(signal).single()


def test_magnitude_matches_numpy(profile, raw_signal) -> None:
    """Magnitude equals abs and flips the value kind to magnitude."""

    out = _run(profile, raw_signal, Magnitude())
    assert out.layout.values == ValueKind.MAGNITUDE
    assert np.allclose(out.values, np.abs(raw_signal.values))


def test_gain_normalize_unit_mean(profile, raw_signal) -> None:
    """After gain normalization each frame's subcarrier mean is ~1."""

    out = _run(profile, raw_signal, Magnitude(), GainNormalize())
    mean = out.values.mean(axis=out.layout.axis_position(AxisName.SUBCARRIER))
    assert np.allclose(mean, 1.0, atol=1e-6)


def test_delay_autocorrelation_is_full_complex(profile, raw_signal) -> None:
    """DelayAutocorrelation gives a full complex delay axis, one value per tap."""

    out = _run(
        profile, raw_signal, Magnitude(), GainNormalize(), DelayAutocorrelation()
    )
    assert out.layout.axis(AxisName.DELAY).size == 64  # nfft for 52 subcarriers
    assert out.layout.representation == Representation.AUTOCORRELATION
    assert np.iscomplexobj(out.values)


def test_delay_taps_truncates_to_real(profile, raw_signal) -> None:
    """DelayTaps keeps 2*num_taps stacked real/imag values."""

    out = _run(
        profile,
        raw_signal,
        Magnitude(),
        GainNormalize(),
        DelayAutocorrelation(),
        DelayTaps(),
    )
    assert out.layout.axis(AxisName.DELAY).size == 6
    assert not np.iscomplexobj(out.values)


def test_fold_antenna_and_delay(profile, raw_signal) -> None:
    """Folding merges the antenna and delay axes, antenna-major (C-order)."""

    out = _run(
        profile,
        raw_signal,
        Magnitude(),
        GainNormalize(),
        DelayAutocorrelation(),
        DelayTaps(num_taps=2),
        FoldAxes(axes=(AxisName.RX_ANTENNA, AxisName.DELAY)),
    )
    assert out.layout.axis(AxisName.FEATURE).size == profile.n_rx_antennas * 4


def test_axis_reference_conjugate_reference_is_real(profile, raw_signal) -> None:
    """The reference antenna conjugated with itself is real and non-negative."""

    step = AxisReference(axis=AxisName.RX_ANTENNA, combine=Combine.CONJUGATE)
    out = _run(profile, raw_signal, step)
    assert out.layout.representation == Representation.CROSS_SPECTRUM
    ref = np.take(out.values, 0, axis=out.layout.axis_position(AxisName.RX_ANTENNA))
    assert np.allclose(ref.imag, 0.0, atol=1e-9)
    assert np.all(ref.real >= -1e-9)


def test_axis_reference_adjacent_shrinks(profile, raw_signal) -> None:
    """Adjacent-mode ratio drops one subcarrier and relabels to a ratio."""

    step = AxisReference(combine=Combine.DIVIDE, mode=Reference.ADJACENT)
    out = _run(profile, raw_signal, step)
    assert out.layout.axis(AxisName.SUBCARRIER).size == profile.n_subcarriers - 1
    assert out.layout.representation == Representation.RATIO


def test_power_then_log(profile, raw_signal) -> None:
    """Power is squared magnitude; log compression yields real values."""

    out = _run(profile, raw_signal, Magnitude(), Power(), LogScale())
    assert out.layout.values == ValueKind.REAL
    expected = np.log(np.abs(raw_signal.values) ** 2 + 1e-12)
    assert np.allclose(out.values, expected)


def test_dyadic_bands_need_frequency_axis(profile, raw_signal) -> None:
    """Dyadic banding refuses to compile without a frequency axis."""

    with pytest.raises(CompileError):
        _run(profile, raw_signal, Magnitude(), DyadicFrequencyBands())


def test_gain_normalize_on_complex_preserves_phase(profile, raw_signal) -> None:
    """Gain normalization keeps complex CSI complex, dividing only by the mean gain."""

    out = _run(profile, raw_signal, GainNormalize())
    assert out.layout.values == ValueKind.COMPLEX  # complex (and phase) preserved
    subcarrier = out.layout.axis_position(AxisName.SUBCARRIER)
    gain = np.abs(raw_signal.values).mean(axis=subcarrier, keepdims=True)
    assert np.allclose(out.values, raw_signal.values / gain, atol=1e-9)
    assert np.allclose(np.angle(out.values), np.angle(raw_signal.values), atol=1e-6)


def test_global_max_normalize_peak_is_one(profile, raw_signal) -> None:
    """After global-max normalization the largest value over the whole array is 1."""

    out = _run(profile, raw_signal, Magnitude(), GlobalMaxNormalize())
    assert np.isclose(out.values.max(), 1.0)
    expected = np.abs(raw_signal.values) / np.abs(raw_signal.values).max()
    assert np.allclose(out.values, expected)


def test_complex_stft_magnitude_adds_frequency_axis(profile, raw_signal) -> None:
    """The complex STFT folds channels to one axis, then adds a magnitude freq axis."""

    pytest.importorskip("scipy")
    out = _run(
        profile,
        raw_signal,
        fold_channels_into_feature(),
        ComplexStftMagnitude(window_size=256, hop_size=10, freq_bins=16),
    )
    assert out.layout.values == ValueKind.MAGNITUDE
    assert out.layout.representation == Representation.TIME_FREQUENCY
    assert out.layout.axis(AxisName.FREQUENCY).size == 16
    assert AxisName.TIME in out.layout.axis_names
    assert np.all(out.values >= 0.0)


def test_gain_normalize_over_a_configurable_axis(profile, raw_signal) -> None:
    """Gain normalization is generic in `axis`: normalize over antennas instead."""

    out = _run(profile, raw_signal, GainNormalize(axis=AxisName.RX_ANTENNA))
    assert out.layout.values == ValueKind.COMPLEX  # complex preserved
    rx = out.layout.axis_position(AxisName.RX_ANTENNA)
    assert np.allclose(np.abs(out.values).mean(axis=rx), 1.0, atol=1e-6)


def test_gain_normalize_rejects_a_missing_axis(profile, raw_signal) -> None:
    """The configured axis is required: normalizing over an absent axis fails."""

    with pytest.raises(CompileError, match="gain-normalize"):
        _run(profile, raw_signal, GainNormalize(axis=AxisName.DELAY))


def test_global_max_normalize_preserves_layout_and_survives_zeros(
    profile, raw_signal
) -> None:
    """Layout is untouched and the epsilon floor keeps all-zero input finite."""

    zeros = raw_signal.with_values(np.zeros_like(raw_signal.values), raw_signal.layout)
    reference = _run(profile, zeros, Magnitude())
    out = _run(profile, zeros, Magnitude(), GlobalMaxNormalize())

    assert out.layout.axis_names == reference.layout.axis_names
    assert out.layout.values == reference.layout.values  # structure unchanged
    assert np.all(np.isfinite(out.values))  # no divide-by-zero
    assert np.all(out.values == 0.0)
