"""Unit tests for the batch behaviour of individual steps."""

import numpy as np
import pytest
from conftest import fold_channels_into_feature
from scipy.signal import butter, filtfilt

from csiphon import AcquisitionProfile, Pipeline, Signal
from csiphon.core import AxisName, CompileError, Representation, ValueKind
from csiphon.steps import (
    AxisReference,
    ButterworthFilter,
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
from csiphon.steps.filtering.butterworth_filter import Band


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


def test_delay_autocorrelation_of_a_flat_spectrum_is_a_delta() -> None:
    """A flat magnitude over every FFT bin autocorrelates to 1 at delay 0, else 0.

    Eight subcarriers filling an 8-point FFT: the inverse DFT of a constant power
    is a delta, and the mean-normalization puts its height at the mean power (1).
    """

    profile = AcquisitionProfile(n_rx_antennas=1, subcarrier_indices=tuple(range(8)))
    flat = np.ones((2, 1, 8), dtype=complex)
    signal = profile.raw_signal(flat, np.arange(2) / 1000.0)
    out = _run(profile, signal, Magnitude(), DelayAutocorrelation(nfft=8))

    delay = out.layout.axis_position(AxisName.DELAY)
    expected = np.zeros(8)
    expected[0] = 1.0
    assert np.allclose(np.moveaxis(out.values, delay, -1), expected)


def test_delay_taps_pick_the_requested_taps_real_then_imag(profile, raw_signal) -> None:
    """DelayTaps(first_tap, num_taps) is those ACF bins' real parts, then imaginary."""

    acf = _run(profile, raw_signal, Magnitude(), DelayAutocorrelation())
    taps = _run(
        profile,
        raw_signal,
        Magnitude(),
        DelayAutocorrelation(),
        DelayTaps(num_taps=2, first_tap=1),
    )

    delay = acf.layout.axis_position(AxisName.DELAY)
    kept = np.take(acf.values, [1, 2], axis=delay)
    expected = np.concatenate([kept.real, kept.imag], axis=delay)
    assert np.allclose(taps.values, expected)


def test_delay_autocorrelation_tap_range_is_a_slice_of_the_full_axis(
    profile, raw_signal
) -> None:
    """DelayAutocorrelation(first_tap, num_taps) carries exactly those taps, with
    the same values the full axis has at them."""

    full = _run(profile, raw_signal, Magnitude(), DelayAutocorrelation())
    ranged = _run(
        profile,
        raw_signal,
        Magnitude(),
        DelayAutocorrelation(first_tap=1, num_taps=3),
    )

    delay = ranged.layout.axis(AxisName.DELAY)
    assert delay.size == 3
    assert delay.coordinates == (1, 2, 3)
    assert delay.unit == "tap"
    position = full.layout.axis_position(AxisName.DELAY)
    assert np.allclose(ranged.values, np.take(full.values, [1, 2, 3], axis=position))


def test_delay_autocorrelation_open_tap_range_runs_to_the_last_tap(
    profile, raw_signal
) -> None:
    """With num_taps left None, first_tap keeps every tap from there to nfft-1."""

    out = _run(profile, raw_signal, Magnitude(), DelayAutocorrelation(first_tap=60))
    assert out.layout.axis(AxisName.DELAY).coordinates == (60, 61, 62, 63)


def test_delay_taps_after_a_tap_range_address_positions(profile, raw_signal) -> None:
    """DelayTaps counts positions on the delay axis it is given, so after a
    restricted DelayAutocorrelation the kept taps start at position 0."""

    from_full = _run(
        profile,
        raw_signal,
        Magnitude(),
        DelayAutocorrelation(),
        DelayTaps(first_tap=1, num_taps=3),
    )
    from_range = _run(
        profile,
        raw_signal,
        Magnitude(),
        DelayAutocorrelation(first_tap=1, num_taps=3),
        DelayTaps(first_tap=0, num_taps=3),
    )
    assert from_range.layout == from_full.layout
    assert np.allclose(from_range.values, from_full.values)


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


def test_butterworth_filters_complex_csi_like_its_parts(profile, raw_signal) -> None:
    """Complex CSI is filtered as is: the same as filtering real and imaginary apart."""

    out = _run(profile, raw_signal, ButterworthFilter(cutoff_hz=5.0, btype=Band.HIGH))
    numerator, denominator = butter(2, 5.0, btype="high", fs=profile.sampling_rate_hz)
    real = filtfilt(numerator, denominator, raw_signal.values.real, axis=0)
    imag = filtfilt(numerator, denominator, raw_signal.values.imag, axis=0)
    assert np.iscomplexobj(out.values)
    assert np.allclose(out.values, real + 1j * imag)
