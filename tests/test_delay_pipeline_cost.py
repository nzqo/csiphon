"""Steps of the delay pipeline: exact results, and no needless copies."""

import numpy as np
import pytest

from csiphon import AcquisitionProfile, Pipeline, Signal
from csiphon.steps import (
    DelayAutocorrelation,
    GainNormalize,
    Magnitude,
    Power,
    RealPart,
    SubsampleEvery,
)

_INDICES = tuple(range(-6, -1)) + tuple(range(2, 7))
_PROFILE = AcquisitionProfile(
    subcarrier_indices=_INDICES, n_rx_antennas=2, sampling_rate_hz=100.0
)


def _raw(n_samples: int = 60) -> Signal:
    rng = np.random.default_rng(0)
    shape = (n_samples, 2, len(_INDICES))
    csi = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return _PROFILE.raw_signal(csi, np.arange(n_samples) / 100.0)


def _values_after(first_step: object) -> np.ndarray:
    """The values after one step, the input of the step under test."""

    return Pipeline().then(first_step).compile(_PROFILE).pour(_raw()).single().values


def test_subsample_returns_a_view_of_the_input() -> None:
    """Keeping every Nth sample must not copy the kept samples. A copy of the
    complex raw CSI is the largest array in the delay pipeline."""

    signal = _raw()
    output = Pipeline().then(SubsampleEvery(8)).compile(_PROFILE).pour(signal).single()

    assert np.shares_memory(output.values, signal.values)
    np.testing.assert_array_equal(output.values, signal.values[::8])


@pytest.mark.parametrize(
    "first_step", [None, RealPart(), Magnitude(), Power()], ids=lambda s: str(s)[:9]
)
def test_gain_normalize_divides_by_the_mean_absolute_value(
    first_step: object,
) -> None:
    """For every admissible value kind, including signed real values (where the
    mean must use |x|), the output is x / (mean(|x|) + epsilon) per time sample."""

    pipeline = Pipeline()
    if first_step is not None:
        pipeline = pipeline.then(first_step)
    siphon = pipeline.then(GainNormalize()).compile(_PROFILE)
    before = (
        pipeline.compile(_PROFILE).pour(_raw()).single().values
        if first_step is not None
        else _raw().values
    )

    output = siphon.pour(_raw()).single().values

    mean = np.abs(before).mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(output, before / (mean + 1e-12), rtol=1e-12)


@pytest.mark.parametrize("first_step", [Magnitude(), RealPart()], ids=["mag", "real"])
def test_delay_autocorrelation_is_the_inverse_dft_of_the_power(
    first_step: object,
) -> None:
    """r[t] = mean over subcarriers k of |x[k]|^2 * exp(2j*pi*k*t/nfft), also for
    signed real input, and for a restricted range of taps."""

    nfft, first_tap, num_taps = 16, 2, 5
    before = _values_after(first_step)
    siphon = (
        Pipeline()
        .then(first_step)
        .then(DelayAutocorrelation(nfft=nfft, first_tap=first_tap, num_taps=num_taps))
        .compile(_PROFILE)
    )

    output = siphon.pour(_raw()).single().values

    power = np.abs(before) ** 2
    expected = np.empty((*before.shape[:-1], num_taps), dtype=complex)
    for position, tap in enumerate(range(first_tap, first_tap + num_taps)):
        basis = np.exp(2j * np.pi * np.asarray(_INDICES) * tap / nfft)
        expected[..., position] = (power * basis).mean(axis=-1)
    np.testing.assert_allclose(output, expected, rtol=1e-10, atol=1e-12)
