"""CumulativeMeanSubtract: the plain mean so far, its exponential tail, streaming."""

import numpy as np
import pytest

from csiphon import AcquisitionProfile, Pipeline
from csiphon.core.errors import CompileError
from csiphon.core.signal import Signal
from csiphon.steps import CumulativeMeanSubtract, Magnitude

PROFILE = AcquisitionProfile(
    subcarrier_indices=tuple(range(-4, 4)), n_rx_antennas=2, sampling_rate_hz=100.0
)


def _raw(n: int = 500, seed: int = 0) -> Signal:
    rng = np.random.default_rng(seed)
    csi = rng.normal(size=(n, 2, 8)) + 1j * rng.normal(size=(n, 2, 8)) + 3.0
    return PROFILE.raw_signal(csi, np.arange(n) / 100.0)


def _magnitudes(raw) -> np.ndarray:
    return Pipeline().then(Magnitude()).compile(PROFILE).pour(raw).single().values


def _run(min_alpha: float, raw) -> np.ndarray:
    pipe = Pipeline().then(Magnitude()).then(CumulativeMeanSubtract(min_alpha))
    return pipe.compile(PROFILE).pour(raw).single().values


def test_plain_mean_so_far() -> None:
    """Each sample minus the mean of all samples up to and including it."""

    raw = _raw()
    x = _magnitudes(raw).astype(np.float64)
    n = np.arange(1, len(x) + 1).reshape((-1,) + (1,) * (x.ndim - 1))
    expected = x - np.cumsum(x, axis=0) / n
    np.testing.assert_allclose(_run(0.0, raw), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_run(0.0, raw)[0], 0.0, atol=1e-12)


def test_exponential_tail() -> None:
    """After 1 / min_alpha samples the mean turns exponential."""

    raw, alpha = _raw(), 0.05
    x = _magnitudes(raw).astype(np.float64)
    expected, mean = np.empty_like(x), np.zeros_like(x[0])
    for t in range(len(x)):
        mean = mean + max(1.0 / (t + 1), alpha) * (x[t] - mean)
        expected[t] = x[t] - mean
    np.testing.assert_allclose(_run(alpha, raw), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("min_alpha", [0.0, 0.05])
def test_streaming_matches_batch(min_alpha: float) -> None:
    """Uneven chunks, including ones that straddle the switch to exponential."""

    raw = _raw()
    pipe = Pipeline().then(Magnitude()).then(CumulativeMeanSubtract(min_alpha))
    siphon = pipe.compile(PROFILE)
    batch = siphon.pour(raw).single().values
    stream, parts, start = siphon.stream(), [], 0
    for size in (1, 7, 12, 3, 40, 97, 340):
        chunk = PROFILE.raw_signal(
            raw.values[start : start + size], raw.times[start : start + size]
        )
        parts.append(stream.flow(chunk).single().values)
        start += size
    parts.append(stream.flush().single().values)
    np.testing.assert_allclose(np.concatenate(parts), batch, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("min_alpha", [-0.1, 1.0])
def test_rejects_bad_gain(min_alpha: float) -> None:
    """min_alpha outside [0, 1) fails when the pipeline is compiled."""

    with pytest.raises(CompileError, match="min_alpha"):
        Pipeline().then(Magnitude()).then(CumulativeMeanSubtract(min_alpha)).compile(
            PROFILE
        )
