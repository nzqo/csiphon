"""Statistics steps: window reduction, correctness, and batch/stream equivalence."""

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
from csiphon.steps import CovarianceSpectrum, LocalPcaBias


def _feature_signal(n: int = 200, d: int = 8, seed: int = 0) -> Signal:
    """A real `(time, feature[d])` signal to feed the statistics steps."""

    rng = np.random.default_rng(seed)
    layout = Layout(
        axes=(Axis.dynamic(AxisName.TIME, unit="s"), Axis.sized(AxisName.FEATURE, d)),
        representation=Representation.FEATURE_VECTOR,
        values=ValueKind.REAL,
    )
    return create_signal(rng.standard_normal((n, d)), np.arange(n) / 1000.0, layout)


def test_covariance_spectrum_windows_and_times(profile: AcquisitionProfile) -> None:
    """Output has one 5-vector per window, timestamped at the window center."""

    signal = _feature_signal(n=200, d=8)
    compiled = (
        Pipeline()
        .then(CovarianceSpectrum(window_size=20, hop_size=5))
        .compile(profile, inlet=signal.layout)
    )
    out = compiled.pour(signal).single()

    starts = np.arange(0, 200 - 20 + 1, 5)
    assert out.values.shape == (len(starts), 5)
    assert out.layout.axis(AxisName.FEATURE).size == 5
    assert np.allclose(out.times, signal.times[starts + 20 // 2])


def test_covariance_spectrum_broadcasts_over_a_channel_axis(
    profile: AcquisitionProfile,
) -> None:
    """A (time, rx_antenna, feature) input equals stacking each antenna run alone.

    The extra axis is a parallel channel: broadcasting it must give bit-for-bit the
    same per-antenna statistics as running the step on that antenna's slice by itself.
    """

    rng = np.random.default_rng(1)
    n, antennas, d = 200, 3, 8
    data = rng.standard_normal((n, antennas, d))
    times = np.arange(n) / 1000.0
    step = CovarianceSpectrum(window_size=20, hop_size=5)

    layout_nd = Layout(
        axes=(
            Axis.dynamic(AxisName.TIME, unit="s"),
            Axis.sized(AxisName.RX_ANTENNA, antennas),
            Axis.sized(AxisName.FEATURE, d),
        ),
        representation=Representation.FEATURE_VECTOR,
        values=ValueKind.REAL,
    )
    out_nd = (
        Pipeline()
        .then(step)
        .compile(profile, inlet=layout_nd)
        .pour(create_signal(data, times, layout_nd))
        .single()
    )
    assert out_nd.layout.axis_names == (
        AxisName.TIME,
        AxisName.RX_ANTENNA,
        AxisName.FEATURE,
    )
    assert out_nd.values.shape[1:] == (antennas, 5)

    layout_1d = Layout(
        axes=(Axis.dynamic(AxisName.TIME, unit="s"), Axis.sized(AxisName.FEATURE, d)),
        representation=Representation.FEATURE_VECTOR,
        values=ValueKind.REAL,
    )
    for antenna in range(antennas):
        one = (
            Pipeline()
            .then(step)
            .compile(profile, inlet=layout_1d)
            .pour(create_signal(data[:, antenna, :], times, layout_1d))
            .single()
        )
        assert np.allclose(out_nd.values[:, antenna, :], one.values)


def test_covariance_spectrum_rank_one_has_effective_rank_one(
    profile: AcquisitionProfile,
) -> None:
    """A rank-1 signal has a single nonzero eigenvalue, so effective rank ~ 1."""

    rng = np.random.default_rng(1)
    n, d = 200, 6
    values = rng.standard_normal((n, 1)) * rng.standard_normal((1, d))  # rank one
    layout = Layout(
        axes=(Axis.dynamic(AxisName.TIME, unit="s"), Axis.sized(AxisName.FEATURE, d)),
        representation=Representation.FEATURE_VECTOR,
        values=ValueKind.REAL,
    )
    signal = create_signal(values, np.arange(n) / 1000.0, layout)
    compiled = (
        Pipeline()
        .then(CovarianceSpectrum(window_size=30, shrinkage=0.0))
        .compile(profile, inlet=layout)
    )
    out = compiled.pour(signal).single()
    assert np.allclose(out.values[:, 1], 1.0, atol=1e-6)  # effective_rank column


@pytest.mark.parametrize("chunk", [17, 64])
def test_covariance_spectrum_stream_equals_batch(
    profile: AcquisitionProfile, chunk: int
) -> None:
    """The windowed statistics stream identically to batch."""

    signal = _feature_signal(n=200, d=8)
    compiled = (
        Pipeline()
        .then(CovarianceSpectrum(window_size=20))
        .compile(profile, inlet=signal.layout)
    )
    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, chunk)
    assert batch.values.shape == streamed.values.shape
    assert np.allclose(batch.values, streamed.values)
    assert np.allclose(batch.times, streamed.times)


@pytest.mark.parametrize("chunk", [13, 50])
def test_local_pca_bias_stream_equals_batch(
    profile: AcquisitionProfile, chunk: int
) -> None:
    """Rotation-bias statistics reduce to 3 columns and stream identically."""

    signal = _feature_signal(n=200, d=6)
    compiled = (
        Pipeline()
        .then(LocalPcaBias(window_size=25))
        .compile(profile, inlet=signal.layout)
    )
    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, chunk)
    assert batch.layout.axis(AxisName.FEATURE).size == 3
    assert batch.values.shape == streamed.values.shape
    assert np.allclose(batch.values, streamed.values)
    assert np.all(np.isfinite(batch.values))
    assert np.all(batch.values[:, 2] >= 0.0)  # median step size is non-negative


def test_local_pca_bias_needs_two_features(profile: AcquisitionProfile) -> None:
    """A single feature dimension cannot support a 2-D PCA."""

    signal = _feature_signal(n=60, d=1)
    with pytest.raises(CompileError):
        Pipeline().then(LocalPcaBias(window_size=20)).compile(
            profile, inlet=signal.layout
        )
