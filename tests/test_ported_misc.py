"""Ported misc steps: Savitzky-Golay, unit phase, amp/phase stacking, robust PCA."""

# Feature-signal setup overlaps with other test modules by design.
# pylint: disable=duplicate-code
from __future__ import annotations

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
from csiphon.steps import Magnitude, RobustPca, SavitzkyGolay, StackAmpPhase, UnitPhase


def _magnitude(profile: AcquisitionProfile, n: int = 300) -> Signal:
    """A real magnitude signal for the real-valued steps."""

    compiled = Pipeline().then(Magnitude()).compile(profile)
    rng = np.random.default_rng(2)
    shape = (n, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    raw = profile.raw_signal(csi, np.arange(n) / 1000.0)
    return compiled.pour(raw).single()


def _feature_signal(n: int = 300, d: int = 8) -> Signal:
    """A real `(time, feature)` signal for robust PCA."""

    rng = np.random.default_rng(0)
    layout = Layout(
        axes=(Axis.dynamic(AxisName.TIME, unit="s"), Axis.sized(AxisName.FEATURE, d)),
        representation=Representation.FEATURE_VECTOR,
        values=ValueKind.REAL,
    )
    return create_signal(rng.standard_normal((n, d)), np.arange(n) / 1000.0, layout)


def test_savitzky_golay_matches_scipy(profile: AcquisitionProfile) -> None:
    """The step reproduces scipy's centered savgol_filter along time."""

    pytest.importorskip("scipy")
    from scipy.signal import savgol_filter  # pylint: disable=import-outside-toplevel

    signal = _magnitude(profile)
    compiled = (
        Pipeline()
        .then(SavitzkyGolay(window_length=11, polyorder=2))
        .compile(profile, inlet=signal.layout)
    )
    out = compiled.pour(signal).single()
    expected = savgol_filter(signal.values, 11, 2, axis=0, mode="interp")
    assert np.allclose(out.values, expected)


def test_unit_phase_is_unit_modulus_and_keeps_phase(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """`exp(j*angle(z))` has magnitude one and the same angle as the input."""

    out = Pipeline().then(UnitPhase()).compile(profile).pour(raw_signal).single()
    assert np.allclose(np.abs(out.values), 1.0)
    assert np.allclose(np.angle(out.values), np.angle(raw_signal.values))


@pytest.mark.parametrize("chunk", [23, 128])
def test_stack_amp_phase_doubles_axis_and_streams(
    profile: AcquisitionProfile, raw_signal: Signal, chunk: int
) -> None:
    """Stacking doubles the subcarrier axis into [amplitude | phase] and streams."""

    compiled = Pipeline().then(StackAmpPhase()).compile(profile)
    out = compiled.pour(raw_signal).single()
    assert out.layout.axis(AxisName.SUBCARRIER).size == 2 * profile.n_subcarriers
    assert out.layout.values == ValueKind.REAL

    half = profile.n_subcarriers
    position = out.layout.axis_position(AxisName.SUBCARRIER)

    # First half is the amplitude, second half the phase, of the same input.
    amplitude = np.take(out.values, range(half), axis=position)
    assert np.allclose(amplitude, np.abs(raw_signal.values))
    phase = np.take(out.values, range(half, 2 * half), axis=position)
    assert np.allclose(phase, np.angle(raw_signal.values))

    streamed = stream_in_chunks(compiled, raw_signal, chunk)
    assert np.allclose(out.values, streamed.values)


def test_robust_pca_reduces_and_separates_low_rank(profile: AcquisitionProfile) -> None:
    """A low-rank signal plus sparse spikes projects to the requested components."""

    rng = np.random.default_rng(7)
    n, d, rank = 300, 8, 3
    low_rank = rng.standard_normal((n, rank)) @ rng.standard_normal((rank, d))
    spikes = np.zeros((n, d))
    spikes[rng.integers(0, n, 15), rng.integers(0, d, 15)] = 25.0
    layout = Layout(
        axes=(Axis.dynamic(AxisName.TIME, unit="s"), Axis.sized(AxisName.FEATURE, d)),
        representation=Representation.FEATURE_VECTOR,
        values=ValueKind.REAL,
    )
    signal = create_signal(low_rank + spikes, np.arange(n) / 1000.0, layout)

    compiled = (
        Pipeline()
        .then(RobustPca(num_components=rank, max_iter=200))
        .compile(profile, inlet=layout)
    )
    out = compiled.pour(signal).single()
    assert out.layout.axis(AxisName.COMPONENT).size == rank
    assert out.values.shape == (n, rank)
    assert np.all(np.isfinite(out.values))

    # Robustness (behavioural expectation, not a benchmark): the components should
    # be learned from the clean low-rank part, ignoring the spikes. At the samples
    # with no spike the recovered scores are an uncontaminated projection, so they
    # must linearly reconstruct the true low-rank signal there with small error --
    # which only holds if the learned basis stayed clean (a plain PCA would tilt).
    clean_rows = ~np.any(spikes != 0.0, axis=1)
    scores, target = out.values[clean_rows], low_rank[clean_rows]
    coeffs, *_ = np.linalg.lstsq(scores, target, rcond=None)
    assert np.linalg.norm(scores @ coeffs - target) / np.linalg.norm(target) < 0.1


def test_robust_pca_broadcasts_over_a_channel_axis(
    profile: AcquisitionProfile,
) -> None:
    """A (time, rx_antenna, feature) input equals decomposing each antenna alone.

    RobustPca's matrix decomposition runs once per extra axis; the extra axis is a
    parallel channel, so the result must match the per-antenna decomposition exactly.
    """

    rng = np.random.default_rng(7)
    n, antennas, d, rank = 300, 2, 8, 3
    data = rng.standard_normal((n, antennas, d))
    times = np.arange(n) / 1000.0
    step = RobustPca(num_components=rank, max_iter=200)

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
    assert out_nd.values.shape == (n, antennas, rank)

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
