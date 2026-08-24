"""Calibration steps: axis-reference combinations and phase-ramp removal."""

from __future__ import annotations

import numpy as np

from csiphon import AcquisitionProfile, AxisName, Pipeline, Signal
from csiphon.core.semantics import Representation
from csiphon.steps.calibration import (
    AxisReference,
    Combine,
    LinearPhaseCorrection,
    Method,
    Reference,
)


def _run(profile: AcquisitionProfile, signal: Signal, step: object) -> Signal:
    """Compile a single step against the raw layout and run it in batch."""

    return Pipeline().then(step).compile(profile).pour(signal).single()


def test_axis_reference_divide_fixed_matches_numpy(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """combine=DIVIDE, mode=FIXED is value / reference along the axis."""

    out = _run(profile, raw_signal, AxisReference(combine=Combine.DIVIDE, index=0))
    position = raw_signal.layout.axis_position(AxisName.SUBCARRIER)
    reference = np.take(raw_signal.values, [0], axis=position)
    assert np.allclose(out.values, raw_signal.values / (reference + 1e-12))
    assert out.layout.representation == Representation.RATIO


def test_axis_reference_conjugate_adjacent_matches_numpy(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """combine=CONJUGATE, mode=ADJACENT is h[k] * conj(h[k-1]), one position shorter."""

    step = AxisReference(combine=Combine.CONJUGATE, mode=Reference.ADJACENT)
    out = _run(profile, raw_signal, step)
    position = raw_signal.layout.axis_position(AxisName.SUBCARRIER)
    leading = np.take(raw_signal.values, range(1, profile.n_subcarriers), axis=position)
    trailing = np.take(
        raw_signal.values, range(0, profile.n_subcarriers - 1), axis=position
    )
    assert np.allclose(out.values, leading * np.conj(trailing))
    assert out.layout.axis(AxisName.SUBCARRIER).size == profile.n_subcarriers - 1
    assert out.layout.representation == Representation.CROSS_SPECTRUM


def test_axis_reference_drop_reference_shrinks(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """Dropping the reference removes exactly one position."""

    step = AxisReference(combine=Combine.CONJUGATE, index=1, drop_reference=True)
    out = _run(profile, raw_signal, step)
    assert out.layout.axis(AxisName.SUBCARRIER).size == profile.n_subcarriers - 1


def test_axis_reference_works_on_antenna_axis(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """The same step covers the antenna axis (the former AntennaConjugate)."""

    step = AxisReference(axis=AxisName.RX_ANTENNA, combine=Combine.CONJUGATE)
    out = _run(profile, raw_signal, step)
    position = raw_signal.layout.axis_position(AxisName.RX_ANTENNA)
    reference = np.take(raw_signal.values, [0], axis=position)
    assert np.allclose(out.values, raw_signal.values * np.conj(reference))


def test_linear_phase_correction_flattens_a_pure_ramp(
    profile: AcquisitionProfile,
) -> None:
    """A pure linear ramp exp(j*(a*x + b)) collapses to a flat phase."""

    x = np.asarray(profile.subcarrier_indices, dtype=float)
    frame = np.exp(1j * (0.2 * x + 0.5))
    n = 20
    values = np.broadcast_to(
        frame, (n, profile.n_rx_antennas, profile.n_subcarriers)
    ).copy()
    signal = profile.raw_signal(values, np.arange(n) / 1000.0)

    for method in (Method.CIRCULAR_DIFF, Method.WEIGHTED_LS):
        step = LinearPhaseCorrection(method=method, keep_quantile=None)
        out = _run(profile, signal, step)
        assert out.values.shape == signal.values.shape
        # The ramp is gone, so every subcarrier holds the same value.
        assert np.allclose(out.values, out.values[..., :1], atol=1e-6)
