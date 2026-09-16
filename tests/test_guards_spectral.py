"""Negative tests for step input-validation guards.

Every guard lives in a step's `output_layout`, so a bad parameter is rejected at
compile time: the `LayoutError` surfaces as a `CompileError` naming the step's
`spec.name`. Each test builds a minimal *valid* input (right value kind and axes)
and flips one parameter out of range, asserting the compile fails on that step.
"""

import pytest

from csiphon import (
    AcquisitionProfile,
    Axis,
    AxisName,
    Layout,
    Pipeline,
    Representation,
    ValueKind,
)
from csiphon.core import CompileError
from csiphon.steps import (
    AxisReference,
    FoldAxes,
    LinearPhaseCorrection,
    Multitaper,
    RunningMeanSubtract,
    SynchrosqueezedPower,
    ToDecibels,
    WindowedFFTPower,
)


def _feature_layout(d: int = 6) -> Layout:
    """A real `(time, feature)` layout for the time-frequency / scaling steps."""

    return Layout(
        axes=(Axis.dynamic(AxisName.TIME, unit="s"), Axis.sized(AxisName.FEATURE, d)),
        representation=Representation.FEATURE_VECTOR,
        values=ValueKind.REAL,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_s": 0.0},
        {"hop_s": 0.0},
        {"band_hz": 0.0},
    ],
)
def test_windowed_fft_power_rejects_nonpositive_geometry(
    profile: AcquisitionProfile, kwargs: dict[str, float]
) -> None:
    """A zero window/hop/band fails at compile, naming the step."""

    with pytest.raises(CompileError, match="windowed-fft-power"):
        Pipeline().then(WindowedFFTPower(**kwargs)).compile(
            profile, inlet=_feature_layout()
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_s": 0.0},
        {"hop_s": 0.0},
        {"band_hz": 0.0},
        {"time_bandwidth": 0.0},
        {"num_tapers": 0},
    ],
)
def test_multitaper_rejects_nonpositive_parameters(
    profile: AcquisitionProfile, kwargs: dict[str, float]
) -> None:
    """Non-positive window/hop/band/bandwidth or a sub-one taper count fails."""

    with pytest.raises(CompileError, match="multitaper-power"):
        Pipeline().then(Multitaper(**kwargs)).compile(profile, inlet=_feature_layout())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voices_per_octave": 0},
        {"block_size": 0},
    ],
)
def test_synchrosqueezed_rejects_bad_parameters(
    profile: AcquisitionProfile, kwargs: dict[str, int]
) -> None:
    """A sub-one voice count or block size fails at compile.

    `block_size=0` would otherwise spin the block-local transform in an
    infinite loop, so this guard is a real bug fix, not only hygiene.
    """

    with pytest.raises(CompileError, match="synchrosqueezed-power"):
        Pipeline().then(SynchrosqueezedPower(**kwargs)).compile(
            profile, inlet=_feature_layout()
        )


@pytest.mark.parametrize("reference", [0.0, -1.0])
def test_to_decibels_rejects_nonpositive_reference(
    profile: AcquisitionProfile, reference: float
) -> None:
    """A zero or negative 0 dB reference fails at compile."""

    with pytest.raises(CompileError, match="to-decibels"):
        Pipeline().then(ToDecibels(reference=reference)).compile(
            profile, inlet=_feature_layout()
        )


@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_running_mean_subtract_rejects_alpha_out_of_range(
    profile: AcquisitionProfile, alpha: float
) -> None:
    """An EWMA factor outside (0, 1] fails at compile."""

    with pytest.raises(CompileError, match="running-mean-subtract"):
        Pipeline().then(RunningMeanSubtract(alpha=alpha)).compile(
            profile, inlet=_feature_layout()
        )


@pytest.mark.parametrize("keep_quantile", [1.5, -0.1])
def test_linear_phase_correction_rejects_bad_keep_quantile(
    profile: AcquisitionProfile, keep_quantile: float
) -> None:
    """A keep_quantile outside [0, 1] fails at compile.

    Left unguarded it would crash `np.quantile` at run time instead.
    """

    with pytest.raises(CompileError, match="linear-phase-correction"):
        Pipeline().then(LinearPhaseCorrection(keep_quantile=keep_quantile)).compile(
            profile
        )


def test_linear_phase_correction_rejects_a_single_position_axis() -> None:
    """A phase ramp needs two points; a single-subcarrier input fails at compile."""

    one_subcarrier = AcquisitionProfile(n_rx_antennas=1, subcarrier_indices=(0,))
    with pytest.raises(CompileError, match="two positions"):
        Pipeline().then(LinearPhaseCorrection()).compile(one_subcarrier)


def test_axis_reference_rejects_out_of_range_index(
    profile: AcquisitionProfile,
) -> None:
    """A FIXED reference index past the axis end fails at compile."""

    with pytest.raises(CompileError, match="axis-reference"):
        Pipeline().then(AxisReference(index=1000)).compile(profile)


def test_fold_axes_rejects_fewer_than_two_axes(
    profile: AcquisitionProfile,
) -> None:
    """Folding needs at least two axes."""

    with pytest.raises(CompileError, match="fold-axes"):
        Pipeline().then(FoldAxes(axes=(AxisName.SUBCARRIER,))).compile(profile)


def test_fold_axes_rejects_duplicate_axes(
    profile: AcquisitionProfile,
) -> None:
    """Folding the same axis twice is rejected."""

    with pytest.raises(CompileError, match="fold-axes"):
        Pipeline().then(
            FoldAxes(axes=(AxisName.SUBCARRIER, AxisName.SUBCARRIER))
        ).compile(profile)
