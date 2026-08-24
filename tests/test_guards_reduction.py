"""Negative tests: input-validation guards reject invalid step parameters.

Each guard lives in a step's `output_layout` and raises `LayoutError`, which the
compiler wraps as a `CompileError` naming the step's spec name. So every test
builds a minimal valid input layout, then a bad parameter, and asserts the
compile fails with the step name in the message. Compilation alone triggers the
guard -- no data is poured -- so the covariance/local-pca hop_size=0 cases can
never reach the runtime loop they would otherwise hang.
"""
# A test requesting a fixture by its name (step(profile)) is the pytest idiom, which
# pylint reads as shadowing.
# pylint: disable=redefined-outer-name

from __future__ import annotations

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
    CovarianceSpectrum,
    DelayTaps,
    DyadicFrequencyBands,
    FixedSizeWindowSum,
    Hold,
    LocalPcaBias,
    PrincipalComponents,
    Resample,
    RobustPca,
    SelectAxis,
)


@pytest.fixture
def profile() -> AcquisitionProfile:
    """A small profile with a nominal sampling rate for the windowed steps."""

    return AcquisitionProfile(
        n_rx_antennas=1, subcarrier_indices=tuple(range(8)), sampling_rate_hz=1000.0
    )


def _feature_layout(d: int = 6) -> Layout:
    """A `(time, feature[d])` real layout for the statistics/RPCA steps."""

    return Layout(
        axes=(Axis.dynamic(AxisName.TIME, unit="s"), Axis.sized(AxisName.FEATURE, d)),
        representation=Representation.FEATURE_VECTOR,
        values=ValueKind.REAL,
    )


def _subcarrier_layout(d: int = 8) -> Layout:
    """A `(time, subcarrier[d])` magnitude layout for PCA / select / interpolate."""

    return Layout(
        axes=(Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, d)),
        representation=Representation.CHANNEL_FREQUENCY_RESPONSE,
        values=ValueKind.MAGNITUDE,
    )


def _frequency_layout(n: int = 8) -> Layout:
    """A `(time, frequency[n])` power layout for the dyadic banding step."""

    return Layout(
        axes=(
            Axis.dynamic(AxisName.TIME),
            Axis.static(
                AxisName.FREQUENCY, tuple(float(i) for i in range(n)), unit="Hz"
            ),
        ),
        representation=Representation.TIME_FREQUENCY,
        values=ValueKind.POWER,
    )


def _delay_layout(d: int = 8) -> Layout:
    """A `(time, delay[d])` complex layout for the delay-taps step."""

    return Layout(
        axes=(Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.DELAY, d, unit="tap")),
        representation=Representation.DELAY_RESPONSE,
        values=ValueKind.COMPLEX,
    )


# --- FixedSizeWindowSum: window_s / hop_s must be > 0 ---


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_s": 0.0},
        {"window_s": -0.1},
        {"hop_s": 0.0},
        {"hop_s": -0.1},
    ],
)
def test_fixed_size_window_sum_rejects_nonpositive(
    profile: AcquisitionProfile, kwargs: dict[str, float]
) -> None:
    """A zero or negative window/hop length fails at compile."""

    with pytest.raises(CompileError, match="fixed-size-window-sum"):
        Pipeline().then(FixedSizeWindowSum(**kwargs)).compile(
            profile, inlet=_subcarrier_layout()
        )


# --- DyadicFrequencyBands: num_bands / first_band must be >= 1 ---


@pytest.mark.parametrize("kwargs", [{"num_bands": 0}, {"first_band": 0}])
def test_dyadic_bands_rejects_below_one(
    profile: AcquisitionProfile, kwargs: dict[str, int]
) -> None:
    """A band count or first index below one fails at compile."""

    with pytest.raises(CompileError, match="dyadic-frequency-bands"):
        Pipeline().then(DyadicFrequencyBands(**kwargs)).compile(
            profile, inlet=_frequency_layout()
        )


# --- CovarianceSpectrum: window_size / hop_size must be >= 1 ---


@pytest.mark.parametrize("kwargs", [{"window_size": 0}, {"hop_size": 0}])
def test_covariance_spectrum_rejects_below_one(
    profile: AcquisitionProfile, kwargs: dict[str, int]
) -> None:
    """A zero window or hop fails at compile.

    hop_size=0 would otherwise spin forever at runtime, so the guard firing at
    compile is the whole point -- this test returns immediately.
    """

    with pytest.raises(CompileError, match="covariance-spectrum"):
        Pipeline().then(CovarianceSpectrum(**kwargs)).compile(
            profile, inlet=_feature_layout()
        )


# --- LocalPcaBias: window_size / hop_size must be >= 1 ---


@pytest.mark.parametrize(
    "kwargs", [{"window_size": 0}, {"window_size": 1}, {"hop_size": 0}]
)
def test_local_pca_bias_rejects_tiny_window_or_hop(
    profile: AcquisitionProfile, kwargs: dict[str, int]
) -> None:
    """A sub-two window (NaN) or a zero hop (hang) fails at compile."""

    with pytest.raises(CompileError, match="local-pca-bias"):
        Pipeline().then(LocalPcaBias(**kwargs)).compile(
            profile, inlet=_feature_layout()
        )


# --- DelayTaps: num_taps >= 1 and first_tap >= 0 ---


@pytest.mark.parametrize("kwargs", [{"num_taps": 0}, {"first_tap": -1}])
def test_delay_taps_rejects_bad_geometry(
    profile: AcquisitionProfile, kwargs: dict[str, int]
) -> None:
    """A non-positive tap count or a negative first tap fails at compile."""

    with pytest.raises(CompileError, match="delay-taps"):
        Pipeline().then(DelayTaps(**kwargs)).compile(profile, inlet=_delay_layout())


# --- PrincipalComponents: n_components >= 1 ---


def test_principal_components_rejects_zero_components(
    profile: AcquisitionProfile,
) -> None:
    """Keeping zero components is meaningless and fails at compile."""

    with pytest.raises(CompileError, match="principal-components"):
        Pipeline().then(PrincipalComponents(n_components=0)).compile(
            profile, inlet=_subcarrier_layout()
        )


# --- RobustPca: first_component >= 0 and fit_max_samples >= 1 ---


@pytest.mark.parametrize("kwargs", [{"first_component": -1}, {"fit_max_samples": 0}])
def test_robust_pca_rejects_bad_params(
    profile: AcquisitionProfile, kwargs: dict[str, int]
) -> None:
    """A negative first component or a sub-one fit budget fails at compile."""

    with pytest.raises(CompileError, match="robust-pca"):
        Pipeline().then(RobustPca(**kwargs)).compile(
            profile, inlet=_feature_layout(d=16)
        )


# --- SelectAxis: indices must be in range ---


@pytest.mark.parametrize("indices", [(-1,), (0, 100)])
def test_select_axis_rejects_out_of_range(
    profile: AcquisitionProfile, indices: tuple[int, ...]
) -> None:
    """A negative index or one past the axis size fails at compile."""

    with pytest.raises(CompileError, match="select-axis"):
        Pipeline().then(SelectAxis(indices=indices)).compile(
            profile, inlet=_subcarrier_layout(d=8)
        )


# --- Resample: rate_hz must be > 0 ---


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_resample_rejects_nonpositive_rate(
    profile: AcquisitionProfile, rate: float
) -> None:
    """A zero or negative target rate fails at compile (avoids a numpy crash)."""

    with pytest.raises(CompileError, match="resample"):
        Pipeline().then(Resample(rate_hz=rate, fill=Hold())).compile(
            profile, inlet=_subcarrier_layout()
        )
