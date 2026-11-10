"""Delay steps build their basis from the subcarriers the signal actually carries."""

import numpy as np
import pytest

from csiphon import AcquisitionProfile, AxisName, Pipeline, Signal
from csiphon.core.errors import LayoutError
from csiphon.steps import (
    ChannelImpulseResponse,
    DelayAutocorrelation,
    Magnitude,
    SelectAxis,
)
from tests.conftest import stream_in_chunks

# A 40 MHz layout: 114 used subcarriers on a 128-point FFT grid, with a DC gap.
_INDICES = tuple(range(-58, -1)) + tuple(range(2, 59))
_KEPT_POSITIONS = tuple(range(0, 114, 2))
_KEPT_INDICES = tuple(_INDICES[position] for position in _KEPT_POSITIONS)
_RATE_HZ = 100.0


def _profile(indices: tuple[int, ...]) -> AcquisitionProfile:
    return AcquisitionProfile(
        subcarrier_indices=indices, n_rx_antennas=1, sampling_rate_hz=_RATE_HZ
    )


def _csi(n_samples: int = 50) -> np.ndarray:
    rng = np.random.default_rng(0)
    shape = (n_samples, 1, len(_INDICES))
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _select() -> SelectAxis:
    return SelectAxis(axis=AxisName.SUBCARRIER, indices=_KEPT_POSITIONS)


def _narrowed_signal(csi: np.ndarray) -> tuple[AcquisitionProfile, Signal]:
    """The same data as a profile that only ever had the kept subcarriers."""

    profile = _profile(_KEPT_INDICES)
    times = np.arange(csi.shape[0]) / _RATE_HZ
    return profile, profile.raw_signal(csi[..., list(_KEPT_POSITIONS)], times)


def _full_signal(csi: np.ndarray) -> tuple[AcquisitionProfile, Signal]:
    profile = _profile(_INDICES)
    return profile, profile.raw_signal(csi, np.arange(csi.shape[0]) / _RATE_HZ)


@pytest.mark.parametrize(
    ("first", "last"),
    [
        (Magnitude(), DelayAutocorrelation(nfft=128, num_taps=10)),
        (None, ChannelImpulseResponse(nfft=128, num_taps=10)),
    ],
    ids=["delay-autocorrelation", "channel-impulse-response"],
)
def test_reduced_subcarrier_axis_matches_narrowed_profile(first, last) -> None:
    """After SelectAxis, the output equals that of a profile narrowed to the kept
    subcarriers (same explicit nfft). Without the fix, the kernel is built for
    every profile subcarrier and the matmul fails on a shape mismatch."""

    csi = _csi()
    profile, signal = _full_signal(csi)
    pipeline = Pipeline()
    if first is not None:
        pipeline = pipeline.then(first)
    reduced = pipeline.then(_select()).then(last).compile(profile).pour(signal)

    narrow_profile, narrow_signal = _narrowed_signal(csi)
    narrow = Pipeline()
    if first is not None:
        narrow = narrow.then(first)
    expected = narrow.then(last).compile(narrow_profile).pour(narrow_signal)

    np.testing.assert_allclose(
        reduced.single().values, expected.single().values, atol=1e-10
    )


@pytest.mark.parametrize("step_cls", [DelayAutocorrelation, ChannelImpulseResponse])
def test_nfft_is_inferred_from_the_profile_not_the_kept_count(step_cls) -> None:
    """The indices live on the original channel's FFT grid (128 for 114 used
    subcarriers), so keeping 57 of them must not change the inferred nfft to 64."""

    csi = _csi()
    profile, signal = _full_signal(csi)
    kwargs = {"num_taps": 100}
    pipeline = Pipeline().then(_select())
    if step_cls is DelayAutocorrelation:
        pipeline = pipeline.then(Magnitude())
    inferred = pipeline.then(step_cls(**kwargs)).compile(profile).pour(signal).single()
    explicit = (
        pipeline.then(step_cls(nfft=128, **kwargs)).compile(profile).pour(signal)
    ).single()

    np.testing.assert_allclose(inferred.values, explicit.values, atol=1e-10)


def test_reduced_subcarrier_axis_streams_like_batch() -> None:
    """DelayAutocorrelation is BATCH_EQUIVALENT also behind a SelectAxis."""

    csi = _csi(60)
    profile, signal = _full_signal(csi)
    compiled = (
        Pipeline()
        .then(Magnitude())
        .then(_select())
        .then(DelayAutocorrelation(nfft=128, num_taps=10))
        .compile(profile)
    )

    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, 7)

    np.testing.assert_allclose(streamed.values, batch.values, atol=1e-10)


def test_axis_without_coordinates_falls_back_to_the_profile_when_sizes_match() -> None:
    """An axis that lost its coordinates but still has every profile subcarrier
    uses the profile's indices."""

    csi = _csi()
    profile, signal = _full_signal(csi)
    plain = (
        Pipeline()
        .then(Magnitude())
        .then(DelayAutocorrelation(nfft=128, num_taps=10))
        .compile(profile)
        .pour(signal)
        .single()
    )
    assert plain.values.shape[-1] == 10


def test_axis_without_coordinates_and_wrong_size_is_rejected() -> None:
    """Missing coordinates and a size that differs from the profile leave the
    subcarrier indices unknown, so compile must fail and say so."""

    from csiphon.core.axes import Axis
    from csiphon.steps.delay.delay_autocorrelation import subcarrier_indices_of

    profile = _profile(_INDICES)
    layout = profile.raw_csi_layout().replace_axis(
        AxisName.SUBCARRIER, Axis.sized(AxisName.SUBCARRIER, 57)
    )

    with pytest.raises(LayoutError, match="57"):
        subcarrier_indices_of(layout, profile)
