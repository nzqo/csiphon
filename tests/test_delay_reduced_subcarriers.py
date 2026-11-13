"""Delay steps build their basis from the subcarriers the signal actually carries."""

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon import AcquisitionProfile, Axis, AxisName, Pipeline, Signal
from csiphon.core.errors import CompileError, LayoutError
from csiphon.steps import (
    ChannelImpulseResponse,
    DelayAutocorrelation,
    Magnitude,
    SelectAxis,
    resolve_nfft,
)

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
def test_nfft_is_inferred_from_the_span_of_the_kept_indices(step_cls) -> None:
    """Every second subcarrier of a 40 MHz capture still spans -58 to 56, so the
    inferred nfft is 128, the same as when nfft is given."""

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


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        (tuple(range(-58, -1)) + tuple(range(2, 59)), 128),  # 40 MHz, full
        (tuple(range(-26, 0)) + tuple(range(1, 27)), 64),  # 20 MHz, full
        (tuple(range(-58, -1)), 64),  # a contiguous block of 57 fits 64 bins
        (tuple(range(-122, -1)) + tuple(range(2, 123)), 256),  # 80 MHz, full
    ],
    ids=["40MHz", "20MHz", "contiguous-block", "80MHz"],
)
def test_resolve_nfft_picks_the_smallest_size_above_the_index_span(
    indices: tuple[int, ...], expected: int
) -> None:
    """Indices that differ by a multiple of nfft would share a bin, so nfft must
    exceed the span of the indices."""

    assert resolve_nfft(np.asarray(indices, dtype=float)) == expected


def test_resolve_nfft_asks_for_nfft_when_the_span_is_too_large() -> None:
    """Beyond 512 bins we do not guess."""

    with pytest.raises(LayoutError, match="pass nfft explicitly"):
        resolve_nfft(np.asarray([-300.0, 300.0]))


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


@pytest.mark.parametrize("step", [DelayAutocorrelation(), ChannelImpulseResponse()])
def test_axis_without_coordinates_is_rejected(step) -> None:
    """Without coordinates the layout does not say which subcarriers it has, so
    compile fails and says what is missing."""

    profile = _profile(_INDICES)
    layout = profile.raw_csi_layout().replace_axis(
        AxisName.SUBCARRIER, Axis.sized(AxisName.SUBCARRIER, len(_INDICES))
    )
    pipeline = Pipeline()
    if isinstance(step, DelayAutocorrelation):
        pipeline = pipeline.then(Magnitude())
    pipeline = pipeline.then(step)

    with pytest.raises(CompileError, match="no coordinates"):
        pipeline.compile(profile, inlet=layout)
