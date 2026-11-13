"""SparseFitSanitize cleans the per-packet phase offsets and keeps the motion."""

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon import (
    AcquisitionProfile,
    Axis,
    AxisName,
    Layout,
    Pipeline,
    Signal,
    create_signal,
)
from csiphon.core.errors import CompileError
from csiphon.steps import SparseFitSanitize

# A 40 MHz layout: 114 used subcarriers with a gap around the DC subcarrier.
_INDICES = tuple(range(-58, -1)) + tuple(range(2, 59))
_FREQUENCIES_HZ = np.asarray(_INDICES) * 312.5e3
_RATE_HZ = 100.0
_PACKETS = 200


def _profile(n_rx_antennas: int = 1) -> AcquisitionProfile:
    return AcquisitionProfile(
        subcarrier_indices=_INDICES,
        n_rx_antennas=n_rx_antennas,
        sampling_rate_hz=_RATE_HZ,
    )


def _path(delay_s: float, amplitude: complex = 1.0) -> np.ndarray:
    """What one path looks like across the subcarriers, as (1, subcarriers)."""

    return amplitude * np.exp(-2j * np.pi * _FREQUENCIES_HZ * delay_s)[None, :]


def _with_random_offsets(csi: np.ndarray, seed: int = 0) -> np.ndarray:
    """Give every packet its own random phase, like the offsets of real hardware."""

    offsets = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, len(csi))
    return csi * np.exp(1j * offsets)[:, None]


def _clean(csi: np.ndarray, profile: AcquisitionProfile | None = None) -> np.ndarray:
    """Run the step on (packets, subcarriers) CSI and return the same shape."""

    profile = profile or _profile()
    times = np.arange(len(csi)) / _RATE_HZ
    signal = profile.raw_signal(csi[:, None, :], times)
    compiled = Pipeline().then(SparseFitSanitize()).compile(profile)
    return compiled.pour(signal).single().values.reshape(csi.shape)


def test_one_path_comes_out_with_flat_phase_whatever_the_packet_offset() -> None:
    """With a single path, referencing it to itself leaves no phase at all. Every
    packet comes out flat and equal, although each went in with its own offset."""

    csi = _with_random_offsets(np.repeat(_path(1.0e-7), _PACKETS, axis=0))

    cleaned = _clean(csi)

    assert np.abs(np.angle(cleaned)).max() < 0.05
    assert np.abs(cleaned - cleaned[0]).max() < 0.1


def test_motion_of_a_second_path_keeps_its_doppler() -> None:
    """A weak path whose phase turns at 5 Hz, next to a static strong path, still
    shows that 5 Hz in the cleaned CSI. The random per-packet offsets are gone."""

    doppler_hz = 5.0
    times = np.arange(_PACKETS) / _RATE_HZ
    moving = np.exp(2j * np.pi * doppler_hz * times)[:, None]
    csi = _path(1.0e-7) + _path(3.0e-7, 0.4) * moving

    cleaned = _clean(_with_random_offsets(csi))

    # Look at one subcarrier over time, without its constant part.
    one_subcarrier = cleaned[:, 57]
    spectrum = np.abs(np.fft.fft(one_subcarrier - one_subcarrier.mean()))
    frequencies = np.fft.fftfreq(_PACKETS, d=1.0 / _RATE_HZ)
    assert frequencies[spectrum[: _PACKETS // 2].argmax()] == pytest.approx(doppler_hz)


def test_packets_of_zeros_stay_zeros() -> None:
    """A packet with no signal has no paths. It stays zero and does not become NaN.
    Guards against dividing by a mean magnitude of zero."""

    cleaned = _clean(np.zeros((5, len(_INDICES)), dtype=complex))

    assert np.array_equal(cleaned, np.zeros_like(cleaned))


def test_antennas_are_cleaned_independently() -> None:
    """Every vector of subcarriers is cleaned on its own, so two antennas together
    give the same as each one alone."""

    first = _with_random_offsets(np.repeat(_path(1.0e-7), 20, axis=0), seed=2)
    second = _with_random_offsets(np.repeat(_path(2.0e-7), 20, axis=0), seed=3)
    profile = _profile(n_rx_antennas=2)
    times = np.arange(20) / _RATE_HZ
    both = profile.raw_signal(np.stack([first, second], axis=1), times)

    together = Pipeline().then(SparseFitSanitize()).compile(profile).pour(both).single()

    values = together.values.reshape(20, 2, len(_INDICES))
    np.testing.assert_allclose(values[:, 0], _clean(first), atol=1e-9)
    np.testing.assert_allclose(values[:, 1], _clean(second), atol=1e-9)


def test_the_subcarrier_axis_may_be_anywhere() -> None:
    """The step finds the subcarrier axis by name, so it also works when that axis
    is not the last one."""

    csi = _with_random_offsets(np.repeat(_path(1.0e-7), 20, axis=0))
    layout = _profile().raw_csi_layout()
    time_axis, subcarrier_axis = layout.axes[0], layout.axes[-1]
    custom = Layout(
        (time_axis, subcarrier_axis, Axis.sized(AxisName.RX_ANTENNA, 1)),
        layout.representation,
        layout.values,
    )
    signal = create_signal(csi[:, :, None], np.arange(20) / _RATE_HZ, custom)

    out = (
        Pipeline()
        .then(SparseFitSanitize())
        .compile(_profile(), inlet=custom)
        .pour(signal)
        .single()
    )

    np.testing.assert_allclose(out.values[:, :, 0], _clean(csi), atol=1e-9)


def test_streaming_matches_batch() -> None:
    """Each packet is cleaned alone, so the chunk size cannot change the result."""

    csi = _with_random_offsets(np.repeat(_path(1.0e-7), 30, axis=0))
    profile = _profile()
    signal: Signal = profile.raw_signal(csi[:, None, :], np.arange(30) / _RATE_HZ)
    compiled = Pipeline().then(SparseFitSanitize()).compile(profile)

    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, 7)

    np.testing.assert_allclose(streamed.values, batch.values, atol=1e-9)


@pytest.mark.parametrize(
    "step",
    [
        SparseFitSanitize(delay_range_s=(0.5e-6, -0.3e-6)),
        SparseFitSanitize(coarse_step_s=0.0),
        SparseFitSanitize(fine_step_s=-1e-9),
        SparseFitSanitize(fit_every=0),
        SparseFitSanitize(iterations=0),
    ],
    ids=["range", "coarse-step", "fine-step", "fit-every", "iterations"],
)
def test_unusable_parameters_are_rejected_when_compiling(step) -> None:
    """These settings would give an empty or endless delay grid, or no fit at all."""

    with pytest.raises(CompileError, match="sparse-fit-sanitize"):
        Pipeline().then(step).compile(_profile())


def test_a_subcarrier_axis_without_coordinates_is_rejected() -> None:
    """Without the indices the step cannot tell which frequencies the data has."""

    profile = _profile()
    layout = profile.raw_csi_layout().replace_axis(
        AxisName.SUBCARRIER, Axis.sized(AxisName.SUBCARRIER, len(_INDICES))
    )

    with pytest.raises(CompileError, match="no coordinates"):
        Pipeline().then(SparseFitSanitize()).compile(profile, inlet=layout)
