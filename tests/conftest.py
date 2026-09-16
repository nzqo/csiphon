"""Shared fixtures and helpers for the csiphon test suite."""
# One fixture requesting another by name (raw_signal(profile)) is the pytest
# idiom, which pylint reads as shadowing.
# pylint: disable=redefined-outer-name

import dataclasses
import inspect

import numpy as np
import pytest

import csiphon.steps as steps_package
from csiphon import AcquisitionProfile, AxisName, Signal, Siphon
from csiphon.pipeline import concat_signals
from csiphon.pipeline.step import Step
from csiphon.steps import FoldAxes

# The channel axes every raw capture carries. The 2-D-only steps (STFT, SST, PCA)
# want a plain (time, feature) matrix, so their tests fold all of these -- plus the
# relevant domain axis -- into one feature axis via `fold_channels_into_feature`.
CHANNEL_AXES = (AxisName.RECEIVER, AxisName.TX_ANTENNA, AxisName.RX_ANTENNA)


def fold_channels_into_feature(domain: AxisName = AxisName.SUBCARRIER) -> FoldAxes:
    """Fold every channel axis plus `domain` into one feature axis (time, feature)."""

    return FoldAxes(axes=(*CHANNEL_AXES, domain))


def concrete_steps() -> list[type[Step]]:
    """Every publicly exported, instantiable step class.

    Enumerating from the package's `__all__` gives the canonical class objects;
    walking `Step.__subclasses__` would also surface the transient pre-slots
    classes that `@dataclass(slots=True)` leaves behind until garbage collection.
    """

    found: list[type[Step]] = []
    for name in steps_package.__all__:
        obj = getattr(steps_package, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Step)
            and not inspect.isabstract(obj)
            and dataclasses.is_dataclass(obj)
        ):
            found.append(obj)
    return found


@pytest.fixture
def profile() -> AcquisitionProfile:
    """A small 3-antenna / 52-subcarrier profile at 1 kHz."""

    return AcquisitionProfile(
        n_rx_antennas=3,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=1000.0,
    )


@pytest.fixture
def raw_signal(profile: AcquisitionProfile) -> Signal:
    """A random complex raw-CSI recording of 800 samples."""

    rate = profile.sampling_rate_hz or 1000.0
    rng = np.random.default_rng(1234)
    shape = (800, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    times = np.arange(shape[0]) / rate
    # raw_signal inserts the singleton receiver/tx axes to match the full layout.
    return profile.raw_signal(csi, times)


def stream_in_chunks(compiled: Siphon, signal: Signal, chunk: int) -> Signal:
    """Feed `signal` through a Stream session in fixed-size chunks."""

    stream = compiled.stream()
    outputs = []
    for start in range(0, signal.n_samples, chunk):
        piece = Signal(
            values=signal.values[start : start + chunk],
            times=signal.times[start : start + chunk],
            layout=signal.layout,
        )
        outputs.append(stream.flow(piece).single())
    outputs.append(stream.flush().single())
    return concat_signals(outputs, compiled.outlet_layout)
