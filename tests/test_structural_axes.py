"""Structural axes (receiver / tx / rx) are always present, so they always merge.

These are the regression tests for the squeeze flaw: the raw layout used to drop
size-1 structural axes, which made a single-antenna or single-receiver capture
look as if it had *no* such axis -- so it could not be concatenated with another
capture along that axis. A size-1 structural axis means "one", not "absent", and
these tests pin that down at the level that actually broke: merging.
"""

from __future__ import annotations

import numpy as np

from csiphon import (
    AcquisitionProfile,
    AxisName,
    Concatenate,
    Hold,
    Pipeline,
    Signal,
    create_signal,
)
from csiphon.steps import Magnitude


def _capture(profile: AcquisitionProfile, seed: int, n: int = 64) -> Signal:
    """A random complex raw-CSI recording for `profile`, on its full raw layout."""

    layout = profile.raw_csi_layout()
    rng = np.random.default_rng(seed)
    shape = (n, *[axis.size for axis in layout.axes[1:]])
    values = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return create_signal(values, np.arange(n) / 1000.0, layout)


def test_single_antenna_single_receiver_capture_has_every_structural_axis() -> None:
    """The plainest possible capture still carries receiver / tx / rx, each at one."""

    layout = AcquisitionProfile(subcarrier_indices=tuple(range(4))).raw_csi_layout()
    sizes = {axis.name: axis.size for axis in layout.axes}
    assert sizes[AxisName.RECEIVER] == 1
    assert sizes[AxisName.TX_ANTENNA] == 1
    assert sizes[AxisName.RX_ANTENNA] == 1


def test_two_single_receiver_captures_concatenate_onto_the_receiver_axis() -> None:
    """Two one-receiver captures merge into a two-receiver signal.

    This is the case the squeeze broke outright: each capture's lone receiver used
    to vanish, so there was no receiver axis to join, and "1 + 1 receivers" could
    not be expressed at all.
    """

    device = AcquisitionProfile(
        subcarrier_indices=tuple(range(4)), sampling_rate_hz=1000.0
    )
    out = (
        Pipeline.from_inlets("a", "b")
        .merge(using=Concatenate(axis=AxisName.RECEIVER), align=Hold())
        .then(Magnitude())
        .compile(a=device, b=device)
        .pour(a=_capture(device, 0), b=_capture(device, 1))
        .single()
    )
    assert out.layout.axis(AxisName.RECEIVER).size == 2


def test_single_antenna_receiver_concatenates_with_a_multi_antenna_one() -> None:
    """A 1-antenna capture joins a 3-antenna one on rx_antenna -> 4 antennas.

    The exact shape that failed before: the single-antenna device had no rx_antenna
    axis to concatenate onto.
    """

    one = AcquisitionProfile(
        n_rx_antennas=1, subcarrier_indices=tuple(range(4)), sampling_rate_hz=1000.0
    )
    three = AcquisitionProfile(
        n_rx_antennas=3, subcarrier_indices=tuple(range(4)), sampling_rate_hz=1000.0
    )
    out = (
        Pipeline.from_inlets("one", "three")
        .merge(using=Concatenate(axis=AxisName.RX_ANTENNA), align=Hold())
        .then(Magnitude())
        .compile(one=one, three=three)
        .pour(one=_capture(one, 0), three=_capture(three, 1))
        .single()
    )
    assert out.layout.axis(AxisName.RX_ANTENNA).size == 4  # 1 + 3


def test_raw_signal_fills_in_whichever_structural_axes_are_omitted() -> None:
    """Full, fully-compact, and partially-compact arrays all reach the full layout.

    The caller may spell every structural axis out, omit all the size-1 ones, or omit
    only some -- raw_signal inserts whatever is missing, since moving size-1 axes never
    disturbs the data.
    """

    profile = AcquisitionProfile(n_rx_antennas=2, subcarrier_indices=tuple(range(4)))
    times = np.arange(5) / 1000.0
    forms = {
        "full": np.zeros((5, 1, 1, 2, 4), dtype=np.complex128),
        "compact": np.zeros((5, 2, 4), dtype=np.complex128),  # (time, rx, subcarrier)
        "partial": np.zeros((5, 1, 2, 4), dtype=np.complex128),  # keeps a size-1 axis
    }
    signals = {name: profile.raw_signal(array, times) for name, array in forms.items()}

    for signal in signals.values():
        assert signal.values.shape == (5, 1, 1, 2, 4)
    assert signals["compact"].layout.axis_names == signals["full"].layout.axis_names
