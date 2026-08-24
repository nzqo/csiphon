"""The raw layout always carries the structural axes, sized to the setup.

Receiver, transmit antenna, and receive antenna are *structural*: a capture
always has at least one of each, so the raw layout keeps all three even at size
one. Only the subcarrier count and the domain axes a step later adds vary in
presence.
"""

from __future__ import annotations

import numpy as np

from csiphon import AcquisitionProfile


def _layout(**counts: int) -> str:
    profile = AcquisitionProfile(subcarrier_indices=tuple(range(52)), **counts)
    return profile.raw_csi_layout().describe_axes()


def test_single_antenna_single_device_still_carries_the_structural_axes() -> None:
    """A plain capture keeps receiver/tx/rx at size one -- one is not absent."""

    assert _layout() == (
        "time[?], receiver[1], tx_antenna[1], rx_antenna[1], subcarrier[52]"
    )


def test_antenna_array_sizes_the_rx_axis() -> None:
    """A 3-antenna array grows rx_antenna to 3; receiver and tx stay at one."""

    assert _layout(n_rx_antennas=3) == (
        "time[?], receiver[1], tx_antenna[1], rx_antenna[3], subcarrier[52]"
    )


def test_mimo_link_sizes_both_antenna_axes() -> None:
    """A 4 x 4 link sizes the transmit and receive antenna axes to four."""

    assert _layout(n_tx_antennas=4, n_rx_antennas=4) == (
        "time[?], receiver[1], tx_antenna[4], rx_antenna[4], subcarrier[52]"
    )


def test_three_devices_each_four_by_four() -> None:
    """Three 4 x 4 receivers give receiver, tx_antenna, rx_antenna, subcarrier."""

    assert _layout(n_receivers=3, n_tx_antennas=4, n_rx_antennas=4) == (
        "time[?], receiver[3], tx_antenna[4], rx_antenna[4], subcarrier[52]"
    )


def test_raw_signal_inserts_the_singleton_structural_axes() -> None:
    """A compact (time, rx, subcarrier) array gains the singleton receiver/tx axes."""

    profile = AcquisitionProfile(subcarrier_indices=tuple(range(52)), n_rx_antennas=3)
    compact = np.zeros((10, 3, 52), dtype=np.complex128)  # no receiver / tx axis
    signal = profile.raw_signal(compact, np.arange(10) / 1000.0)

    assert signal.layout.describe_axes() == (
        "time[?], receiver[1], tx_antenna[1], rx_antenna[3], subcarrier[52]"
    )
    assert signal.values.shape == (10, 1, 1, 3, 52)


def test_raw_signal_accepts_the_already_full_array() -> None:
    """An array that already carries every structural axis passes through unchanged."""

    profile = AcquisitionProfile(subcarrier_indices=tuple(range(52)), n_rx_antennas=3)
    full = np.zeros((10, 1, 1, 3, 52), dtype=np.complex128)
    signal = profile.raw_signal(full, np.arange(10) / 1000.0)

    assert signal.values.shape == (10, 1, 1, 3, 52)
