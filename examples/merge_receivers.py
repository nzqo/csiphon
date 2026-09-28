"""Merge several receivers into one pipeline, aligned by timestamps or packet numbers.

Each receiver enters through its own named inlet. A merge lines the receivers up
in time and combines them, and from there the pipeline treats them as one signal:

1. two receivers on different clocks and rates, aligned on timestamps with `Hold`
   and concatenated on the antenna axis,
2. three receivers of one capture, aligned on packet (sequence) numbers, so clock
   drift between them does not matter.

`describe()` draws each receiver as its own source card.

Runs with the base (numpy-only) install, no optional extras required.
"""

# Examples share small profile/signal setup blocks by design.
# pylint: disable=duplicate-code
from dataclasses import replace

import numpy as np

from csiphon import (
    AcquisitionProfile,
    AxisName,
    Concatenate,
    Hold,
    Outlets,
    Pipeline,
    Sequence,
    Signal,
    Stack,
    create_signal,
)
from csiphon.steps import Magnitude, WindowedSlope, WindowedVariance


def print_outlets(outlets: Outlets) -> None:
    """Print each outlet's name and axes."""

    print("\noutlets from pour:")
    for name, out in outlets.items():
        print(f"  {name:10s} {out.layout.describe_axes()}")


def merge_on_timestamps() -> None:
    """Merge a 3-antenna receiver at 1 kHz with a 1-antenna receiver at 500 Hz."""

    rx0_profile = AcquisitionProfile(
        n_rx_antennas=3,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=1000.0,
    )
    rx1_profile = AcquisitionProfile(
        n_rx_antennas=1,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=500.0,
    )

    # Hold carries rx1's latest value onto rx0's timeline, then the receivers
    # concatenate into one signal with four antennas.
    siphon = (
        Pipeline.from_inlets("rx0", "rx1")
        .merge(using=Concatenate(axis=AxisName.RX_ANTENNA), align=Hold())
        .then(Magnitude())
        .then(WindowedVariance(win_size_s=0.05))
        .compile(rx0=rx0_profile, rx1=rx1_profile)
    )
    print("Two receivers on different clocks, aligned on timestamps:\n")
    print(siphon.describe())

    # A single-antenna receiver's compact array is just (time, subcarrier);
    # raw_signal inserts the singleton receiver / tx / rx axes.
    rng = np.random.default_rng(0)
    rx0_shape = (2000, rx0_profile.n_rx_antennas, rx0_profile.n_subcarriers)
    rx1_shape = (1000, rx1_profile.n_subcarriers)
    rx0_csi = rng.standard_normal(rx0_shape) + 1j * rng.standard_normal(rx0_shape)
    rx1_csi = rng.standard_normal(rx1_shape) + 1j * rng.standard_normal(rx1_shape)
    rx0 = rx0_profile.raw_signal(rx0_csi, np.arange(2000) / 1000.0)
    rx1 = rx1_profile.raw_signal(rx1_csi, np.arange(1000) / 500.0)

    print_outlets(siphon.pour(rx0=rx0, rx1=rx1))


def merge_on_sequence_numbers() -> None:
    """Merge three receivers of one capture on their packet numbers."""

    rate = 1000.0
    # One capture with three receivers; each inlet is a single-receiver slice of it.
    profile = AcquisitionProfile(
        n_rx_antennas=2,
        subcarrier_indices=tuple(range(52)),
        n_receivers=3,
        sampling_rate_hz=rate,
    )
    inlet = replace(profile, n_receivers=1).raw_csi_layout()

    siphon = (
        Pipeline.from_inlets("rx0", "rx1", "rx2")
        # Each inlet already carries a size-1 receiver axis, so Concatenate grows
        # that axis (Stack, which adds a new axis, would reject it).
        .merge(
            using=Concatenate(axis=AxisName.RECEIVER),
            align=Hold(on=Sequence(period=4096)),
        )
        .then(Magnitude())
        # A normal fork/merge on the combined multi-receiver signal.
        .branch(
            variance=Pipeline().then(WindowedVariance(win_size_s=0.05)),
            slope=Pipeline().then(WindowedSlope()),
        )
        .merge(using=Stack(into=AxisName.FEATURE))
    ).compile(profile)

    print("\nThree receivers, aligned on sequence numbers:\n")
    print(siphon.describe())

    rng = np.random.default_rng(1)

    def receiver(drift: float) -> Signal:
        """One receiver's raw CSI, carrying wrapped packet numbers, slightly drifted."""

        length = 2000
        packet = np.arange(length, dtype=float)
        shape = (length, *[axis.size for axis in inlet.axes[1:]])
        csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        return create_signal(csi, packet / rate + drift, inlet, sequence=packet % 4096)

    print_outlets(
        siphon.pour(rx0=receiver(0.0), rx1=receiver(2e-4), rx2=receiver(-1e-4))
    )


def main() -> None:
    """Merge receivers on timestamps, then on sequence numbers."""

    merge_on_timestamps()
    merge_on_sequence_numbers()


if __name__ == "__main__":
    main()
