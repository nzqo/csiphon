"""Batch example: build a pipeline once, run it over a whole recording.

Runs with the base (numpy-only) install, no optional extras required.
"""

# The batch and streaming examples deliberately build the same pipeline.
# pylint: disable=duplicate-code
from __future__ import annotations

import numpy as np

from csiphon import AcquisitionProfile, AxisName, Pipeline
from csiphon.steps import (
    DelayAutocorrelation,
    DelayTaps,
    FoldAxes,
    GainNormalize,
    Magnitude,
    WindowedFFTPower,
)


def main() -> None:
    """Extract Doppler-power features from a synthetic recording."""

    rate = 1000.0
    profile = AcquisitionProfile(
        n_rx_antennas=3,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=rate,
    )

    pipeline = (
        Pipeline()
        .then(Magnitude())
        .then(GainNormalize())
        .then(DelayAutocorrelation())
        .then(DelayTaps(num_taps=3, first_tap=1))
        .then(FoldAxes(axes=(AxisName.RX_ANTENNA, AxisName.DELAY)))
        .then(WindowedFFTPower(window_s=0.256, hop_s=0.05, band_hz=60.0))
    )
    compiled = pipeline.compile(profile)
    print(compiled.describe())

    rng = np.random.default_rng(0)
    shape = (5000, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    times = np.arange(shape[0]) / rate
    signal = profile.raw_signal(csi, times)

    features = compiled.pour(signal).single()
    print("\nfeatures:", features.values.shape, features.layout.describe_axes())


if __name__ == "__main__":
    main()
