"""Micro-Doppler spectrogram from a classic STFT pipeline.

Needs the ``[filters]`` extra (scipy) for the STFT step.
"""

import numpy as np

from csiphon import AcquisitionProfile, AxisName, Pipeline
from csiphon.steps import (
    AxisReference,
    ComplexStftMagnitude,
    FoldAxes,
    GainNormalize,
    GlobalMaxNormalize,
    MeanOverAxes,
    NoiseFloorClip,
    TimeDifference,
    ToDecibels,
)
from csiphon.steps.calibration import Combine, Reference
from csiphon.steps.temporal_features import Mode


def build_doppler_pipeline() -> Pipeline:
    """Build the STFT micro-Doppler feature pipeline."""

    return (
        Pipeline()
        .then(GainNormalize())
        .then(
            AxisReference(
                axis=AxisName.RX_ANTENNA,
                combine=Combine.CONJUGATE,
                mode=Reference.FIXED,
                index=0,
                drop_reference=True,
            )
        )
        .then(
            FoldAxes(
                axes=(
                    AxisName.RECEIVER,
                    AxisName.TX_ANTENNA,
                    AxisName.RX_ANTENNA,
                    AxisName.SUBCARRIER,
                )
            )
        )
        .then(TimeDifference(mode=Mode.CONJUGATE))
        .then(ComplexStftMagnitude(window_size=512, hop_size=10, freq_bins=31))
        .then(MeanOverAxes(axes=(AxisName.FEATURE,)))
        .then(GlobalMaxNormalize())
        .then(NoiseFloorClip(level_log10=-4.0))
        .then(ToDecibels(reference=1.0, epsilon=0.0))
    )


def main() -> None:
    """Build the pipeline, print its contract, and run it on synthetic CSI."""

    rate = 1000.0
    profile = AcquisitionProfile(
        n_rx_antennas=4,
        subcarrier_indices=tuple(range(114)),
        sampling_rate_hz=rate,
    )

    # build and verify pipeline
    pipeline = build_doppler_pipeline()
    compiled = pipeline.compile(profile)
    print(compiled.describe())

    # generate some random (time, antennas=4, subcarriers=114) CSI
    rng = np.random.default_rng(0)
    shape = (4000, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    times = np.arange(shape[0]) / rate
    signal = profile.raw_signal(csi, times)

    # Process!
    features = compiled.pour(signal).single()
    print("\nfeatures:", features.values.shape, features.layout.describe_axes())
    print("dB range:", float(features.values.min()), "->", float(features.values.max()))


if __name__ == "__main__":
    main()
