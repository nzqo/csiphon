"""Doppler spectrogram of the SHARP front end (Meneghello et al., IEEE TMC 2023).

SHARP cleans the phase of every packet, so that raw CSI can be used without a
reference antenna, and then takes the Doppler spectrum with an STFT. The phase
cleaning is `SparseFitSanitize`. It expects magnitudes over their mean and leaves
an arbitrary scale, so a `GainNormalize` goes before and after it.

Needs the ``[filters]`` extra (scipy) for the STFT step.
"""

import numpy as np

from csiphon import AcquisitionProfile, AxisName, Pipeline
from csiphon.steps import (
    ComplexStftMagnitude,
    GainNormalize,
    NoiseFloorClip,
    PerFrameMaxNormalize,
    SparseFitSanitize,
    TemporalMeanSubtract,
)

# A 40 MHz capture: 114 used subcarriers around the DC subcarrier.
SUBCARRIER_INDICES = tuple(range(-58, -1)) + tuple(range(2, 59))


def build_sharp_pipeline() -> Pipeline:
    """Build SHARP's front end from csiphon steps."""

    return (
        Pipeline()
        .then(GainNormalize())
        .then(SparseFitSanitize())
        .then(GainNormalize())
        .then(
            ComplexStftMagnitude(
                window_size=31,
                hop_size=1,
                freq_bins=100,
                nfft=100,
                signed_bins=True,
                symmetric_window=True,
                edges="valid",
                power=True,
                mean_over=(AxisName.SUBCARRIER,),
            )
        )
        .then(PerFrameMaxNormalize())
        .then(NoiseFloorClip(level_log10=-1.2))
        # SHARP's own centering, over the whole recording (future included).
        .then(TemporalMeanSubtract())
    )


def main() -> None:
    """Build the pipeline, print its contract, and run it on synthetic CSI."""

    rate = 173.0
    profile = AcquisitionProfile(
        n_rx_antennas=2,
        subcarrier_indices=SUBCARRIER_INDICES,
        sampling_rate_hz=rate,
    )

    compiled = build_sharp_pipeline().compile(profile)
    print(compiled.describe())

    # random (time, antennas=2, subcarriers=114) CSI
    rng = np.random.default_rng(0)
    shape = (400, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    signal = profile.raw_signal(csi, np.arange(shape[0]) / rate)

    features = compiled.pour(signal).single()
    print("\nfeatures:", features.values.shape, features.layout.describe_axes())


if __name__ == "__main__":
    main()
