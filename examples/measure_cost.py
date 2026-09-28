"""Measure cost: find out what each step of a run costs.

`siphon.measure()` pours a recording once and reports, per step, the wall time
and the shape of the array it produced; `memory=True` adds the peak and added
bytes (via tracemalloc, which slows the run). `groups=` names spans of
consecutive steps to read as one stage, by step number or step name.

Runs with the base (numpy-only) install, no optional extras required.
"""

# The recording, live-stream, and measure examples deliberately build the same pipeline.
# pylint: disable=duplicate-code
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
    """Run the Doppler-power pipeline once and print what each stage cost."""

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

    rng = np.random.default_rng(0)
    shape = (5000, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    signal = profile.raw_signal(csi, np.arange(shape[0]) / rate)

    # Two stages: the clean-up front-end, by step name (a step's `spec.name`, the
    # id describe() prints; a name that occurs twice in the pipeline is rejected,
    # use the number then), and the delay-domain feature extraction, by step
    # number (as describe() numbers the steps).
    run = compiled.measure(
        signal,
        memory=True,
        groups={
            "clean-up": ("magnitude", "gain-normalize"),
            "delay-features": (3, 5),
        },
    )
    print(run)

    # The outlets are the same as pour() would return; the costs are plain data.
    features = run.outlets.single()
    axes = ", ".join(axis.value for axis in features.layout.axis_names)
    print(f"\nfeatures: shape {features.values.shape}, axes ({axes})")
    slowest = max(run.steps, key=lambda step: step.seconds)
    print(f"slowest step: {slowest.name} ({slowest.seconds * 1e3:.1f} ms)")
    clean_up = run.groups["clean-up"]
    print(
        f"clean-up stage: {clean_up.seconds * 1e3:.1f} ms, peak {clean_up.peak_bytes} B"
    )


if __name__ == "__main__":
    main()
