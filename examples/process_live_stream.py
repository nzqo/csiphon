"""Process a live stream: the same pipeline, fed chunks as they arrive.

Demonstrates that a pipeline compiled once runs live over arbitrary chunks, and
that (for these exact-streaming steps) the streamed result matches batch.
"""

# This and process_whole_recording.py deliberately build the same pipeline.
# pylint: disable=duplicate-code
import numpy as np

from csiphon import AcquisitionProfile, AxisName, Pipeline, Signal
from csiphon.pipeline import concat_signals
from csiphon.steps import (
    DelayAutocorrelation,
    DelayTaps,
    FoldAxes,
    GainNormalize,
    Magnitude,
    WindowedFFTPower,
)


def main() -> None:
    """Feed a synthetic recording through a Stream in small chunks."""

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

    # Live: push arriving frames in bursts of 42; collect emitted feature frames.
    stream = compiled.stream()
    emitted = []
    for start in range(0, signal.n_samples, 42):
        chunk = Signal(
            values=signal.values[start : start + 42],
            times=signal.times[start : start + 42],
            layout=signal.layout,
        )
        out = stream.flow(chunk).single()
        if out.n_samples:
            emitted.append(out)
    emitted.append(stream.flush().single())
    streamed = concat_signals(emitted, compiled.outlet_layout)

    batch = compiled.pour(signal).single()
    print("streamed features:", streamed.values.shape)
    print("matches batch:", np.allclose(streamed.values, batch.values))


if __name__ == "__main__":
    main()
