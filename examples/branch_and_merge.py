"""Branch and merge: fork into parallel branches, probe, merge, and fuse.

`describe()` draws the pipeline as a 2-D data-flow graph, so the structure is
visible at a glance. This file builds pipelines of increasing shape, with
genuinely different processing on each branch:

1. a simple fork/merge (a rolling variance beside a local slope, stacked),
2. a three-way merge: three subcarrier features averaged into one,
3. a heterogeneous fuse: a subcarrier feature beside a frequency feature,
   combined into one feature vector by `Fuse` (different axes and value kinds),
4. a staged "dual merge": three different branches, two averaged together first,
   then that result fused with the third.

The simple siphon is poured (whole recording) and streamed (chunk by chunk) to
show the two run modes give identical results; every other shape is drawn and
then poured too, so the example proves each one actually runs. Merging several
receivers lives in merge_receivers.py.

Runs with the base (numpy-only) install, no optional extras required.
"""

# Examples share small profile/signal setup blocks by design.
# pylint: disable=duplicate-code
import numpy as np

from csiphon import (
    AcquisitionProfile,
    AxisName,
    Fuse,
    Hold,
    Mean,
    Pipeline,
    Signal,
    Siphon,
    Stack,
)
from csiphon.pipeline import concat_signals
from csiphon.steps import (
    GainNormalize,
    Magnitude,
    WindowedFFTPower,
    WindowedSlope,
    WindowedVariance,
)
from csiphon.steps.baseline import RunningMeanSubtract
from csiphon.steps.pooling import MeanOverAxes


def _detrended_variance(win_s: float) -> Pipeline:
    """A subcarrier feature: detrend, then a rolling variance -> (time, rx, sc)."""

    return (
        Pipeline().then(RunningMeanSubtract()).then(WindowedVariance(win_size_s=win_s))
    )


def _spectral() -> Pipeline:
    """A frequency feature: FFT power, averaged over subcarriers -> (time, freq, rx)."""

    return (
        Pipeline()
        .then(GainNormalize())
        .then(WindowedFFTPower())
        .then(MeanOverAxes(axes=(AxisName.SUBCARRIER,)))
    )


def show_shapes(profile: AcquisitionProfile, signal: Signal) -> None:
    """Draw a few pipeline shapes and actually run each one to prove it works."""

    def draw_and_run(title: str, siphon: Siphon) -> None:
        """Print the siphon's flow graph, then pour `signal` through it."""

        print(f"\n{title}\n")
        print(siphon.describe())
        outputs = siphon.pour(signal)
        produced = "  ".join(
            f"{name} {out.layout.describe_axes()}" for name, out in outputs.items()
        )
        print(f"  runs, produces:  {produced}")

    # Three different subcarrier features, all averaged together at one three-way
    # merge (the three branches reunite at a single node).
    three_way = (
        Pipeline()
        .then(Magnitude())
        .branch(
            variance=_detrended_variance(0.1),
            fast=Pipeline().then(WindowedVariance(win_size_s=0.05)),
            slope=Pipeline().then(WindowedSlope()),
        )
        .merge(using=Mean())
        .compile(profile)
    )
    draw_and_run("Three-way merge (three features averaged into one):", three_way)

    # A heterogeneous fuse: the variance branch stays in the subcarrier domain
    # (magnitude), the spectral branch is in the frequency domain (power). Fuse
    # lays their differing axes end to end into one real feature vector. The two
    # branches are also on different time grids (per-sample vs window-centre
    # times), so align=Hold() carries each spectral value across the per-sample
    # timeline instead of dropping the samples that do not line up exactly.
    fused = (
        Pipeline()
        .then(Magnitude())
        .branch(variance=_detrended_variance(0.1), spectral=_spectral())
        .merge(using=Fuse(), align=Hold())
        .compile(profile)
    )
    draw_and_run("Heterogeneous fuse (subcarrier variance + spectral power):", fused)

    # A staged dual merge with three different branches: the two subcarrier-domain
    # features (slope, variance) are averaged into one, which is then fused with
    # the frequency-domain branch.
    staged = (
        Pipeline()
        .then(Magnitude())
        .branch(
            slope=Pipeline().then(WindowedSlope()),
            variance=_detrended_variance(0.1),
            spectral=_spectral(),
        )
        # slope and variance are both per-sample, so they share a grid (Exact).
        .merge(["slope", "variance"], using=Mean(), name="temporal")
        # temporal (per-sample) and spectral (windowed) are on different grids.
        .merge(["temporal", "spectral"], using=Fuse(), align=Hold())
        .compile(profile)
    )
    draw_and_run(
        "Staged dual merge (temporal averaged, then fused with spectral):", staged
    )


def main() -> None:
    """Build one branching siphon, then pour it and stream it."""

    rate = 1000.0
    profile = AcquisitionProfile(
        n_rx_antennas=3,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=rate,
    )

    siphon = (
        Pipeline()
        .then(Magnitude())
        .probe("amplitude")  # keep the plain magnitude as a side outlet
        .branch(
            variance=Pipeline().then(WindowedVariance(win_size_s=0.05)),
            slope=Pipeline().then(WindowedSlope()),
        )
        .merge(using=Stack(into=AxisName.FEATURE))  # two features, side by side
        .compile(profile)
    )
    print("Simple fork/merge:\n")
    print(siphon.describe())

    # Generate some random CSI
    rng = np.random.default_rng(0)
    shape = (2000, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    signal = profile.raw_signal(csi, np.arange(shape[0]) / rate)

    # Pour: the whole recording at once. A run returns Outlets (name -> signal).
    outlets = siphon.pour(signal)
    print("\noutlets from pour:")
    for name, out in outlets.items():
        print(f"  {name:10s} {out.layout.describe_axes()}")

    # Stream: the same siphon, fed in chunks, gives the same merged result.
    stream = siphon.stream()
    chunks = []
    for start in range(0, signal.n_samples, 256):
        piece = signal.with_values(
            signal.values[start : start + 256],
            signal.layout,
            times=signal.times[start : start + 256],
        )
        chunks.append(stream.flow(piece)["out"])
    chunks.append(stream.flush()["out"])
    streamed = concat_signals(chunks, siphon.outlet_layouts["out"])

    matches = np.allclose(streamed.values, outlets["out"].values)
    print(f"\nstreamed 'out' matches poured 'out': {matches}")

    # Further shapes: a three-way merge, heterogeneous fuses, a staged dual merge.
    show_shapes(profile, signal)


if __name__ == "__main__":
    main()
