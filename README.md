<p align="center">
  <img
    src="https://raw.githubusercontent.com/nzqo/csiphon/main/assets/ai_slop_mascot.png"
    alt="slop Kanna doing Wi-Fi plumbing"
    width="400"
  >
</p>

# csiphon

`csiphon` is a Python library for preprocessing WiFi Channel State Information.
You plug DSP steps together into a pipeline, csiphon checks that the pieces fit
before you run anything, and the finished pipeline handles recordings and live
streams alike. Every signal carries its axes by name, so you never have to
wonder which one was the subcarrier again.

```python
from csiphon import AcquisitionProfile, Pipeline
from csiphon.steps import GainNormalize, Magnitude, WindowedVariance

# Describe the capture setup
profile = AcquisitionProfile(
    n_rx_antennas=3,
    subcarrier_indices=tuple(range(52)),
    sampling_rate_hz=1000.0,
)

# Define the pipeline steps
siphon = (
    Pipeline()
    .then(Magnitude())
    .then(GainNormalize())
    .then(WindowedVariance(win_size_s=0.1))
    .compile(profile)
)

# Generate a summary of the pipeline
print(siphon.describe())

# Push some data through it to run
features = siphon.pour(profile.raw_signal(csi, timestamps)).single()
```

A compiled pipeline is a `Siphon`. You `pour()` a recording through it or
`stream()` live data, and collect results from named `Outlets`, which you can tap
anywhere along the pipeline.

## What it does

- **Catches structural mistakes before execution.** Compilation checks axes,
  shapes, value semantics, and physical representations before data starts
  flowing.
- **Runs the same recipe in batch or live.** Every step states whether its
  streaming computation matches batch, differs intentionally, or requires the
  complete recording.
- **Supports branching pipelines.** Split into parallel feature paths, merge
  them again, and expose named intermediate outlets.
- **Merges receivers.** Feed several receivers through their own inlets and line
  them up on timestamps or on packet sequence numbers, which survive clock drift.
- **Keeps real timestamps.** Non-uniform CSI sampling is expected; resampling is
  explicit rather than silently assumed. You can also drop packets on purpose,
  with independent or bursty loss, to test how a pipeline copes.
- **Describes itself.** `describe()` draws a compiled siphon as a data-flow graph
  in the terminal. Steps and siphons expose their contracts, layouts, parameters,
  and streaming behavior, and can save that information alongside results for
  reproducibility.
- **Measures itself.** `siphon.measure(signal)` pours once and reports each
  step's time and output shape (memory on request), per step or for a named
  group of consecutive steps.

The built-in steps cover components, scaling, cleaning, calibration,
normalization, baseline removal, filtering, temporal features, delay and
time-frequency transforms, pooling, statistics, reduction, restructuring, and
resampling. The
[steps README](https://github.com/nzqo/csiphon/blob/main/src/csiphon/steps/README.md)
lists the categories. Custom steps use the same contracts and inspection tools.

## Install

```bash
pip install csiphon
```

The core only depends on NumPy. A few steps (filters, STFT, multitaper,
cubic-spline resampling) need the `[filters]` extra (SciPy), and the
synchrosqueezed transform needs `[sst]`. Install everything with:

```bash
pip install "csiphon[all]"
```

Runnable recipes live in [`examples/`](https://github.com/nzqo/csiphon/tree/main/examples).
