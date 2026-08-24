<p align="center">
  <img
    src="assets/ai_slop_mascot.png"
    alt="slop Kanna doing Wi-Fi plumbing"
    width="300"
  >
</p>

# csiphon

`csiphon` is an online-first processing library for WiFi Channel State
Information. Compose reusable DSP steps, compile them against your capture
setup, then pour a complete recording or stream live chunks through the same
validated pipeline.

```python
from csiphon import AcquisitionProfile, Pipeline
from csiphon.steps import GainNormalize, Magnitude, WindowedVariance

profile = AcquisitionProfile(
    n_rx_antennas=3,
    subcarrier_indices=tuple(range(52)),
    sampling_rate_hz=1000.0,
)

siphon = (
    Pipeline()
    .then(Magnitude())
    .then(GainNormalize())
    .then(WindowedVariance(win_size_s=0.1))
    .compile(profile)
)

features = siphon.pour(profile.raw_signal(csi, timestamps)).single()
```

## What it does

- **Catches structural mistakes before execution.** Compilation checks axes,
  shapes, value semantics, and physical representations before data starts
  flowing.
- **Runs the same recipe in batch or live.** Every step states whether its
  streaming computation matches batch, differs intentionally, or requires the
  complete recording.
- **Supports branching pipelines.** Split into parallel feature paths, merge
  them again, and expose named intermediate outlets.
- **Keeps real timestamps.** Non-uniform CSI sampling is expected; resampling is
  explicit rather than silently assumed.
- **Describes itself.** Steps and compiled siphons expose their contracts,
  layouts, parameters, and streaming behavior, and can save that information
  alongside results for reproducibility.

The built-in steps cover calibration, cleaning, filtering, delay and
time-frequency transforms, temporal features, pooling, statistics, reduction,
and restructuring. Custom steps use the same contracts and inspection tools.

## Install

```bash
pip install -e .
```

The core only depends on NumPy. Install every optional transform with:

```bash
pip install -e ".[all]"
```

Requires Python 3.13+. Runnable recipes live in [`examples/`](examples/).
