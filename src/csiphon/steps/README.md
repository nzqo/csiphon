# Steps

Each step is one small, self-documenting DSP building block, in its own file,
grouped by category:

| Category | What lives here |
|---|---|
| `components/` | a direct per-value component: magnitude, phase, unit phase, power, real and imaginary part |
| `scaling/` | fixed pointwise scale / compression: log, decibels |
| `cleaning/` | data hygiene: NaN scrubbing, noise-floor clipping |
| `calibration/` | reference/phase-model artifact removal across a non-time axis |
| `normalization/` | rescale relative to a frame or an axis's statistics |
| `baseline/` | estimate and subtract a temporal baseline (running / whole-recording mean) |
| `filtering/` | filter or smooth along time: Butterworth, Savitzky-Golay (`[filters]` extra) |
| `temporal_features/` | local time-domain change: difference, rolling variance, slope |
| `delay/` | frequency-domain CSI to a delay / CIR representation |
| `time_frequency/` | windowed-FFT / complex STFT / multitaper / synchrosqueezed power (`[sst]`, `[filters]` extras) |
| `pooling/` | combine samples or bins by direct reduction (window sums, dyadic bands) |
| `statistics/` | higher-order window statistics: covariance eigenspectrum, PCA bias |
| `reduction/` | reduce feature dimensionality: PCA, robust PCA, tap/axis selection |
| `restructuring/` | rearrange or fold axes without losing information |
| `resampling/` | change the temporal sampling grid |

`_support/` holds shared internals (window geometry, the ring-buffer streaming
operator), not steps.

## The per-step file template

Every step file has the same shape, so opening any one tells you the same things
in the same order. Use `describe(TheStep)` to print this contract at runtime.

```python
@dataclass(frozen=True, slots=True)
class TheStep(PointwiseStep):  # or Step for non-pointwise
    """One-line summary, then any detail."""

    some_param: int = field(default=3, metadata={"doc": "what it controls"})

    spec: ClassVar[StepSpec] = StepSpec(
        name="the-step",  # short id (used in errors + describe)
        summary="what it does",
        category=Category.COMPONENTS,
        admissible_values=(ValueKind.MAGNITUDE,),  # None = anything
        admissible_reprs=None,
        requires_axes=(),  # statically required axes
        layout_effect=LayoutEffect(note="values now power"),  # how the layout changes
        streaming=Streaming.BATCH_EQUIVALENT,  # BATCH_EQUIVALENT | BATCH_DIVERGENT | UNAVAILABLE
    )

    def output_layout(self, layout, profile):
        self.require_inputs(layout)  # guards first: enforces the spec
        ...  # then build and return the output layout

    def transform_values(
        self, values, signal, out_layout, profile
    ): ...  # PointwiseStep: batch + exact streaming for free
```

Rules of thumb:
- **`require_inputs` first.** It enforces exactly what `spec` declares, so the
  documented contract and the runtime check can never drift. Put any extra
  structural guard (e.g. `require_static_axis`) right after it, guards grouped
  up front, early failure. Reject a bad parameter with a message that gives the
  valid range and the value received: `f"hop_size must be >= 1, got {hop}."`
  Some guards prevent real failures (an infinite loop, a crash), so they are
  worth writing even when the bad value looks unlikely.
- **The spec is the contract.** `describe()` reads it; the test suite
  (`tests/test_inspection.py`) enforces that every step declares a complete spec
  and documents every parameter.
- **Generic vs physical axes.** Operations that work on any axis take an
  `axis=` parameter and override `resolve_required_axes()`; only steps that are
  genuinely tied to a physical dimension (delay-ACF needs subcarrier indices,
  dyadic bands need frequencies) hard-code an axis name.
- **Batch is always the best algorithm.** `process` runs the full
  whole-recording computation; `stream` returns a `StreamOperator` (or `None` if
  batch-only). A `PointwiseStep` derives both from `transform_values`.
- **Declare streaming honestly.** `BATCH_EQUIVALENT` means the streamed output
  matches batch to numerical precision; `BATCH_DIVERGENT` means it differs by
  design (say how in `streaming_note`); `UNAVAILABLE` means batch-only. Where a
  step claims interior windows match batch, add a test that asserts it.
- **Export a new step.** Re-export it from its category `__init__.py` and add it
  to `csiphon.steps` (`__all__` in `steps/__init__.py`). This makes
  `from csiphon.steps import TheStep` work and lets the inspection tests, which
  enumerate `__all__`, find it. A step missing from `__all__` is untested.
```
