# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [Unreleased]

## [0.3.1] - 2026-09-28

### Added

- `DelayAutocorrelation` can compute just a range of taps. Computing every
  tap and keeping a handful afterwards wastes most of the work, and for a
  typical recording that stage dominated the whole pipeline. Pass `first_tap`
  and `num_taps` to get only those taps; the delay axis then carries their
  indices. Without them the step behaves as before and produces every tap. A
  `DelayTaps` placed after a restricted step counts positions on that shorter
  axis, so start it at `first_tap=0`.

### Changed

- `DelayAutocorrelation` is faster for the full axis as well, with identical
  values. The projection onto the taps ran as one small matrix product per
  sample; it now runs as a single large product over all samples, and as two
  real products (for the real and the imaginary part of the kernel) instead
  of one complex product.

## [0.3.0] - 2026-09-28

### Added

- `RealPart` and `ImagPart` steps: the real and the imaginary part of complex
  values, as real-valued signals.
- `ComplexFromParts` merge: joins a real-part branch and an imaginary-part branch
  back into complex values. The first branch is the real part, so name them at
  the merge: `.merge(["re", "im"], using=ComplexFromParts())`.

### Changed

- `ButterworthFilter` accepts complex input and filters its real and imaginary
  parts, the same filter on each.
- `pour()` and `measure()` drop an intermediate result once no later step reads
  it and it is not an outlet, instead of holding every intermediate until the run
  ends. Outlets and probes are unaffected.
- README: new introduction, PyPI install instructions, and coverage of receiver
  merging, packet-loss simulation, and the `describe()` flow graph.
- Examples are named after what they do (`batch_example.py` is now
  `process_whole_recording.py`, `streaming_example.py` is now
  `process_live_stream.py`, and so on). Merging receivers moved out of the
  branching example into its own `merge_receivers.py`.

## [0.2.0] - 2026-09-16

### Added

- `Siphon.measure()`: pour a whole recording once and get, per step in run
  order, its wall time and output shape (`MeasuredRun.steps`), with peak and
  added memory behind `memory=True`. `groups=` names spans of consecutive steps
  (by step number or name) to report as one unit; `total()` covers the run.

### Changed

- `SynchrosqueezedPower`: with a block size set, `pour()` now runs the
  block-local transform (consecutive blocks transformed independently and
  joined in time order, trailing partial block dropped) instead of the
  whole-recording one, so batch and streaming agree and the step declares
  `BATCH_EQUIVALENT` in that configuration. The parameter is renamed from
  `streaming_window` to `block_size` since it now governs both modes.

## [0.1.1] - 2026-08-24

### Fixed

- `csiphon[sst]` now pins `numpy<2.5`. `ssqueezepy` pulls in numba, which does
  not yet support NumPy 2.5+, so the extra previously installed a NumPy/numba
  combination that failed to import.

## [0.1.0] - 2026-08-24

### Added

- Initial release: an online-first, schema-validated CSI preprocessing library.
- `Pipeline` / `Siphon` model: compose immutable steps, compile against an
  `AcquisitionProfile`, then `pour()` a whole recording or `stream()` it in
  chunks through the same validated pipeline.
- A library of DSP steps grouped by category (components, scaling, cleaning,
  calibration, normalization, baseline, filtering, temporal features, delay,
  time-frequency, pooling, statistics, reduction, restructuring, resampling),
  each carrying an inspectable `StepSpec` contract.
- Optional extras: `filters` (scipy) and `sst` (ssqueezepy); the base install is
  numpy-only. Ships `py.typed`.

[Unreleased]: https://github.com/nzqo/csiphon/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/nzqo/csiphon/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nzqo/csiphon/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/nzqo/csiphon/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nzqo/csiphon/releases/tag/v0.1.0
