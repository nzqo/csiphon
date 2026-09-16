# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/nzqo/csiphon/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/nzqo/csiphon/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/nzqo/csiphon/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nzqo/csiphon/releases/tag/v0.1.0
