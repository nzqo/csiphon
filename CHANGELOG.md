# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/nzqo/csiphon/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nzqo/csiphon/releases/tag/v0.1.0
