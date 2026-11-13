"""Effective sampling rate and jitter handling.

Timestamps are the source of truth. When a step needs a scalar sampling rate
(to interpret an FFT, for example) it asks for the *effective* rate: the
nominal rate of the time axis if one was declared, otherwise an estimate from
the actual inter-sample intervals. Because real CSI is never perfectly uniform,
jitter is tolerated by default; a step may warn when the jitter is excessive,
and `strict` mode upgrades that warning to an error.
"""

import warnings

import numpy as np

from csiphon.core.arrays import RealArray
from csiphon.core.errors import DataError, LayoutError
from csiphon.core.layout import Layout

# Default tolerance: warn when the standard deviation of inter-sample intervals
# exceeds this fraction of their median. 0.5 is deliberately loose. Normal CSI
# jitter sits well below it, so tripping it means something is genuinely wrong.
DEFAULT_JITTER_TOLERANCE = 0.5


class JitterWarning(UserWarning):
    """Emitted when observed timestamp jitter exceeds the tolerance."""


def estimate_rate_hz(times_s: RealArray) -> float:
    """Estimate the sampling rate from timestamps (median interval)."""

    if times_s.size < 2:
        raise DataError("At least two timestamps are needed to estimate a rate.")

    intervals = np.diff(times_s)
    median = float(np.median(intervals))
    if median <= 0.0:
        raise DataError("Timestamps must be strictly increasing to estimate a rate.")

    return 1.0 / median


def require_nominal_rate(layout: Layout, step: str) -> float:
    """Return the nominal rate of the layout's time axis, or fail naming the step."""

    rate_hz = layout.nominal_rate_hz
    if rate_hz is None:
        raise LayoutError(
            f"Step '{step}' needs a nominal sampling rate. Set "
            "AcquisitionProfile.sampling_rate_hz (actual timestamp jitter is "
            "still tolerated at run time)."
        )
    return rate_hz


def divide_nominal_rate(layout: Layout, factor: float) -> Layout:
    """Return the layout with its nominal rate divided by `factor`.

    For steps that keep every Nth sample, or emit one sample per hop. An unknown
    rate stays unknown.
    """

    rate_hz = layout.nominal_rate_hz
    return layout.with_nominal_rate(None if rate_hz is None else rate_hz / factor)


def effective_rate_hz(times_s: RealArray, nominal_hz: float | None) -> float:
    """Return the nominal rate if given, else estimate it from timestamps."""

    if nominal_hz is not None:
        return nominal_hz
    return estimate_rate_hz(times_s)


def jitter_ratio(times_s: RealArray) -> float:
    """Return std / median of inter-sample intervals (0.0 if fewer than 3)."""

    if times_s.size < 3:
        return 0.0

    intervals = np.diff(times_s)
    median = float(np.median(intervals))
    if median <= 0.0:
        return float("inf")

    return float(np.std(intervals) / median)


def check_jitter(
    times_s: RealArray,
    *,
    step: str,
    strict: bool,
    tolerance: float = DEFAULT_JITTER_TOLERANCE,
) -> None:
    """Warn (or raise in strict mode) when timestamp jitter is excessive."""

    ratio = jitter_ratio(times_s)
    if ratio <= tolerance:
        return

    message = (
        f"Step '{step}' sees irregular timestamps "
        f"(interval std/median = {ratio:.2f} > {tolerance:.2f}). "
        "Results assume an approximately uniform grid; consider a Resample "
        "step if this is unexpected."
    )
    if strict:
        raise DataError(message)
    warnings.warn(message, JitterWarning, stacklevel=2)
