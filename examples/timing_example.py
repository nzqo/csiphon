"""Timing playground: see what the resampling / loss steps do to a stream's timing.

Runs each timing-related step on one regular 1 kHz recording and draws the result
straight in the terminal:

- a TIMELINE of the first slice, one column per time bin, so you can see the
  *pattern* of kept vs missing samples (scattered vs clustered vs regular), and
- an inter-arrival HISTOGRAM over the whole recording, so you can see the timing
  *distribution* (a single spike for a regular stream, a decaying tail for loss).

Only the timestamps matter here, so the CSI values are random filler. Numpy-only,
no plotting libraries -- it prints.
"""

# Examples share small profile/signal setup blocks by design.
# pylint: disable=duplicate-code
import sys
from dataclasses import dataclass

import numpy as np

from csiphon import AcquisitionProfile, Pipeline
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step
from csiphon.steps import (
    Bursty,
    DropSamples,
    Hold,
    Independent,
    Resample,
    SubsampleEvery,
)

# Light ANSI colour, but only when writing to a real terminal (piped output stays
# plain so it does not fill up with escape codes).
_TTY = sys.stdout.isatty()


def _paint(text: str, code: str) -> str:
    """Wrap `text` in an ANSI colour when attached to a terminal, else leave it."""

    return f"\033[{code}m{text}\033[0m" if _TTY else text


@dataclass(frozen=True)
class Scenario:
    """One named timing transformation to run and draw."""

    title: str
    step: Step | None  # None = the untouched original


def _timeline(times: np.ndarray, window_s: float, width: int = 84) -> str:
    """One row per time bin over [0, window_s]: a tick where a sample lands.

    A column is a small slice of time; it shows a tick if any sample falls in it and
    a faint dot otherwise. So regular streams look evenly striped, random loss looks
    speckled, and bursty loss shows solid runs of dots.
    """

    columns = [_paint("·", "2")] * width  # faint dots for the empty slices
    for time in times[times <= window_s]:
        column = min(int(time / window_s * width), width - 1)
        columns[column] = _paint("│", "32")  # green tick where a sample survives
    return "".join(columns)


def _interval_histogram(times: np.ndarray, width: int = 44) -> list[str]:
    """Horizontal bars of the inter-arrival intervals (ms) over the whole recording."""

    intervals_ms = np.diff(times) * 1000.0
    if intervals_ms.size == 0:
        return ["  (only one sample)"]

    low, high = float(intervals_ms.min()), float(intervals_ms.max())
    if high - low < 1e-6:  # a perfectly regular stream is a single value
        return [
            f"  every {low:.2f} ms  ({_paint('regular', '36')}, {1000 / low:.0f} Hz)"
        ]

    # Roughly one bin per millisecond, capped so the block stays compact.
    bins = max(4, min(12, round(high - low) + 1))
    counts, edges = np.histogram(intervals_ms, bins=bins)
    peak = max(counts.max(), 1)
    rows = []
    for count, left, right in zip(counts, edges[:-1], edges[1:], strict=False):
        filled = round(float(count) / peak * width)
        bar = _paint("█" * filled, "34")
        rows.append(f"  {left:5.1f}-{right:4.1f} ms |{bar:<{width}}| {count}")
    return rows


def _summary(times: np.ndarray) -> str:
    """A one-line count / mean-interval / effective-rate summary."""

    if times.size < 2:
        return f"{times.size} samples"
    intervals_ms = np.diff(times) * 1000.0
    rate = 1000.0 / float(intervals_ms.mean())
    return (
        f"{times.size} samples, "
        f"mean gap {intervals_ms.mean():.2f} +/- {intervals_ms.std():.2f} ms, "
        f"~{rate:.0f} Hz"
    )


def _draw(title: str, times: np.ndarray, window_s: float) -> None:
    """Print one scenario: its heading, timeline, summary, and interval histogram."""

    print(f"\n{_paint(title, '1')}")
    print("  " + _timeline(times, window_s))
    print("  " + _paint(_summary(times), "2"))
    for row in _interval_histogram(times):
        print(row)


def _run(step: Step | None, profile: AcquisitionProfile, signal: Signal) -> np.ndarray:
    """The output timestamps after applying `step` (or the input's, if step is None)."""

    if step is None:
        return signal.times
    return Pipeline().then(step).compile(profile).pour(signal).single().times


def main() -> None:
    """Build a regular 1 kHz recording and draw each timing step's effect."""

    rate = 1000.0
    length = 2000  # two seconds at 1 kHz
    profile = AcquisitionProfile(
        n_rx_antennas=1, subcarrier_indices=tuple(range(8)), sampling_rate_hz=rate
    )
    rng = np.random.default_rng(0)
    shape = (length, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    signal = profile.raw_signal(csi, np.arange(length) / rate)

    scenarios = [
        Scenario("Original -- regular 1 kHz", None),
        Scenario(
            "SubsampleEvery(every=4) -- plain decimation to 250 Hz",
            SubsampleEvery(every=4),
        ),
        Scenario(
            "Resample(200 Hz, Hold) -- onto an exact 200 Hz grid",
            Resample(rate_hz=200.0, fill=Hold()),
        ),
        Scenario(
            "DropSamples(30%, Independent) -- scattered loss",
            DropSamples(loss_rate=0.3, model=Independent(), seed=1),
        ),
        Scenario(
            "DropSamples(30%, Bursty, mean_burst=8) -- clustered loss",
            DropSamples(loss_rate=0.3, model=Bursty(mean_burst=8.0), seed=1),
        ),
    ]

    window_s = 0.09  # the first 90 ms, so each 1 kHz sample gets about one column
    print(
        _paint(
            "Timing of each step (first 90 ms as a timeline, then the "
            "inter-arrival distribution):",
            "1",
        )
    )
    for scenario in scenarios:
        _draw(scenario.title, _run(scenario.step, profile, signal), window_s)
    print()


if __name__ == "__main__":
    main()
