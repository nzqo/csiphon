"""Interpolation up close: what Resample does to the timestamps and the values.

A slow wave is sampled at *irregular* times with a gap in the middle, then resampled
onto a uniform grid three ways. The point is to see, in the terminal:

- the TIMESTAMPS go from irregular (varying gaps) to an exact uniform grid, and
- each fill treats the gap differently -- Linear draws a straight line across it,
  Hold keeps the last value flat, Nearest jumps at the midpoint.

Only one subcarrier is used so the wave plots as a simple line. Numpy-only.
"""

# Examples share small profile/signal setup blocks by design.
# pylint: disable=duplicate-code
from __future__ import annotations

import sys

import numpy as np

from csiphon import AcquisitionProfile, Pipeline
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step
from csiphon.steps import Hold, Linear, Nearest, Resample

_TTY = sys.stdout.isatty()


def _paint(text: str, code: str) -> str:
    """Wrap `text` in an ANSI colour on a real terminal, else leave it plain."""

    return f"\033[{code}m{text}\033[0m" if _TTY else text


def _irregular_wave() -> tuple[AcquisitionProfile, Signal]:
    """A 1 Hz wave sampled at jittery times, with a deliberate gap from 0.4 to 0.6 s."""

    rng = np.random.default_rng(0)
    before = np.sort(rng.uniform(0.0, 0.4, 16))
    after = np.sort(rng.uniform(0.6, 1.0, 16))
    times = np.concatenate([[0.0], before, after, [1.0]])  # endpoints + a middle gap

    profile = AcquisitionProfile(subcarrier_indices=(0,), sampling_rate_hz=100.0)
    wave = np.sin(2.0 * np.pi * 1.0 * times)  # one slow cycle, so it reads as a curve
    return profile, profile.raw_signal(wave[:, None].astype(np.complex128), times)


def _wave_of(signal: Signal) -> np.ndarray:
    """The single subcarrier's real part, one value per sample."""

    return signal.values.reshape(signal.n_samples, -1)[:, 0].real


def _chart(
    original: Signal, resampled: Signal, width: int = 70, height: int = 11
) -> None:
    """Plot the original samples (o) and the resampled ones (.) on shared axes."""

    grid = [[" "] * width for _ in range(height)]

    def place(times: np.ndarray, values: np.ndarray, mark: str, colour: str) -> None:
        for time, value in zip(times, values, strict=True):
            column = int(time * (width - 1))  # time runs 0..1 s
            row = int((1.15 - value) / 2.3 * (height - 1))  # value runs -1..1
            if 0 <= row < height and 0 <= column < width:
                grid[row][column] = _paint(mark, colour)

    # Resampled first, so an original sample sitting on top stays visible.
    place(resampled.times, _wave_of(resampled), ".", "34")
    place(original.times, _wave_of(original), "o", "33")
    for line in grid:
        print("  " + "".join(line))


def _timeline_row(times: np.ndarray, width: int = 70) -> str:
    """A tick per sample along a 0..1 s row -- to eyeball the spacing at a glance."""

    columns = [_paint("·", "2")] * width
    for time in times:
        columns[min(int(time * (width - 1)), width - 1)] = _paint("│", "36")
    return "".join(columns)


def _run(step: Step, profile: AcquisitionProfile, signal: Signal) -> Signal:
    """Apply one resample step to the whole recording."""

    return Pipeline().then(step).compile(profile).pour(signal).single()


def main() -> None:
    """Draw the irregular input and each fill's uniform resampling of it."""

    profile, signal = _irregular_wave()
    grid_hz = 40.0

    print(
        _paint("Input: a 1 Hz wave sampled irregularly, with a gap at 0.4-0.6 s", "1")
    )
    print("  " + _timeline_row(signal.times))
    gaps_ms = np.diff(signal.times) * 1000.0
    print(
        _paint(
            f"  {signal.n_samples} samples; gaps range "
            f"{gaps_ms.min():.0f}-{gaps_ms.max():.0f} ms (the big one is the hole)",
            "2",
        )
    )

    for name, fill in [("Linear", Linear()), ("Hold", Hold()), ("Nearest", Nearest())]:
        out = _run(Resample(rate_hz=grid_hz, fill=fill), profile, signal)
        print(_paint(f"\nResample({grid_hz:.0f} Hz, {name})", "1"))
        _chart(signal, out)
        print("  " + _timeline_row(out.times))
        step_ms = float(np.diff(out.times)[0]) * 1000.0
        print(
            _paint(
                f"  {out.n_samples} samples on an exact {step_ms:.1f} ms grid  "
                f"(o = original, . = resampled)",
                "2",
            )
        )
    print()


if __name__ == "__main__":
    main()
