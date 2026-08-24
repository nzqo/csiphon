"""Inspect blocks and whole pipelines with describe().

describe() works on:

- a step class      -> its default contract,
- a step instance   -> the same, with the values you set marked,
- a compiled pipeline -> a summary of every structural transition
  (pass verbose=True to expand each step).

Runs on the base (numpy-only) install. Run: python examples/inspect_example.py
"""

# The example builds a small pipeline, like the batch/streaming examples do.
# pylint: disable=duplicate-code
from __future__ import annotations

from csiphon import AcquisitionProfile, AxisName, Pipeline, describe
from csiphon.steps import (
    DelayAutocorrelation,
    DelayTaps,
    FoldAxes,
    GainNormalize,
    Magnitude,
    WindowedFFTPower,
)


def inspect_blocks() -> None:
    """Show a block's contract, by class and instantiated."""

    # Just the block: its default contract.
    print(describe(DelayTaps))

    # The same block, configured. describe() marks the values you set
    # (the parameter shows `= default → your_value`).
    print(describe(DelayTaps(num_taps=5, use_tap_abs=True)))


def inspect_pipeline() -> None:
    """Compile a pipeline and summarize every transition."""

    profile = AcquisitionProfile(
        n_rx_antennas=3, subcarrier_indices=tuple(range(52)), sampling_rate_hz=1000.0
    )
    compiled = (
        Pipeline()
        .then(Magnitude())
        .then(GainNormalize())
        .then(DelayAutocorrelation())
        .then(DelayTaps(num_taps=3))
        .then(FoldAxes(axes=(AxisName.RX_ANTENNA, AxisName.DELAY)))
        .then(WindowedFFTPower(window_s=0.256, hop_s=0.05, band_hz=60.0))
        .compile(profile)
    )

    # Concise: one line per step (input and output layouts at top and bottom).
    print(describe(compiled))

    # Verbose: each step's contract plus the running layout.
    print(describe(compiled).render(verbose=True))


def main() -> None:
    """Run both demos. Output is colored in a terminal, plain when piped."""

    inspect_blocks()
    inspect_pipeline()


if __name__ == "__main__":
    main()
