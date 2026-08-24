"""Contract failures: csiphon validates a pipeline's structure at compile time.

`compile()` walks every step's declared contract (accepted value kinds and
representations, required axes, the resulting layout) before any data flows, so a
mismatched pipeline fails immediately with a clear message instead of blowing up
mid-recording. Streaming adds one more gate: a batch-only step is refused when you
try to stream it.

Runs with the base (numpy-only) install.
"""

# Examples share small profile/signal setup blocks by design.
# pylint: disable=duplicate-code
from __future__ import annotations

from collections.abc import Callable

from csiphon import AcquisitionProfile, AxisName, Pipeline
from csiphon.core import CompileError, StreamingError
from csiphon.steps import DelayAutocorrelation, FoldAxes, Magnitude, PrincipalComponents


def show(title: str, attempt: Callable[[], object]) -> None:
    """Run `attempt`, printing the contract error it raises (or noting success)."""

    try:
        attempt()
    except (CompileError, StreamingError) as error:
        print(f"{title}\n  {type(error).__name__}: {error}\n")
    else:
        print(f"{title}\n  (compiled with no error)\n")


def main() -> None:
    """Trigger the three kinds of contract failure the pipeline catches."""

    profile = AcquisitionProfile(
        n_rx_antennas=3,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=1000.0,
    )

    # 1. Wrong value kind. DelayAutocorrelation consumes magnitude/real values, but raw
    #    CSI is complex. Caught at compile, before any samples are read.
    show(
        "1. DelayAutocorrelation fed raw (complex) CSI:",
        lambda: Pipeline().then(DelayAutocorrelation()).compile(profile),
    )

    # 2. Missing axis. FoldAxes asks for a delay axis, but nothing has produced
    #    one yet (no DelayAutocorrelation in the chain).
    show(
        "2. FoldAxes over an axis the layout does not have:",
        lambda: (
            Pipeline()
            .then(Magnitude())
            .then(FoldAxes(axes=(AxisName.RX_ANTENNA, AxisName.DELAY)))
            .compile(profile)
        ),
    )

    # 3. Streaming refusal. Fit-on-the-recording PCA needs every sample, so it
    #    has no streaming variant: it compiles fine but cannot be streamed.
    def stream_a_batch_only_pipeline() -> None:
        compiled = (
            Pipeline()
            .then(Magnitude())
            .then(PrincipalComponents(axis=AxisName.SUBCARRIER, n_components=2))
            .compile(profile)
        )
        compiled.stream()  # raises: this step needs the whole recording

    show("3. Streaming a fit-on-data PCA pipeline:", stream_a_batch_only_pipeline)


if __name__ == "__main__":
    main()
