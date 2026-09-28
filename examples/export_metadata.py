"""Export a compiled pipeline's full metadata (for provenance / reproducibility).

`describe(compiled).as_dict()` is plain JSON-able data: the streaming verdict, the
input and output layouts, and every step with its configured parameters and
resulting layout. `.to_json()` serializes it; `compiled.save_metadata(path)` writes
it to a file.

Runs with the base (numpy-only) install.
"""

# Examples share small profile/signal setup blocks by design.
# pylint: disable=duplicate-code
import pathlib
import tempfile

from csiphon import AcquisitionProfile, AxisName, Pipeline, describe
from csiphon.steps import DelayAutocorrelation, DelayTaps, FoldAxes, Magnitude


def main() -> None:
    """Build a pipeline and dump its metadata as data and as a JSON file."""

    profile = AcquisitionProfile(
        n_rx_antennas=3,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=1000.0,
    )
    compiled = (
        Pipeline()
        .then(Magnitude())
        .then(DelayAutocorrelation())
        .then(DelayTaps(num_taps=3))
        .then(FoldAxes(axes=(AxisName.RX_ANTENNA, AxisName.DELAY)))
    ).compile(profile)

    meta = describe(compiled).as_dict()
    print("top-level keys:", list(meta))
    print("streams:", meta["streams"])
    third = meta["steps"][2]
    print("step 3:", third["name"], "->", third["layout_change_description"])
    print("step 3 params:", [(p["name"], p["value"]) for p in third["params"]])

    path = pathlib.Path(tempfile.gettempdir()) / "pipeline.json"
    compiled.save_metadata(path)
    print(f"\nwrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
