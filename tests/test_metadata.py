"""describe(...).as_dict() / .save() capture full pipeline metadata."""

from __future__ import annotations

import json
from pathlib import Path

from csiphon import AcquisitionProfile, Pipeline, Siphon, describe
from csiphon.steps import DelayAutocorrelation, DelayTaps, Magnitude


def _compiled(profile: AcquisitionProfile) -> Siphon:
    return (
        Pipeline()
        .then(Magnitude())
        .then(DelayAutocorrelation())
        .then(DelayTaps(num_taps=5))
    ).compile(profile)


def test_pipeline_as_dict_has_full_info(profile: AcquisitionProfile) -> None:
    """The metadata records steps, configured params, and the resulting layouts."""

    data = describe(_compiled(profile)).as_dict()
    assert set(data) == {"streams", "inlet", "outlet", "steps"}
    assert data["inlet"]["axes"][0]["name"] == "time"

    taps = data["steps"][2]
    assert taps["name"] == "delay-taps"
    num_taps = next(p for p in taps["params"] if p["name"] == "num_taps")
    assert (
        num_taps["value"] == 5 and num_taps["default"] == 3
    )  # configured value captured
    assert taps["layout"]["values"] == "real-valued"


def test_save_metadata_writes_valid_json(
    profile: AcquisitionProfile, tmp_path: Path
) -> None:
    """save_metadata writes a JSON file that round-trips to the same data."""

    compiled = _compiled(profile)
    path = tmp_path / "pipeline.json"
    compiled.save_metadata(path)

    loaded = json.loads(path.read_text())
    assert loaded == describe(compiled).as_dict()


def test_block_as_dict_records_configured_value() -> None:
    """A single block's metadata captures the value you set."""

    data = describe(DelayTaps(num_taps=7)).as_dict()
    num_taps = next(p for p in data["params"] if p["name"] == "num_taps")
    assert num_taps["value"] == 7
