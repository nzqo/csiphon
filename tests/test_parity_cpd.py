"""Parity: the csiphon SST pipeline reproduces cpd.compute_features.

Skipped unless the sibling `split-decision-lite` package (`cpd`) is
importable and its WISE recordings are present, since csiphon's own environment
is numpy-only. Run it from the split-decision-lite venv with both `src`
directories on the path. Validated manually on three recordings to a max
absolute difference of ~1e-7.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import fold_channels_into_feature

from csiphon import (
    AcquisitionProfile,
    AxisName,
    Pipeline,
    Representation,
)
from csiphon.steps import (
    DelayAutocorrelation,
    DelayTaps,
    DyadicFrequencyBands,
    FixedSizeWindowSum,
    FoldAxes,
    GainNormalize,
    LogScale,
    Magnitude,
    SynchrosqueezedPower,
)

_CPD_ROOT = Path(__file__).resolve().parents[2] / "split-decision-lite"
_CPD_SRC = _CPD_ROOT / "src"

if not _CPD_SRC.is_dir():
    pytest.skip(
        f"split-decision-lite source directory not found: {_CPD_SRC}",
        allow_module_level=True,
    )

sys.path.insert(0, str(_CPD_SRC))

cpd_pipeline = pytest.importorskip(
    "cpd.pipeline",
    reason=f"Could not import cpd.pipeline from {_CPD_SRC}",
)
cpd_feature = pytest.importorskip(
    "cpd.feature",
    reason=f"Could not import cpd.feature from {_CPD_SRC}",
)
cpd_load = pytest.importorskip(
    "cpd.load",
    reason=f"Could not import cpd.load from {_CPD_SRC}",
)

_RECORDINGS = sorted(
    glob.glob(str(_CPD_ROOT / "data/mega_dataset/fabian/session_0[0-2]/meta.parquet"))
)

if not _RECORDINGS:
    pytest.skip(
        f"WISE recordings not found below {_CPD_ROOT / 'data/mega_dataset'}",
        allow_module_level=True,
    )


def _csiphon_pipeline(params) -> Pipeline:
    """Rebuild the DELAY_ACF_SST SplitDecision pipeline in csiphon."""

    return (
        Pipeline()
        .then(Magnitude())
        .then(GainNormalize())
        .then(DelayAutocorrelation(nfft=params.nfft))
        .then(
            DelayTaps(
                num_taps=params.num_taps,
                first_tap=params.first_tap,
                use_tap_abs=params.use_tap_abs,
            )
        )
        .then(fold_channels_into_feature(AxisName.DELAY))
        .then(SynchrosqueezedPower(voices_per_octave=params.voices_per_octave))
        .then(
            DyadicFrequencyBands(
                num_bands=params.num_frequency_bands,
                first_band=params.first_frequency_band,
            )
        )
        .then(FixedSizeWindowSum(window_s=params.window_s, hop_s=params.hop_s))
        .then(LogScale())
        .then(
            FoldAxes(
                axes=(AxisName.BAND, AxisName.FEATURE),
                representation=Representation.FEATURE_VECTOR,
            )
        )
    )


@pytest.mark.skipif(not _RECORDINGS, reason="WISE recordings not available")
@pytest.mark.parametrize("meta", _RECORDINGS)
def test_sst_pipeline_matches_cpd(meta: str) -> None:
    """csiphon batch features equal cpd features for one recording."""

    params = cpd_feature.FeatureParams()
    data = cpd_load.load_wise(
        Path(meta).parent, receivers=["asus1"], subsample_factor=1
    )
    reference, _ = cpd_pipeline.compute_features(data, params)

    fps = float(1e6 / np.median(np.diff(data.times)))
    profile = AcquisitionProfile(
        n_rx_antennas=data.csi.shape[1],
        subcarrier_indices=tuple(int(x) for x in data.subcarrier_idxs),
        sampling_rate_hz=fps,
    )
    compiled = _csiphon_pipeline(params).compile(profile)
    # data.csi is compact (n, n_rx, n_sc); raw_signal inserts the singleton
    # receiver/tx axes onto the full raw layout.
    signal = profile.raw_signal(data.csi, data.times / 1e6)
    got = compiled.pour(signal).single()

    assert got.values.shape == reference.features.shape
    assert np.allclose(reference.features, got.values, rtol=1e-5, atol=1e-5)
