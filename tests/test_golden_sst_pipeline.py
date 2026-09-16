"""Golden check of the delay-ACF / SST feature pipeline, end to end.

The per-step tests pin each step on its own; this pins the numbers the whole
chain produces on a fixed synthetic recording, so a change in one step's output
convention that the next step silently accepts still shows up. The values were
generated while the pipeline matched the reference implementation it was ported
from (`cpd.compute_features` in split-decision-lite).
"""

import numpy as np
import pytest
from conftest import fold_channels_into_feature

from csiphon import AcquisitionProfile, AxisName, Pipeline, Representation
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

# fmt: off
# Two 250 ms windows of (3 bands x 4 tap features), band-major.
EXPECTED = np.array([
    [-1.591298, -1.395337, -1.676767, -1.559529, -2.729075, -2.301942,
     -2.460099, -2.626653, -3.938123, -3.115853, -3.508632, -3.425709],
    [-1.728918, -1.739668, -1.424937, -1.755563, -2.695767, -2.310867,
     -2.504002, -2.421142, -3.492083, -2.781069, -3.689180, -3.512946],
])
# fmt: on


def _pipeline() -> Pipeline:
    """The DELAY_ACF_SST feature chain, kept small enough to pin by hand."""

    return (
        Pipeline()
        .then(Magnitude())
        .then(GainNormalize())
        .then(DelayAutocorrelation())
        .then(DelayTaps(num_taps=2))
        .then(fold_channels_into_feature(AxisName.DELAY))
        .then(SynchrosqueezedPower())
        .then(DyadicFrequencyBands(num_bands=3))
        .then(FixedSizeWindowSum(window_s=0.25, hop_s=0.25))
        .then(LogScale())
        .then(
            FoldAxes(
                axes=(AxisName.BAND, AxisName.FEATURE),
                representation=Representation.FEATURE_VECTOR,
            )
        )
    )


def test_sst_pipeline_matches_golden_features() -> None:
    """The feature chain reproduces the pinned features on a seeded recording."""

    pytest.importorskip("ssqueezepy")
    profile = AcquisitionProfile(
        n_rx_antennas=1, subcarrier_indices=tuple(range(52)), sampling_rate_hz=1000.0
    )
    rng = np.random.default_rng(7)
    shape = (512, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    signal = profile.raw_signal(csi, np.arange(shape[0]) / 1000.0)

    got = _pipeline().compile(profile).pour(signal).single().values
    assert got.shape == EXPECTED.shape
    assert np.allclose(got, EXPECTED, atol=1e-5)
