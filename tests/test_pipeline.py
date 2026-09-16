"""Compilation, description, and input-validation tests."""

import numpy as np
import pytest

from csiphon import AcquisitionProfile, Pipeline
from csiphon.core import CompileError, DataError
from csiphon.steps import DelayAutocorrelation, GainNormalize, Magnitude


def test_describe_draws_the_flow(profile) -> None:
    """describe() draws each step in a flow graph flowing down from the inlet."""

    compiled = Pipeline().then(Magnitude()).then(GainNormalize()).compile(profile)
    text = compiled.describe()
    assert "magnitude" in text and "gain-normalize" in text  # the steps
    assert "inlet" in text  # the flow starts at the inlet
    assert "│" in text  # steps are joined by a flow rail
    assert "streaming  supported" in text  # the pipeline-level streaming line


def test_compile_error_names_failing_step(profile) -> None:
    """A step whose requirement is unmet fails compilation with its name."""

    # DelayAutocorrelation needs magnitude/real input, but raw CSI is complex.
    with pytest.raises(CompileError, match="delay-autocorrelation"):
        Pipeline().then(DelayAutocorrelation()).compile(profile)


def test_inlet_mismatch_is_rejected(profile) -> None:
    """Running a signal with the wrong subcarrier count is rejected."""

    compiled = Pipeline().then(Magnitude()).compile(profile)
    rng = np.random.default_rng(0)
    wrong = rng.standard_normal((10, profile.n_rx_antennas, 8)) + 0j
    other = AcquisitionProfile(
        n_rx_antennas=profile.n_rx_antennas, subcarrier_indices=tuple(range(8))
    )
    # A valid signal for `other` (8 subcarriers) is still wrong for `compiled`
    # (52 subcarriers), so the mismatch is caught at pour, not at construction.
    bad = other.raw_signal(wrong, np.arange(10) / 1000.0)
    with pytest.raises(DataError):
        compiled.pour(bad).single()


def test_pipeline_is_reusable_across_recordings(profile) -> None:
    """One compiled pipeline runs on recordings of different lengths."""

    compiled = Pipeline().then(Magnitude()).then(GainNormalize()).compile(profile)
    rng = np.random.default_rng(3)
    for length in (100, 250, 500):
        shape = (length, profile.n_rx_antennas, profile.n_subcarriers)
        csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        signal = profile.raw_signal(csi, np.arange(length) / 1000.0)
        out = compiled.pour(signal).single()
        assert out.n_samples == length
