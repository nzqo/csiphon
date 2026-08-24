"""Dimensionality-reduction steps: subcarrier selection and PCA."""
# Tests repeat the batch/stream compile-and-compare pattern by design.
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon import Pipeline
from csiphon.core import AxisName, StreamingError
from csiphon.steps import (
    Magnitude,
    PrincipalComponents,
    SelectAxis,
    fit_pca_basis,
)


def test_select_axis_keeps_meaning(profile, raw_signal) -> None:
    """SelectAxis shrinks the subcarrier axis but keeps it a subcarrier axis."""

    keep = (0, 2, 4, 6, 8)
    pipeline = Pipeline().then(Magnitude()).then(SelectAxis(indices=keep))
    out = pipeline.compile(profile).pour(raw_signal).single()
    axis = out.layout.axis(AxisName.SUBCARRIER)
    assert axis.size == len(keep)
    assert axis.coordinates == keep


@pytest.mark.parametrize("chunk", [1, 25, 400])
def test_select_axis_streams_exactly(profile, raw_signal, chunk) -> None:
    """SelectAxis is pointwise, so streaming equals batch."""

    pipeline = Pipeline().then(Magnitude()).then(SelectAxis(indices=(1, 3, 5)))
    compiled = pipeline.compile(profile)
    batch = compiled.pour(raw_signal).single()
    streamed = stream_in_chunks(compiled, raw_signal, chunk)
    assert np.allclose(batch.values, streamed.values)


def test_pca_renames_axis_to_component(profile, raw_signal) -> None:
    """PCA over subcarriers yields a component axis, not a subcarrier axis."""

    pipeline = Pipeline().then(Magnitude()).then(PrincipalComponents(n_components=4))
    out = pipeline.compile(profile).pour(raw_signal).single()
    assert not out.layout.has_axis(AxisName.SUBCARRIER)
    assert out.layout.axis(AxisName.COMPONENT).size == 4
    assert out.layout.axis(AxisName.COMPONENT).coordinates == (0, 1, 2, 3)


def test_pca_fit_on_data_is_batch_only(profile) -> None:
    """Fit-on-recording PCA cannot stream (non-causal)."""

    pipeline = Pipeline().then(Magnitude()).then(PrincipalComponents(n_components=4))
    compiled = pipeline.compile(profile)
    with pytest.raises(StreamingError, match="principal-components"):
        compiled.stream()


@pytest.mark.parametrize("chunk", [1, 30, 300])
def test_pca_with_prefit_basis_streams_exactly(profile, raw_signal, chunk) -> None:
    """With a pre-fit basis PCA is a per-sample projection and streams exactly."""

    # Fit the basis once on the magnitude of the recording.
    mag = Pipeline().then(Magnitude()).compile(profile).pour(raw_signal).single()
    basis = fit_pca_basis(mag, AxisName.SUBCARRIER, n_components=4)

    pipeline = (
        Pipeline()
        .then(Magnitude())
        .then(PrincipalComponents(n_components=4, basis=basis))
    )
    compiled = pipeline.compile(profile)
    batch = compiled.pour(raw_signal).single()
    streamed = stream_in_chunks(compiled, raw_signal, chunk)
    assert batch.values.shape == streamed.values.shape
    assert np.allclose(batch.values, streamed.values)
