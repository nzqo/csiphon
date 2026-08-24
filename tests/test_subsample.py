"""SubsampleEvery: plain decimation, keeping every Nth sample and its timestamp."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon.core import CompileError
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.signal import Signal
from csiphon.pipeline import Pipeline
from csiphon.steps import SubsampleEvery


def test_every_n_keeps_every_nth_sample(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """`every=20` keeps `values[::20]` / `times[::20]` exactly, timestamps unchanged."""

    out = (
        Pipeline().then(SubsampleEvery(every=20)).compile(profile).pour(raw_signal)
    ).single()

    kept = np.arange(0, raw_signal.n_samples, 20)
    assert out.n_samples == kept.size  # ceil(800 / 20) == 40
    assert np.array_equal(out.times, raw_signal.times[kept])
    assert np.array_equal(out.values, np.take(raw_signal.values, kept, axis=0))


def test_every_one_is_the_identity(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """`every=1` keeps everything -- a no-op subset."""

    out = (
        Pipeline().then(SubsampleEvery(every=1)).compile(profile).pour(raw_signal)
    ).single()

    assert np.array_equal(out.values, raw_signal.values)
    assert np.array_equal(out.times, raw_signal.times)


@pytest.mark.parametrize("chunk", [1, 7, 100, 800])
def test_streams_like_batch(
    profile: AcquisitionProfile, raw_signal: Signal, chunk: int
) -> None:
    """The global stride is causal, so streaming equals batch for any chunking.

    The stride is counted across the whole stream, not reset per chunk -- so a chunk
    size that is not a multiple of `every` still keeps exactly every 7th sample.
    """

    compiled = Pipeline().then(SubsampleEvery(every=7)).compile(profile)
    batch = compiled.pour(raw_signal).single()
    streamed = stream_in_chunks(compiled, raw_signal, chunk)

    assert np.array_equal(batch.values, streamed.values)
    assert np.array_equal(batch.times, streamed.times)


@pytest.mark.parametrize("every", [0, -1])
def test_rejects_nonpositive_every(profile: AcquisitionProfile, every: int) -> None:
    """A stride of zero or negative is a clear error at compile time."""

    with pytest.raises(CompileError, match="every"):
        Pipeline().then(SubsampleEvery(every=every)).compile(profile)
