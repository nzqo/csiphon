"""Merge alignment: Exact (strict, loud on empty) and Hold (sample-and-hold)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon import (
    AcquisitionProfile,
    AxisName,
    Exact,
    Fuse,
    Hold,
    Pipeline,
    Stack,
)
from csiphon.core.axes import Axis
from csiphon.core.errors import ClogError, DataError, LayoutError
from csiphon.core.layout import Layout
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline import concat_signals
from csiphon.pipeline.merges import Junction
from csiphon.steps import (
    GainNormalize,
    Magnitude,
    WindowedFFTPower,
    WindowedVariance,
)
from csiphon.steps.pooling import MeanOverAxes

_LAYOUT = Layout(
    (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 2)),
    Representation.CHANNEL_FREQUENCY_RESPONSE,
    ValueKind.MAGNITUDE,
)


def _signal(times, base: float) -> Signal:
    """A 2-subcarrier signal whose values encode their timestamp (for easy checking)."""

    t = np.asarray(times, dtype=float)
    values = np.stack([t + base, t + base + 100.0], axis=1)
    return Signal(values=values, times=t, layout=_LAYOUT)


def test_exact_raises_when_branches_share_no_timestamps() -> None:
    """A whole-recording exact merge fails loudly instead of emptying silently."""

    a = _signal(np.arange(10), 0.0)
    b = _signal(np.arange(10) + 0.5, 1000.0)  # offset grid: no shared timestamps
    with pytest.raises(DataError, match="different time grids"):
        Exact().align([a, b])


def test_exact_keeps_only_shared_timestamps() -> None:
    """Exact trims to the timestamps present in every branch."""

    a = _signal(np.arange(10), 0.0)
    b = _signal(np.arange(0, 10, 2), 1000.0)  # 0,2,4,6,8 -- a subset of a
    arrays, times = Exact().align([a, b])

    assert np.array_equal(times, np.arange(0, 10, 2))
    assert all(array.shape == (5, 2) for array in arrays)


def test_hold_takes_each_branch_last_value_at_or_before_reference() -> None:
    """Hold samples every branch onto the reference timeline, holding the last value."""

    a = _signal(np.arange(10), 0.0)  # reference: 0..9
    b = _signal([1.5, 4.5, 7.5], 1000.0)  # sparse, offset
    arrays, times = Hold(reference=0).align([a, b])

    assert np.array_equal(times, np.arange(10))
    # b held onto a's times: before 1.5 -> first value; then last <= t.
    expected = np.array([1001.5] * 5 + [1004.5] * 3 + [1007.5] * 2)
    assert np.allclose(arrays[1][:, 0], expected)


def test_hold_reference_out_of_range_fails_at_compile() -> None:
    """A reference branch index beyond the branch count fails at compile, by name."""

    junction = Junction(Stack(into=AxisName.FEATURE), Hold(reference=5))
    with pytest.raises(LayoutError, match="reference branch 5"):
        junction.output_layout([_LAYOUT, _LAYOUT])


def test_hold_forward_fills_before_the_other_branch_starts() -> None:
    """Reference times before an other branch's first sample take that first value."""

    a = _signal([0.0, 1.0, 2.0, 3.0], 0.0)
    b = _signal([2.0], 1000.0)  # a single sample, only at t=2
    arrays, _ = Hold(reference=0).align([a, b])
    assert np.allclose(arrays[1][:, 0], [1002.0] * 4)  # its one value, held throughout


def test_hold_includes_a_sample_at_the_exact_reference_time() -> None:
    """'At or before' includes an exact timestamp match, not just strictly earlier."""

    a = _signal([0.0, 1.0, 2.0], 0.0)
    b = _signal([1.0, 2.0], 1000.0)  # values 1001 at t=1, 1002 at t=2
    arrays, _ = Hold(reference=0).align([a, b])
    # t=0 -> first (1001); t=1 -> the sample exactly at 1.0 (1001); t=2 -> 1002.
    assert np.allclose(arrays[1][:, 0], [1001.0, 1001.0, 1002.0])


def test_hold_across_three_branches() -> None:
    """Hold works with more than two branches, each held independently."""

    a = _signal([0.0, 1.0, 2.0, 3.0], 0.0)  # reference
    b = _signal([0.5, 2.5], 10.0)  # -> 10.5, 12.5
    c = _signal([1.5], 20.0)  # a single sample -> 21.5
    arrays, times = Hold(reference=0).align([a, b, c])

    assert np.array_equal(times, a.times)
    # b held at 0,1,2,3: 2.5 only counts once t reaches 3 (it is 2.5 > 2).
    assert np.allclose(arrays[1][:, 0], [10.5, 10.5, 10.5, 12.5])
    assert np.allclose(arrays[2][:, 0], [21.5] * 4)


@pytest.mark.parametrize("chunk_a, chunk_b", [(4, 2), (1, 5), (7, 1), (3, 3)])
def test_hold_streaming_matches_batch(  # pylint: disable=too-many-locals
    chunk_a: int, chunk_b: int
) -> None:
    """Streaming a Hold merge gives the exact same result as the whole-recording one.

    This must hold no matter how the two branches are chopped up, so the two
    branches are fed in *different-sized* chunks that do not arrive in lock-step.
    """

    # A dense per-sample reference; B sparse and off its grid (window-centre-like).
    a = _signal(np.arange(30.0), 0.0)
    b = _signal(np.arange(1.5, 30, 3.3), 1000.0)
    junction = Junction(Stack(into=AxisName.FEATURE), Hold(reference=0))
    out_layout = junction.output_layout([_LAYOUT, _LAYOUT])

    # The whole-recording answer we must reproduce chunk by chunk.
    batch = junction.process([a, b], out_layout)

    trap = junction.trap(out_layout)
    empty = _signal(np.zeros(0), 0.0)

    def piece(signal: Signal, start: int, size: int) -> Signal:
        """One chunk of a branch: the samples in [start, start + size)."""

        stop = start + size
        return Signal(
            values=signal.values[start:stop],
            times=signal.times[start:stop],
            layout=_LAYOUT,
        )

    # Step through both branches at their own chunk sizes. Each step pushes
    # whatever new samples each branch has this round (an empty chunk once a
    # branch is exhausted), and the trap emits whatever has become safe.
    outputs = []
    for step in range(max(len(a.times), len(b.times))):
        at_a, at_b = step * chunk_a, step * chunk_b
        chunk_a_signal = piece(a, at_a, chunk_a) if at_a < a.n_samples else empty
        chunk_b_signal = piece(b, at_b, chunk_b) if at_b < b.n_samples else empty
        outputs.append(trap.push([chunk_a_signal, chunk_b_signal]))
    # Flush releases the tail (reference times past the other branch's last sample).
    outputs.append(trap.flush())

    streamed = concat_signals([out for out in outputs if out.n_samples], out_layout)
    assert np.array_equal(batch.times, streamed.times)
    assert np.allclose(batch.values, streamed.values)


def test_hold_streaming_with_a_lagging_branch_and_empty_chunks() -> None:
    """The reference can arrive well ahead of the other branch (with empty chunks)."""

    a = _signal(np.arange(20.0), 0.0)
    b = _signal([2.0, 9.0, 15.0], 1000.0)
    junction = Junction(Stack(into=AxisName.FEATURE), Hold(reference=0))
    out_layout = junction.output_layout([_LAYOUT, _LAYOUT])
    batch = junction.process([a, b], out_layout)

    trap = junction.trap(out_layout)
    empty = _signal(np.zeros(0), 0.0)
    outputs = [
        trap.push([a, empty]),  # all of the reference first; other still silent
        trap.push([empty, _signal([2.0, 9.0], 1000.0)]),  # other catches up in pieces
        trap.push([empty, _signal([15.0], 1000.0)]),
        trap.flush(),
    ]
    streamed = concat_signals([out for out in outputs if out.n_samples], out_layout)
    assert np.array_equal(batch.times, streamed.times)
    assert np.allclose(batch.values, streamed.values)


def test_hold_streaming_clogs_when_a_branch_stalls() -> None:
    """If an other branch never produces, the reference piles up and clogs (loudly)."""

    junction = Junction(Stack(into=AxisName.FEATURE), Hold(reference=0), max_hold=20)
    out_layout = junction.output_layout([_LAYOUT, _LAYOUT])
    trap = junction.trap(out_layout)
    silent = _signal(np.zeros(0), 0.0)

    with pytest.raises(ClogError):
        for step in range(30):
            trap.push([_signal(np.arange(step * 3, step * 3 + 3), 0.0), silent])


def test_hold_streaming_matches_batch_across_three_branches() -> None:
    """A Hold merge of three branches streams identically to the whole-recording run.

    Each branch is on its own grid and arrives in its own chunk size, so the trap
    must hold and release three independent carries at once -- not just a pair.
    """

    reference = _signal(np.arange(30.0), 0.0)  # dense per-sample reference
    middle = _signal(np.arange(1.5, 30, 3.3), 1000.0)  # sparse, offset grid
    slow = _signal(np.arange(0.7, 30, 5.1), 2000.0)  # sparser still
    branches = [reference, middle, slow]

    junction = Junction(Stack(into=AxisName.FEATURE), Hold(reference=0))
    out_layout = junction.output_layout([_LAYOUT, _LAYOUT, _LAYOUT])
    batch = junction.process(branches, out_layout)

    trap = junction.trap(out_layout)
    empty = _signal(np.zeros(0), 0.0)

    def piece(signal: Signal, start: int, size: int) -> Signal:
        """One chunk of a branch: the samples in [start, start + size)."""

        stop = start + size
        return Signal(
            values=signal.values[start:stop],
            times=signal.times[start:stop],
            layout=_LAYOUT,
        )

    # Each branch advances at a different pace (3, 2, and 1 samples per round),
    # pushing an empty chunk once it runs out. The trap emits what has become safe.
    sizes = [3, 2, 1]
    outputs = []
    for step in range(len(reference.times)):
        chunks = [
            piece(branch, step * size, size)
            if step * size < branch.n_samples
            else empty
            for branch, size in zip(branches, sizes, strict=True)
        ]
        outputs.append(trap.push(chunks))
    outputs.append(trap.flush())

    streamed = concat_signals([out for out in outputs if out.n_samples], out_layout)
    assert np.array_equal(batch.times, streamed.times)
    assert np.allclose(batch.values, streamed.values)


def test_merge_clogs_when_one_of_three_branches_never_arrives() -> None:
    """Clogging is detected under a multi-way merge, not just a two-way one.

    Two of three branches keep producing while the third stays silent, so the two
    pile up unmatched until they exceed max_hold and the merge raises loudly.
    """

    junction = Junction(Stack(into=AxisName.FEATURE), max_hold=20)
    out_layout = junction.output_layout([_LAYOUT, _LAYOUT, _LAYOUT])
    trap = junction.trap(out_layout)
    silent = _signal(np.zeros(0), 0.0)

    with pytest.raises(ClogError):
        for step in range(30):
            live = _signal(np.arange(step * 3, step * 3 + 3), 0.0)
            trap.push([live, live, silent])  # the third branch never produces


def test_exact_across_three_branches_keeps_the_common_subset() -> None:
    """Exact intersects the timestamps of all branches, not just two."""

    a = _signal(np.arange(12), 0.0)
    b = _signal(np.arange(0, 12, 2), 100.0)  # 0,2,4,6,8,10
    c = _signal(np.arange(0, 12, 3), 200.0)  # 0,3,6,9
    _, times = Exact().align([a, b, c])
    assert np.array_equal(times, np.array([0.0, 6.0]))  # shared by all three


def test_hold_reference_picks_the_output_timeline() -> None:
    """`reference` chooses whose timestamps the output lands on."""

    dense = _signal(np.arange(10), 0.0)
    sparse = _signal([2.0, 6.0], 1000.0)

    _, on_dense = Hold(reference=0).align([dense, sparse])
    _, on_sparse = Hold(reference=1).align([dense, sparse])
    assert np.array_equal(on_dense, dense.times)
    assert np.array_equal(on_sparse, sparse.times)


def test_merge_defaults_to_exact() -> None:
    """A merge with no `align` uses Exact."""

    assert isinstance(Junction(Stack()).align, Exact)


def test_cross_grid_merge_needs_hold(profile: AcquisitionProfile) -> None:
    """A per-sample branch fused with a windowed one: Exact drops rows, Hold keeps them.

    The variance branch is per-sample; the spectral branch is on the coarser
    window-centre grid. Exact keeps only the timestamps they happen to share (the
    window centres), silently shrinking the output; Hold carries each spectral
    value across the per-sample timeline, so the output keeps the full length.
    """

    length = 2000
    rng = np.random.default_rng(0)
    shape = (length, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 0j
    signal = profile.raw_signal(csi, np.arange(length) / 1000.0)

    def fused(align: Exact | Hold) -> int:
        siphon = (
            Pipeline()
            .then(Magnitude())
            .branch(
                variance=Pipeline().then(WindowedVariance(win_size_s=0.05)),
                spectral=Pipeline()
                .then(GainNormalize())
                .then(WindowedFFTPower())
                .then(MeanOverAxes(axes=(AxisName.SUBCARRIER,))),
            )
            .merge(using=Fuse(), align=align)
            .compile(profile)
        )
        return siphon.pour(signal).single().n_samples

    assert fused(Exact()) < length // 10  # only the shared window centres survive
    assert fused(Hold()) == length  # the per-sample reference length is kept


@pytest.mark.parametrize("chunk", [128, 256, 333])
def test_cross_grid_hold_pipeline_streams_like_batch(
    profile: AcquisitionProfile, chunk: int
) -> None:
    """A whole windowed+per-sample pipeline with a Hold merge streams like batch.

    This is the full stack: real steps produce real timestamps on two grids, fed
    through the streaming session in chunks, and the result must match the
    whole-recording run exactly.
    """

    length = 1500
    rng = np.random.default_rng(0)
    shape = (length, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    signal = profile.raw_signal(csi, np.arange(length) / 1000.0)

    siphon = (
        Pipeline()
        .then(Magnitude())
        .branch(
            variance=Pipeline().then(WindowedVariance(win_size_s=0.05)),
            spectral=Pipeline()
            .then(GainNormalize())
            .then(WindowedFFTPower())
            .then(MeanOverAxes(axes=(AxisName.SUBCARRIER,))),
        )
        .merge(using=Fuse(), align=Hold())
        .compile(profile)
    )

    batch = siphon.pour(signal).single()
    streamed = stream_in_chunks(siphon, signal, chunk)
    assert np.array_equal(batch.times, streamed.times)
    assert np.allclose(batch.values, streamed.values)
