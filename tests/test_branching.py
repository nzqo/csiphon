"""Branching: fan-out, fork/merge, probes, and stream-equals-batch through junctions."""

# Branch/merge setup overlaps with other modules by design.
# pylint: disable=duplicate-code
import numpy as np
import pytest

from csiphon import (
    AcquisitionProfile,
    Axis,
    AxisName,
    Concatenate,
    Layout,
    Mean,
    Pipeline,
    Representation,
    Signal,
    Stack,
    ValueKind,
    create_signal,
)
from csiphon.core.errors import ClogError, CompileError, LayoutError
from csiphon.pipeline import align_on_time, concat_signals
from csiphon.pipeline.merges import Junction
from csiphon.steps import (
    ChannelImpulseResponse,
    GainNormalize,
    Magnitude,
    Power,
    WindowedVariance,
)


def _var(win: float) -> Pipeline:
    """A one-step branch: a rolling variance with the given window."""

    return Pipeline().then(WindowedVariance(win_size_s=win))


def test_branch_fans_out_to_named_outlets(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """
    A branch with no merge yields one outlet per branch, matching linear pipelines.
    """

    siphon = (
        Pipeline().then(Magnitude()).branch(a=_var(0.05), b=_var(0.2)).compile(profile)
    )
    out = siphon.pour(raw_signal)
    assert set(out.keys()) == {"a", "b"}

    only_a = (
        Pipeline()
        .then(Magnitude())
        .then(WindowedVariance(win_size_s=0.05))
        .compile(profile)
    )
    only_b = (
        Pipeline()
        .then(Magnitude())
        .then(WindowedVariance(win_size_s=0.2))
        .compile(profile)
    )
    assert np.allclose(out["a"].values, only_a.pour(raw_signal).single().values)
    assert np.allclose(out["b"].values, only_b.pour(raw_signal).single().values)


def test_stack_merge_adds_a_branch_axis(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """Stack puts the branches side by side on a new axis of length = branch count."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .branch(a=_var(0.05), b=_var(0.2))
        .merge(using=Stack(into=AxisName.FEATURE))
        .compile(profile)
    )
    out = siphon.pour(raw_signal).single()
    assert out.layout.axis(AxisName.FEATURE).size == 2

    fanned = (
        Pipeline().then(Magnitude()).branch(a=_var(0.05), b=_var(0.2)).compile(profile)
    )
    parts = fanned.pour(raw_signal)
    assert np.allclose(out.values[..., 0], parts["a"].values)
    assert np.allclose(out.values[..., 1], parts["b"].values)


def test_describe_shows_branch_structure(profile: AcquisitionProfile) -> None:
    """describe() draws the branches, the merge, and its inputs by name."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .branch(fast=_var(0.05), slow=_var(0.2))
        .merge(using=Stack(), name="joined")
        .compile(profile)
    )
    text = siphon.describe()

    # The flow graph draws both branch aliases side by side and names the merge op.
    assert any("fast" in line and "slow" in line for line in text.splitlines())
    assert "stack" in text


def test_concatenate_merge_grows_an_axis(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """Concatenate lays branches end to end along one existing axis."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .branch(a=_var(0.05), b=_var(0.2))
        .merge(using=Concatenate(axis=AxisName.SUBCARRIER))
        .compile(profile)
    )
    out = siphon.pour(raw_signal).single()
    assert out.layout.axis(AxisName.SUBCARRIER).size == 2 * profile.n_subcarriers


def test_mean_merge_averages_branches(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """Mean reduces identical-shaped branches element by element."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .branch(a=_var(0.05), b=_var(0.2))
        .merge(using=Mean())
        .compile(profile)
    )
    out = siphon.pour(raw_signal).single()

    fanned = (
        Pipeline().then(Magnitude()).branch(a=_var(0.05), b=_var(0.2)).compile(profile)
    )
    parts = fanned.pour(raw_signal)
    assert np.allclose(out.values, 0.5 * (parts["a"].values + parts["b"].values))


def test_probe_keeps_an_intermediate_as_an_outlet(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """A probe adds a named outlet without stopping the line."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .probe("mag")
        .then(WindowedVariance(win_size_s=0.05))
    ).compile(profile)
    out = siphon.pour(raw_signal)
    assert set(out.keys()) == {"mag", "out"}

    magnitude = Pipeline().then(Magnitude()).compile(profile).pour(raw_signal).single()
    assert np.allclose(out["mag"].values, magnitude.values)


def test_probe_on_a_line_that_branches_keeps_its_values(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """A probed line read by two later branches is still returned whole by pour."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .probe("mag")
        .branch(a=_var(0.05), b=_var(0.2))
        .merge(using=Mean())
    ).compile(profile)
    out = siphon.pour(raw_signal)

    magnitude = Pipeline().then(Magnitude()).compile(profile).pour(raw_signal).single()
    assert np.allclose(out["mag"].values, magnitude.values)


def test_nested_branches_compose(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """A branch can itself branch and merge; the whole tree runs to one outlet."""

    inner = _var(0.05).branch(a=_var(0.1), b=_var(0.2)).merge(using=Mean())
    siphon = (
        Pipeline()
        .then(Magnitude())
        .branch(deep=inner, plain=_var(0.3))
        .merge(using=Stack(into=AxisName.FEATURE))
        .compile(profile)
    )
    out = siphon.pour(raw_signal).single()
    assert out.layout.axis(AxisName.FEATURE).size == 2


def test_then_after_open_branch_is_rejected() -> None:
    """A bare .then with several live lines fails with a clear message."""

    forked = Pipeline().then(Magnitude()).branch(a=_var(0.05), b=_var(0.2))
    with pytest.raises(LayoutError):
        forked.then(Magnitude())


def test_incompatible_merge_fails_to_compile(profile: AcquisitionProfile) -> None:
    """Merging branches with mismatched axes is caught at compile time."""

    pipeline = (
        Pipeline()
        .branch(
            taps=Pipeline().then(ChannelImpulseResponse(num_taps=4)),
            raw=Pipeline(),
        )
        .merge(using=Concatenate(axis=AxisName.SUBCARRIER))
    )
    with pytest.raises(CompileError):
        pipeline.compile(profile)


def test_merge_needs_at_least_two_branches() -> None:
    """A merge combines several lines, so a single input is rejected up front."""

    # After one step only one line is live; merging it with nothing is meaningless.
    with pytest.raises(LayoutError, match="at least two"):
        Pipeline().then(Magnitude()).merge(using=Mean())


def test_stack_rejects_branches_of_different_value_kinds(
    profile: AcquisitionProfile,
) -> None:
    """Stack combines like with like: mixing value kinds (magnitude vs power) fails.

    (Fuse is the strategy that deliberately allows this; Stack/Mean/Concatenate
    require the branches to be the same kind of quantity.)
    """

    pipeline = (
        Pipeline()
        .then(Magnitude())
        .branch(
            amplitude=Pipeline().then(GainNormalize()),  # stays magnitude
            energy=Pipeline().then(Power()),  # becomes power
        )
        .merge(using=Stack(into=AxisName.FEATURE))
    )
    with pytest.raises(CompileError, match="value kind"):
        pipeline.compile(profile)


@pytest.mark.parametrize("chunk", [37, 128])
def test_branched_stream_equals_batch(
    profile: AcquisitionProfile, raw_signal: Signal, chunk: int
) -> None:
    """A branched, merged pipeline streams identically to pouring it whole."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .branch(a=_var(0.05), b=_var(0.2))
        .merge(using=Stack(into=AxisName.FEATURE))
        .compile(profile)
    )
    batch = siphon.pour(raw_signal).single()

    stream = siphon.stream()
    pieces = []
    for start in range(0, raw_signal.n_samples, chunk):
        piece = Signal(
            values=raw_signal.values[start : start + chunk],
            times=raw_signal.times[start : start + chunk],
            layout=raw_signal.layout,
        )
        pieces.append(stream.flow(piece).single())
    pieces.append(stream.flush().single())
    streamed = concat_signals(pieces, siphon.outlet_layout)

    assert np.allclose(batch.values, streamed.values)
    assert np.allclose(batch.times, streamed.times)


def _stream_outlets(siphon, signal: Signal, chunk: int) -> dict[str, Signal]:
    """Feed a signal through a session in chunks and reassemble every outlet."""

    stream = siphon.stream()
    collected: dict[str, list[Signal]] = {}
    for start in range(0, signal.n_samples, chunk):
        piece = Signal(
            values=signal.values[start : start + chunk],
            times=signal.times[start : start + chunk],
            layout=signal.layout,
        )
        for name, part in stream.flow(piece).items():
            collected.setdefault(name, []).append(part)
    for name, part in stream.flush().items():
        collected.setdefault(name, []).append(part)
    return {
        name: concat_signals(
            [part for part in parts if part.n_samples], siphon.outlet_layouts[name]
        )
        for name, parts in collected.items()
    }


@pytest.mark.parametrize("chunk", [37, 128])
def test_fanned_branches_stream_to_both_outlets(
    profile: AcquisitionProfile, raw_signal: Signal, chunk: int
) -> None:
    """An unmerged branch is allowed and streams: each of its two outlets matches batch.

    This is the positive side of the unmerged-branch rule -- a branch with no merge
    is not an error, it simply exposes one outlet per line -- and it must behave the
    same in streaming as in batch, independently per outlet.
    """

    siphon = (
        Pipeline().then(Magnitude()).branch(a=_var(0.05), b=_var(0.2)).compile(profile)
    )
    batch = siphon.pour(raw_signal)
    assert set(batch.keys()) == {"a", "b"}  # two outlets, no merge

    streamed = _stream_outlets(siphon, raw_signal, chunk)
    assert set(streamed.keys()) == {"a", "b"}
    for name in batch:
        assert np.allclose(streamed[name].values, batch[name].values)
        assert np.allclose(streamed[name].times, batch[name].times)


@pytest.mark.parametrize("chunk", [37, 128])
def test_probe_outlet_streams_alongside_the_main_line(
    profile: AcquisitionProfile, raw_signal: Signal, chunk: int
) -> None:
    """A probe outlet and the main outlet both stream identically to batch."""

    siphon = (
        Pipeline()
        .then(Magnitude())
        .probe("mag")
        .then(WindowedVariance(win_size_s=0.05))
    ).compile(profile)
    batch = siphon.pour(raw_signal)
    assert set(batch.keys()) == {"mag", "out"}

    streamed = _stream_outlets(siphon, raw_signal, chunk)
    assert set(streamed.keys()) == {"mag", "out"}
    for name in batch:
        assert np.allclose(streamed[name].values, batch[name].values)


def _feature_frame(times: list[float], fill: float) -> Signal:
    """A tiny `(time, subcarrier=3)` real branch signal, every value = fill."""

    layout = Layout(
        axes=(
            Axis.dynamic(AxisName.TIME, unit="s"),
            Axis.sized(AxisName.SUBCARRIER, 3),
        ),
        representation=Representation.CHANNEL_FREQUENCY_RESPONSE,
        values=ValueKind.REAL,
    )
    values = np.full((len(times), 3), fill, dtype=np.float64)
    return create_signal(values, np.asarray(times, dtype=np.float64), layout)


def test_trap_aligns_branches_with_different_time_supports() -> None:
    """The junction trap holds unmatched samples and emits only aligned timestamps."""

    branch_layout = _feature_frame([0.0], 0.0).layout
    junction = Junction(Stack(into=AxisName.FEATURE))
    out_layout = junction.output_layout([branch_layout, branch_layout])
    trap = junction.trap(out_layout)

    # Branch a and b arrive in mismatched chunks with different time supports.
    emitted = [
        trap.push(
            [_feature_frame([0.0, 1.0], 1.0), _feature_frame([0.0, 1.0, 2.0], 2.0)]
        ),
        trap.push([_feature_frame([2.0, 3.0], 1.0), _feature_frame([3.0, 4.0], 2.0)]),
        trap.flush(),
    ]
    streamed = concat_signals(emitted, out_layout)

    # Batch reference: align the full branches and stack them.
    full_a = _feature_frame([0.0, 1.0, 2.0, 3.0], 1.0)
    full_b = _feature_frame([0.0, 1.0, 2.0, 3.0, 4.0], 2.0)
    arrays, times = align_on_time([full_a, full_b])
    expected = np.stack(arrays, axis=-1)

    assert np.allclose(streamed.times, times)
    assert np.allclose(streamed.values, expected)


def test_trap_clogs_when_a_branch_stalls() -> None:
    """A branch that never aligns piles up past max_hold and raises ClogError."""

    junction = Junction(Stack(into=AxisName.FEATURE), max_hold=3)
    branch_layout = _feature_frame([0.0], 0.0).layout
    trap = junction.trap(junction.output_layout([branch_layout, branch_layout]))

    empty = _feature_frame([], 0.0)  # branch 1 never arrives, so branch 0 piles up

    # 2 samples held is within max_hold=3, so this push must not clog.
    trap.push([_feature_frame([0.0, 1.0], 1.0), empty])

    # A second push takes it to 4 held, past max_hold=3, so now it clogs.
    with pytest.raises(ClogError):
        trap.push([_feature_frame([2.0, 3.0], 1.0), empty])
