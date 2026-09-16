"""Measured runs: per-step time, output shape, memory, and named groups."""

import numpy as np
import pytest

from csiphon import AcquisitionProfile, GroupCost, Mean, Pipeline, Siphon, StepCost
from csiphon.pipeline.measure import combine
from csiphon.steps import GainNormalize, Magnitude, WindowedSlope, WindowedVariance


def _branching_siphon(profile: AcquisitionProfile) -> Siphon:
    """magnitude -> gain-normalize -> (slope | variance) -> mean, five nodes."""

    return (
        Pipeline()
        .then(Magnitude())
        .then(GainNormalize())
        .branch(
            slope=Pipeline().then(WindowedSlope()),
            variance=Pipeline().then(WindowedVariance()),
        )
        .merge(using=Mean())
        .compile(profile)
    )


def test_measure_returns_the_same_outlets_as_pour(profile, raw_signal) -> None:
    """measure() is pour() plus bookkeeping: the outlets are identical."""

    siphon = _branching_siphon(profile)
    run = siphon.measure(raw_signal)
    poured = siphon.pour(raw_signal)

    assert list(run.outlets) == list(poured)
    assert np.array_equal(run.outlets.single().values, poured.single().values)


def test_measure_reports_every_node_in_run_order(profile, raw_signal) -> None:
    """One StepCost per node (merges included), numbered as describe() numbers them."""

    siphon = _branching_siphon(profile)
    run = siphon.measure(raw_signal)

    assert [step.number for step in run.steps] == [1, 2, 3, 4, 5]
    assert [step.name for step in run.steps] == [
        "magnitude",
        "gain-normalize",
        "windowed-slope",
        "windowed-variance",
        "mean",
    ]
    assert [step.output for step in run.steps] == [n.output for n in siphon.nodes]


def test_measure_records_time_and_output_shape(profile, raw_signal) -> None:
    """Each step's time is positive and its shape is the array it produced."""

    siphon = Pipeline().then(Magnitude()).then(GainNormalize()).compile(profile)
    run = siphon.measure(raw_signal)

    for step in run.steps:
        assert step.seconds > 0.0
        assert step.shape == raw_signal.values.shape
        assert step.peak_bytes is None and step.added_bytes is None


def test_measure_memory_is_opt_in(profile, raw_signal) -> None:
    """memory=True fills peak and added bytes; the output array is at least added."""

    siphon = Pipeline().then(Magnitude()).compile(profile)
    run = siphon.measure(raw_signal, memory=True)
    (step,) = run.steps

    assert step.peak_bytes is not None and step.added_bytes is not None
    assert step.peak_bytes >= step.added_bytes
    # The magnitude array is retained by the run, so at least its bytes were added.
    assert step.added_bytes >= run.outlets.single().values.nbytes


def test_groups_by_number_or_name_combine_consecutive_steps(
    profile, raw_signal
) -> None:
    """A group span accepts step numbers or names; its cost is the members' sum."""

    siphon = _branching_siphon(profile)
    run = siphon.measure(
        raw_signal,
        groups={"clean-up": ("magnitude", "gain-normalize"), "features": (3, "mean")},
    )

    clean_up = run.groups["clean-up"]
    assert (clean_up.first, clean_up.last) == (1, 2)
    assert clean_up.seconds == pytest.approx(sum(s.seconds for s in run.steps[:2]))
    assert clean_up.shape == run.steps[1].shape

    features = run.groups["features"]
    assert (features.first, features.last) == (3, 5)
    assert features.seconds == pytest.approx(sum(s.seconds for s in run.steps[2:]))

    total = run.total()
    assert (total.first, total.last) == (1, 5)
    assert total.seconds == pytest.approx(sum(s.seconds for s in run.steps))


def test_group_peak_is_rebased_on_the_group_start() -> None:
    """The group peak accounts for what earlier members left allocated.

    Step 1 adds 100 bytes and peaks at 100; step 2 peaks at 150 above its own
    start, which is 100 above the group's, so the group peak is 250, not 150.
    """

    steps = (
        StepCost(1, "a", "_0", 1.0, (2,), peak_bytes=100, added_bytes=100),
        StepCost(2, "b", "_1", 2.0, (3,), peak_bytes=150, added_bytes=-40),
    )
    group = combine("g", steps)

    assert group == GroupCost("g", 1, 2, 3.0, (3,), peak_bytes=250, added_bytes=60)


def test_group_without_memory_has_no_memory() -> None:
    """Combining unmeasured steps leaves the group's memory fields None."""

    steps = (
        StepCost(1, "a", "_0", 1.0, (2,), None, None),
        StepCost(2, "b", "_1", 2.0, (3,), None, None),
    )
    assert combine("g", steps).peak_bytes is None


@pytest.mark.parametrize(
    ("span", "message"),
    [
        ((0, 2), "numbered 1 to 5"),
        ((1, 6), "numbered 1 to 5"),
        ((3, 1), "runs backwards"),
        (("nope", 2), "matches 0 steps"),
    ],
)
def test_bad_group_spans_are_rejected_before_running(
    profile, raw_signal, span, message
) -> None:
    """A span outside the pipeline, backwards, or with an unknown name fails."""

    siphon = _branching_siphon(profile)
    with pytest.raises(ValueError, match=message):
        siphon.measure(raw_signal, groups={"g": span})


def test_ambiguous_step_name_in_a_group_is_rejected(profile, raw_signal) -> None:
    """A name shared by several steps cannot bound a group; use the number."""

    siphon = Pipeline().then(Magnitude()).then(Magnitude()).compile(profile)
    with pytest.raises(ValueError, match="matches 2 steps"):
        siphon.measure(raw_signal, groups={"g": ("magnitude", "magnitude")})


def test_render_lists_steps_groups_and_total(profile, raw_signal) -> None:
    """The table names every step, each group with its span, and the total."""

    siphon = _branching_siphon(profile)
    run = siphon.measure(raw_signal, memory=True, groups={"clean-up": (1, 2)})
    text = str(run)

    assert "magnitude" in text and "mean" in text
    assert "1-2   clean-up" in text
    assert "1-5   total" in text
    assert "peak" in text and "added" in text
