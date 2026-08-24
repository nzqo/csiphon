"""DropSamples: reproducibly drop samples to simulate independent or bursty loss."""

from __future__ import annotations

import numpy as np
import pytest

from csiphon.core import CompileError, StreamingError
from csiphon.core.errors import LayoutError
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.signal import Signal
from csiphon.pipeline import Pipeline
from csiphon.steps import Bursty, DropSamples, Independent


def _ramp_signal(n: int, n_subcarriers: int = 4) -> tuple[AcquisitionProfile, Signal]:
    """A signal of `n` samples whose every value equals its own (integer) timestamp."""

    profile = AcquisitionProfile(subcarrier_indices=tuple(range(n_subcarriers)))
    times = np.arange(n, dtype=float)
    values = np.tile(times[:, None], (1, n_subcarriers)).astype(np.complex128)
    return profile, profile.raw_signal(values, times)


def _dropped(step: DropSamples, n: int = 20_000) -> Signal:
    """Run `step` on an `n`-sample ramp and return the surviving signal."""

    profile, signal = _ramp_signal(n)
    return Pipeline().then(step).compile(profile).pour(signal).single()


def test_independent_drops_about_the_loss_rate() -> None:
    """Independent loss keeps roughly `1 - loss_rate` of the samples."""

    out = _dropped(DropSamples(loss_rate=0.3, seed=1))
    # Fixed seed -> deterministic; a tight band (not the full ~2 sigma) so an
    # off-by-one or a threshold flip would actually show. (exact survivors: 0.6966)
    assert 0.694 < out.n_samples / 20_000 < 0.700


def test_survivors_keep_their_own_timestamps() -> None:
    """Dropping thins the set; it never moves a surviving sample in time."""

    out = _dropped(DropSamples(loss_rate=0.5, seed=2))
    # Each value equals its source time, so values and timestamps must still agree.
    assert np.array_equal(out.values.reshape(out.n_samples, -1)[:, 0].real, out.times)
    assert np.all(np.diff(out.times) > 0)  # still in order, still a subset of 0..N-1


def test_keeps_the_first_and_last_sample_by_default() -> None:
    """Even at heavy loss the endpoints survive, so the time span is preserved."""

    out = _dropped(DropSamples(loss_rate=0.95, seed=3))
    assert out.times[0] == 0.0
    assert out.times[-1] == 20_000 - 1


def test_keep_endpoints_only_forces_the_two_endpoints() -> None:
    """keep_endpoints pins the first and last sample and leaves the interior untouched.

    Same seed -> the two runs share the exact interior mask, so the forced version can
    only *add* the endpoints. At 90 % loss the free version almost surely drops at least
    one end, which makes the contrast real rather than vacuous.
    """

    forced = set(_dropped(DropSamples(loss_rate=0.9, seed=4)).times.tolist())
    free = set(
        _dropped(
            DropSamples(loss_rate=0.9, keep_endpoints=False, seed=4)
        ).times.tolist()
    )
    endpoints = {0.0, float(20_000 - 1)}

    assert free <= forced  # forcing only ever adds samples
    assert forced - free <= endpoints  # ...and only the two endpoints
    assert endpoints <= forced  # the forced run keeps both ends
    assert not endpoints <= free  # the free run does not (it dropped at least one)


def test_is_deterministic_in_the_seed() -> None:
    """The same seed drops the same samples; a different seed does not."""

    same_a = _dropped(DropSamples(loss_rate=0.4, seed=7))
    same_b = _dropped(DropSamples(loss_rate=0.4, seed=7))
    other = _dropped(DropSamples(loss_rate=0.4, seed=8))

    assert np.array_equal(same_a.times, same_b.times)
    assert not np.array_equal(same_a.times, other.times)


def test_loss_rate_zero_keeps_everything() -> None:
    """A zero loss rate is a no-op -- every sample survives."""

    out = _dropped(DropSamples(loss_rate=0.0, seed=5), n=500)
    assert out.n_samples == 500


def test_bursty_loss_comes_in_runs_of_the_right_mean_length() -> None:
    """Gilbert-Elliott drops arrive in bursts averaging `mean_burst` samples.

    A dropped run shows up as a gap between surviving timestamps; its length is the
    gap minus one. Endpoints are freed so a burst at either end is not clipped.
    """

    out = _dropped(
        DropSamples(
            loss_rate=0.3, model=Bursty(mean_burst=10.0), keep_endpoints=False, seed=11
        ),
        n=50_000,
    )
    gaps = np.diff(out.times)
    run_lengths = gaps[gaps > 1] - 1  # lengths of the dropped bursts
    assert run_lengths.size > 100  # plenty of bursts to average over
    assert 9.0 < run_lengths.mean() < 11.5  # ~10, the requested mean burst (obs 10.23)


def test_bursty_still_drops_about_the_loss_rate() -> None:
    """Over a long run the bursty model drops close to `loss_rate` overall."""

    out = _dropped(
        DropSamples(loss_rate=0.3, model=Bursty(mean_burst=8.0), seed=12), n=50_000
    )
    assert 0.700 < out.n_samples / 50_000 < 0.710  # ~0.70 survive (obs 0.70468)


def test_independent_and_bursty_differ_at_the_same_rate() -> None:
    """Same loss rate, different clustering -- the two models drop different samples."""

    independent = _dropped(DropSamples(loss_rate=0.3, model=Independent(), seed=9))
    bursty = _dropped(DropSamples(loss_rate=0.3, model=Bursty(mean_burst=10.0), seed=9))
    assert not np.array_equal(independent.times, bursty.times)


@pytest.mark.parametrize("loss_rate", [1.0, 1.5, -0.1])
def test_rejects_loss_rate_outside_zero_to_one(loss_rate: float) -> None:
    """A loss rate must be in [0, 1) (1.0 would drop everything)."""

    profile, _ = _ramp_signal(10)
    with pytest.raises(CompileError, match="loss_rate"):
        Pipeline().then(DropSamples(loss_rate=loss_rate)).compile(profile)


def test_refuses_to_stream() -> None:
    """Dropping is an offline augmentation, so a streaming pipeline is refused."""

    profile, _ = _ramp_signal(10)
    compiled = Pipeline().then(DropSamples(loss_rate=0.3)).compile(profile)
    with pytest.raises(StreamingError, match="drop-samples"):
        compiled.stream()


def test_bursty_is_deterministic_in_the_seed() -> None:
    """Like Independent, the bursty model repeats exactly for a given seed."""

    model = Bursty(mean_burst=8.0)
    same_a = _dropped(DropSamples(loss_rate=0.3, model=model, seed=3))
    same_b = _dropped(DropSamples(loss_rate=0.3, model=model, seed=3))
    assert np.array_equal(same_a.times, same_b.times)


def test_bursty_mean_burst_one_gives_single_sample_bursts() -> None:
    """mean_burst=1 recovers every step, so no dropped run is longer than one sample."""

    out = _dropped(
        DropSamples(
            loss_rate=0.3, model=Bursty(mean_burst=1.0), keep_endpoints=False, seed=5
        ),
        n=5_000,
    )
    run_lengths = np.diff(out.times)
    run_lengths = run_lengths[run_lengths > 1] - 1
    assert run_lengths.size > 0  # there are bursts...
    assert run_lengths.max() == 1  # ...and each is exactly one sample


def test_bursty_at_zero_loss_keeps_everything() -> None:
    """The loss_rate<=0 short-circuit keeps every sample (no good->bad transitions)."""

    out = _dropped(
        DropSamples(loss_rate=0.0, model=Bursty(mean_burst=8.0), seed=6), n=500
    )
    assert out.n_samples == 500


@pytest.mark.parametrize("mean_burst", [0.0, 0.5])
def test_bursty_rejects_mean_burst_below_one(mean_burst: float) -> None:
    """A burst is at least one sample, so a mean below one is rejected up front."""

    with pytest.raises(LayoutError, match="mean_burst"):
        Bursty(mean_burst=mean_burst)
