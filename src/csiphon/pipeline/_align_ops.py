"""Low-level operations for lining signals up in time and holding them.

These are the primitives shared by both merge alignments: the time-based one
(`Exact` / `Hold` and the streaming `Trap` in merges.py) and the packet-number one
(`Sequence` / `SequenceTrap` in sequence.py). Each acts on a single signal's time
axis; the strategy that decides *which* samples line up lives one level up.

The "keys" a few of these take are just a monotonic coordinate per sample: the
signal's own timestamps, or its unwrapped packet numbers. Everything here works the
same whichever coordinate is used.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_real_array, as_signal_array
from csiphon.core.errors import ClogError
from csiphon.core.signal import Signal


def align_on_time(signals: SequenceABC[Signal]) -> tuple[list[SignalArray], RealArray]:
    """Trim every signal to the timestamps present in all of them, in time order.

    Frame signals (no time axis) are returned untouched. This is the time case of
    `align_on_keys`, aligning on each signal's own timestamps.
    """

    return align_on_keys(signals, [as_real_array(signal.times) for signal in signals])


def align_on_keys(
    signals: SequenceABC[Signal], keys: list[RealArray], reference: int = 0
) -> tuple[list[SignalArray], RealArray]:
    """Keep only the keys shared by every branch.

    The output timestamps come from the reference branch itself, even when the
    keys we aligned on are packet numbers rather than real times.
    """

    dynamic = signals[0].layout.dynamic_index
    if dynamic is None:
        return [as_signal_array(s.values) for s in signals], signals[0].times

    # Keep only the keys that appear in every branch.
    common = keys[0]
    for key in keys[1:]:
        common = np.intersect1d(common, key)

    # For each branch, take the values sitting at those shared keys.
    arrays = [
        as_signal_array(np.take(s.values, np.searchsorted(key, common), axis=dynamic))
        for s, key in zip(signals, keys, strict=True)
    ]

    # The emitted timestamps come from the reference branch's own time axis.
    matched = np.searchsorted(keys[reference], common)
    return arrays, as_real_array(signals[reference].times[matched])


def hold_on_keys(
    signals: SequenceABC[Signal], keys: list[RealArray], reference: int
) -> tuple[list[SignalArray], RealArray]:
    """Sample-and-hold every branch onto the reference branch's key timeline."""

    dynamic = signals[0].layout.dynamic_index
    if dynamic is None:
        return [as_signal_array(s.values) for s in signals], signals[0].times

    reference_keys = keys[reference]
    arrays = [
        hold_values_at(signal, key, reference_keys)
        for signal, key in zip(signals, keys, strict=True)
    ]
    return arrays, as_real_array(signals[reference].times)


def hold_values_at(signal: Signal, keys: RealArray, at_keys: RealArray) -> SignalArray:
    """Each value held at `at_keys`: the most recent sample at or before each key.

    Keys can be any monotonic coordinate (timestamps, packet numbers). A key before
    the first sample takes the first; one after the last takes the last.
    """

    axis = signal.layout.dynamic_index
    assert axis is not None
    index = np.clip(np.searchsorted(keys, at_keys, side="right") - 1, 0, keys.size - 1)
    return as_signal_array(np.take(signal.values, index, axis=axis))


def sample_hold(signal: Signal, at: RealArray) -> SignalArray:
    """Sample-and-hold onto the times `at`, keyed by the signal's own timestamps."""

    return hold_values_at(signal, signal.times, at)


def keep_times(signal: Signal, keep: np.ndarray) -> Signal:
    """Return only the samples where `keep` is True, along the time axis."""

    axis = signal.layout.dynamic_index
    assert axis is not None
    values = as_signal_array(np.compress(keep, signal.values, axis=axis))
    return Signal(
        values=values, times=as_real_array(signal.times[keep]), layout=signal.layout
    )


def concat_time(first: Signal, second: Signal) -> Signal:
    """Concatenate two chunks of one branch along its time axis."""

    axis = first.layout.dynamic_index
    # Traps only run on streams, which always carry a time axis.
    assert axis is not None
    values = np.concatenate([first.values, second.values], axis=axis)
    times = np.concatenate([first.times, second.times])
    return Signal(
        values=as_signal_array(values), times=as_real_array(times), layout=first.layout
    )


def ensure_not_clogged(held: list[Signal | None], max_hold: int) -> None:
    """Fail if any branch has piled up past `max_hold` without ever aligning.

    Unbounded growth means a branch never lines up (a stalled branch, or grids that
    never intersect); past the cap that is a clear error rather than a memory leak.
    """

    for index, buffer in enumerate(held):
        if buffer is None or buffer.n_samples <= max_hold:
            continue
        raise ClogError(
            f"Merge branch {index} is clogged: {buffer.n_samples} samples held "
            "without aligning (a stalled branch, or branches that never line up). "
            "Raise the junction's max_hold if this is expected."
        )
