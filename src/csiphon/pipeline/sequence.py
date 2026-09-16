"""Aligning merge branches on packet (sequence) numbers instead of timestamps.

The same packet carries the same number on every receiver, so matching on that
number lines receivers up even when their clocks drift. This module holds the whole
sequence-number feature: the `Sequence` coordinate (batch), the packet math it needs
(unwrap + cross-branch calibration), and `SequenceTrap` (the streaming version).

The time-based merge machinery (strategies, `Exact` / `Hold`, `Junction`, `Trap`)
lives in merges.py and uses this module for the sequence case.
"""

from collections.abc import Callable
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_real_array
from csiphon.core.errors import DataError, StreamingError
from csiphon.core.layout import Layout
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline._align_ops import (
    concat_time,
    ensure_not_clogged,
    hold_values_at,
    keep_times,
)

# A merge strategy's `combine`, taken as a plain callable so this module does not
# depend on merges.py (which would be a cycle: merges builds the traps here).
CombineFn = Callable[[SequenceABC[SignalArray], SequenceABC[Layout]], SignalArray]


@dataclass(frozen=True, slots=True)
class Sequence:
    """Align branches on per-sample packet (sequence) numbers.

    The same packet carries the same number on every receiver, so this matches
    packets even when receiver clocks drift, better than timestamps for that.

    Sequence numbers wrap (mod `period`, e.g. 4096). Give `period` and each branch
    is unwrapped into a monotonic packet index first, so a recording longer than one
    period still aligns. Leave `period` as None and the raw numbers are used as-is,
    which is fine only when the recording does not wrap within itself. Either way the
    branches are calibrated against each other: a same-numbered, close-in-time sample
    must exist within `tolerance_s`, so a wrapped number from another epoch is never
    mistaken for a match; if none is found the merge raises. (Streaming needs a
    period, for the incremental unwrap.)
    """

    period: int | None = None
    tolerance_s: float = 0.05

    def keys(self, signals: SequenceABC[Signal]) -> list[RealArray]:
        """One alignment key per branch: its calibrated (and unwrapped) packet index."""

        if any(signal.sequence is None for signal in signals):
            raise DataError(
                "Aligning on sequence numbers needs Signal.sequence on every branch."
            )
        # Without a period the raw numbers are the keys; with one, unwrap them.
        if self.period is None:
            numbers = [_sequence_of(signal) for signal in signals]
        else:
            numbers = [_unwrap(_sequence_of(signal), self.period) for signal in signals]
        return _calibrate_offsets(signals, numbers, self.tolerance_s)


def reject_sequence_streaming(on: object) -> None:
    """Sequence alignment is batch-only for some cases; refuse it in a stream."""

    if isinstance(on, Sequence):
        raise StreamingError(
            "Streaming with sequence-number alignment is not supported yet; "
            "pour the recording instead."
        )


# --- packet-number math ------------------------------------------------------


def _sequence_of(signal: Signal) -> RealArray:
    """A branch's sequence numbers (the caller has checked they are present)."""

    assert signal.sequence is not None
    return signal.sequence


def _unwrap(raw: RealArray, period: int) -> RealArray:
    """Undo modular wrap-around: add `period` at every true wrap.

    A wrap drops the number by almost a whole `period` (e.g. period-1 -> 0), so only a
    large backward jump (more than half a period) counts. A small backward step is a
    reordered or duplicated packet, not a wrap, and must not shift the epoch.
    """

    if raw.size < 2:
        return as_real_array(raw)
    wraps = np.concatenate([[0], np.cumsum(np.diff(raw) < -period / 2)])
    return as_real_array(raw + period * wraps)


def _calibrate_offsets(
    signals: SequenceABC[Signal], unwrapped: list[RealArray], tolerance: float
) -> list[RealArray]:
    """Shift each branch so a same-packet sample coincides with the first branch.

    Unwrapping is per-branch, so two branches that started on different sides of a
    wrap land in different epochs for the same packet. Anchor on the first sample
    the branches share by number *and* whose timestamps agree (within `tolerance`),
    and subtract the resulting offset so they land on one absolute packet index.
    """

    aligned = [unwrapped[0]]
    for other, other_unwrapped in zip(signals[1:], unwrapped[1:], strict=True):
        offset = _first_match_offset(
            signals[0], unwrapped[0], other, other_unwrapped, tolerance
        )
        aligned.append(as_real_array(other_unwrapped - offset))
    return aligned


def _match_within_tolerance(
    numbers: RealArray,
    times: RealArray,
    target: float,
    at_time: float,
    tolerance: float,
) -> int | None:
    """First index carrying `target` whose time is within `tolerance` of `at_time`.

    Same packet, close in time, so a wrapped recurrence of the number, far away in
    time, is never mistaken for a match. Returns None if there is no such sample.
    """

    for index in np.nonzero(numbers == target)[0]:
        if abs(float(times[index]) - at_time) <= tolerance:
            return int(index)
    return None


def _first_match_offset(
    reference: Signal,
    reference_keys: RealArray,
    other: Signal,
    other_keys: RealArray,
    tolerance: float,
) -> float:
    """Offset (in packets) that lines `other` up with the earliest shared packet.

    Walk the reference's samples in time order; for each, look for the other
    branch's sample with the same raw number whose timestamp is within
    `tolerance`. The first such pair is the same physical packet, so their unwrapped
    keys should coincide: the difference is the epoch offset to remove.
    """

    # The raw (still-wrapped) packet numbers of both branches.
    other_raw = _sequence_of(other)
    reference_raw = _sequence_of(reference)

    # Find the first packet both branches have (same number, close in time). At
    # that packet the gap between their indices is the offset to line them up.
    for i in range(reference.n_samples):
        match = _match_within_tolerance(
            other_raw,
            other.times,
            float(reference_raw[i]),
            float(reference.times[i]),
            tolerance,
        )
        if match is not None:
            return float(other_keys[match] - reference_keys[i])

    raise DataError(
        "No sequence number matches within the time tolerance, so the branches "
        "cannot be aligned. They may be on different epochs, unrelated, or the "
        "tolerance is too small."
    )


# --- streaming ---------------------------------------------------------------


def _no_samples() -> RealArray:
    """An empty sample buffer, the starting state of a branch."""

    return as_real_array(np.zeros(0))


@dataclass
class _Branch:
    """One branch's buffered, incrementally-unwrapped stream inside a SequenceTrap.

    Holds the samples that have arrived but not yet been emitted, with their wrapped
    packet numbers (`raw`) and the absolute packet index computed for each (`keys`).
    It owns its own unwrap, seeded across chunk boundaries so a wrap that straddles
    two chunks is still caught. `offset` is the calibration shift; `aligned_keys` is
    `keys` shifted onto the shared absolute index.
    """

    # fmt: off
    period    : int
    signal    : Signal | None = None
    raw       : RealArray     = field(default_factory=_no_samples)
    keys      : RealArray     = field(default_factory=_no_samples)
    offset    : float         = 0.0
    _last_raw : float         = float("nan")  # last wrapped number, seeds the unwrap
    _epoch    : int           = 0             # whole periods wrapped through so far
    # fmt: on

    @property
    def aligned_keys(self) -> RealArray:
        """The absolute packet index shifted onto the shared, calibrated timeline."""

        return as_real_array(self.keys - self.offset)

    def extend(self, chunk: Signal) -> None:
        """Append a chunk, continuing this branch's incremental unwrap."""

        raw = _sequence_of(chunk)
        # Seed wrap detection with the number carried from the previous chunk, so a
        # wrap across the boundary counts; the nan seed on the first-ever chunk makes
        # that first difference nan, which never counts as a wrap. Only a large drop
        # (> half a period) is a wrap. A small backward step is a reordered packet.
        seeded = np.concatenate([[self._last_raw], raw])
        wraps = np.cumsum(np.diff(seeded) < -self.period / 2)
        absolute = as_real_array(raw + self.period * (self._epoch + wraps))
        self._epoch += int(wraps[-1])
        self._last_raw = float(raw[-1])

        if self.signal is None:
            self.signal = chunk
            self.raw = as_real_array(raw)
            self.keys = absolute
        else:
            self.signal = concat_time(self.signal, chunk)
            self.raw = as_real_array(np.concatenate([self.raw, raw]))
            self.keys = as_real_array(np.concatenate([self.keys, absolute]))

    def keep(self, mask: np.ndarray) -> None:
        """Drop the samples where `mask` is False (already emitted, or consumed)."""

        assert self.signal is not None
        self.signal = keep_times(self.signal, mask)
        self.raw = as_real_array(self.raw[mask])
        self.keys = as_real_array(self.keys[mask])


class SequenceTrap:
    """Streaming hold-and-align on packet numbers (Hold + Sequence, with a period).

    Branches line up on their calibrated absolute packet index, not timestamps. The
    cross-branch offset is calibrated once (on the first same-numbered, close-in-
    time match) and each branch unwraps incrementally as chunks arrive. Everything
    is held until that first anchor is found; if the stream ends without one, it
    raises. Given a fixed calibration, this then matches the batch result.
    """

    def __init__(
        self,
        combine: CombineFn,
        max_hold: int,
        reference: int,
        sequence: Sequence,
        out_layout: Layout,
    ) -> None:
        """Bind the combine step, the reference branch, and the sequence coordinate."""

        assert sequence.period is not None
        self._combine = combine
        self._max_hold = max_hold
        self._reference = reference
        self._sequence = sequence
        self._out_layout = out_layout
        self._branches: list[_Branch] = []
        self._calibrated = False

    def push(self, inputs: list[Signal]) -> Signal:
        """Add each branch's new samples (unwrapping), then emit aligned rows."""

        if not self._branches:
            period = self._sequence.period
            assert period is not None
            self._branches = [_Branch(period) for _ in inputs]
        for branch, chunk in zip(self._branches, inputs, strict=True):
            if chunk.n_samples:
                branch.extend(chunk)
        return self._release(final=False)

    def flush(self) -> Signal:
        """Release whatever aligns at end of stream."""

        return self._release(final=True)

    def _release(self, *, final: bool) -> Signal:
        """Emit the ready rows once calibrated; otherwise hold and wait."""

        # Don't buffer forever: a branch stuck past max_hold means nothing is lining up.
        ensure_not_clogged([branch.signal for branch in self._branches], self._max_hold)

        # Need at least one sample on every branch before anything can line up.
        if not self._branches or any(b.signal is None for b in self._branches):
            return empty_signal(self._out_layout)

        # Hold until the one-time calibration succeeds; never getting it is a fault.
        if not self._calibrated and not self._calibrate():
            if final:
                raise DataError(
                    "Sequence alignment found no matching packets within the time "
                    "tolerance before the stream ended."
                )
            return empty_signal(self._out_layout)

        return self._emit(final=final)

    def _calibrate(self) -> bool:
        """Anchor on the first reference packet every branch shares, in time."""

        reference = self._branches[self._reference]
        assert reference.signal is not None

        # Anchor on the first reference packet every other branch also has; that fixes
        # the offsets once, for the whole stream.
        for i in range(reference.signal.n_samples):
            offsets = self._offsets_at(
                float(reference.raw[i]),
                float(reference.signal.times[i]),
                float(reference.keys[i]),
            )
            if offsets is not None:
                for branch, offset in zip(self._branches, offsets, strict=True):
                    branch.offset = offset
                self._calibrated = True
                return True
        return False

    def _offsets_at(
        self, number: float, time: float, reference_key: float
    ) -> list[float] | None:
        """Per-branch offsets that line every branch up with this packet, or None."""

        offsets = [0.0] * len(self._branches)
        for index, branch in enumerate(self._branches):
            # The reference lines up with itself, so it contributes no shift.
            if index == self._reference:
                continue

            # Every other branch must have this same packet, close in time; if one
            # doesn't, this packet isn't the shared anchor.
            assert branch.signal is not None
            match = _match_within_tolerance(
                branch.raw,
                branch.signal.times,
                number,
                time,
                self._sequence.tolerance_s,
            )
            if match is None:
                return None
            offsets[index] = float(branch.keys[match] - reference_key)
        return offsets

    def _emit(self, *, final: bool) -> Signal:
        """Emit the reference packets the others have caught up to; keep the rest."""

        reference = self._branches[self._reference]
        assert reference.signal is not None
        reference_keys = reference.aligned_keys

        # The reference branch sets the pace. We can emit one of its packets once every
        # other branch has data reaching it (at end of stream, just emit the rest).
        others = [
            b.aligned_keys for i, b in enumerate(self._branches) if i != self._reference
        ]
        reached = np.inf if final else min(float(keys[-1]) for keys in others)
        ready = reference_keys <= reached
        if not bool(ready.any()):
            return empty_signal(self._out_layout)

        # For each emitted packet, the other branches give their latest value, reusing
        # the previous one if nothing newer arrived (sample-and-hold). Then combine.
        ready_keys = reference_keys[ready]
        arrays = [
            hold_values_at(_signal_of(b), b.aligned_keys, ready_keys)
            for b in self._branches
        ]
        times = as_real_array(reference.signal.times[ready])
        combined = self._combine(arrays, [_signal_of(b).layout for b in self._branches])

        # Drop what we just emitted, then hand back the combined rows.
        self._trim(float(ready_keys[-1]))
        return Signal(values=combined, times=times, layout=self._out_layout)

    def _trim(self, emitted_upto: float) -> None:
        """Drop emitted reference samples; keep each other branch's carry and after."""

        for index, branch in enumerate(self._branches):
            key = branch.aligned_keys
            if index == self._reference:
                keep = key > emitted_upto
            else:
                # Keep this branch's carried value (last sample at or before the cut)
                # and everything after it.
                carry = int(np.searchsorted(key, emitted_upto, side="right"))
                keep = np.zeros(key.size, dtype=bool)
                keep[max(carry - 1, 0) :] = True
            branch.keep(keep)


def _signal_of(branch: _Branch) -> Signal:
    """A branch's buffered signal (present once alignment is under way)."""

    assert branch.signal is not None
    return branch.signal
