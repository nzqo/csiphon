"""Aligning merges on packet (sequence) numbers, with wrap-around handling."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from csiphon import (
    AcquisitionProfile,
    AxisName,
    Concatenate,
    Hold,
    Pipeline,
    Sequence,
    Stack,
    create_signal,
)
from csiphon.core.axes import Axis
from csiphon.core.errors import DataError, StreamingError
from csiphon.core.layout import Layout
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline import concat_signals
from csiphon.pipeline.merges import Exact, Junction
from csiphon.pipeline.sequence import _unwrap
from csiphon.steps import Magnitude

_LAYOUT = Layout(
    (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 2)),
    Representation.CHANNEL_FREQUENCY_RESPONSE,
    ValueKind.MAGNITUDE,
)


def _branch(
    packets,
    *,
    period: int,
    base_value: float = 0.0,
    spacing: float = 0.001,
    drift: float = 0.0,
) -> Signal:
    """A branch whose value *encodes its absolute packet index*, so alignment is
    checkable, carrying the wrapped sequence number (`packet % period`).

    Packet `k` sits at time `k * spacing + drift`; `base_value` offsets the value
    so two branches are distinguishable.
    """

    absolute = np.asarray(packets, dtype=float)
    raw = absolute % period
    times = absolute * spacing + drift
    values = np.stack([absolute + base_value, absolute + base_value + 1000.0], axis=1)
    return create_signal(values, times, _LAYOUT, sequence=raw)


def test_hold_on_sequence_matches_packets_by_number() -> None:
    """Two receivers of the same packets align by number, not by (drifting) clocks."""

    period = 10
    a = _branch(range(10), period=period)
    b = _branch(range(10), period=period, base_value=0.0, drift=2e-4)  # clock drift
    arrays, times = Hold(on=Sequence(period=period)).align([a, b])

    # Held onto branch a's timeline; every packet matches its twin.
    assert np.array_equal(times, a.times)
    assert np.array_equal(arrays[0][:, 0], np.arange(10))  # a's packets
    assert np.array_equal(arrays[1][:, 0], np.arange(10))  # b's, matched by number


def test_sequence_alignment_unwraps_and_calibrates_across_a_wrap() -> None:
    """A recording spans a wrap and one branch starts late; both are still aligned.

    Branch a is packets 0..14, branch b is packets 5..19 -- both cross the period-10
    wrap (packet 10 has raw number 0 again). Unwrapping keeps the post-wrap packets
    distinct, and the timestamp-anchored calibration lines b's packet numbering up
    with a's despite the 5-packet head start.
    """

    period = 10
    tolerance = 0.005  # 5 ms < the 10 ms wrap interval, so it disambiguates
    a = _branch(range(15), period=period)  # packets 0..14
    b = _branch(range(5, 20), period=period, drift=2e-4)  # packets 5..19

    arrays, times = Hold(on=Sequence(period=period, tolerance_s=tolerance)).align(
        [a, b]
    )

    assert np.array_equal(times, a.times)
    assert np.array_equal(arrays[0][:, 0], np.arange(15))  # a: packets 0..14
    # b held onto a's packets 0..14: before b starts (packets 0..4) it holds its
    # first sample (packet 5); packets 5..14 match exactly -- including 10..14,
    # which carry raw numbers 0..4 but are *not* mistaken for the early packets.
    expected_b = np.array([5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], float)
    assert np.array_equal(arrays[1][:, 0], expected_b)


def test_exact_on_sequence_keeps_shared_packets() -> None:
    """Exact on sequence keeps only the packet numbers every branch has."""

    period = 100
    a = _branch(range(10), period=period)  # packets 0..9
    b = _branch(range(3, 13), period=period, drift=1e-4)  # packets 3..12
    arrays, _ = Exact(on=Sequence(period=period)).align([a, b])

    # Shared packets are 3..9.
    assert np.array_equal(arrays[0][:, 0], np.arange(3, 10))
    assert np.array_equal(arrays[1][:, 0], np.arange(3, 10))


def test_sequence_alignment_raises_when_no_match_within_tolerance() -> None:
    """If no same-numbered packet lands within the time window, it refuses to guess."""

    period = 10
    a = _branch(range(10), period=period)
    # Same packet numbers, but a full second apart: no honest match exists.
    b = _branch(range(10), period=period, drift=1.0)
    with pytest.raises(DataError, match="within the time tolerance"):
        Hold(on=Sequence(period=period, tolerance_s=0.005)).align([a, b])


def test_sequence_alignment_needs_sequence_numbers_on_every_branch() -> None:
    """Aligning on sequence requires the branches to actually carry them."""

    period = 10
    with_seq = _branch(range(10), period=period)
    without = create_signal(with_seq.values, with_seq.times, _LAYOUT)  # no sequence
    with pytest.raises(DataError, match="needs Signal"):
        Hold(on=Sequence(period=period)).align([with_seq, without])


def test_sequence_without_a_period_aligns_by_raw_number() -> None:
    """Without a period the raw numbers are used as-is, fine when it never wraps."""

    period = 1000  # so packets 0..12 never wrap: raw number == absolute packet
    a = _branch(range(10), period=period)  # packets 0..9
    b = _branch(range(3, 13), period=period, drift=1e-4)  # packets 3..12
    arrays, _ = Hold(on=Sequence(period=None, tolerance_s=0.005)).align([a, b])

    assert np.array_equal(arrays[0][:, 0], np.arange(10))  # a: packets 0..9
    # b held onto a's packets: 0..2 hold b's first (packet 3), then 3..9 match.
    expected_b = np.array([3, 3, 3, 3, 4, 5, 6, 7, 8, 9], float)
    assert np.array_equal(arrays[1][:, 0], expected_b)


def _receiver_pipeline(align) -> tuple[object, Layout]:
    """A 2-receiver Mode-1 siphon that merges on the given alignment, and its inlet."""

    profile = AcquisitionProfile(
        n_rx_antennas=1,
        subcarrier_indices=tuple(range(2)),
        n_receivers=2,
        sampling_rate_hz=1000.0,
    )
    inlet = replace(profile, n_receivers=1).raw_csi_layout()
    siphon = (
        Pipeline.from_inlets("rx0", "rx1")
        # Each inlet already carries a size-1 receiver axis, so concatenate to grow
        # it rather than stacking a new one.
        .merge(using=Concatenate(axis=AxisName.RECEIVER), align=align)
        .then(Magnitude())
    ).compile(profile)
    return siphon, inlet


def _receiver_branch(
    inlet: Layout, count: int, base: float, drift: float, period: int
) -> Signal:
    """A raw-CSI receiver stream carrying wrapped sequence numbers."""

    absolute = np.arange(count, dtype=float)
    raw = absolute % period
    shape = (count, *[axis.size for axis in inlet.axes[1:]])
    # Put the per-packet value on the time axis and let it broadcast across every
    # trailing (structural + subcarrier) axis of the full inlet layout.
    per_packet = (absolute + base).reshape((count,) + (1,) * (len(shape) - 1))
    values = np.broadcast_to(per_packet, shape).astype(complex)
    return create_signal(values, absolute / 1000.0 + drift, inlet, sequence=raw)


@pytest.mark.parametrize("chunk", [17, 50])
def test_streaming_hold_on_sequence_equals_batch(chunk: int) -> None:
    """Streaming a Hold+Sequence merge matches pouring it: the offset is calibrated
    once and unwrapping continues incrementally, so chunking makes no difference."""

    period = 8  # small period -> the 100-sample streams wrap many times
    siphon, inlet = _receiver_pipeline(
        Hold(on=Sequence(period=period, tolerance_s=0.003))
    )
    a = _receiver_branch(inlet, 100, 0.0, 0.0, period)
    b = _receiver_branch(inlet, 100, 1000.0, 3e-4, period)
    batch = siphon.pour(rx0=a, rx1=b).single()

    def piece(signal: Signal, start: int) -> Signal:
        stop = start + chunk
        return Signal(
            values=signal.values[start:stop],
            times=signal.times[start:stop],
            layout=inlet,
            sequence=signal.sequence[start:stop],
        )

    stream = siphon.stream()
    parts = []
    for start in range(0, 100, chunk):
        parts.append(stream.flow(rx0=piece(a, start), rx1=piece(b, start)).single())
    parts.append(stream.flush().single())
    streamed = concat_signals([p for p in parts if p.n_samples], siphon.outlet_layout)

    assert streamed.values.shape == batch.values.shape
    assert np.allclose(batch.values, streamed.values)
    assert np.allclose(batch.times, streamed.times)


def test_streaming_refuses_exact_on_sequence() -> None:
    """Streaming sequence alignment supports Hold only; Exact refuses up front."""

    siphon, _ = _receiver_pipeline(Exact(on=Sequence(period=4096)))
    with pytest.raises(StreamingError, match="supports Hold only"):
        siphon.stream()


def test_streaming_refuses_sequence_without_a_period() -> None:
    """Streaming needs a wrap period to unwrap; without one it refuses."""

    siphon, _ = _receiver_pipeline(Hold(on=Sequence(period=None)))
    with pytest.raises(StreamingError, match="needs a wrap period"):
        siphon.stream()


def test_junction_process_aligns_on_sequence_end_to_end() -> None:
    """A junction merges two branches on sequence numbers across a wrap."""

    period = 8
    a = _branch(range(12), period=period)  # 0..11, wraps at 8
    b = _branch(range(12), period=period, base_value=1000.0, drift=3e-4)
    junction = Junction(
        Stack(into=AxisName.RECEIVER),
        Hold(on=Sequence(period=period, tolerance_s=0.003)),
    )
    out_layout = junction.output_layout([_LAYOUT, _LAYOUT])
    out = junction.process([a, b], out_layout)

    assert out.n_samples == 12
    assert np.array_equal(out.values[:, 0, 0], np.arange(12))  # a receiver
    assert np.array_equal(out.values[:, 0, 1], np.arange(12) + 1000.0)  # b receiver


def test_unwrap_ignores_a_reordered_packet() -> None:
    """A small backward step is a reordered/duplicate packet, not a wrap.

    Regression: unwrapping used to add a whole period at *any* decrease, so a single
    out-of-order packet jumped into the next epoch and dragged the rest with it.
    """

    raw = np.array([100.0, 99.0, 101.0, 102.0])  # 99 arrives late
    assert np.array_equal(_unwrap(raw, period=4096), [100.0, 99.0, 101.0, 102.0])


def test_unwrap_still_detects_a_real_wrap() -> None:
    """A drop of nearly a whole period is a genuine wrap and does shift the epoch."""

    raw = np.array([4094.0, 4095.0, 0.0, 1.0])  # wraps at 4096
    assert np.array_equal(_unwrap(raw, period=4096), [4094.0, 4095.0, 4096.0, 4097.0])
