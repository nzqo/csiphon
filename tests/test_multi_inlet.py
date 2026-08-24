"""Multiple inlets: load N sources, merge, then process (Modes 1 and 2)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from csiphon import (
    AcquisitionProfile,
    AxisName,
    Concatenate,
    Hold,
    Layout,
    Pipeline,
    Signal,
    create_signal,
)
from csiphon.core.errors import DataError, LayoutError
from csiphon.pipeline import concat_signals
from csiphon.steps import Magnitude, WindowedVariance


def _complex_signal(layout: Layout, seed: int, n: int = 200) -> Signal:
    """A random complex recording on `layout` (one receiver's raw CSI)."""

    rng = np.random.default_rng(seed)
    shape = (n, *[axis.size for axis in layout.axes[1:]])
    values = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return create_signal(values, np.arange(n) / 1000.0, layout)


def _chunk(signal: Signal, start: int, size: int) -> Signal:
    """One `[start, start + size)` slice of a signal (same layout)."""

    return Signal(
        values=signal.values[start : start + size],
        times=signal.times[start : start + size],
        layout=signal.layout,
    )


# --- from_inlets builder validation -------------------------------------------


def test_from_inlets_needs_at_least_two() -> None:
    """A single inlet is just a normal pipeline, so from_inlets requires two or more."""

    with pytest.raises(LayoutError, match="at least two"):
        Pipeline.from_inlets("rx0")


def test_from_inlets_rejects_duplicate_names() -> None:
    """Inlet names identify the sources, so they must be distinct."""

    with pytest.raises(LayoutError, match="distinct"):
        Pipeline.from_inlets("rx0", "rx0")


def test_describe_draws_every_inlet_as_a_source() -> None:
    """The flow graph draws one source card per inlet, feeding the merge."""

    profile = AcquisitionProfile(
        n_rx_antennas=2,
        subcarrier_indices=tuple(range(8)),
        n_receivers=3,
        sampling_rate_hz=1000.0,
    )
    text = (
        Pipeline.from_inlets("rx0", "rx1", "rx2")
        .merge(using=Concatenate(axis=AxisName.RECEIVER), align=Hold())
        .then(Magnitude())
        .compile(profile)
        .describe()
    )
    # Each inlet is drawn as its own `○ name` source card (the 2-D graph, not the
    # flat fallback), and they converge into the concatenate merge.
    for name in ("rx0", "rx1", "rx2"):
        assert f"○ {name}" in text
    assert "concatenate" in text


# --- Mode 1: one N-receiver profile, N inlets ---------------------------------


def test_mode1_concatenates_receivers_onto_the_receiver_axis() -> None:
    """N inlets from one N-receiver capture concatenate back onto the receiver axis.

    Each inlet already carries a size-1 receiver axis (a single-receiver capture is
    not receiver-less), so the merge grows that axis rather than stacking a new one.
    """

    profile = AcquisitionProfile(
        n_rx_antennas=2,
        subcarrier_indices=tuple(range(8)),
        n_receivers=3,
        sampling_rate_hz=1000.0,
    )
    per_receiver = replace(profile, n_receivers=1).raw_csi_layout()
    signals = {
        name: _complex_signal(per_receiver, seed)
        for seed, name in enumerate(("rx0", "rx1", "rx2"))
    }

    pipe = (
        Pipeline.from_inlets("rx0", "rx1", "rx2")
        .merge(using=Concatenate(axis=AxisName.RECEIVER), align=Hold())
        .then(Magnitude())
    )
    out = pipe.compile(profile).pour(**signals).single()

    # The receiver axis has one slot per inlet, in inlet order. Receiver sits right
    # after time now (position 1), so slice it there -- not on the trailing axis.
    receiver = out.layout.axis_position(AxisName.RECEIVER)
    assert out.layout.axis(AxisName.RECEIVER).size == 3
    for index, name in enumerate(("rx0", "rx1", "rx2")):
        slot = np.take(out.values, [index], axis=receiver)
        assert np.allclose(slot, np.abs(signals[name].values))


def test_mode1_requires_n_receivers_to_equal_the_inlet_count() -> None:
    """A single profile only fits when its receiver count matches the inlets."""

    profile = AcquisitionProfile(
        subcarrier_indices=tuple(range(8)), n_receivers=2, sampling_rate_hz=1000.0
    )
    pipe = Pipeline.from_inlets("rx0", "rx1", "rx2").merge(
        using=Concatenate(axis=AxisName.RECEIVER)
    )
    with pytest.raises(LayoutError, match="n_receivers"):
        pipe.compile(profile)  # 2 receivers, 3 inlets


# --- Mode 2: one profile per inlet (heterogeneous devices) --------------------


def test_mode2_merges_devices_with_different_antenna_counts() -> None:
    """Two routers with 2 and 3 antennas concatenate into 5 -- a single profile can't
    even express this, which is the whole point of per-device profiles."""

    device_a = AcquisitionProfile(
        n_rx_antennas=2, subcarrier_indices=tuple(range(8)), sampling_rate_hz=1000.0
    )
    device_b = AcquisitionProfile(
        n_rx_antennas=3, subcarrier_indices=tuple(range(8)), sampling_rate_hz=1000.0
    )
    signal_a = _complex_signal(device_a.raw_csi_layout(), 0)
    signal_b = _complex_signal(device_b.raw_csi_layout(), 1)

    pipe = (
        Pipeline.from_inlets("a", "b")
        .merge(using=Concatenate(axis=AxisName.RX_ANTENNA), align=Hold())
        .then(Magnitude())
    )
    out = pipe.compile(a=device_a, b=device_b).pour(a=signal_a, b=signal_b).single()
    assert out.layout.axis(AxisName.RX_ANTENNA).size == 5  # 2 + 3


def test_mode2_requires_each_device_profile_to_have_one_receiver() -> None:
    """A per-device profile describes one device, so its receiver count must be 1."""

    two_receivers = AcquisitionProfile(
        subcarrier_indices=tuple(range(8)), n_receivers=2, sampling_rate_hz=1000.0
    )
    single = AcquisitionProfile(
        subcarrier_indices=tuple(range(8)), sampling_rate_hz=1000.0
    )
    pipe = Pipeline.from_inlets("a", "b").merge(
        using=Concatenate(axis=AxisName.RECEIVER)
    )
    with pytest.raises(LayoutError, match="n_receivers=1"):
        pipe.compile(a=two_receivers, b=single)


def test_compile_rejects_mixing_one_profile_with_per_inlet_profiles() -> None:
    """You give either one capture profile or one per inlet, never both."""

    profile = AcquisitionProfile(
        subcarrier_indices=tuple(range(8)), n_receivers=2, sampling_rate_hz=1000.0
    )
    pipe = Pipeline.from_inlets("a", "b").merge(
        using=Concatenate(axis=AxisName.RECEIVER)
    )
    with pytest.raises(LayoutError, match="not both"):
        pipe.compile(profile, a=profile, b=profile)


# --- feeding: mapping and keyword forms are equivalent ------------------------


def test_pour_accepts_mapping_and_keyword_forms() -> None:
    """`pour(rx0=s0, …)` and `pour({"rx0": s0, …})` mean the same thing."""

    profile = AcquisitionProfile(
        n_rx_antennas=2,
        subcarrier_indices=tuple(range(8)),
        n_receivers=2,
        sampling_rate_hz=1000.0,
    )
    layout = replace(profile, n_receivers=1).raw_csi_layout()
    signals = {"rx0": _complex_signal(layout, 0), "rx1": _complex_signal(layout, 1)}
    siphon = (
        Pipeline.from_inlets("rx0", "rx1")
        .merge(using=Concatenate(axis=AxisName.RECEIVER), align=Hold())
        .then(Magnitude())
    ).compile(profile)

    by_kwargs = siphon.pour(rx0=signals["rx0"], rx1=signals["rx1"]).single()
    by_mapping = siphon.pour(signals).single()
    assert np.allclose(by_kwargs.values, by_mapping.values)


def test_pour_rejects_an_unknown_inlet_name() -> None:
    """Feeding a name the pipeline never declared is a clear error."""

    profile = AcquisitionProfile(
        n_rx_antennas=1,
        subcarrier_indices=tuple(range(8)),
        n_receivers=2,
        sampling_rate_hz=1000.0,
    )
    layout = replace(profile, n_receivers=1).raw_csi_layout()
    siphon = (
        Pipeline.from_inlets("rx0", "rx1").merge(
            using=Concatenate(axis=AxisName.RECEIVER), align=Hold()
        )
    ).compile(profile)
    with pytest.raises(DataError, match="Unknown inlet"):
        siphon.pour(rx0=_complex_signal(layout, 0), typo=_complex_signal(layout, 1))


# --- streaming: lockstep and asynchronous feeding -----------------------------


def _mode1_siphon() -> tuple[object, Layout, list[str]]:
    """A small 3-receiver Mode-1 siphon plus its per-receiver inlet layout."""

    profile = AcquisitionProfile(
        n_rx_antennas=2,
        subcarrier_indices=tuple(range(8)),
        n_receivers=3,
        sampling_rate_hz=1000.0,
    )
    layout = replace(profile, n_receivers=1).raw_csi_layout()
    siphon = (
        Pipeline.from_inlets("rx0", "rx1", "rx2")
        .merge(using=Concatenate(axis=AxisName.RECEIVER), align=Hold())
        .then(Magnitude())
        .then(WindowedVariance(win_size_s=0.02))
    ).compile(profile)
    return siphon, layout, ["rx0", "rx1", "rx2"]


@pytest.mark.parametrize("chunk", [37, 100])
def test_streaming_lockstep_equals_batch(chunk: int) -> None:
    """Feeding one chunk per receiver each round streams identically to batch."""

    siphon, layout, names = _mode1_siphon()
    signals = {name: _complex_signal(layout, seed) for seed, name in enumerate(names)}
    batch = siphon.pour(**signals).single()

    stream = siphon.stream()
    parts = []
    length = next(iter(signals.values())).n_samples
    for start in range(0, length, chunk):
        round_ = {name: _chunk(sig, start, chunk) for name, sig in signals.items()}
        parts.append(stream.flow(**round_).single())
    parts.append(stream.flush().single())
    streamed = concat_signals([p for p in parts if p.n_samples], siphon.outlet_layout)

    assert np.allclose(batch.values, streamed.values)
    assert np.allclose(batch.times, streamed.times)


def test_streaming_asynchronous_feeding_equals_batch() -> None:
    """A receiver that runs ahead can be pushed alone; the merge holds it until the
    others catch up, and the end result still equals batch."""

    siphon, layout, names = _mode1_siphon()
    signals = {name: _complex_signal(layout, seed) for seed, name in enumerate(names)}
    batch = siphon.pour(**signals).single()

    stream = siphon.stream()
    parts = []
    # rx0 arrives in halves first, then rx1 and rx2 catch up together, out of step.
    parts.append(stream.flow(rx0=_chunk(signals["rx0"], 0, 120)).single())
    parts.append(stream.flow(rx1=_chunk(signals["rx1"], 0, 200)).single())
    parts.append(stream.flow(rx0=_chunk(signals["rx0"], 120, 80)).single())
    parts.append(stream.flow(rx2=_chunk(signals["rx2"], 0, 200)).single())
    parts.append(stream.flush().single())
    streamed = concat_signals([p for p in parts if p.n_samples], siphon.outlet_layout)

    assert np.allclose(batch.values, streamed.values)
    assert np.allclose(batch.times, streamed.times)
