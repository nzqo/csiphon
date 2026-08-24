"""Executors: pour a whole recording, or stream it chunk by chunk.

Both are the same loop over a dict of intermediate signals ("lines"), keyed by
line id. `pour` runs each node's full whole-recording algorithm. `Stream` builds
a stateful operator per node and threads chunks through them, refusing up front
if any step cannot stream. Either way a run returns `Outlets`, the named results.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np

from csiphon.core.arrays import as_signal_array
from csiphon.core.errors import DataError, StreamingError
from csiphon.core.layout import Layout
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline._lines import INLET, inlet_id
from csiphon.pipeline.merges import Junction, Trap
from csiphon.pipeline.sequence import SequenceTrap
from csiphon.pipeline.step import StreamOperator

if TYPE_CHECKING:
    from csiphon.pipeline.pipeline import Node, Siphon


class Outlets(Mapping[str, Signal]):
    """The named results of a run: `outlets["doppler"]`, `outlets.single()`."""

    def __init__(self, data: Mapping[str, Signal]) -> None:
        """Wrap the display-name -> signal mapping."""

        self._data = dict(data)

    def __getitem__(self, name: str) -> Signal:
        """Return the outlet with this display name."""

        return self._data[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate the outlet names."""

        return iter(self._data)

    def __len__(self) -> int:
        """The number of outlets."""

        return len(self._data)

    def __repr__(self) -> str:
        """Show the outlet names."""

        return f"Outlets({list(self._data)})"

    def single(self) -> Signal:
        """Return the one outlet, or fail if there are several."""

        if len(self._data) != 1:
            raise DataError(f"Expected a single outlet; found {list(self._data)}.")
        return next(iter(self._data.values()))


def _check_inlet(signal: Signal, expected: Layout) -> None:
    """Verify a signal's structure matches the pipeline's inlet layout."""

    # Same physical domain: don't feed a Doppler map where a CFR is expected.
    if signal.layout.representation != expected.representation:
        raise DataError(
            f"Inlet representation {signal.layout.representation.value} does not "
            f"match the pipeline's inlet {expected.representation.value}."
        )

    # Same numeric kind (complex, magnitude, power, ...).
    if signal.layout.values != expected.values:
        raise DataError(
            f"Inlet values {signal.layout.values.value} do not match the "
            f"pipeline's inlet {expected.values.value}."
        )

    # Same axes, by name and order.
    if signal.layout.axis_names != expected.axis_names:
        raise DataError(
            f"Inlet axes {signal.layout.describe_axes()} do not match the "
            f"pipeline's inlet {expected.describe_axes()}."
        )

    # Each fixed axis the same length (the time axis rides with the data).
    for got, want in zip(signal.layout.axes, expected.axes, strict=True):
        if not want.is_dynamic and got.size != want.size:
            raise DataError(
                f"Inlet axis '{want.name}' has size {got.size}, expected {want.size}."
            )


def _inlet_signals(
    signals: Signal | Mapping[str, Signal] | None, named: Mapping[str, Signal]
) -> Signal | Mapping[str, Signal]:
    """Reconcile the two ways a caller supplies inputs into one value.

    A single-inlet run passes one Signal; a multi-inlet run passes one signal per
    inlet, as keywords (`rx0=…`) which land in `named`, or as a mapping in
    `signals`. Exactly one of the two must be given.
    """

    if named:
        if signals is not None:
            raise DataError("Pass inlet signals either positionally or by keyword.")
        return named
    if signals is None:
        raise DataError("No input signal provided.")
    return signals


def _seed_inlets(
    siphon: Siphon, signals: Signal | Mapping[str, Signal], *, allow_partial: bool
) -> dict[str, Signal]:
    """Turn the caller's input(s) into the env dict keyed by inlet line id.

    A bare Signal feeds the single unnamed inlet. A mapping feeds each named
    inlet; when `allow_partial` (streaming), inlets left out this round get an
    empty chunk so the merge simply holds what it has.
    """

    if isinstance(signals, Signal):
        _check_inlet(signals, siphon.inlet_layout)
        return {INLET: signals}

    # Place each given signal on its inlet, checking its structure as we go.
    env: dict[str, Signal] = {}
    for name, signal in signals.items():
        line = inlet_id(name)
        if line not in siphon.inlet_layouts:
            raise DataError(f"Unknown inlet '{name}'.")
        _check_inlet(signal, siphon.inlet_layouts[line])
        env[line] = signal

    # An inlet with no signal this round: an error in batch, an empty chunk when
    # streaming (so the merge just holds what it has).
    for line, layout in siphon.inlet_layouts.items():
        if line not in env:
            if not allow_partial:
                raise DataError(f"Missing a signal for inlet '{line}'.")
            env[line] = empty_signal(layout)
    return env


def pour(siphon: Siphon, signals: Signal | Mapping[str, Signal]) -> Outlets:
    """Run a whole recording through a siphon and return every outlet."""

    env = _seed_inlets(siphon, signals, allow_partial=False)
    for node in siphon.nodes:
        inputs = [env[line] for line in node.inputs]
        env[node.output] = _run_node(node, inputs)
    return Outlets({display: env[line] for display, line in siphon.outlets.items()})


def _run_node(node: Node, inputs: list[Signal]) -> Signal:
    """Run one node in batch: a junction takes all inputs, a step takes the one."""

    if isinstance(node.op, Junction):
        return node.op.process(inputs, node.out_layout)
    return node.op.process(inputs[0], node.out_layout, node.out_profile)


def concat_signals(signals: list[Signal], layout: Layout) -> Signal:
    """Concatenate chunk outputs along the dynamic axis."""

    dynamic = layout.dynamic_index
    non_empty = [signal for signal in signals if signal.n_samples > 0]

    # Nothing to join: hand back an empty signal on the target layout.
    if not non_empty:
        return empty_signal(layout)

    # A frame signal has no time axis to concatenate along, so pass the one through.
    if dynamic is None:
        return non_empty[0]

    values = np.concatenate([signal.values for signal in non_empty], axis=dynamic)
    times = np.concatenate([signal.times for signal in non_empty])
    return Signal(values=as_signal_array(values), times=times, layout=layout)


class Stream:
    """A stateful streaming session over one siphon: feed chunks, then flush."""

    def __init__(self, siphon: Siphon) -> None:
        """Build one streaming operator per node, refusing batch-only steps."""

        self._siphon = siphon
        self._operators: dict[str, StreamOperator | Trap | SequenceTrap] = {}
        for node in siphon.nodes:
            self._operators[node.output] = _build_operator(node, siphon)

    def flow(
        self,
        signals: Signal | Mapping[str, Signal] | None = None,
        **named: Signal,
    ) -> Outlets:
        """Feed one input chunk per inlet and return whatever output is ready.

        Single-inlet: `flow(chunk)`. Multi-inlet: `flow(rx0=c0, rx1=c1)` or a
        mapping. Inlets you leave out this round get an empty chunk, so a receiver
        that ran ahead can be pushed alone. The merge holds it until the others
        reach the same point.
        """

        env = _seed_inlets(
            self._siphon, _inlet_signals(signals, named), allow_partial=True
        )
        for node in self._siphon.nodes:
            inputs = [env[line] for line in node.inputs]
            env[node.output] = self._push(node, inputs)
        return Outlets(
            {display: env[line] for display, line in self._siphon.outlets.items()}
        )

    def flush(self) -> Outlets:
        """Drain buffered state at end of stream, propagating tails downstream."""

        # End of stream: no more input, so feed empty and let each node drain.
        tails: dict[str, Signal] = {
            line: empty_signal(layout)
            for line, layout in self._siphon.inlet_layouts.items()
        }

        # Take the nodes in order; each one's drained tail feeds the next, so a tail
        # flows through to the outlets.
        for node in self._siphon.nodes:
            inputs = [tails[line] for line in node.inputs]
            operator = self._operators[node.output]
            pushed = self._push(node, inputs, skip_empty=True)
            flushed = operator.flush()
            tails[node.output] = concat_signals([pushed, flushed], node.out_layout)

        return Outlets(
            {display: tails[line] for display, line in self._siphon.outlets.items()}
        )

    def _push(
        self, node: Node, inputs: list[Signal], *, skip_empty: bool = False
    ) -> Signal:
        """Push a chunk through one node's operator (junction takes all inputs)."""

        operator = self._operators[node.output]
        if isinstance(operator, (Trap, SequenceTrap)):
            return operator.push(inputs)
        if skip_empty and inputs[0].n_samples == 0:
            return empty_signal(node.out_layout)
        return operator.push(inputs[0])


def _build_operator(node: Node, siphon: Siphon) -> StreamOperator | Trap | SequenceTrap:
    """Make the streaming operator for a node, or refuse a batch-only step."""

    if isinstance(node.op, Junction):
        return node.op.trap(node.out_layout)
    in_layout = siphon.layouts[node.inputs[0]]
    operator = node.op.stream(in_layout, node.out_layout, node.out_profile)
    if operator is None:
        raise StreamingError(
            f"Step '{node.op.name}' cannot run in streaming mode (it needs the whole "
            "recording). Replace it with a causal variant or pour this siphon instead."
        )
    return operator
