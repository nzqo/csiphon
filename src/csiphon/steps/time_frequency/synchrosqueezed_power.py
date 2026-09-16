"""Synchrosqueezed wavelet power (optional `[sst]` extra -> ssqueezepy).

Without a block size the batch run is the full whole-recording synchrosqueezed
CWT, the best-quality transform, and the step cannot stream. With `block_size`
set, batch and streaming both run the *block-local* variant: each fixed-size
block of the time axis is transformed independently and the results are joined
back in time order. That variant is an approximation of the whole-recording one,
but batch and streaming then agree exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_real_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError, MissingDependencyError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.sampling import effective_rate_hz
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import CONFIG_DEPENDENT, Category, LayoutEffect, StepSpec, Streaming


def _ssq_cwt() -> Callable[..., Any]:
    """Import ssqueezepy lazily with a helpful error if it is missing."""

    try:
        # Lazy so the base install stays numpy-only.
        from ssqueezepy import ssq_cwt  # pylint: disable=import-outside-toplevel
    except ImportError as error:  # pragma: no cover - exercised via message
        raise MissingDependencyError("ssqueezepy", "sst") from error
    return cast("Callable[..., Any]", ssq_cwt)


def _transform_rows(
    rows: RealArray, rate_hz: float, voices_per_octave: int
) -> tuple[RealArray, RealArray]:
    """Run one synchrosqueezed CWT over a stack of equal-length rows."""

    ssq_cwt = _ssq_cwt()
    spectrum, _, frequencies, *_ = ssq_cwt(
        rows, wavelet="cmhat", nv=voices_per_octave, fs=rate_hz, scales="log"
    )
    spectrum = np.asarray(spectrum)
    if spectrum.ndim == 2:
        spectrum = spectrum[None, :, :]
    return spectrum, np.asarray(frequencies, dtype=float)


def _sst_power(
    values: RealArray, rate_hz: float, voices_per_octave: int
) -> tuple[RealArray, RealArray]:
    """Synchrosqueezed power for a `(time, *channels)` block.

    ssqueezepy transforms a 2-D `(rows, samples)` stack, so the channel axes are
    flattened into rows for the transform and restored on the result. Returns the
    `(time, frequency, *channels)` power and its frequency coordinates.
    """

    channel_shape = values.shape[1:]
    rows = values.reshape(values.shape[0], -1).T
    spectrum, frequencies = _transform_rows(rows, rate_hz, voices_per_octave)

    # (channels, frequency, samples) -> (samples, frequency, *channels) power.
    power = np.abs(spectrum.transpose(2, 1, 0)) ** 2
    power = power.reshape(power.shape[0], power.shape[1], *channel_shape)
    return as_real_array(power), frequencies


@dataclass(frozen=True, slots=True)
class SynchrosqueezedPower(Step):
    """Synchrosqueezed CWT power along time, per input channel.

    With `block_size` unset the batch run transforms the whole recording and the
    step is batch-only. With `block_size` set, both batch and streaming transform
    consecutive non-overlapping `block_size`-sample blocks independently and join
    them in time order; a trailing partial block is dropped.
    """

    voices_per_octave: int = field(
        default=2, metadata={"doc": "wavelet voices per octave"}
    )
    block_size: int | None = field(
        default=None,
        metadata={"doc": "block size (samples) for the block-local transform"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="synchrosqueezed-power",
        summary="compute synchrosqueezed CWT power along time, per channel",
        category=Category.TIME_FREQUENCY,
        admissible_values=(ValueKind.REAL, ValueKind.MAGNITUDE),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        # Streams (block-local, matching batch) only with a block_size; see
        # resolve_streaming.
        layout_effect=LayoutEffect(
            adds=(AxisName.FREQUENCY,),
            value_kind=ValueKind.POWER,
            note="time-frequency",
        ),
        streaming=CONFIG_DEPENDENT,
        streaming_note="block-local with block_size set, else batch-only",
    )

    def resolve_streaming(self) -> Streaming:
        """Streams only with a block size set; batch is then block-local too."""

        if self.block_size is None:
            return Streaming.UNAVAILABLE
        return Streaming.BATCH_EQUIVALENT

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Insert a runtime-sized frequency axis; produce time-frequency power."""

        self.require_inputs(layout)
        if layout.dynamic_index != 0:
            raise LayoutError("SynchrosqueezedPower expects the time axis first.")
        if self.voices_per_octave < 1:
            raise LayoutError(
                f"voices_per_octave must be >= 1, got {self.voices_per_octave}."
            )
        if self.block_size is not None and self.block_size < 1:
            raise LayoutError(f"block_size must be >= 1, got {self.block_size}.")

        frequency = Axis(AxisName.FREQUENCY, size=None, unit="Hz")
        return (
            layout.insert_axis_after(AxisName.TIME, frequency)
            .with_representation(Representation.TIME_FREQUENCY)
            .with_values(ValueKind.POWER)
        )

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Whole-recording power, or block-local power when `block_size` is set."""

        values = as_real_array(signal.values)
        if self.block_size is None:
            rate_hz = effective_rate_hz(signal.times, profile.sampling_rate_hz)
            power, frequencies = _sst_power(values, rate_hz, self.voices_per_octave)
            return signal.with_values(
                power, out_layout, coords={AxisName.FREQUENCY: frequencies}
            )

        block_local, _ = self.block_local(
            values, signal.times, profile.sampling_rate_hz, out_layout
        )
        return block_local

    def block_local(
        self,
        values: RealArray,
        times: RealArray,
        rate_hz: float | None,
        out_layout: Layout,
    ) -> tuple[Signal, int]:
        """Transform every whole `block_size` block of `values` on its own.

        Returns the blocks' power joined in time order and the number of input
        samples they cover. A trailing partial block is left out: ssqueezepy
        picks its number of frequency bins from the block length, so a shorter
        block would not fit the same frequency axis. Both batch and streaming
        use this, which is what makes them agree.
        """

        if self.block_size is None:
            raise LayoutError("block_local needs a block_size.")

        blocks: list[RealArray] = []
        frequencies: RealArray | None = None
        for start in range(0, values.shape[0] - self.block_size + 1, self.block_size):
            stop = start + self.block_size

            # Use the known rate if we have one, otherwise estimate it from the block.
            block_rate_hz = (
                rate_hz
                if rate_hz is not None
                else effective_rate_hz(times[start:stop], None)
            )
            power, frequencies = _sst_power(
                values[start:stop], block_rate_hz, self.voices_per_octave
            )
            blocks.append(power)

        # Shorter than one block: nothing to emit yet.
        if frequencies is None:
            return empty_signal(out_layout), 0

        covered = len(blocks) * self.block_size
        joined: SignalArray = as_real_array(np.concatenate(blocks, axis=0))
        signal = Signal(
            values=joined,
            times=times[:covered],
            layout=out_layout,
            coords={AxisName.FREQUENCY: frequencies},
        )
        return signal, covered

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Return a block-local operator, or `None` if no block size is set."""

        if self.block_size is None:
            return None
        return _BlockLocalSst(self, profile.sampling_rate_hz, out_layout)


class _BlockLocalSst(StreamOperator):
    """Transforms each full block of `block_size` samples independently."""

    def __init__(
        self, step: SynchrosqueezedPower, rate_hz: float | None, out_layout: Layout
    ) -> None:
        """Buffer samples until a whole block is available."""

        self._step = step
        self._rate_hz = rate_hz
        self._out_layout = out_layout
        self._buffer: RealArray | None = None
        self._times = np.zeros(0)

    def push(self, chunk: Signal) -> Signal:
        """Emit transforms for every full block now buffered."""

        incoming = as_real_array(chunk.values)
        self._buffer = (
            incoming
            if self._buffer is None
            else np.concatenate([self._buffer, incoming], axis=0)
        )
        self._times = np.concatenate([self._times, chunk.times])

        # Transform every whole block that has arrived and drop those samples from
        # the front of the buffer. A partial block waits for more samples.
        emitted, covered = self._step.block_local(
            self._buffer, self._times, self._rate_hz, self._out_layout
        )
        self._buffer = self._buffer[covered:]
        self._times = self._times[covered:]
        return emitted

    def flush(self) -> Signal:
        """Drop any trailing partial block (matches the batch run)."""

        return empty_signal(self._out_layout)
