"""Synchrosqueezed wavelet power (optional `[sst]` extra -> ssqueezepy).

Batch runs the full whole-recording synchrosqueezed CWT, the best-quality
transform. Streaming uses a *block-local* variant (each fixed-size block
transformed independently), a genuine approximation, so batch and streaming are
intentionally not identical here.
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

    Batch: whole-recording transform. Streaming: block-local transform over
    non-overlapping `streaming_window`-sample blocks (an approximation); with
    `streaming_window` unset the step is batch-only.
    """

    voices_per_octave: int = field(
        default=2, metadata={"doc": "wavelet voices per octave"}
    )
    streaming_window: int | None = field(
        default=None,
        metadata={"doc": "block size (samples) enabling block-local streaming"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="synchrosqueezed-power",
        summary="compute synchrosqueezed CWT power along time, per channel",
        category=Category.TIME_FREQUENCY,
        admissible_values=(ValueKind.REAL, ValueKind.MAGNITUDE),
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        # Streams (block-local) only with a streaming_window; see resolve_streaming.
        layout_effect=LayoutEffect(
            adds=(AxisName.FREQUENCY,),
            value_kind=ValueKind.POWER,
            note="time-frequency",
        ),
        streaming=CONFIG_DEPENDENT,
        streaming_note="block-local; batch-only unless streaming_window is set",
    )

    def resolve_streaming(self) -> Streaming:
        """Streams (a block-local approximation) only with a block size set."""

        if self.streaming_window is None:
            return Streaming.UNAVAILABLE
        return Streaming.BATCH_DIVERGENT

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Insert a runtime-sized frequency axis; produce time-frequency power."""

        self.require_inputs(layout)
        if layout.dynamic_index != 0:
            raise LayoutError("SynchrosqueezedPower expects the time axis first.")
        if self.voices_per_octave < 1:
            raise LayoutError(
                f"voices_per_octave must be >= 1, got {self.voices_per_octave}."
            )
        if self.streaming_window is not None and self.streaming_window < 1:
            raise LayoutError(
                f"streaming_window must be >= 1, got {self.streaming_window}."
            )

        frequency = Axis(AxisName.FREQUENCY, size=None, unit="Hz")
        return (
            layout.insert_axis_after(AxisName.TIME, frequency)
            .with_representation(Representation.TIME_FREQUENCY)
            .with_values(ValueKind.POWER)
        )

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Whole-recording synchrosqueezed power (best quality)."""

        rate_hz = effective_rate_hz(signal.times, profile.sampling_rate_hz)
        power, frequencies = _sst_power(
            as_real_array(signal.values), rate_hz, self.voices_per_octave
        )
        return signal.with_values(
            power, out_layout, coords={AxisName.FREQUENCY: frequencies}
        )

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Return a block-local operator, or `None` if no block size is set."""

        if self.streaming_window is None:
            return None
        return _BlockLocalSst(
            window=self.streaming_window,
            rate_hz=profile.sampling_rate_hz,
            voices_per_octave=self.voices_per_octave,
            out_layout=out_layout,
        )


class _BlockLocalSst(StreamOperator):
    """Transforms each full block of `window` samples independently."""

    def __init__(
        self,
        window: int,
        rate_hz: float | None,
        voices_per_octave: int,
        out_layout: Layout,
    ) -> None:
        """Buffer samples until a whole block is available."""

        self._window = window
        self._rate_hz = rate_hz
        self._voices_per_octave = voices_per_octave
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

        # Transform each whole block that has arrived, one at a time, and drop it
        # from the front of the buffer. A partial block waits for more samples.
        blocks: list[SignalArray] = []
        block_times: list[RealArray] = []
        frequencies: RealArray | None = None
        while self._buffer.shape[0] >= self._window:
            block = self._buffer[: self._window]

            # Use the known rate if we have one, otherwise estimate it from the block.
            rate_hz = (
                self._rate_hz
                if self._rate_hz is not None
                else effective_rate_hz(self._times[: self._window], None)
            )
            power, frequencies = _sst_power(block, rate_hz, self._voices_per_octave)
            blocks.append(power)
            block_times.append(self._times[: self._window])
            self._buffer = self._buffer[self._window :]
            self._times = self._times[self._window :]

        if not blocks:
            return empty_signal(self._out_layout)

        coords = {} if frequencies is None else {AxisName.FREQUENCY: frequencies}
        return Signal(
            values=as_real_array(np.concatenate(blocks, axis=0)),
            times=np.concatenate(block_times),
            layout=self._out_layout,
            coords=coords,
        )

    def flush(self) -> Signal:
        """Drop any trailing partial block (matches valid-block semantics)."""

        return empty_signal(self._out_layout)
