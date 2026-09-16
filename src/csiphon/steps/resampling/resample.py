"""Resample onto a uniform time grid, with a pluggable fill method.

`Resample(rate_hz=500, fill=Hold())` puts a (possibly irregular) stream onto the
exact grid `t0, t0 + 1/rate, ...`: the output timestamps are that grid, so the
result is precisely `rate_hz`. `fill=` picks how each grid value is drawn from the
source samples, the same strategy-object pattern the merges use:

- `Hold`      : the most recent value at or before the grid point (zero-order hold)
- `Nearest`   : the closest source sample in time
- `Linear`    : straight line between the bracketing samples
- `PolarLinear` : CSI-aware, interpolate magnitude and unwrapped phase separately
- `CubicSpline` : a cubic spline (needs scipy; falls back to linear below 4 points)

The first three are causal and stream exactly; the last two look at the whole
recording (phase unwrap / global spline) and are batch-only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np

from csiphon.core.arrays import (
    RealArray,
    SignalArray,
    as_complex_array,
    as_real_array,
    as_signal_array,
)
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError, MissingDependencyError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline._align_ops import concat_time, hold_values_at
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import CONFIG_DEPENDENT, Category, LayoutEffect, StepSpec, Streaming


class FillMethod(ABC):  # pylint: disable=too-few-public-methods
    """How a resampler draws each grid value from the (irregular) source samples.

    A one-method strategy (like the merge strategies): pick a subclass and hand it to
    `Resample(fill=...)`. Each also declares whether it can stream.
    """

    # Whether resampling with this fill streams (see Streaming). Causal fills that
    # only need the samples bracketing a grid point are BATCH_EQUIVALENT.
    streaming: ClassVar[Streaming]
    # A fill that only makes sense on complex CSI (magnitude/phase) sets this, so
    # Resample can reject a real input up front instead of failing cryptically later.
    requires_complex: ClassVar[bool] = False

    @abstractmethod
    def values_on_grid(
        self, signal: Signal, source_times: RealArray, grid: RealArray
    ) -> SignalArray:
        """The values at each `grid` time, from `signal` sampled at `source_times`."""


@dataclass(frozen=True, slots=True)
class Hold(FillMethod):
    """Zero-order hold: the most recent value at or before each grid point."""

    streaming: ClassVar[Streaming] = Streaming.BATCH_EQUIVALENT

    def values_on_grid(
        self, signal: Signal, source_times: RealArray, grid: RealArray
    ) -> SignalArray:
        """Hold the latest value at or before each grid point."""

        return hold_values_at(signal, source_times, grid)


@dataclass(frozen=True, slots=True)
class Nearest(FillMethod):
    """The value of the source sample closest in time to each grid point."""

    streaming: ClassVar[Streaming] = Streaming.BATCH_EQUIVALENT

    def values_on_grid(
        self, signal: Signal, source_times: RealArray, grid: RealArray
    ) -> SignalArray:
        """Take each grid point's nearest source sample (ties go to the earlier one)."""

        after = np.searchsorted(source_times, grid, side="left")
        before = np.clip(after - 1, 0, source_times.size - 1)
        after = np.clip(after, 0, source_times.size - 1)
        closer_after = (grid - source_times[before]) > (source_times[after] - grid)
        index = np.where(closer_after, after, before)
        axis = _time_axis(signal)
        return as_signal_array(np.take(signal.values, index, axis=axis))


@dataclass(frozen=True, slots=True)
class Linear(FillMethod):
    """Straight-line interpolation between the bracketing samples, per channel."""

    streaming: ClassVar[Streaming] = Streaming.BATCH_EQUIVALENT

    def values_on_grid(
        self, signal: Signal, source_times: RealArray, grid: RealArray
    ) -> SignalArray:
        """Interpolate each channel linearly; complex data goes real and imag apart."""

        def fill(column: SignalArray) -> SignalArray:
            """Linear interpolation of one channel, real and imaginary parts apart."""

            if np.iscomplexobj(column):
                real = np.interp(grid, source_times, column.real)
                imag = np.interp(grid, source_times, column.imag)
                return as_signal_array(real + 1j * imag)
            return as_signal_array(np.interp(grid, source_times, as_real_array(column)))

        return _per_channel(signal, grid, fill)


@dataclass(frozen=True, slots=True)
class PolarLinear(FillMethod):
    """CSI-aware linear interpolation: magnitude and unwrapped phase, separately.

    Interpolating a complex value's real and imaginary parts shrinks its magnitude
    across a phase rotation; interpolating magnitude and (unwrapped) phase instead
    keeps the magnitude and follows the rotation. Batch-only, because unwrapping the
    phase reads the whole recording.
    """

    streaming: ClassVar[Streaming] = Streaming.UNAVAILABLE
    # Magnitude and phase only mean something for a complex signal.
    requires_complex: ClassVar[bool] = True

    def values_on_grid(
        self, signal: Signal, source_times: RealArray, grid: RealArray
    ) -> SignalArray:
        """Interpolate |x| and the unwrapped angle of x, then recombine."""

        def fill(column: SignalArray) -> SignalArray:
            """Interpolate one channel's magnitude and unwrapped phase apart."""

            complex_column = as_complex_array(column)
            magnitude = np.interp(grid, source_times, np.abs(complex_column))
            phase = np.interp(grid, source_times, np.unwrap(np.angle(complex_column)))
            return as_signal_array(magnitude * np.exp(1j * phase))

        return _per_channel(signal, grid, fill)


@dataclass(frozen=True, slots=True)
class CubicSpline(FillMethod):
    """A cubic spline per channel (needs scipy; below 4 points it falls to linear)."""

    streaming: ClassVar[Streaming] = Streaming.UNAVAILABLE

    def values_on_grid(
        self, signal: Signal, source_times: RealArray, grid: RealArray
    ) -> SignalArray:
        """Fit a cubic spline per channel; too few points fall back to linear."""

        if source_times.size < 4:
            return Linear().values_on_grid(signal, source_times, grid)
        spline = _cubic_spline()

        def fill(column: SignalArray) -> SignalArray:
            """Cubic-spline one channel, real and imaginary parts apart."""

            if np.iscomplexobj(column):
                real = spline(source_times, column.real, extrapolate=True)(grid)
                imag = spline(source_times, column.imag, extrapolate=True)(grid)
                return as_signal_array(real + 1j * imag)
            fitted = spline(source_times, as_real_array(column), extrapolate=True)
            return as_signal_array(fitted(grid))

        return _per_channel(signal, grid, fill)


@dataclass(frozen=True, slots=True)
class Resample(Step):
    """Resample onto a uniform time grid at a fixed rate, filling by `fill`.

    Give the grid as `rate_hz` (e.g. 500) or the equivalent `step_s` (e.g. 0.02),
    exactly one. The output timestamps are the grid `t0 + k*step`, so the stream comes
    out at precisely that rate whatever the input timing. `fill` chooses how each grid
    value is drawn from the source samples (Hold, Nearest, Linear, PolarLinear,
    CubicSpline); its choice also decides whether the step can stream.
    """

    fill: FillMethod = field(
        default_factory=Hold, metadata={"doc": "how each grid value is drawn"}
    )
    rate_hz: float | None = field(
        default=None, metadata={"doc": "uniform output rate in Hz"}
    )
    step_s: float | None = field(
        default=None, metadata={"doc": "uniform grid spacing in seconds"}
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="resample",
        summary="resample onto a uniform time grid, filled by the chosen method",
        category=Category.RESAMPLING,
        admissible_values=None,
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(note="resampled onto a uniform time grid"),
        streaming=CONFIG_DEPENDENT,
        streaming_note="streams for causal fills (Hold/Nearest/Linear), else batch",
    )

    def resolve_streaming(self) -> Streaming:
        """The fill decides: causal fills stream, whole-recording ones do not."""

        return self.fill.streaming

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Time stays the (dynamic) axis; structure and semantics are unchanged."""

        self.require_inputs(layout)
        if (self.rate_hz is None) == (self.step_s is None):
            raise LayoutError(
                "Resample needs exactly one of `rate_hz` or `step_s`, not both "
                "or neither."
            )
        if self.rate_hz is not None and self.rate_hz <= 0:
            raise LayoutError(f"rate_hz must be > 0, got {self.rate_hz}.")
        if self.step_s is not None and self.step_s <= 0:
            raise LayoutError(f"step_s must be > 0, got {self.step_s}.")
        if self.fill.requires_complex and layout.values is not ValueKind.COMPLEX:
            raise LayoutError(
                f"{type(self.fill).__name__} fill needs a complex signal (it "
                f"interpolates magnitude and phase); got {layout.values.value}."
            )
        return layout

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Build the grid over the recording and fill each point from the source."""

        if signal.layout.dynamic_index is None or signal.n_samples == 0:
            return signal.with_values(signal.values, out_layout)
        times = as_real_array(signal.times)
        grid = _time_grid(float(times[0]), float(times[-1]), self._step_seconds())
        values = self.fill.values_on_grid(signal, times, grid)
        return signal.with_values(values, out_layout, times=grid)

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """A causal grid operator for streaming fills; None for batch-only ones."""

        if self.fill.streaming is Streaming.UNAVAILABLE:
            return None
        return _GridOperator(self.fill, self._step_seconds(), out_layout)

    def _step_seconds(self) -> float:
        """The grid spacing in seconds, from whichever of rate_hz / step_s was given."""

        if self.step_s is not None:
            return self.step_s
        assert self.rate_hz is not None
        return 1.0 / self.rate_hz


def _cubic_spline() -> Callable[..., Any]:
    """Import scipy's CubicSpline lazily, with a helpful error if it is missing."""

    try:
        # Lazy so the base install stays numpy-only; aliased so it does not shadow the
        # CubicSpline fill class above.
        from scipy.interpolate import (  # pylint: disable=import-outside-toplevel
            CubicSpline as ScipyCubicSpline,
        )
    except ImportError as error:  # pragma: no cover - exercised via message
        raise MissingDependencyError("scipy", "filters") from error
    return cast("Callable[..., Any]", ScipyCubicSpline)


def _time_axis(signal: Signal) -> int:
    """The signal's time axis position (present on any signal a resampler sees)."""

    axis = signal.layout.dynamic_index
    assert axis is not None
    return axis


def _per_channel(
    signal: Signal, grid: RealArray, fill: Callable[[SignalArray], SignalArray]
) -> SignalArray:
    """Apply a per-column fill along the time axis, restoring the channel shape.

    Moves time to the front, flattens the channel axes into columns, fills each
    onto `grid`, then restores the original channel shape with time back in place.
    """

    axis = _time_axis(signal)
    moved = np.moveaxis(signal.values, axis, 0)
    columns = moved.reshape(moved.shape[0], -1)
    filled = np.stack([fill(columns[:, c]) for c in range(columns.shape[1])], axis=1)
    restored = np.moveaxis(filled.reshape((grid.size, *moved.shape[1:])), 0, axis)
    return as_signal_array(restored)


def _last_grid_index(last_time: float, start: float, step_s: float) -> int:
    """The largest k whose grid point `start + k*step_s` is at or before `last_time`.

    The `+1e-9` absorbs float error when the span is an exact multiple of the step, so
    the final grid point is not silently dropped (e.g. a 1 s run stepped at 0.1 s would
    otherwise stop at 0.9 s because 1.0 / 0.1 evaluates to just under 10).
    """

    return int(np.floor((last_time - start) / step_s + 1e-9))


def _time_grid(start: float, stop: float, step_s: float) -> RealArray:
    """The uniform grid `start, start + step_s, ...` up to (and within) `stop`."""

    count = _last_grid_index(stop, start, step_s) + 1
    return as_real_array(start + np.arange(count) * step_s)


class _GridOperator(StreamOperator):
    """Streams a causal fill onto the grid, carrying one sample for look-back.

    A grid point is emitted once a sample at or after it has arrived (so its
    bracketing samples are known); the previous sample is carried between chunks so
    the fill can look back across a chunk boundary.
    """

    def __init__(self, fill: FillMethod, step_s: float, out_layout: Layout) -> None:
        """Wait for the first sample to anchor the grid, then track the next point."""

        self._fill = fill
        self._step_s = step_s
        self._out_layout = out_layout
        self._start: float | None = None
        # The next grid index not yet emitted, and the most recent sample kept
        # for look-back.
        self._next_k = 0
        self._carry: Signal | None = None

    def push(self, chunk: Signal) -> Signal:
        """Emit the grid points now covered, filled from the carried + new samples."""

        time_index = chunk.layout.dynamic_index
        if time_index is None or chunk.n_samples == 0:
            return empty_signal(self._out_layout)

        # The grid is anchored on the first sample ever seen.
        start = self._start
        if start is None:
            start = self._start = float(chunk.times[0])
        last_k = _last_grid_index(float(chunk.times[-1]), start, self._step_s)

        # The look-back source is the carried previous sample plus this chunk; only
        # build it when a grid point is actually due (else just carry forward).
        carry, self._carry = self._carry, _last_sample(chunk, time_index)
        # No new grid point reached yet.
        if last_k < self._next_k:
            return empty_signal(self._out_layout)

        source = chunk if carry is None else concat_time(carry, chunk)
        grid = as_real_array(start + np.arange(self._next_k, last_k + 1) * self._step_s)
        self._next_k = last_k + 1
        values = self._fill.values_on_grid(source, as_real_array(source.times), grid)
        return chunk.with_values(values, self._out_layout, times=grid)

    def flush(self) -> Signal:
        """Grid points are emitted as they are reached, so nothing is left waiting."""

        return empty_signal(self._out_layout)


def _last_sample(signal: Signal, time_index: int) -> Signal:
    """The signal trimmed to its final time sample (a 1-sample look-back carry)."""

    last = np.take(signal.values, [signal.n_samples - 1], axis=time_index)
    return signal.with_values(
        as_signal_array(last), signal.layout, times=as_real_array(signal.times[-1:])
    )
