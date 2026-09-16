"""Shared sliding-window geometry and the ring-buffer streaming operator.

Several steps place windows on the same grid: time-window pooling, windowed-FFT
power, and the complex STFT. In streaming mode they also buffer samples until a
whole window is ready.

That shared geometry and buffering live here, so the steps (and their batch and
streaming paths) can't drift apart. The buffer follows the input's value kind, so
it carries complex samples (the STFT) as happily as real ones (FFT power).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from csiphon.core.arrays import RealArray, SignalArray, as_real_array, as_signal_array
from csiphon.core.layout import Layout
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import StreamOperator

# A frame function maps one window block (window_size, *other_axes) to one output
# frame (*frame_axes, *other_axes), e.g. a sum over the window, or an FFT power.
# The block may be real or complex; the emitted frame is real.
FrameFn = Callable[[SignalArray], RealArray]

# Integer index arrays for window starts and centers.
IndexArray = npt.NDArray[np.intp]


def sliding_window_indices(
    n_samples: int, window_size: int, hop_stride: float
) -> tuple[IndexArray, IndexArray]:
    """Return valid window start and center indices.

    A fractional `hop_stride` becomes an alternating integer step via
    `round(k * hop_stride)`, hitting the requested rate on average instead of
    silently rounding the step down.
    """

    last_start = n_samples - window_size
    if last_start < 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty

    # A window exists for every k whose rounded start `round(k*hop)` still fits, i.e.
    # `round(k*hop) <= last_start`. The `+0.5` matches that rounding boundary, so the
    # batch grid keeps the final window the streaming operator also emits (a plain
    # `floor(last_start/hop)` drops it when the hop is fractional).
    window_count = int(np.floor((last_start + 0.5) / hop_stride)) + 1
    starts = np.rint(np.arange(window_count) * hop_stride).astype(np.intp)
    starts = starts[starts <= last_start]
    return starts, starts + window_size // 2


@dataclass(frozen=True, slots=True)
class WindowGeometry:
    """The window size (samples) and hop stride (samples) for a sliding grid."""

    window_size: int
    hop_stride: float

    @classmethod
    def from_seconds(
        cls, rate_hz: float, window_s: float, hop_s: float
    ) -> WindowGeometry:
        """Resolve the physical window/hop (seconds) to samples at `rate_hz`."""

        window_size = max(1, round(window_s * rate_hz))
        hop_stride = max(1.0, hop_s * rate_hz)
        return cls(window_size=window_size, hop_stride=hop_stride)

    def indices(self, n_samples: int) -> tuple[IndexArray, IndexArray]:
        """Return the (starts, centers) for a signal of `n_samples` samples."""

        return sliding_window_indices(n_samples, self.window_size, self.hop_stride)


def rfft_band(
    window_size: int, rate_hz: float, band_hz: float
) -> tuple[RealArray, npt.NDArray[np.bool_]]:
    """Return the kept (in-band) rFFT frequencies and their boolean mask."""

    frequencies = np.fft.rfftfreq(window_size, d=1.0 / rate_hz)
    keep = frequencies <= band_hz
    return as_real_array(frequencies[keep]), keep


def _empty_time_buffer(layout: Layout) -> SignalArray:
    """An empty `(0, *non_time_axes)` buffer sized and typed from a layout.

    The buffer stores samples with the time axis first, so its non-time shape is
    fixed by the input layout and known before any data arrives. Its dtype follows
    the layout's value kind, so a complex input keeps its phase.
    """

    time_index = layout.dynamic_index
    dims: list[int] = []
    for position, size in enumerate(layout.shape):
        if position == time_index:
            continue
        if size is None:
            raise ValueError("WindowedOperator needs statically sized non-time axes.")
        dims.append(size)
    dtype = np.complex128 if layout.values == ValueKind.COMPLEX else np.float64
    return as_signal_array(np.zeros((0, *dims), dtype=dtype))


# Ring-buffer state (below) justifies the attribute count.
class WindowedOperator(StreamOperator):  # pylint: disable=too-many-instance-attributes
    """Buffer samples across chunks and apply a frame function to each window.

    Data arrives in chunks, but windows are placed on the whole stream, so a
    window can span two chunks. This operator holds the samples an upcoming
    window still needs and drops the rest, keeping memory bounded however long
    the stream runs. The windows it produces match the batch ones exactly.
    """

    def __init__(
        self,
        geometry: WindowGeometry,
        frame_fn: FrameFn,
        in_layout: Layout,
        out_layout: Layout,
    ) -> None:
        """Bind the window geometry, per-window frame function, and layouts."""

        self._window_size = geometry.window_size
        self._hop_stride = geometry.hop_stride
        self._frame_fn = frame_fn
        self._out_layout = out_layout

        # The buffer holds input samples still needed by an upcoming window, time
        # axis first. Its non-time shape comes from the input layout, so it starts
        # empty rather than needing a special case for the first chunk.
        self._buffer: SignalArray = _empty_time_buffer(in_layout)
        self._buffer_times: RealArray = np.zeros(0)
        # Counters that let us map buffer positions back to absolute sample and
        # window numbers even after we drop consumed samples off the front: the
        # absolute index of the buffer's first sample, and how many windows we
        # have produced so far.
        self._buffer_origin = 0
        self._next_window_index = 0

    def push(self, chunk: Signal) -> Signal:
        """Add the next samples and return any windows they complete.

        Chunks arrive at any size, but the window size is fixed, so after adding a
        chunk the buffer may hold no full window yet, or one or more full windows
        plus a leftover tail.
        """

        time_index = chunk.layout.dynamic_index
        if time_index is None:
            raise ValueError("WindowedOperator requires a signal with a time axis.")

        # Append the new samples (time axis first) and their timestamps.
        incoming = as_signal_array(np.moveaxis(chunk.values, time_index, 0))
        self._buffer = np.concatenate([self._buffer, incoming], axis=0)
        self._buffer_times = np.concatenate([self._buffer_times, chunk.times])

        # Collect all complete windows that are in the buffer now
        frames: list[RealArray] = []
        centers: list[float] = []

        # Iterate until there are no full windows left
        while True:
            # Where this window begins, counted from the start of the buffer.
            window_start = round(self._next_window_index * self._hop_stride)
            offset = window_start - self._buffer_origin

            # Current section does not hold a full window yet, frames/centers complete.
            if offset + self._window_size > self._buffer.shape[0]:
                break

            # - Take the window's samples
            # - process them (apply whatever streaming operating)
            # - note its center time.
            block = self._buffer[offset : offset + self._window_size]
            frames.append(self._frame_fn(block))
            centers.append(float(self._buffer_times[offset + self._window_size // 2]))
            self._next_window_index += 1

        # Drop the leading samples no upcoming window will need any more, and move
        # the buffer origin forward to where the next window starts.
        next_start = round(self._next_window_index * self._hop_stride)
        discard = next_start - self._buffer_origin
        if discard > 0:
            self._buffer = self._buffer[discard:]
            self._buffer_times = self._buffer_times[discard:]
            self._buffer_origin = next_start

        # Emit just stacks all those windows into one long signal.
        return self._emit(frames, centers)

    def flush(self) -> Signal:
        """Valid-window semantics keep no partial final window."""

        return empty_signal(self._out_layout)

    def _emit(self, frames: list[RealArray], centers: list[float]) -> Signal:
        """Stack emitted frames along the output layout's time axis."""

        if not frames:
            return empty_signal(self._out_layout)

        stacked = np.stack(frames, axis=0)
        time_index = self._out_layout.dynamic_index
        time_index = 0 if time_index is None else time_index
        values = np.moveaxis(stacked, 0, time_index)
        return Signal(
            values=as_real_array(values),
            times=np.asarray(centers, dtype=float),
            layout=self._out_layout,
        )
