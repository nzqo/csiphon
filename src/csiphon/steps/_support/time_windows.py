"""Windows placed by timestamp instead of by sample count.

Window `k` covers the time span `[t0 + k * hop_s, t0 + k * hop_s + window_s)`,
where `t0` is the first timestamp. A window is complete once a sample at or past
its end has arrived. Lost samples change what a window holds, never how many
windows there are. Batch and streaming use the same operator.
"""

from __future__ import annotations

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_real_array
from csiphon.core.layout import Layout
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import StreamOperator
from csiphon.steps._support.windowing import IndexArray, empty_time_buffer

# Rounding in `k * hop_s` can put a sample that sits exactly on a window edge on
# the wrong side of it. Timestamps within a nanosecond of an edge count as past it.
_EDGE_TOLERANCE_S = 1e-9


def _count_complete_windows(
    first_time_s: float, newest_time_s: float, window_s: float, hop_s: float
) -> int:
    """Return how many windows end at or before the newest timestamp."""

    # The first window ends at `first_time_s + window_s`, and each later one ends
    # one hop after the previous.
    time_past_first_end = newest_time_s - first_time_s - window_s + _EDGE_TOLERANCE_S
    if time_past_first_end < 0.0:
        return 0
    return int(time_past_first_end // hop_s) + 1


class TimeWindowSumOperator(StreamOperator):
    """Sums the samples inside each timestamp window. An empty window gives NaN."""

    def __init__(
        self, window_s: float, hop_s: float, in_layout: Layout, out_layout: Layout
    ) -> None:
        """Bind the window length and hop (seconds) and the layouts."""

        self._window_s = window_s
        self._hop_s = hop_s
        self._out_layout = out_layout

        # The samples (time first) and timestamps that a later window may still need.
        self._buffer: SignalArray = empty_time_buffer(in_layout)
        self._buffer_times: RealArray = np.zeros(0)

        # Window 0 starts at the first timestamp ever seen. `_next_window` counts
        # the windows already returned.
        self._first_time: float | None = None
        self._next_window = 0

    def push(self, chunk: Signal) -> Signal:
        """Add the chunk, and return the windows that are complete now.

        A window is complete once a sample at or past its end has arrived. Until
        then, a later sample could still fall into it.
        """

        time_index = chunk.layout.dynamic_index
        if time_index is None:
            raise ValueError("TimeWindowSumOperator needs a signal with a time axis.")
        if chunk.n_samples == 0:
            return empty_signal(self._out_layout)

        # Window 0 starts at the very first timestamp, and every other window is
        # placed relative to it.
        if self._first_time is None:
            self._first_time = float(chunk.times[0])
        first_time = self._first_time

        self._add_to_buffer(chunk, time_index)

        # Number the windows that are complete now, but were not returned before.
        # Window 0 is the first, and each further one starts a hop later.
        complete_count = _count_complete_windows(
            first_time, float(chunk.times[-1]), self._window_s, self._hop_s
        )
        new_windows = np.arange(self._next_window, complete_count)
        if new_windows.size == 0:
            return empty_signal(self._out_layout)
        self._next_window = complete_count

        # Sum each new window. Its output time is the middle of its span.
        starts = as_real_array(first_time + new_windows * self._hop_s)
        sums = self._sums_in_windows(starts)
        centers = as_real_array(starts + self._window_s / 2.0)

        # The windows still to come start at the next window's start, so earlier
        # samples are not needed anymore.
        self._drop_samples_before(first_time + complete_count * self._hop_s)

        return self._as_signal(sums, centers)

    def flush(self) -> Signal:
        """Windows that never completed are dropped, as with sample windows."""

        return empty_signal(self._out_layout)

    def _add_to_buffer(self, chunk: Signal, time_index: int) -> None:
        """Append the chunk's samples and timestamps, with time as the first axis."""

        incoming = np.moveaxis(chunk.values, time_index, 0)
        self._buffer = np.concatenate([self._buffer, incoming], axis=0)
        self._buffer_times = np.concatenate([self._buffer_times, chunk.times])

    def _drop_samples_before(self, time_s: float) -> None:
        """Forget the buffered samples that come before `time_s`."""

        keep_from = self._first_index_at(time_s)
        self._buffer = self._buffer[keep_from:]
        self._buffer_times = self._buffer_times[keep_from:]

    def _as_signal(self, sums: RealArray, centers: RealArray) -> Signal:
        """Wrap the window sums as a signal, with the window axis in the time slot."""

        out_index = self._out_layout.dynamic_index
        restored = np.moveaxis(sums, 0, 0 if out_index is None else out_index)
        return Signal(
            values=as_real_array(restored), times=centers, layout=self._out_layout
        )

    def _first_index_at(self, time_s: float | RealArray) -> IndexArray:
        """Return the buffer position of the first sample at or after `time_s`."""

        position = np.searchsorted(
            self._buffer_times, time_s - _EDGE_TOLERANCE_S, side="left"
        )
        return np.asarray(position)

    def _sums_in_windows(self, starts: RealArray) -> RealArray:
        """Return the sum of the buffered samples in each window (NaN if empty)."""

        # A window holds the buffered samples from `first` up to `past_last`.
        first = self._first_index_at(starts)
        past_last = self._first_index_at(starts + self._window_s)

        # A running total with a leading zero row turns each window sum into a
        # difference of two rows, so all windows are summed at once.
        running_total = np.cumsum(self._buffer, axis=0)
        zero_row = np.zeros_like(running_total[:1])
        running_total = np.concatenate([zero_row, running_total], axis=0)
        sums = running_total[past_last] - running_total[first]

        sums[past_last == first] = np.nan
        return as_real_array(sums)
