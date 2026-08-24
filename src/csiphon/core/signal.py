"""The concrete signal that flows through a pipeline.

A Signal is a NumPy array plus:

- its Layout (the structure: named axes and their semantics),
- the timestamps of its time axis (Signal.times),
- any runtime-resolved coordinates (Signal.coords), e.g. the SST frequencies
  whose count depends on the recording length.

In batch mode the array is a whole recording; in streaming mode it is one chunk.
The layout fixes every coordinate except the few that ride along here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from csiphon.core.arrays import RealArray, SignalArray, as_real_array, as_signal_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import DataError
from csiphon.core.layout import Layout
from csiphon.core.semantics import ValueKind


@dataclass(frozen=True, slots=True)
class Signal:
    """Numerical values with their layout, timestamps, and runtime coordinates."""

    # fmt: off
    values   : SignalArray
    times    : RealArray
    layout   : Layout
    coords   : dict[AxisName, RealArray] = field(default_factory=dict)
    sequence : RealArray | None = None  # per-sample packet number, if known
    # fmt: on

    def __post_init__(self) -> None:
        """Verify shape, timestamps, and complex-versus-real semantics."""

        # Sequence numbers, when present, are one per time sample (parallel to times).
        if self.sequence is not None and self.sequence.size != self.times.size:
            raise DataError(
                f"There are {self.sequence.size} sequence numbers "
                f"but {self.times.size} timestamps."
            )

        if self.values.ndim != len(self.layout.axes):
            raise DataError(
                f"Array has {self.values.ndim} dimensions but the layout has "
                f"{len(self.layout.axes)} axes."
            )

        time_index = self.layout.dynamic_index
        for position, axis in enumerate(self.layout.axes):
            # timestamp vs data length
            actual = self.values.shape[position]
            if position == time_index:
                if actual != self.times.size:
                    raise DataError(
                        f"Time axis has length {actual} "
                        f"but there are {self.times.size} timestamps."
                    )

            # Other axes with potential fixed specification must also match
            elif axis.size is not None and actual != axis.size:
                raise DataError(
                    f"Axis '{axis.name}' has length {actual} but the layout "
                    f"declares size {axis.size}."
                )

        # A layout with no time axis is a single frame: 0 or 1 timestamps, no more.
        if time_index is None and self.times.size not in (0, 1):
            raise DataError("A layout without a time axis carries 0 or 1 timestamps.")

        # The array's complexity must match the value kind the layout declares.
        is_complex = np.iscomplexobj(self.values)
        expects_complex = self.layout.values == ValueKind.COMPLEX
        if is_complex != expects_complex:
            raise DataError(
                f"Layout declares {self.layout.values.value}, "
                f"but the dtype is {self.values.dtype}."
            )

    @property
    def n_samples(self) -> int:
        """Return the length of the time axis (0 if the layout has none)."""

        time_index = self.layout.dynamic_index
        if time_index is None:
            return 0
        return int(self.values.shape[time_index])

    def with_values(
        self,
        values: npt.ArrayLike,
        layout: Layout,
        times: npt.ArrayLike | None = None,
        coords: dict[AxisName, RealArray] | None = None,
    ) -> Signal:
        """Return a new signal after a step has produced new values.

        `times` defaults to the current timestamps (correct for the many steps
        that leave the time axis untouched). `coords` defaults to empty:
        runtime coordinates are dropped unless a step explicitly forwards them.
        """

        new_times = self.times if times is None else as_real_array(times)
        # Carry sequence numbers through steps that leave the time axis untouched;
        # a step that reshapes time (windowing) drops them (align before those).
        new_sequence = self.sequence if times is None else None
        return Signal(
            values=as_signal_array(values),
            times=new_times,
            layout=layout,
            coords={} if coords is None else coords,
            sequence=new_sequence,
        )


def create_signal(
    values: npt.ArrayLike,
    times: npt.ArrayLike,
    layout: Layout,
    coords: dict[AxisName, RealArray] | None = None,
    sequence: npt.ArrayLike | None = None,
) -> Signal:
    """Build a signal from raw arrays, normalizing dtypes.

    Pass `sequence` (one packet number per sample) to align on sequence numbers
    rather than timestamps when this signal is later merged.
    """

    return Signal(
        values=as_signal_array(values),
        times=as_real_array(times),
        layout=layout,
        coords={} if coords is None else coords,
        sequence=None if sequence is None else as_real_array(sequence),
    )


def empty_signal(layout: Layout) -> Signal:
    """Build a zero-length signal for a layout (used by streaming operators).

    The time axis (and any unknown-size axis) has length zero; every known
    static axis keeps its declared size, so downstream steps can process it
    without special-casing "no data ready yet".
    """

    shape = tuple(0 if size is None else size for size in layout.shape)
    if layout.values == ValueKind.COMPLEX:
        values: SignalArray = np.zeros(shape, dtype=np.complex128)
    else:
        values = np.zeros(shape, dtype=np.float64)
    return Signal(values=values, times=np.zeros(0, dtype=np.float64), layout=layout)
