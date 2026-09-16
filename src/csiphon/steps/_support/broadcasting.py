"""Apply a 2-D per-channel operation along one axis, broadcasting the rest.

Some steps run a genuine matrix algorithm across one axis for every time sample:
a covariance eigenspectrum, an SVD-based rotation bias, a robust-PCA
decomposition. Unlike an FFT or a magnitude, these have no elementwise `axis=`
form. Each needs a full 2-D `(time, feature)` slice at once. When the signal
carries extra axes (several receivers, antennas, ...), each of those axes is just
an independent channel: run the operation once per channel and stack the results.

This helper is that loop. It keeps every step's core math a plain, readable 2-D
function and gathers the axis bookkeeping in one place.
"""

from collections.abc import Callable

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_real_array, as_signal_array
from csiphon.core.axes import AxisName
from csiphon.core.layout import Layout


def operated_axis_position(layout: Layout, axis: AxisName) -> int:
    """Position of `axis` in a block whose time axis has been moved to the front.

    Both the windowed operator and the batch steps put time on axis 0 before they
    work, so the operated axis sits one past its rank among the non-time axes.
    """

    non_time = [entry.name for entry in layout.axes if not entry.is_dynamic]
    return 1 + non_time.index(axis)


def broadcast_channels(
    block: SignalArray,
    feature_pos: int,
    core: Callable[[SignalArray], RealArray],
) -> RealArray:
    """Run `core` on the `(axis 0, feature)` matrix once per remaining channel.

    `block` holds time (or a window) on axis 0 and the operated axis at
    `feature_pos`; every other axis is an independent channel. `core` receives a
    clean 2-D `(axis 0, feature)` slice and returns its result, whose last axis
    replaces the operated axis and whose leading axes (if any) become new front
    axes. The channel axes keep their positions.
    """

    # Bring the operated axis next to time, then fold the remaining axes into one
    # flat channel dimension so we can iterate over them.
    moved = np.moveaxis(block, feature_pos, 1)
    channel_shape = moved.shape[2:]
    flat = moved.reshape(moved.shape[0], moved.shape[1], -1)

    # Run the 2-D core once per channel and stack the outputs on a new last axis.
    columns = [
        core(as_signal_array(flat[:, :, index])) for index in range(flat.shape[2])
    ]
    stacked = np.stack(columns, axis=-1)

    # Unfold that flat channel axis back into the original channel axes, then move
    # the core's result (its last axis) into the operated axis's old slot.
    leading = stacked.shape[:-2]
    result = stacked.reshape(*leading, stacked.shape[-2], *channel_shape)
    return as_real_array(
        np.moveaxis(result, len(leading), len(leading) + feature_pos - 1)
    )
