"""Stable radio properties of a capture setup.

An AcquisitionProfile describes the *hardware / capture* configuration
that is constant across every recording from the same setup. A pipeline is
compiled against a profile, so one compiled pipeline serves all recordings from
that setup (batch or streaming).

Sampling rate is deliberately **optional**: WiFi CSI timestamps are never
perfectly uniform, and the timestamps carried on each Signal are the
source of truth. Provide `sampling_rate_hz` as a *nominal* value only. Steps
that lay out a frequency axis (STFT / SST / filters) need this nominal rate at
compile time to size their output; steps that do not touch time spacing ignore
it entirely.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from csiphon.core.arrays import RealArray, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal, create_signal


@dataclass(frozen=True, slots=True)
class AcquisitionProfile:
    """Radio configuration shared by every recording from one setup.

    This describes a general MIMO, possibly multi-device capture. The spatial
    structure is three counts: how many receiver devices, how many transmit
    antennas, and how many receive antennas each link carries. The raw layout
    (see `raw_csi_layout`) always carries all three as axes (even at size one)
    because a capture always has at least one of each; a size-1 axis means "one",
    not "absent". So the raw layout is always::

        (time, receiver, tx_antenna, rx_antenna, subcarrier)

    with the structural axes sized 1 for a single-device / single-antenna setup.
    (Only *domain* axes, frequency, delay, ..., are genuinely optional, and a
    step adds them when it produces them.) Keeping the structural axes present lets
    captures merge (concatenate receivers or antennas) without one branch's
    axis having been squeezed away.
    """

    # fmt: off
    subcarrier_indices  : tuple[int, ...]
    n_rx_antennas       : int          = 1
    n_tx_antennas       : int          = 1
    n_receivers         : int          = 1
    sampling_rate_hz    : float | None = None
    center_frequency_hz : float | None = None
    bandwidth_hz        : float | None = None
    # fmt: on

    def __post_init__(self) -> None:
        """Validate the spatial and subcarrier configuration."""

        if self.n_rx_antennas < 1:
            raise LayoutError("A profile needs at least one receive antenna.")
        if self.n_tx_antennas < 1:
            raise LayoutError("A profile needs at least one transmit antenna.")
        if self.n_receivers < 1:
            raise LayoutError("A profile needs at least one receiver device.")
        if not self.subcarrier_indices:
            raise LayoutError("A profile needs at least one subcarrier index.")

    @property
    def n_subcarriers(self) -> int:
        """Return the number of subcarriers in the capture."""

        return len(self.subcarrier_indices)

    def require_sampling_rate(self, step: str) -> float:
        """Return the nominal rate or fail explaining why a step needs it."""

        if self.sampling_rate_hz is None:
            raise LayoutError(
                f"Step '{step}' needs a nominal sampling rate to lay out its "
                "frequency axis. Set AcquisitionProfile.sampling_rate_hz "
                "(actual timestamp jitter is still tolerated at run time)."
            )
        return self.sampling_rate_hz

    def raw_csi_layout(self) -> Layout:
        """Return the raw-CSI input layout for this setup.

        The axes are always, in order, time, receiver, transmit antenna, receive
        antenna, subcarrier, the three structural axes kept even at size one (the
        class docstring explains why).

        To feed an array that omits the singleton structural axes, use `raw_signal`,
        which inserts them for you.
        """

        axes = [
            Axis.dynamic(AxisName.TIME, unit="s"),
            Axis.static(AxisName.RECEIVER, tuple(range(self.n_receivers))),
            Axis.static(AxisName.TX_ANTENNA, tuple(range(self.n_tx_antennas))),
            Axis.static(AxisName.RX_ANTENNA, tuple(range(self.n_rx_antennas))),
            Axis.static(AxisName.SUBCARRIER, self.subcarrier_indices, unit="index"),
        ]

        return Layout(
            axes=tuple(axes),
            representation=Representation.CHANNEL_FREQUENCY_RESPONSE,
            values=ValueKind.COMPLEX,
        )

    def raw_signal(
        self,
        values: npt.ArrayLike,
        times: npt.ArrayLike,
        coords: dict[AxisName, RealArray] | None = None,
    ) -> Signal:
        """Wrap a raw-CSI array as a Signal, filling in any omitted structural axes.

        The full layout is `(time, receiver, tx_antenna, rx_antenna, subcarrier)`.
        You may spell every structural axis out, or omit any that this setup has only
        one of: a single-receiver 3-antenna capture works as the full array, as
        `(time, rx_antenna, subcarrier)`, or as anything between. Whichever size-1
        structural axes the array omits are inserted at their proper positions so it
        lands on the layout from `raw_csi_layout`.
        """

        array = as_signal_array(values)
        layout = self.raw_csi_layout()

        # The array may spell out the size-1 structural axes or leave them out.
        # Rebuild the full shape so it matches the layout. This only adds and
        # removes length-1 axes, so no data moves; a real size or order mismatch
        # still fails later in create_signal.
        if array.ndim < len(layout.axes):
            # Strip the size-1 structural axes the array did carry.
            interior = tuple(a for a in range(1, array.ndim - 1) if array.shape[a] == 1)
            core = np.squeeze(array, axis=interior)

            # The multi-element structural sizes it kept, in layout order.
            present = list(core.shape[1:-1])

            # Put a structural axis back in each slot: 1 where this setup has
            # only one, otherwise the size the array carried for it.
            structural = [
                1 if count == 1 else (present.pop(0) if present else count)
                for count in (self.n_receivers, self.n_tx_antennas, self.n_rx_antennas)
            ]
            target = (core.shape[0], *structural, core.shape[-1])
            array = as_signal_array(core.reshape(target))

        return create_signal(array, times, layout, coords)
