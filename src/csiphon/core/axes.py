"""Named tensor axes; mostly for semantics-based processing (vs. raw axis numbers)

An Axis is a named tensor dimension. Steps address dimensions by name
(`AxisName.SUBCARRIER`), never by integer position, so unrelated dimensions can
sit in any order.

Size and coordinates come in three cases:

- Statically sized.
    `size` and (optionally) `coordinates` are known at compile time from the
    profile and each step's config: subcarriers, delay taps, dyadic bands,
    the frequency bins of a windowed FFT, and so on.
- Runtime sized.
    `size is None` because the length only shows up once data arrives.
    Not just time: the number of SST frequency bins depends on the recording
    length, so that axis is runtime sized and carries its coordinates on the
    Signal (`coords`) instead of in the layout. Any number of axes can be
    runtime sized.
- The time axis, named AxisName.TIME.
    Its coordinates ride with the data as `Signal.times`. It is runtime sized
    too, since you don't know a recording's or a stream's length up front. A
    layout has at most one time axis.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from csiphon.core.errors import LayoutError

# The axis name gives a dimension its semantics.
# Its coordinates give each position along it a physical meaning.
# For example:
#   - the exact frequency values along a `FREQUENCY` axis
#   - the subcarrier index each `SUBCARRIER` position corresponds to.
# Any hashable value works; axes with no meaningful label (eg. PCA components)
# just use their integer index.
type Coordinate = Hashable


class AxisName(StrEnum):
    """Semantic dimensions used by CSI preprocessing steps.

    A curated, closed set (which we might extend)
    The fixed vocabulary keeps describe() and cross-step validation type-safe.
    The spatial names describe a general MIMO / multi-device capture,
    such as (receivers, transmit/receive antennas, spatial streams).

    When a step produces a genuinely new kind of dimension (say, reducing subcarriers
    to abstract components), pick the closest name here (`COMPONENT`, `FEATURE`, or
    `LATENT` for learned reductions) rather than mislabelling the output. Add a
    member when you need a new physical dimension.
    """

    # fmt: off
    # time
    TIME           = "time"

    # spatial (MIMO / multi-device)
    RECEIVER       = "receiver"        # receiver device, for a multi-device capture
    RX_ANTENNA     = "rx_antenna"      # receive antenna
    TX_ANTENNA     = "tx_antenna"      # transmit antenna
    SPATIAL_STREAM = "spatial_stream"  # spatial stream (Nss)

    # spectral / delay / doppler
    SUBCARRIER     = "subcarrier"
    FREQUENCY      = "frequency"
    DELAY          = "delay"
    DOPPLER        = "doppler"
    BAND           = "band"
    WAVELET_BAND   = "wavelet_band"
    LAG            = "lag"

    # derived / learned
    COMPONENT      = "component"        # e.g. a PCA component
    LATENT         = "latent"
    FEATURE        = "feature"          # several axes folded into one
    # fmt: on


@dataclass(frozen=True, slots=True)
class Axis:
    """One named tensor dimension.

    `size` is `None` when the length is only known at run time (the time axis, or
    an axis like the SST frequency bins whose count depends on the data).

    Statically sized axes provide a positive `size` and, optionally, matching
    `coordinates`.
    """

    # fmt: off
    name        : AxisName
    size        : int | None
    coordinates : tuple[Coordinate, ...] | None = None
    unit        : str | None                    = None
    # fmt: on

    def __post_init__(self) -> None:
        """Validate the size / coordinates relationship."""

        if self.size is None:
            if self.coordinates is not None:
                raise LayoutError(
                    f"Dynamic axis '{self.name}' must not carry coordinates."
                )
            return

        if self.size < 1:
            raise LayoutError(f"Axis '{self.name}' must have a positive size.")

        if self.coordinates is not None and len(self.coordinates) != self.size:
            raise LayoutError(
                f"Axis '{self.name}' has {len(self.coordinates)} coordinates "
                f"but size {self.size}."
            )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def is_dynamic(self) -> bool:
        """Return whether the size is only known at runtime (`size is None`)."""

        return self.size is None

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------
    @classmethod
    def dynamic(cls, name: AxisName = AxisName.TIME, unit: str | None = "s") -> Axis:
        """Create the dynamic time / window axis."""

        return cls(name=name, size=None, coordinates=None, unit=unit)

    @classmethod
    def static(
        cls,
        name: AxisName,
        coordinates: Sequence[Coordinate],
        unit: str | None = None,
    ) -> Axis:
        """Create a static axis from explicit coordinates."""

        values = tuple(coordinates)
        return cls(name=name, size=len(values), coordinates=values, unit=unit)

    @classmethod
    def sized(cls, name: AxisName, size: int, unit: str | None = None) -> Axis:
        """Create a static axis of a known size without explicit coordinates."""

        return cls(name=name, size=size, coordinates=None, unit=unit)

    # -------------------------------------------------------------------------
    # Transformations
    # -------------------------------------------------------------------------
    def relabel(self, coordinates: Sequence[Coordinate]) -> Axis:
        """Return this axis with new coordinates (and matching size)."""

        return Axis.static(self.name, coordinates, unit=self.unit)
