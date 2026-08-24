"""The structural contract carried through a pipeline.

A Layout is the structure of a signal: its ordered named axes and their semantics.
It does not hold the recording's timestamps or data.

A pipeline is "compiled" against an `AcquisitionProfile` (antennas, subcarriers,
bandwidth, ...), which builds this layout. Because the layout carries no
per-recording data, one compiled pipeline then works for every recording from
that profile, and for an unbounded live stream where we don't know the
timestamps up front.

The concrete data and timestamps live on the runtime Signal.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.semantics import Representation, ValueKind


@dataclass(frozen=True, slots=True)
class Layout:
    """Axes and semantic information known before numerical execution."""

    # fmt: off
    axes           : tuple[Axis, ...]
    representation : Representation
    values         : ValueKind
    # fmt: on

    def __post_init__(self) -> None:
        """Reject duplicate axis names.

        An axis may have an unknown size (`size is None`) when it only resolves
        at run time, like the SST frequency axis whose bin count depends on the
        recording length.

        The time axis (AxisName.TIME) is special: its coordinates are the
        per-sample timestamps, and they live on the Signal (Signal.times), not
        in the layout.
        """

        names = tuple(axis.name for axis in self.axes)
        if len(names) != len(set(names)):
            raise LayoutError("Every axis name must be unique within a signal.")

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    @property
    def axis_names(self) -> tuple[AxisName, ...]:
        """Return current semantic axes in tensor order."""

        return tuple(axis.name for axis in self.axes)

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Return the tensor shape, with `None` for the dynamic axis."""

        return tuple(axis.size for axis in self.axes)

    @property
    def dynamic_index(self) -> int | None:
        """Return the position of the time axis, or `None` if there is none.

        The time axis carries the timestamps on the Signal; every other
        axis is fixed by the layout (even if its size is only known at run time).
        """

        for position, axis in enumerate(self.axes):
            if axis.name == AxisName.TIME:
                return position
        return None

    def axis(self, name: AxisName) -> Axis:
        """Return one axis or fail with the available names."""

        for axis in self.axes:
            if axis.name == name:
                return axis

        available = ", ".join(self.axis_names) or "<none>"
        raise LayoutError(
            f"Required axis '{name}' is missing. Available axes: {available}."
        )

    def axis_position(self, name: AxisName) -> int:
        """Return the current tensor position of a named axis."""

        for position, axis in enumerate(self.axes):
            if axis.name == name:
                return position

        # The axis is missing: this call raises with a helpful message. The
        # AssertionError after it never runs; it is only here so the type checker
        # sees that this branch cannot fall through without a return.
        self.axis(name)
        raise AssertionError("")

    def has_axis(self, name: AxisName) -> bool:
        """Return whether an axis with this name exists."""

        return name in self.axis_names

    # -------------------------------------------------------------------------
    # Requirements (used by Step.output_layout)
    # -------------------------------------------------------------------------

    def require_values(self, allowed: Iterable[ValueKind]) -> None:
        """Fail when current value semantics are unsupported."""

        allowed_values = tuple(allowed)
        if self.values in allowed_values:
            return

        expected = ", ".join(value.value for value in allowed_values)
        raise LayoutError(f"Expected {expected}; received {self.values.value}.")

    def require_representation(self, allowed: Iterable[Representation]) -> None:
        """Fail when the current scientific representation is unsupported."""

        allowed_reprs = tuple(allowed)
        if self.representation in allowed_reprs:
            return

        expected = ", ".join(value.value for value in allowed_reprs)
        raise LayoutError(
            f"Expected representation {expected}; received {self.representation.value}."
        )

    def require_axis(self, name: AxisName) -> Axis:
        """Return a required axis (raising a helpful error if absent)."""

        return self.axis(name)

    def require_axis_absent(self, name: AxisName) -> None:
        """Fail when a step would create a duplicate axis."""

        if name in self.axis_names:
            raise LayoutError(f"Axis '{name}' already exists.")

    def require_static_axis(self, name: AxisName) -> Axis:
        """Return a required axis, failing if it is the dynamic axis."""

        axis = self.axis(name)
        if axis.is_dynamic:
            raise LayoutError(f"Axis '{name}' must be static for this step.")
        return axis

    # -------------------------------------------------------------------------
    # Transformations (return new layouts)
    # -------------------------------------------------------------------------

    def with_values(self, values: ValueKind) -> Layout:
        """Return this layout with different value semantics."""

        return replace(self, values=values)

    def with_representation(self, representation: Representation) -> Layout:
        """Return this layout with a different representation."""

        return replace(self, representation=representation)

    def replace_axis(self, old_name: AxisName, new_axis: Axis) -> Layout:
        """Replace an axis without changing its tensor position."""

        if new_axis.name != old_name:
            self.require_axis_absent(new_axis.name)

        position = self.axis_position(old_name)
        axes = list(self.axes)
        axes[position] = new_axis
        return replace(self, axes=tuple(axes))

    def insert_axis_after(self, existing: AxisName, new_axis: Axis) -> Layout:
        """Insert a new axis directly after an existing one."""

        self.require_axis_absent(new_axis.name)
        position = self.axis_position(existing) + 1
        axes = list(self.axes)
        axes.insert(position, new_axis)
        return replace(self, axes=tuple(axes))

    def remove_axes(self, names: Sequence[AxisName]) -> Layout:
        """Remove named axes while preserving remaining order."""

        target = set(names)
        for name in target:
            # Raise now if any named axis is missing, before we drop anything.
            self.axis(name)
        remaining = tuple(axis for axis in self.axes if axis.name not in target)
        return replace(self, axes=remaining)

    def set_axes(self, axes: Sequence[Axis]) -> Layout:
        """Return this layout with an explicit new axis tuple."""

        return replace(self, axes=tuple(axes))

    def describe_axes(self) -> str:
        """Return a compact human-readable axis summary."""

        parts: list[str] = []
        for axis in self.axes:
            size = "?" if axis.size is None else str(axis.size)
            parts.append(f"{axis.name}[{size}]")
        return ", ".join(parts) or "<scalar>"
