"""Declarative, inspectable contracts for steps.

Every Step carries a StepSpec: a small record that says, without running
anything, what the step accepts, what it produces, how it streams, and what it
is for.

It has two uses:

- describe() renders it (plus the step's parameters) as a readable block.
- Step.require_inputs enforces it at compile time, so the documented contract
  and the runtime check can never drift apart.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, assert_never, cast

from csiphon.core.axes import Axis, AxisName
from csiphon.core.layout import Layout
from csiphon.core.semantics import Representation, ValueKind
from csiphon.graphics.style import Status, Styler, Theme, should_color
from csiphon.graphics.tree import FlowNode, render_flow


# -----------------------------------------------------------------------------
# Streaming semantics
# -----------------------------------------------------------------------------
class Streaming(StrEnum):
    """Whether a block has an online (streaming) version, and how it relates to batch.

    - BATCH_EQUIVALENT.: the online version gives the same result as batch.
    - BATCH_DIVERGENT..: an online version exists, but it is a different computation
      whose result differs from batch (e.g. a causal filter, a block-local
      transform).
    - UNAVAILABLE......: there is no online version; needs the whole recording.
    """

    # fmt: off
    BATCH_EQUIVALENT = "batch-equivalent"
    BATCH_DIVERGENT  = "batch-divergent"
    UNAVAILABLE      = "unavailable"
    # fmt: on

    @property
    def status(self) -> Status:
        """The traffic-light status used to color this state."""

        match self:
            case Streaming.BATCH_EQUIVALENT:
                return Status.OK
            case Streaming.BATCH_DIVERGENT:
                return Status.WARN
            case Streaming.UNAVAILABLE:
                return Status.BAD
            case _:  # a new Streaming member without a status here
                assert_never(self)

    @property
    def gloss(self) -> str:
        """A short, human explanation of what this state means."""

        match self:
            case Streaming.BATCH_EQUIVALENT:
                return "online result equals batch, any chunk size"
            case Streaming.BATCH_DIVERGENT:
                return "a different online computation; result differs from batch"
            case Streaming.UNAVAILABLE:
                return "no online version; needs the whole recording"
            case _:  # a new Streaming member without a gloss here
                assert_never(self)


# -----------------------------------------------------------------------------
# Step categories
# -----------------------------------------------------------------------------
class Category(StrEnum):
    """The kind of work a step does, used to group steps for the reader."""

    # fmt: off
    COMPONENTS        = "components"
    SCALING           = "scaling"
    CLEANING          = "cleaning"
    CALIBRATION       = "calibration"
    NORMALIZATION     = "normalization"
    BASELINE          = "baseline"
    FILTERING         = "filtering"
    TEMPORAL_FEATURES = "temporal_features"
    DELAY             = "delay"
    TIME_FREQUENCY    = "time_frequency"
    POOLING           = "pooling"
    STATISTICS        = "statistics"
    REDUCTION         = "reduction"
    RESTRUCTURING     = "restructuring"
    RESAMPLING        = "resampling"
    # fmt: on


# -----------------------------------------------------------------------------
# Contract types
# -----------------------------------------------------------------------------
class Deferred(Enum):
    """Marks a contract field that is only known once the step is configured.

    Some parts of a step's contract cannot be stated at the class level because
    they depend on constructor arguments (which axis a reducer removes, whether a
    reducer has a pre-fit basis and can stream). A field set to CONFIG_DEPENDENT
    says exactly that: "not knowable here, ask the configured instance" (via the
    step's resolve_* methods). This keeps an empty/None field honest: it then
    means "genuinely none / any / unchanged", never "depends, ask the instance".
    """

    CONFIG_DEPENDENT = "config-dependent"


# The one sentinel, shared by every contract field that a step resolves per
# configured instance (admissible values, streaming, and the layout effect).
CONFIG_DEPENDENT = Deferred.CONFIG_DEPENDENT


@dataclass(frozen=True, slots=True)
class LayoutEffect:
    """How a step changes the layout, as structured data instead of prose.

    Axis changes are named: `adds` are axes the step creates, `removes` axes it
    consumes, `replaces` a rename (old -> new), and `resizes` an axis that keeps
    its name but changes length. Only *names* are pinned: sizes are usually
    config- or runtime-dependent, so they are left to `note`. `value_kind=None`
    means the value kind is unchanged. An empty LayoutEffect() means the layout
    is unchanged (a pure value-domain step, or a no-op on structure).
    """

    # fmt: off
    adds       : tuple[AxisName, ...] = ()                    # axes created
    removes    : tuple[AxisName, ...] = ()                    # axes consumed
    replaces   : tuple[tuple[AxisName, AxisName], ...] = ()   # rename old -> new
    resizes    : tuple[AxisName, ...] = ()                    # same name, new size
    value_kind : ValueKind | None = None                     # None = unchanged
    note       : str = ""                                    # human nuance
    # fmt: on

    def describe(self) -> str:
        """Render the change as a short prose phrase (for describe())."""

        parts: list[str] = []
        for old, new in self.replaces:
            parts.append(f"{old.value} → {new.value}")
        if self.adds:
            parts.append("add " + ", ".join(a.value for a in self.adds))
        if self.removes:
            parts.append("drop " + ", ".join(a.value for a in self.removes))
        if self.resizes:
            parts.append("resize " + ", ".join(a.value for a in self.resizes))
        if self.value_kind is not None:
            parts.append(f"values now {self.value_kind.value}")
        text = "; ".join(parts) if parts else "structure unchanged"
        return f"{text} ({self.note})" if self.note else text


@dataclass(frozen=True, slots=True)
class StepSpec:  # pylint: disable=too-many-instance-attributes  # a data contract
    """The declarative contract of a step: what it accepts, produces, and streams.

    Fields:
    - `name`...............: short id of the step, shown in errors and describe()
    - `summary`............: prose about the step's *intent* (what it does, and why)
    - `category`...........: which group it belongs to (reduction, filtering, ...)
    - `admissible_values`..: value kinds it accepts (None = any; CONFIG_DEPENDENT = per instance)
    - `admissible_reprs`...: representations it accepts, e.g. only a CFR (None = any)
    - `requires_axes`......: axes that must be present, e.g. an antenna axis to conjugate
    - `streaming`..........: how it runs online (a Streaming mode, or CONFIG_DEPENDENT)
    - `layout_effect`......: the structured layout change (a LayoutEffect, or CONFIG_DEPENDENT)
    - `streaming_note`.....: optional caveat about streaming behaviour

    On representation vs. value kind:

    > A *representation* is the physical domain (Channel Frequency Response as
    > measured, a delay-domain response, a Doppler spectrum, ...), separate from
    > the *value kind* (complex, magnitude, power, ...), the realized numeric form.
    """  # noqa: E501

    # fmt: off
    name              : str  # used in errors + describe()
    summary           : str  # what it does, for a human
    category          : Category  # e.g. reduction, filtering
    admissible_values : tuple[ValueKind, ...] | Deferred | None  # None = any
    admissible_reprs  : tuple[Representation, ...] | None  # None = any
    requires_axes     : tuple[AxisName, ...]  # statically required axes
    streaming         : Streaming | Deferred  # mode, or CONFIG_DEPENDENT
    layout_effect     : LayoutEffect | Deferred  # structured layout change
    streaming_note    : str = ""  # optional streaming caveat
    # fmt: on


@dataclass(frozen=True, slots=True)
class ParamInfo:
    """One configurable parameter of a step, for display."""

    # fmt: off
    name      : str
    type      : str
    default   : Any
    doc       : str
    value     : Any  = None    # the configured value when describing an instance
    has_value : bool = False
    # fmt: on


# -----------------------------------------------------------------------------
# Rendered description
# -----------------------------------------------------------------------------
_WIDTH = 68  # divider width in columns
_LABELS = ("accepts", "requires", "produces", "streaming")
_LABEL_W = max(len(label) for label in _LABELS)


class _Serializable:
    """Adds JSON export to anything with an `as_dict()`."""

    __slots__ = ()

    def as_dict(self) -> dict[str, Any]:
        """Return the description as plain, JSON-able data."""

        raise NotImplementedError

    def to_json(self, indent: int = 2) -> str:
        """Return the full description as a JSON string."""

        return json.dumps(
            self.as_dict(), indent=indent, default=str, ensure_ascii=False
        )

    def save(self, path: str | Path) -> None:
        """Write the full description to a `.json` file."""

        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Description(_Serializable):
    """The rendered contract of a step, printable and machine-readable."""

    # fmt: off
    spec              : StepSpec
    params            : tuple[ParamInfo, ...]
    requires_axes     : tuple[AxisName, ...]
    admissible_values : tuple[ValueKind, ...] | Deferred | None  # resolved or from spec
    streaming         : Streaming | Deferred                     # resolved or from spec
    layout_effect     : LayoutEffect | Deferred                  # resolved or from spec
    # fmt: on

    def _produces_text(self) -> str:
        """Prose for the layout change, rendered from the structured effect."""

        return _effect_prose(self.layout_effect)

    def as_dict(self) -> dict[str, Any]:
        """Return the description as plain data (for tooling / tests)."""

        return {
            "name": self.spec.name,
            "category": self.spec.category.value,
            "summary": self.spec.summary,
            "admissible_values": (
                [self.admissible_values.value]
                if isinstance(self.admissible_values, Deferred)
                else _value_names(self.admissible_values)
            ),
            "admissible_reprs": _repr_names(self.spec.admissible_reprs),
            "requires_axes": [axis.value for axis in self.requires_axes],
            "layout_change_description": self._produces_text(),
            "streaming": self.streaming.value,
            "streaming_note": self.spec.streaming_note,
            "params": _params_dict(self.params),
        }

    def __str__(self) -> str:
        """Render with color auto-detected from the output stream."""

        return self.render()

    def render(self, color: bool | None = None, theme: Theme | None = None) -> str:
        """Render the contract block.

        `color` is auto-detected from the terminal by default; pass False for
        plain text or True to force ANSI. `theme` recolors the output (defaults
        to the vibrant palette).
        """

        style = Styler(should_color(color), theme)
        spec = self.spec

        # A full-width double rule tops each block, so stacked blocks are clearly
        # separate; the lighter single-line rules below divide sections within.
        lines = [
            style.paint("═" * _WIDTH, style.theme.rule, dim=True),
            self._header(style),
            "  " + spec.summary,
            "",
            _divider(style, "contract"),
            _contract_row(style, "accepts", self._accepts_text()),
        ]

        # Only steps that need particular axes get a "requires" row.
        requires = self._requires_text()
        if requires:
            lines.append(_contract_row(style, "requires", requires))
        lines.append(_contract_row(style, "produces", _arrows(self._produces_text())))
        lines.append(_streaming_row(style, self.streaming, spec.streaming_note))

        # Parameters, when the step has any, get their own section below.
        if self.params:
            lines.append("")
            lines.append(_divider(style, "parameters"))
            lines.extend(self._param_rows(style))

        # Trailing blank line so consecutive printed blocks stay visually separated.
        lines.append("")
        return "\n".join(lines)

    # -- line builders ----------------------------------------------------
    def _header(self, style: Styler) -> str:
        """The `name ... category` title line."""

        pad = max(1, _WIDTH - 2 - len(self.spec.name) - len(self.spec.category))
        name = style.paint(self.spec.name, style.theme.title, bold=True)
        category = style.paint(self.spec.category, style.theme.subtitle, dim=True)
        return "  " + name + " " * pad + category

    def _param_rows(self, style: Styler) -> list[str]:
        """Aligned parameter table: name, type, default, description."""

        # Column widths (the _w values), so the name, type, and default line up.
        # _default_cell renders a parameter's default as a short display string.
        defaults = [_default_cell(param) for param in self.params]
        name_w = max(len(param.name) for param in self.params)
        type_w = max(len(param.type) for param in self.params)
        default_w = max(len(cell) for cell in defaults)

        rows: list[str] = []
        for param, default_cell in zip(self.params, defaults, strict=True):
            name = style.paint(param.name.ljust(name_w), bold=True)
            type_ = style.paint(
                param.type.ljust(type_w), style.theme.subtitle, dim=True
            )

            # A parameter that was set to something other than its default is
            # shown in the override color; an unchanged one stays dim.
            changed = param.has_value and param.value != param.default
            padded = default_cell.ljust(default_w)
            if changed:
                default = style.paint(padded, style.theme.override)
            else:
                default = style.paint(padded, style.theme.subtitle, dim=True)

            rows.append(f"    {name}  {type_}  {default}  {param.doc}".rstrip())
        return rows

    # -- text bits --------------------------------------------------------
    def _accepts_text(self) -> str:
        """One line describing accepted values and representation."""

        reprs = _repr_names(self.spec.admissible_reprs)
        if isinstance(self.admissible_values, Deferred):
            value_part = "values depend on configuration"
        else:
            values = _value_names(self.admissible_values)
            value_part = f"{', '.join(values)} values" if values else "any values"
        repr_part = (
            f"{', '.join(reprs)} representation" if reprs else "any representation"
        )
        return f"{value_part}, {repr_part}"

    def _requires_text(self) -> str:
        """One line naming the required axes, or empty when none are required."""

        names = [axis.value for axis in self.requires_axes]
        if not names:
            return ""
        if len(names) == 1:
            return f"a {names[0]} axis"
        return ", ".join(names) + " axes"


def _divider(style: Styler, label: str) -> str:
    """A `── label ──…` section rule; labels align at the first letter."""

    prefix = f"  ── {label} "
    fill = "─" * max(3, _WIDTH - len(prefix))
    return style.paint(prefix + fill, style.theme.rule, dim=True)


def _contract_row(style: Styler, label: str, value: str) -> str:
    """A `label   value` row with the label dim and column-aligned."""

    return (
        f"  {style.paint(label.ljust(_LABEL_W), style.theme.subtitle, dim=True)}"
        f"  {value}"
    )


def _effect_prose(effect: LayoutEffect | Deferred) -> str:
    """Render a layout effect as prose, or note that it depends on configuration."""

    if isinstance(effect, Deferred):
        return "depends on configuration"
    return effect.describe()


def _streaming_row(style: Styler, mode: Streaming | Deferred, note: str) -> str:
    """The color-coded streaming line for one step's contract."""

    label = style.paint("streaming".ljust(_LABEL_W), style.theme.subtitle, dim=True)
    if isinstance(mode, Deferred):
        token = style.paint(mode.value, style.theme.subtitle, bold=True)
        gloss = style.paint(
            "resolved per configured instance", style.theme.subtitle, dim=True
        )
    else:
        token = style.paint(
            mode.value, style.theme.status_color(mode.status), bold=True
        )
        gloss = style.paint(mode.gloss, style.theme.subtitle, dim=True)
    row = f"  {label}  {token}  {gloss}"
    if note:
        row += style.paint(f"  ({note})", style.theme.subtitle, dim=True)
    return row


# -----------------------------------------------------------------------------
# Pipeline description
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PipelineStepView:  # pylint: disable=too-many-instance-attributes  # a data record
    """
    One step in a pipeline summary: its wiring, spec, params, and
    (if compiled) layouts. Callers build these directly, one per node.
    """

    # fmt: off
    number      : int
    spec        : StepSpec
    params      : tuple[ParamInfo, ...]
    inputs      : tuple[str, ...]        # line ids read (several for a merge)
    output      : str                    # line id this node writes
    display     : str                    # op name for the graph (merge: strategy)
    in_layout   : Layout | None
    out_layout  : Layout | None
    # fmt: on


@dataclass(frozen=True, slots=True)
class PipelineDescription(_Serializable):
    """A printable summary of a whole pipeline (concise or verbose).

    Layouts are present only for a compiled pipeline; for an un-compiled recipe
    each step shows its declared effect instead of a concrete transition.
    """

    # fmt: off
    inlet_layout  : Layout | None  # one representative source layout
    outlet_layout : Layout | None  # the sole outlet, when there is one
    steps         : tuple[PipelineStepView, ...]  # one view per node, in order
    labels        : dict[str, str] = dataclasses.field(default_factory=dict)
    outlets       : dict[str, str] = dataclasses.field(default_factory=dict)
    inlet_layouts : dict[str, Layout] = dataclasses.field(default_factory=dict)
    # fmt: on

    def as_dict(self) -> dict[str, Any]:
        """Return the full pipeline metadata as plain, JSON-able data."""

        return {
            "streams": _streaming_summary(self.steps)[0],
            "inlet": _layout_dict(self.inlet_layout),
            "outlet": _layout_dict(self.outlet_layout),
            "steps": [
                {
                    "number": view.number,
                    "name": view.spec.name,
                    "category": view.spec.category.value,
                    "summary": view.spec.summary,
                    "layout_change_description": _effect_prose(view.spec.layout_effect),
                    "streaming": view.spec.streaming.value,
                    "streaming_note": view.spec.streaming_note,
                    "params": _params_dict(view.params),
                    "layout": _layout_dict(view.out_layout),
                }
                for view in self.steps
            ],
        }

    def __str__(self) -> str:
        """Render the concise view with auto-detected color."""

        return self.render()

    def render(
        self,
        color: bool | None = None,
        theme: Theme | None = None,
        verbose: bool = False,
    ) -> str:
        """Render the pipeline. `verbose` expands each step's contract."""

        style = Styler(should_color(color), theme)
        return self._verbose(style) if verbose else self._concise(style)

    def _concise(self, style: Styler) -> str:
        """A top-to-bottom data-flow diagram: steps, branches, merges, and outlets."""

        lines = [
            style.paint("═" * _WIDTH, style.theme.rule, dim=True),
            self._header(style, _WIDTH),
            self._streaming_line(style),
            self._legend(style),
            "",
            *(f"  {line}" for line in self._flow_lines(style)),
            "",
        ]
        return "\n".join(lines)

    def _legend(self, style: Styler) -> str:
        """A dim one-line key for the flow-graph glyphs."""

        key = "flow ↓     ┌┴┐ fork     └┬┘ merge     ├─▶ outlet"
        return "  " + style.paint(key, style.theme.subtitle, dim=True)

    def _flow_lines(self, style: Styler) -> list[str]:
        """The drawn flow graph, or a plain numbered list if it is not drawable."""

        nodes = [
            FlowNode(
                number=view.number,
                name=view.display,
                inputs=view.inputs,
                output=view.output,
                axes=_axes_short(view.out_layout),
                kinds=_kinds_short(view.out_layout),
            )
            for view in self.steps
        ]
        inlets = {
            line: (_axes_short(layout), _kinds_short(layout))
            for line, layout in self.inlet_layouts.items()
        }
        flow = render_flow(
            style,
            nodes,
            labels=self.labels,
            outlets=self.outlets,
            inlet=(_axes_short(self.inlet_layout), _kinds_short(self.inlet_layout)),
            inlets=inlets,
        )
        return flow if flow is not None else self._flat_flow(style)

    def _flat_flow(self, style: Styler) -> list[str]:
        """Fallback numbered listing for a graph the flow cannot draw."""

        lines: list[str] = []
        for view in self.steps:
            marker = (
                style.paint(f"  » {self.outlets[view.output]}", style.theme.ok)
                if view.output in self.outlets
                else ""
            )
            name = style.paint(view.display, bold=True)

            # A merge (several inputs) names the lines it joins (branch names, or
            # inlet names for a multi-inlet source merge) so the sources are visible.
            sources = ""
            if len(view.inputs) > 1:
                joined = ", ".join(self.labels.get(line, line) for line in view.inputs)
                sources = style.paint(f"  ◀ {joined}", style.theme.subtitle, dim=True)
            lines.append(f"{view.number}. {name}{sources}{marker}")

            # Under the step name, show its output axes and value kind.
            detail = " ".join(
                part
                for part in (
                    _axes_short(view.out_layout),
                    _kinds_short(view.out_layout),
                )
                if part
            )
            if detail:
                lines.append(
                    "   " + style.paint(detail, style.theme.subtitle, dim=True)
                )

        return lines

    def _verbose(self, style: Styler) -> str:
        """
        A full contract card per step (summary, accepts/produces, streaming, params).
        """

        lines = [self._header(style, _WIDTH), self._streaming_line(style)]
        for view in self.steps:
            # The pipeline card shows each step's declared (static) contract; the
            # per-instance resolved contract is what describe(step) reports.
            card = Description(
                spec=view.spec,
                params=view.params,
                requires_axes=view.spec.requires_axes,
                admissible_values=view.spec.admissible_values,
                streaming=view.spec.streaming,
                layout_effect=view.spec.layout_effect,
            )
            lines.append("")
            lines.append(
                card.render(color=style.enabled, theme=style.theme).rstrip("\n")
            )

        lines.append("")
        return "\n".join(lines)

    def _header(self, style: Styler, width: int) -> str:
        """The `pipeline ... N steps` title line, the count right-aligned to `width`."""

        count = f"{len(self.steps)} step" + ("" if len(self.steps) == 1 else "s")
        pad = max(1, width - 2 - len("pipeline") - len(count))
        title = style.paint("pipeline", style.theme.title, bold=True)

        return (
            "  "
            + title
            + " " * pad
            + style.paint(count, style.theme.subtitle, dim=True)
        )

    def _streaming_line(self, style: Styler) -> str:
        """The `streaming  supported: ...` line for the whole pipeline."""

        text, status = _streaming_summary(self.steps)
        label = style.paint("streaming", style.theme.subtitle, dim=True)
        return f"  {label}  {style.paint(text, style.theme.status_color(status))}"


# Short axis labels for the compact flow-graph cards.
# fmt: off
_AXIS_SHORT: dict[AxisName, str] = {
    AxisName.TIME           : "T",
    AxisName.RECEIVER       : "RXR",
    AxisName.RX_ANTENNA     : "RX",
    AxisName.TX_ANTENNA     : "TX",
    AxisName.SPATIAL_STREAM : "SS",
    AxisName.SUBCARRIER     : "SC",
    AxisName.FREQUENCY      : "FQ",
    AxisName.DELAY          : "DL",
    AxisName.DOPPLER        : "DP",
    AxisName.BAND           : "BD",
    AxisName.WAVELET_BAND   : "WB",
    AxisName.LAG            : "LG",
    AxisName.COMPONENT      : "CP",
    AxisName.LATENT         : "LT",
    AxisName.FEATURE        : "FT",
}
# fmt: on


def _axes_short(layout: Layout | None) -> str:
    """Axes for a flow-graph card, e.g. `(T, RX[3], SC[52])`; empty if uncompiled.

    The time axis shows just its label (its length rides with the data); every
    other axis shows its size, or `?` when that is only known at run time.
    """

    if layout is None:
        return ""
    parts: list[str] = []
    for axis in layout.axes:
        short = _AXIS_SHORT.get(axis.name, axis.name.value[:2].upper())
        if axis.name == AxisName.TIME:
            parts.append(short)
        else:
            parts.append(f"{short}[{'?' if axis.size is None else axis.size}]")
    return "(" + ", ".join(parts) + ")"


def _kinds_short(layout: Layout | None) -> str:
    """A card's `representation: value kind` line, e.g. `CFR: magnitude`."""

    if layout is None:
        return ""
    return f"{layout.representation.short}: {layout.values.short}"


def _streaming_summary(views: Sequence[PipelineStepView]) -> tuple[str, Status]:
    """Whether the whole pipeline can stream, reporting its weakest step."""

    blocked = next(
        (v for v in views if v.spec.streaming is Streaming.UNAVAILABLE), None
    )
    if blocked is not None:
        return (
            f"not supported: step {blocked.number} '{blocked.spec.name}' "
            "needs the whole recording",
            Status.BAD,
        )
    # A config-dependent step could stream or not depending on how it was built,
    # so the whole-pipeline answer is only known for the configured instances.
    if any(isinstance(v.spec.streaming, Deferred) for v in views):
        return ("depends on configuration", Status.WARN)
    if any(v.spec.streaming is Streaming.BATCH_DIVERGENT for v in views):
        return ("supported: batch-divergent", Status.WARN)
    return ("supported: batch-equivalent", Status.OK)


# -----------------------------------------------------------------------------
# Parameters and serialization
# -----------------------------------------------------------------------------
def params_of(step: object) -> tuple[ParamInfo, ...]:
    """Read the display parameters (name, type, default, configured value) of a step."""

    step_class = step if isinstance(step, type) else type(step)
    instance = None if isinstance(step, type) else step
    fields = dataclasses.fields(cast("Any", step_class))
    return tuple(_param_info(field, instance) for field in fields)


def _param_info(field: dataclasses.Field[Any], instance: object | None) -> ParamInfo:
    """Build a ParamInfo from a dataclass field and an optional configured instance."""

    default: Any = None
    if field.default is not dataclasses.MISSING:
        default = field.default
    elif field.default_factory is not dataclasses.MISSING:
        default = field.default_factory()

    value = getattr(instance, field.name) if instance is not None else None
    return ParamInfo(
        name=field.name,
        type=str(field.type),
        default=default,
        doc=str(field.metadata.get("doc", "")),
        value=value,
        has_value=instance is not None,
    )


def _params_dict(params: Sequence[ParamInfo]) -> list[dict[str, Any]]:
    """Serialize parameters, recording the configured value that was actually used."""

    rows: list[dict[str, Any]] = []
    for param in params:
        used = param.value if param.has_value else param.default
        rows.append(
            {
                "name": param.name,
                "type": param.type,
                "value": _json_value(used),
                "default": _json_value(param.default),
                "doc": param.doc,
            }
        )
    return rows


def _layout_dict(layout: Layout | None) -> dict[str, Any] | None:
    """Serialize a layout (axes with sizes/coordinates, representation, values)."""

    if layout is None:
        return None
    return {
        "representation": layout.representation.value,
        "values": layout.values.value,
        "axes": [_axis_dict(axis) for axis in layout.axes],
    }


def _axis_dict(axis: Axis) -> dict[str, Any]:
    """Serialize one axis."""

    data: dict[str, Any] = {"name": axis.name.value, "size": axis.size}
    if axis.unit is not None:
        data["unit"] = axis.unit
    if axis.coordinates is not None:
        data["coordinates"] = [_json_value(coord) for coord in axis.coordinates]
    return data


def _json_value(value: object) -> object:
    """Make a value JSON-safe: enums become their value, tuples become lists."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _default_cell(param: ParamInfo) -> str:
    """`= default`, plus `→ value` when an instance overrides it."""

    cell = f"= {_format_value(param.default)}"
    if param.has_value and param.value != param.default:
        cell += f" → {_format_value(param.value)}"
    return cell


def _arrows(text: str) -> str:
    """Normalize ASCII arrows to a single glyph so output reads uniformly."""

    return text.replace("->", "→")


def _format_value(value: object) -> str:
    """Render a default/configured value compactly (enums as `Type.NAME`)."""

    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    return repr(value)


def _value_names(values: tuple[ValueKind, ...] | None) -> list[str]:
    """Return the value-kind names, or an empty list for 'any'."""

    return [] if values is None else [value.value for value in values]


def _repr_names(reprs: tuple[Representation, ...] | None) -> list[str]:
    """Return the representation names, or an empty list for 'any'."""

    return [] if reprs is None else [representation.value for representation in reprs]
