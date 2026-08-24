"""Building and compiling a pipeline.

You build a Pipeline as an immutable recipe of named streams ("lines"): `.then`
extends the current line, `.branch` splits it into named branches, `.merge`
joins branches back into one, and `.probe` keeps an intermediate line as an
output. Compiling against an AcquisitionProfile lowers the recipe to a flat list
of nodes (each reading some named lines and writing one), validates every
layout before any data flows, and hands back a Siphon you can pour (whole
recording) or stream (chunk by chunk).

The line names and the "current line" pointer live only here, in the builder.
The compiled Siphon just holds nodes; running it is a plain loop over a dict of
intermediate signals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from csiphon.core.axes import AxisName
from csiphon.core.errors import CompileError, LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.signal import Signal
from csiphon.pipeline._lines import INLET, inlet_id
from csiphon.pipeline.merges import Alignment, Junction, MergeStrategy
from csiphon.pipeline.runners import Outlets, Stream, _inlet_signals
from csiphon.pipeline.runners import pour as _run_pour
from csiphon.pipeline.step import Step
from csiphon.spec import (
    PipelineDescription,
    PipelineStepView,
    params_of,
)


def _op_display(op: Step | Junction) -> str:
    """The operation name shown in the flow graph (a merge shows its strategy)."""

    return op.display_name if isinstance(op, Junction) else op.spec.name


@dataclass(frozen=True, slots=True)
class _BuildNode:
    """One recipe operation with its wiring (line ids), before compilation."""

    inputs: tuple[str, ...]
    output: str
    op: Step | Junction


@dataclass(frozen=True, slots=True)
class Node:
    """
    One compiled operation: which lines it reads, the line it writes, and its layout.
    """

    # fmt: off
    inputs      : tuple[str, ...]        # line ids read (1 per step, N per merge)
    output      : str                   # line id this node writes
    op          : Step | Junction       # a 1->1 step or an N->1 junction
    out_layout  : Layout                # validated at compile time
    out_profile : AcquisitionProfile    # the capture profile in force at this node
    # fmt: on


def _step_view(
    number: int,
    node: _BuildNode | Node,
    in_layout: Layout | None,
    out_layout: Layout | None,
) -> PipelineStepView:
    """One description row for `node`; layouts are None for an un-compiled recipe."""

    return PipelineStepView(
        number=number,
        spec=node.op.spec,
        params=params_of(node.op),
        inputs=node.inputs,
        output=node.output,
        display=_op_display(node.op),
        in_layout=in_layout,
        out_layout=out_layout,
    )


@dataclass(frozen=True, slots=True)
class Pipeline:  # pylint: disable=too-many-instance-attributes
    """
    An immutable recipe of named lines, built with `.then`/`.branch`/`.merge`/`.probe`.

    The fields are internal bookkeeping; you never construct or read them directly.
    `_head` is the current line (what a bare `.then` acts on); it is `None` after a
    branch until you merge or select, so a `.then` there fails with a clear message.
    """

    # fmt: off
    _nodes      : tuple[_BuildNode, ...]       = ()
    _head       : str | None                   = INLET
    _head_label : str | None                   = None
    _live       : tuple[str, ...]              = (INLET,)
    _names      : dict[str, str]               = field(default_factory=dict)
    _outlets    : tuple[tuple[str, str], ...]  = ()
    _next_id    : int                          = 0
    _inlets     : tuple[str, ...]              = ()   # names; () = one unnamed inlet
    # fmt: on

    @classmethod
    def from_inlets(cls, *names: str) -> Pipeline:
        """Start a pipeline with several named inlets, to be merged before processing.

        This is a pipeline whose head is a set of independent source lines (the
        "dangling leaves" of a branch that has no root). You merge them (with a
        strategy and alignment, e.g. `.merge(using=Concatenate(axis=AxisName.RECEIVER),
        align=Hold())` to join N single-receiver captures onto the receiver axis)
        and then process the single line as usual.
        """

        if len(names) < 2:
            raise LayoutError("from_inlets needs at least two inlet names.")
        if len(set(names)) != len(names):
            raise LayoutError("Inlet names must be distinct.")
        ids = tuple(inlet_id(name) for name in names)
        return cls(
            _head=None,  # several live lines: merge or select one before .then
            _live=ids,
            _names=dict(zip(names, ids, strict=True)),
            _inlets=names,
        )

    def then(self, step: Step, name: str | None = None) -> Pipeline:
        """Extend the current line with one step (optionally renaming the line)."""

        if self._head is None:
            raise LayoutError(
                "Several lines are live after a branch; merge them or name a target "
                "before calling .then."
            )

        # A Pipeline is immutable: every builder method returns a new copy with the
        # step appended. Each line carries a string id ("_0", "_1", ...); this step
        # reads the current line (_head) and writes a fresh output line, `out`.
        out = f"_{self._next_id}"

        # The output line inherits the current line's label unless the caller renamed
        # it here, so a named line keeps its name as it grows.
        label = name if name is not None else self._head_label
        new_names = dict(self._names)
        if label is not None:
            new_names[label] = out

        # The step consumes _head and produces `out`, so replace _head with `out` in
        # the set of live lines; any other live lines (open branches) are untouched.
        new_live = tuple(out if line == self._head else line for line in self._live)

        return replace(
            self,
            _nodes=(*self._nodes, _BuildNode((self._head,), out, step)),
            _head=out,
            _head_label=label,
            _live=new_live,
            _names=new_names,
            _next_id=self._next_id + 1,
        )

    def probe(self, name: str) -> Pipeline:
        """Keep the current line as an output named `name`, and keep flowing."""

        if self._head is None:
            raise LayoutError(
                "A probe needs a single current line; merge or select one first."
            )
        if any(display == name for display, _ in self._outlets):
            raise LayoutError(f"Outlet '{name}' already exists.")
        return replace(self, _outlets=(*self._outlets, (name, self._head)))

    def branch(self, **branches: Pipeline) -> Pipeline:
        """Split the current line into named branches (each its own sub-pipeline)."""

        # Grafting reads each sub-pipeline's build state (sibling Pipeline instances).
        # pylint: disable=protected-access
        if self._head is None:
            raise LayoutError("Branch from a single line; merge or select one first.")
        if len(branches) < 2:
            raise LayoutError("branch needs at least two named branches.")

        new_nodes = list(self._nodes)
        new_names = dict(self._names)
        new_outlets = list(self._outlets)
        new_live = [line for line in self._live if line != self._head]
        counter = self._next_id

        for label, sub in branches.items():
            if sub._head is None:
                raise LayoutError(
                    f"Branch '{label}' must end on a single line "
                    "(it has an open branch)."
                )

            # Each branch is its own small Pipeline, built against its own INLET. To
            # graft it onto the parent we replay its nodes here, translating every
            # line id from the sub-pipeline's namespace into a fresh parent id.
            # `idmap` holds that sub-id -> parent-id translation; seeding it with the
            # sub's INLET -> parent head makes the branch start where the split was.
            idmap = {INLET: self._head}
            for node in sub._nodes:
                # Mint a fresh parent id for this node's output and record the
                # translation, so later nodes reading this output find the new id.
                out = f"_{counter}"
                counter += 1
                idmap[node.output] = out
                # Re-point the node's inputs through `idmap` so they name parent ids,
                # then append the translated node to the parent's node list.
                new_nodes.append(
                    _BuildNode(tuple(idmap[i] for i in node.inputs), out, node.op)
                )

            # The branch's final line (its head, now a parent id) becomes a live line
            # the caller can later merge or select by this branch's name.
            new_names[label] = idmap[sub._head]
            new_live.append(idmap[sub._head])

            # Carry any probes that were placed inside the branch out to the parent.
            for display, line in sub._outlets:
                new_outlets.append((f"{label}.{display}", idmap[line]))

        return replace(
            self,
            _nodes=tuple(new_nodes),
            _head=None,
            _head_label=None,
            _live=tuple(new_live),
            _names=new_names,
            _outlets=tuple(new_outlets),
            _next_id=counter,
        )

    def merge(
        self,
        inputs: Sequence[str] | None = None,
        *,
        using: MergeStrategy,
        align: Alignment | None = None,
        name: str | None = None,
    ) -> Pipeline:
        """Join branches into one line (defaults to all live branches).

        `using` is how the branches combine (Stack, Mean, Fuse, ...); `align` is
        how they line up in time first (Exact by default, Hold to bridge branches
        on different time grids).
        """

        ids = (
            list(self._live)
            if inputs is None
            else [self._resolve(ref) for ref in inputs]
        )
        if len(ids) < 2:
            raise LayoutError("merge needs at least two branches.")

        # Build the merge node. Junction's `align` field defaults to Exact; passing
        # align=None explicitly would overwrite that default with None, so only pass
        # `align` through when the caller actually gave one.
        junction = Junction(using) if align is None else Junction(using, align)
        out = f"_{self._next_id}"

        # The merge replaces its branch lines with the single new line `out`.
        consumed = set(ids)
        new_live = (*(line for line in self._live if line not in consumed), out)

        new_names = dict(self._names)
        if name is not None:
            new_names[name] = out

        return replace(
            self,
            _nodes=(*self._nodes, _BuildNode(tuple(ids), out, junction)),
            _head=out,
            _head_label=name,
            _live=new_live,
            _names=new_names,
            _next_id=self._next_id + 1,
        )

    def _resolve(self, ref: str) -> str:
        """Turn a user-facing branch name into its current line id."""

        if ref in self._names:
            return self._names[ref]
        if ref in self._live:
            return ref
        raise LayoutError(f"Unknown branch '{ref}' to merge.")

    def compile(
        self,
        profile: AcquisitionProfile | None = None,
        inlet: Layout | None = None,
        **profiles: AcquisitionProfile,
    ) -> Siphon:
        """Validate every layout in declaration order and return a runnable Siphon.

        Single inlet: `compile(profile)`, or `compile(profile, inlet=layout)` to
        feed something other than raw CSI. Multiple inlets (from `from_inlets`)
        come two ways:
          * one profile describing the whole N-receiver capture: `compile(profile)`,
            where `profile.n_receivers` must equal the number of inlets; or
          * one profile per inlet (heterogeneous devices):
            `compile(rx0=profile0, rx1=profile1, …)`, each with `n_receivers == 1`.
        """

        # Start the layout/profile maps off with the inlets.
        inlet_layouts, inlet_profiles = self._inlet_setup(profile, inlet, profiles)
        layouts: dict[str, Layout] = dict(inlet_layouts)
        node_profiles: dict[str, AcquisitionProfile] = dict(inlet_profiles)

        # Compile the nodes in order; each node's output layout and profile feed the
        # later nodes that read its line.
        compiled: list[Node] = []
        for number, node in enumerate(self._nodes, start=1):
            out_layout, out_profile = _compile_node(
                number, node, layouts, node_profiles
            )
            layouts[node.output] = out_layout
            node_profiles[node.output] = out_profile
            compiled.append(
                Node(node.inputs, node.output, node.op, out_layout, out_profile)
            )

        # Invert the builder's label map so the Siphon can name lines (branch names,
        # merge inputs) when it describes itself.
        labels = {line: name for name, line in self._names.items()}
        primary = (
            profile if profile is not None else next(iter(inlet_profiles.values()))
        )
        return Siphon(
            primary,
            inlet_layouts,
            tuple(compiled),
            self._resolve_outlets(),
            layouts,
            labels,
        )

    def _inlet_setup(
        self,
        profile: AcquisitionProfile | None,
        inlet: Layout | None,
        profiles: dict[str, AcquisitionProfile],
    ) -> tuple[dict[str, Layout], dict[str, AcquisitionProfile]]:
        """Resolve each inlet's layout and profile for the three compile modes."""

        # Mode 2: one profile per inlet (heterogeneous devices).
        if profiles:
            if profile is not None or inlet is not None:
                raise LayoutError("Pass one profile, or one per inlet, not both.")
            if set(profiles) != set(self._inlets):
                raise LayoutError(
                    f"Per-inlet profiles must name every inlet: {list(self._inlets)}."
                )
            for name in self._inlets:
                if profiles[name].n_receivers != 1:
                    raise LayoutError(
                        f"Per-device profile '{name}' must have n_receivers=1."
                    )
            return (
                {inlet_id(n): profiles[n].raw_csi_layout() for n in self._inlets},
                {inlet_id(n): profiles[n] for n in self._inlets},
            )

        if profile is None:
            raise LayoutError("compile needs a profile (or one profile per inlet).")

        # Mode 1: one profile for an N-receiver capture; each inlet is one receiver.
        if self._inlets:
            if inlet is not None:
                raise LayoutError(
                    "An inlet layout override needs a single-inlet pipeline."
                )
            if profile.n_receivers != len(self._inlets):
                raise LayoutError(
                    f"profile.n_receivers ({profile.n_receivers}) must equal the "
                    f"number of inlets ({len(self._inlets)})."
                )
            one = replace(profile, n_receivers=1)
            per_receiver = one.raw_csi_layout()
            return (
                {inlet_id(name): per_receiver for name in self._inlets},
                {inlet_id(name): one for name in self._inlets},
            )

        # Single unnamed inlet (the original path).
        inlet_layout = profile.raw_csi_layout() if inlet is None else inlet
        return {INLET: inlet_layout}, {INLET: profile}

    def _resolve_outlets(self) -> dict[str, str]:
        """Map each output's display name to its line id (probes + terminal lines)."""

        # A finished pipeline exposes one outlet per output. Invert the label map so
        # a bare line id can recover the user-facing name it was given.
        id_to_label = {line: label for label, line in self._names.items()}

        # Probes are explicit outlets the user named mid-pipeline; take those first.
        outlets: dict[str, str] = {}
        for display, line in self._outlets:
            outlets[display] = line

        # Every line still live at the end is also an outlet. Name it by its label,
        # or "out" for the main line, or fall back to the raw id. Two outlets must
        # not share a name, so a collision is a build error.
        for line in self._live:
            display = id_to_label.get(line) or ("out" if line == self._head else line)
            if display in outlets:
                raise LayoutError(f"Two outlets share the name '{display}'.")
            outlets[display] = line
        return outlets

    def to_description(self) -> PipelineDescription:
        """A readable summary of the (un-compiled) recipe (see `csiphon.describe`).

        Nothing has been compiled yet, so there are no concrete layouts: the inlet
        and outlet layouts are None and each step carries None for its in/out layout.
        Each step therefore shows its *declared* effect rather than a concrete
        transition; `labels` and `outlets` still name the lines so forks and kept
        outputs are visible.
        """

        steps = tuple(
            _step_view(n, node, None, None) for n, node in enumerate(self._nodes, 1)
        )
        labels = {line: name for name, line in self._names.items()}
        outlets = {line: name for name, line in self._outlets}
        return PipelineDescription(None, None, steps, labels=labels, outlets=outlets)


def _compile_node(
    number: int,
    node: _BuildNode,
    layouts: dict[str, Layout],
    profiles: dict[str, AcquisitionProfile],
) -> tuple[Layout, AcquisitionProfile]:
    """Compile one node into its output layout and the profile in force there.

    A step keeps its single input's profile (the capture rate and subcarriers do
    not change as data flows). A merge reconciles its branches: the rate comes
    from the alignment's reference branch (whose timeline the output lands on),
    and the receiver/antenna counts are read straight off the merged layout, so
    they can never drift from the real shape.
    """

    # A layout error here becomes a CompileError that names the step, so it points at
    # the recipe line that failed.
    try:
        # A junction reconciles all its branch layouts and profiles into one.
        if isinstance(node.op, Junction):
            in_layouts = [layouts[i] for i in node.inputs]
            in_profiles = [profiles[i] for i in node.inputs]
            out_layout = node.op.output_layout(in_layouts)
            return out_layout, _merged_profile(in_profiles, node.op, out_layout)

        # A plain step takes its single input line and keeps that line's profile.
        in_profile = profiles[node.inputs[0]]
        return node.op.output_layout(layouts[node.inputs[0]], in_profile), in_profile
    except LayoutError as error:
        raise CompileError(
            f"Pipeline compilation failed at step {number}, '{node.op.name}': {error}"
        ) from error


def _merged_profile(
    inputs: list[AcquisitionProfile], junction: Junction, out_layout: Layout
) -> AcquisitionProfile:
    """Profile after a merge: rate from the alignment reference, shape from layout."""

    # The alignment decides whose timeline the merged stream lands on. Hold picks a
    # reference branch; Exact sits on the shared grid and keeps the default 0, so the
    # first branch's rate is representative.
    return _profile_from_layout(inputs[junction.align.reference], out_layout)


def _profile_from_layout(
    base: AcquisitionProfile, layout: Layout
) -> AcquisitionProfile:
    """`base`'s rate/subcarriers, but device counts read off `layout`'s axes."""

    def axis_size(name: AxisName) -> int:
        return layout.axis(name).size or 1 if layout.has_axis(name) else 1

    return replace(
        base,
        n_receivers=axis_size(AxisName.RECEIVER),
        n_tx_antennas=axis_size(AxisName.TX_ANTENNA),
        n_rx_antennas=axis_size(AxisName.RX_ANTENNA),
    )


@dataclass(frozen=True, slots=True)
class Siphon:
    """A validated pipeline, ready to pour (batch) or stream (chunk by chunk)."""

    # fmt: off
    profile       : AcquisitionProfile
    inlet_layouts : dict[str, Layout]  # inlet line id -> layout (one, or several)
    nodes         : tuple[Node, ...]   # the compiled operations, in execution order
    outlets       : dict[str, str]     # display name -> line id
    layouts       : dict[str, Layout]  # line id -> layout (for inspection)
    labels        : dict[str, str]     # line id -> branch / line name (for describe)
    # fmt: on

    @property
    def inlet_layout(self) -> Layout:
        """The sole inlet's layout; errors if the pipeline has several inlets."""

        if len(self.inlet_layouts) != 1:
            raise LayoutError(
                "This pipeline has several inlets; use inlet_layouts (a mapping)."
            )
        return next(iter(self.inlet_layouts.values()))

    def pour(
        self, signals: Signal | Mapping[str, Signal] | None = None, **named: Signal
    ) -> Outlets:
        """Run a whole recording through and return every outlet (best quality).

        Pass one Signal for a single-inlet pipeline, or one signal per inlet for a
        multi-inlet one, as keywords `pour(rx0=s0, rx1=s1)` or a mapping
        `pour({"rx0": s0, "rx1": s1})`.
        """

        return _run_pour(self, _inlet_signals(signals, named))

    def stream(self) -> Stream:
        """Open a stateful streaming session; refuses if any step cannot stream."""

        return Stream(self)

    @property
    def outlet_layouts(self) -> dict[str, Layout]:
        """The layout produced at each outlet."""

        return {display: self.layouts[line] for display, line in self.outlets.items()}

    @property
    def outlet_layout(self) -> Layout:
        """The layout of the single outlet (for a one-outlet siphon)."""

        if len(self.outlets) != 1:
            raise LayoutError(
                "outlet_layout is only defined for a single-outlet siphon."
            )
        (line,) = self.outlets.values()
        return self.layouts[line]

    def to_description(self) -> PipelineDescription:
        """A readable summary of every compiled node and its layout.

        Every node carries its real input and output layout. Two inlet fields are
        passed: `inlet_layouts` (one layout per source line) is what the flow graph
        draws when there are several sources; `inlet_layout` (singular) is a single
        representative shown on the compact card and in the JSON summary. For a
        multi-inlet pipeline the sources differ, so the representative is just the
        first. The full per-source detail lives in `inlet_layouts`.
        """

        steps = tuple(
            _step_view(n, node, self.layouts[node.inputs[0]], node.out_layout)
            for n, node in enumerate(self.nodes, 1)
        )
        outlet = self.outlet_layout if len(self.outlets) == 1 else None
        outlets = {line: name for name, line in self.outlets.items()}

        representative_inlet = next(iter(self.inlet_layouts.values()))
        return PipelineDescription(
            representative_inlet,
            outlet,
            steps,
            labels=self.labels,
            outlets=outlets,
            inlet_layouts=self.inlet_layouts,
        )

    def describe(self, verbose: bool = False) -> str:
        """Return a readable summary of every node (equivalent to `describe(this)`)."""

        return self.to_description().render(verbose=verbose)

    def save_metadata(self, path: str | Path) -> None:
        """Write the siphon's full metadata to a JSON file (nodes, params, layouts)."""

        self.to_description().save(path)
