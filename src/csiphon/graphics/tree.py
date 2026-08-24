"""Draw a pipeline as a real 2-D data-flow graph of cards.

A pipeline is a graph of steps: they run one after another, fan out into named
branches at a fork, and come back together at a merge. This module rebuilds that
branch structure from the flat node wiring and draws it, with branches drawn
side by side so parallel work is actually parallel on the page.

Each node is a small card of just its operation and layout:

    windowed-variance       <- the operation (a merge shows its strategy)
    (T, RX[3], SC[52])      <- axes, using short names
    CFR: magnitude          <- representation: value kind

Names live off the cards, where you actually read them. A branch's name sits on
the merge bar that joins it:

        variance               slope
           └─────────┬──────────┘

An outlet taps off to the side with a spout that points out of the pipeline,
`├─▶ name` while the line keeps flowing, `└─▶ name` at the very end.

The drawing is built bottom-up from `_Block`s (a rectangle of text with a top
and a bottom connection point). Steps that run in sequence are stacked and joined
with a `│`. Branches that run in parallel are placed next to each other, split
apart with a fork bar and rejoined with a merge bar; a shorter branch has its
rail padded downward so every branch reaches the merge. Nesting is automatic: a
branch that itself forks is just another block.

Branches can rejoin in stages (merge a and b, then merge that result with c). So
before drawing a fork we first work out where all its branches eventually come
back together (`_rejoin_points`); the merge there tells us which lines are the
fork's real branches, even when some of them merged among themselves first.

Card text is coloured as it is built, so widths are measured on the visible text
only, ignoring the ANSI colour codes (see `_vlen`).

Drawing is defensive: if the wiring is not a shape we can lay out, `render_flow`
returns None and the caller falls back to a plain list, so a description never
breaks.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from csiphon.graphics.style import Styler

# Horizontal gap between neighbouring branches.
_BRANCH_GAP = 3
# Stand-in nodes for the very start and end of the pipeline. They give every real
# node a single common start and end, which `_rejoin_points` needs to work.
_START = "\x00inlet"
_END = "\x00sink"
# Matches ANSI colour codes so widths can be measured on the visible text only.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True, slots=True)
class FlowNode:
    """One node to draw: its number, operation name, wiring, and detail strings.

    `inputs` holds one line id for a normal step and several for a merge (the
    only many-to-one operation), so `len(inputs) > 1` identifies a merge. `axes`
    and `kinds` are the card's dim detail lines (empty for an un-compiled recipe).
    """

    number: int
    name: str
    inputs: tuple[str, ...]
    output: str
    axes: str
    kinds: str


@dataclass
class _Block:
    """A rectangular text grid with a top input column and a bottom output column."""

    grid: list[str]
    in_col: int
    out_col: int


def render_flow(  # pylint: disable=too-many-arguments
    style: Styler,
    nodes: Sequence[FlowNode],
    *,
    labels: Mapping[str, str],
    outlets: Mapping[str, str],
    inlet: tuple[str, str] = ("", ""),
    inlets: Mapping[str, tuple[str, str]] | None = None,
) -> list[str] | None:
    """Draw the flow graph as styled lines, or None if it is not a drawable graph.

    `labels` names lines (branch names) and `outlets` names lines kept as
    outputs. `inlet` is the single inlet card's (axes, kinds) detail; `inlets`
    gives the same per inlet line id when a pipeline has several sources.
    """

    graph = _Graph(style, nodes, labels, outlets, inlet, inlets)
    block = graph.build()
    if block is None:
        return None
    return [_vrstrip(row) for row in block.grid]


# --------------------------------------------------------------------------- #
# Visible-width helpers: measure and pad ignoring the ANSI colour codes.
# --------------------------------------------------------------------------- #
def _vlen(text: str) -> int:
    """The visible width of a (possibly coloured) string."""

    return len(_ANSI.sub("", text))


def _vljust(text: str, width: int) -> str:
    """Right-pad a coloured string to a visible width with plain spaces."""

    return text + " " * max(0, width - _vlen(text))


def _vrstrip(text: str) -> str:
    """Drop trailing spaces, even ones sitting inside a trailing colour code.

    Matches the trailing run of spaces and colour codes, then removes only the
    spaces from it, so the closing reset code is kept.
    """

    return re.sub(
        r"(?:\x1b\[[0-9;]*m|\s)+$",
        lambda match: re.sub(r"\s+", "", match.group()),
        text,
    )


def _vcenter(text: str, width: int) -> str:
    """Centre a coloured string within a visible width with plain spaces."""

    pad = max(0, width - _vlen(text))
    left = pad // 2
    return " " * left + text + " " * (pad - left)


# --------------------------------------------------------------------------- #
# Drawing pieces: a block is a rectangle of text with a top and bottom
# connection point (the columns where a `│` enters and leaves it).
# --------------------------------------------------------------------------- #
def _width(block: _Block) -> int:
    """The block's width (its widest row)."""

    return max((_vlen(row) for row in block.grid), default=0)


def _pad(rows: list[str], width: int) -> list[str]:
    """Right-pad every row to `width` so the grid is a clean rectangle."""

    return [_vljust(row, width) for row in rows]


def _shift(block: _Block, amount: int) -> _Block:
    """Move a block `amount` columns to the right (ports move with it)."""

    if amount <= 0:
        return block
    pad = " " * amount
    return _Block(
        [pad + row for row in block.grid],
        block.in_col + amount,
        block.out_col + amount,
    )


def _rail(style: Styler, column: int, width: int) -> str:
    """A single dim `│` rail at `column`, padded to `width`."""

    text = " " * column + "│" + " " * (width - column - 1)
    return style.paint(text, style.theme.rule, dim=True)


def _stack(style: Styler, blocks: list[_Block]) -> _Block:
    """Stack blocks vertically, aligning and joining each pair of ports with a `│`."""

    kept = [block for block in blocks if block.grid]
    current = kept[0]
    for block in kept[1:]:
        # Slide whichever block is needed so the upper output port and the lower
        # input port share a column, then drop a `│` between them.
        gap = current.out_col - block.in_col
        upper = current if gap >= 0 else _shift(current, -gap)
        lower = _shift(block, gap) if gap >= 0 else block
        width = max(_width(upper), _width(lower))
        current = _Block(
            [
                *_pad(upper.grid, width),
                _rail(style, upper.out_col, width),
                *_pad(lower.grid, width),
            ],
            upper.in_col,
            lower.out_col,
        )
    return current


def _rail_to(style: Styler, block: _Block, height: int) -> _Block:
    """Extend a branch's output rail downward so it reaches a shared merge row."""

    width = _width(block)
    rows = _pad(block.grid, width)
    while len(rows) < height:
        rows.append(_rail(style, block.out_col, width))
    return _Block(rows, block.in_col, block.out_col)


def _tee(existing: str, into_up: bool) -> str:
    """Merge a port stub into a corner already on the connector row."""

    up = {"┬": "┼", "┌": "├", "┐": "┤", "─": "┴", " ": "┴"}
    down = {"┴": "┼", "└": "├", "┘": "┤", "─": "┬", " ": "┬"}
    return (up if into_up else down).get(existing, existing)


def _connector(
    style: Styler, width: int, columns: list[int], port: int, *, is_fork: bool
) -> str:
    """A dim fork/merge row: a bar across the branch columns with a stub at the port.

    A fork bar (branches below) uses `┌┬┐` at the branch columns and `┴` up to the
    port; a merge bar (branches above) uses `└┴┘` and `┬` down to the port.
    """

    chars = [" "] * width
    low, high = min(columns), max(columns)
    for column in range(low, high + 1):
        chars[column] = "─"
    for column in columns:
        if is_fork:
            chars[column] = "┌" if column == low else "┐" if column == high else "┬"
        else:
            chars[column] = "└" if column == low else "┘" if column == high else "┴"
    chars[port] = _tee(chars[port], is_fork)
    return style.paint("".join(chars), style.theme.rule, dim=True)


def _side_by_side(style: Styler, blocks: list[_Block]) -> tuple[list[str], list[int]]:
    """Lay branch blocks out in a row; return the rows and each block's x-offset."""

    height = max(len(block.grid) for block in blocks)
    blocks = [_rail_to(style, block, height) for block in blocks]
    widths = [_width(block) for block in blocks]
    offsets, cursor = [], 0
    for width in widths:
        offsets.append(cursor)
        cursor += width + _BRANCH_GAP
    total = cursor - _BRANCH_GAP
    rows = []
    for index in range(height):
        row = (" " * _BRANCH_GAP).join(
            _vljust(block.grid[index], widths[position])
            for position, block in enumerate(blocks)
        )
        rows.append(_vljust(row, total))
    return rows, offsets


def _label_row(style: Styler, columns: list[int], names: Sequence[str]) -> str:
    """A row naming each branch, centred over its column, packed left to right."""

    row, cursor = "", 0
    for column, name in sorted(zip(columns, names, strict=True)):
        if not name:
            continue
        # Centre over the column, but never touch the previous name: keep at least
        # one space between them when they would otherwise collide.
        gap = 1 if cursor else 0
        start = max(cursor + gap, column - len(name) // 2)
        row += " " * (start - cursor)
        row += style.paint(name, style.theme.alias, italic=True)
        cursor = start + len(name)
    return row


def _parallel(
    style: Styler,
    branches: list[_Block],
    merge: _Block | None,
    names: Sequence[str] = (),
    *,
    fork: bool = True,
) -> _Block:
    """Place branches side by side, split into them, and (if any) join at a merge.

    `names` labels each branch above the merge bar (where you read which incoming
    column is which), in the same order as `branches`. `fork=False` omits the top
    split bar, for branches that are the pipeline's start (its inlets) rather than
    a fork of one line.
    """

    rows, offsets = _side_by_side(style, branches)
    total = max(_vlen(row) for row in rows)
    in_cols = [offsets[i] + branches[i].in_col for i in range(len(branches))]
    out_cols = [offsets[i] + branches[i].out_col for i in range(len(branches))]
    port_in = (min(in_cols) + max(in_cols)) // 2
    port_out = (min(out_cols) + max(out_cols)) // 2

    # The fork row splits the single incoming rail into the branches. The `│` that
    # feeds it comes from the stack above, so it is not drawn here. Inlets have no
    # line above them, so they skip the fork row entirely.
    grid = list(rows)
    if fork:
        grid = [_connector(style, total, in_cols, port_in, is_fork=True), *rows]
    if merge is None:
        return _Block(grid, port_in, port_in)

    # Name the branches just above the bar that joins them, then the merge card.
    if any(names):
        grid.append(_label_row(style, out_cols, names))
    grid.append(_connector(style, total, out_cols, port_out, is_fork=False))
    grid.append(_rail(style, port_out, total))
    merge = _shift(merge, max(0, port_out - merge.in_col))
    width = max(max(_vlen(row) for row in grid), _width(merge))
    return _Block(_pad(grid, width) + _pad(merge.grid, width), port_in, merge.out_col)


# --------------------------------------------------------------------------- #
# Rebuild the branch structure from the flat node list, then draw it.
# --------------------------------------------------------------------------- #
class _Graph:
    # pylint: disable=too-many-instance-attributes  # a graph model with several maps
    # pylint: disable=too-few-public-methods  # one entry point (build); the rest is layout
    """Rebuilds a pipeline's fork/branch/merge structure and draws it."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        style: Styler,
        nodes: Sequence[FlowNode],
        labels: Mapping[str, str],
        outlets: Mapping[str, str],
        inlet: tuple[str, str],
        inlets: Mapping[str, tuple[str, str]] | None = None,
    ) -> None:
        """Index the nodes by the lines they read and write, ready to walk."""

        self._style = style
        self._labels = labels
        self._outlets = outlets
        self._inlet_axes, self._inlet_kinds = inlet
        # Per-inlet (axes, kinds) detail, keyed by inlet line id (multi-inlet only).
        self._inlets = dict(inlets or {})
        self._nodes = {node.output: node for node in nodes}

        produced = set(self._nodes)
        roots = {line for n in nodes for line in n.inputs if line not in produced}
        self._root = next(iter(roots)) if len(roots) == 1 else None

        # children[line]: nodes reading `line` as their first input (fork branches).
        # consumers[line]: every node reading `line` (a merge reads several lines).
        self._children: dict[str, list[FlowNode]] = {}
        self._consumers: dict[str, list[str]] = {}
        for node in nodes:
            self._children.setdefault(node.inputs[0], []).append(node)
            for line in node.inputs:
                self._consumers.setdefault(line, []).append(node.output)

        self._producer = {node.output: node.output for node in nodes}
        if self._root is not None:
            self._producer[self._root] = _START
        self._terminal = {
            node.output for node in nodes if node.output not in self._consumers
        }
        self._ids = [_START, _END, *(node.output for node in nodes)]
        # Several inlets (roots) that all converge into one merge is the drawable
        # multi-inlet shape; `_source_merge` is that merge's line (else None).
        self._source_merge = self._find_source_merge(roots)
        self._start_followers = self._start_from()
        self._rejoin: dict[str, str] = {}
        self._seen: set[int] = set()

    def _find_source_merge(self, roots: set[str]) -> str | None:
        """The single merge that all inlets feed, if that is the pipeline's shape."""

        if self._root is not None or len(roots) < 2:
            return None
        targets = set()
        for line in roots:
            consumers = self._consumers.get(line, [])
            if len(consumers) != 1:  # an inlet feeding more than the one merge
                return None
            targets.add(consumers[0])
        if len(targets) != 1:  # inlets fanning into different nodes
            return None
        merge = self._nodes[next(iter(targets))]
        return merge.output if set(merge.inputs) == roots else None

    def _start_from(self) -> list[str]:
        """The node(s) the start feeds: one inlet's consumers, or the source merge."""

        if self._root is not None:
            followers = list(self._consumers.get(self._root, []))
            if self._root in self._terminal:
                followers.append(_END)
            return followers
        return [self._source_merge] if self._source_merge is not None else []

    def build(self) -> _Block | None:
        """Lay out the whole pipeline, or None if it is not a graph we can draw."""

        dim = self._style.theme.subtitle
        inlet = self._card_lines(
            [
                self._inlet_glyph(),
                self._style.paint(self._inlet_axes, dim, dim=True),
                self._style.paint(self._inlet_kinds, dim, dim=True),
            ]
        )

        # An empty pipeline still draws its inlet.
        if not self._nodes:
            return inlet

        if self._root is not None:
            self._rejoin = self._rejoin_points()
            body, _ = self._series_from(self._root, None)
            block = _stack(self._style, [inlet, body]) if body is not None else inlet
        elif self._source_merge is not None:
            block = self._build_multi_inlet()
        else:
            return None  # not a shape we can lay out

        # If any node was left undrawn, the wiring was not a shape we can lay out.
        if len(self._seen) != len(self._nodes):
            return None
        return block

    def _build_multi_inlet(self) -> _Block:
        """Draw several inlet sources side by side, merging into the first node."""

        assert self._source_merge is not None
        merge = self._nodes[self._source_merge]
        self._seen.add(merge.number)

        # A source card per inlet (in the merge's input order), then the merge bar,
        # then the merge card. `fork=False`: the inlets are the pipeline's start, so
        # there is no split bar above them. Their names sit on the source cards, so
        # the merge bar needs no labels either.
        sources = [self._inlet_card(line) for line in merge.inputs]
        head = _parallel(self._style, sources, self._card(merge), fork=False)

        # Everything below the merge is an ordinary single line.
        self._rejoin = self._rejoin_points()
        rest, _ = self._series_from(merge.output, None)
        return _stack(self._style, [head, rest]) if rest is not None else head

    def _inlet_card(self, line: str) -> _Block:
        """A source card for one inlet: its glyph and name, plus its layout detail."""

        dim = self._style.theme.subtitle
        name = self._labels.get(line, "inlet")
        axes, kinds = self._inlets.get(line, ("", ""))
        return self._card_lines(
            [
                self._style.paint("○", self._style.theme.title)
                + " "
                + self._style.paint(name, bold=True),
                self._style.paint(axes, dim, dim=True),
                self._style.paint(kinds, dim, dim=True),
            ]
        )

    def _inlet_glyph(self) -> str:
        """The inlet marker line: a magenta circle and a bold `inlet`."""

        circle = self._style.paint("○", self._style.theme.title)
        return circle + " " + self._style.paint("inlet", bold=True)

    # -- graph queries ----------------------------------------------------- #
    def _out_line(self, node_id: str) -> str | None:
        """The line a node writes: the first line for the start, nothing for the end."""

        if node_id == _START:
            return self._root
        if node_id == _END:
            return None
        return node_id

    def _followers(self, node_id: str) -> list[str]:
        """The nodes that read this node's output; a final output feeds the end."""

        if node_id == _END:
            return []
        if node_id == _START:
            return list(self._start_followers)
        line = self._out_line(node_id)
        assert line is not None
        followers = list(self._consumers.get(line, []))
        if line in self._terminal:
            followers.append(_END)
        return followers

    def _rejoin_points(self) -> dict[str, str]:
        """For each node N, find the node M where all branches leaving N reunite.

        When N forks, the flow splits down several branches; those branches always
        come back together later at a merge. M is that merge: the first node below
        N that sits on every one of N's branches, so it is where all of them are
        back in a single line, not just where the first two of them meet. To draw
        the fork we need M: everything between N and M is the parallel part.

        M is not always the nearest merge. With three or more branches they can
        merge two at a time (a and b merge, then that result merges with c); M is
        the later merge, where the last branch rejoins the rest.

        (If N does not fork it has no branches, and M is just its next node, which
        the caller ignores.)

        The two steps below compute M for every node.
        """

        all_nodes = set(self._ids)

        # Step 1: for every node, find the nodes that sit on all of its branches
        # at once, call them its "shared" nodes. M will be the nearest of them.
        #
        # A node's shared nodes are: itself, plus every node that is shared by all
        # the nodes it feeds directly (a node on all their branches is on all of
        # this node's branches too). We do not know these sets up front, so start
        # each as "all nodes" and keep applying that rule until a round changes
        # nothing.
        shared = {node_id: all_nodes.copy() for node_id in self._ids}
        shared[_END] = {_END}
        settled = False
        while not settled:
            settled = True
            for node_id in self._ids:
                if node_id == _END:
                    continue
                on_all_branches = set(all_nodes)
                for nxt in self._followers(node_id):
                    on_all_branches &= shared[nxt]
                updated = {node_id} | on_all_branches
                if updated != shared[node_id]:
                    shared[node_id] = updated
                    settled = False

        # Step 2: M is the nearest shared node. A node's shared nodes fall one
        # after another down the pipeline, so the nearest is the one that still
        # has all the others below it, i.e. the one with the most shared nodes
        # of its own.
        rejoin = {}
        for node_id in self._ids:
            if node_id == _END:
                continue
            below = shared[node_id] - {node_id}
            if not below:
                rejoin[node_id] = _END
            else:
                rejoin[node_id] = max(below, key=lambda n: len(shared[n]))
        return rejoin

    def _reaches(self, node_id: str, target: str) -> bool:
        """Whether `target` is reachable from `node_id` following the flow forward."""

        stack, visited = [node_id], set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self._followers(current))
        return False

    def _is_merge(self, node: FlowNode) -> bool:
        """A merge reads more than one line; every plain step reads exactly one."""

        return len(node.inputs) > 1

    # -- cards ------------------------------------------------------------- #
    def _card_lines(self, texts: Sequence[str]) -> _Block:
        """Build a centred block from the card lines that have visible text.

        Filtering on visible width, not the raw string, matters with colour on:
        painting an empty string still returns non-empty ANSI codes, and those
        would otherwise add blank lines that the plain drawing does not have.
        """

        lines = [text for text in texts if _vlen(text) > 0]
        width = max(_vlen(line) for line in lines)
        grid = [_vcenter(line, width) for line in lines]
        return _Block(grid, width // 2, width // 2)

    def _card(self, node: FlowNode) -> _Block:
        """One node's card: operation name, dim detail lines, and any outlet spout.

        Branch and outlet names live off the card (on the merge bar and as a side
        spout), so the card itself is just the operation and its layout.
        """

        name = self._style.paint(node.name, bold=True)
        axes = self._style.paint(node.axes, self._style.theme.subtitle, dim=True)
        kinds = self._style.paint(node.kinds, self._style.theme.subtitle, dim=True)
        block = self._card_lines([name, axes, kinds])
        if node.output in self._outlets:
            block = self._tap(block, node.output)
        return block

    def _tap(self, block: _Block, line: str) -> _Block:
        """Add a side spout `├─▶ name` where a line is kept as an outlet.

        The spout points out of the pipeline, toward the outlet name. A terminal
        outlet (nothing flows on below it) uses `└─▶`; one on a line that keeps
        flowing uses `├─▶`.
        """

        corner = "└─" if line in self._terminal else "├─"
        spout = self._style.paint(corner, self._style.theme.rule, dim=True)
        spout += self._style.paint("▶", self._style.theme.ok)
        spout += " " + self._style.paint(
            self._outlets[line], self._style.theme.alias, italic=True
        )
        row = " " * block.out_col + spout
        width = max(_width(block), _vlen(row))
        grid = [*_pad(block.grid, width), _vljust(row, width)]
        return _Block(grid, block.in_col, block.out_col)

    # -- layout ------------------------------------------------------------ #
    def _series_from(
        self, entry: str, stop: str | None
    ) -> tuple[_Block | None, str | None]:
        """Draw a run of steps that follow the line `entry`, up to (not incl.) `stop`.

        Stops when it hits a fork (handed to `_region`) it cannot pass, or a merge
        that also joins branches from elsewhere (left for the enclosing fork to
        draw), or the end.
        """

        blocks: list[_Block] = []
        current = entry
        while current != stop:
            followers = self._children.get(current, [])
            if not followers:
                break
            if len(followers) == 1 and not self._is_merge(followers[0]):
                node = followers[0]
                self._seen.add(node.number)
                blocks.append(self._card(node))
                current = node.output
                continue

            # A single follower that is a merge also pulls in branches from
            # elsewhere, so leave it for the enclosing fork to draw.
            if len(followers) == 1:
                break

            # Several followers means a fork here. Draw everything from the fork
            # down to where its branches rejoin, then carry on from the line that
            # rejoin writes. If they rejoin only at the end, the branches were
            # outlets and there is nothing after them.
            rejoin = self._rejoin[self._producer[current]]
            blocks.append(self._region(current, rejoin))
            following = self._out_line(rejoin)
            if following is None:
                break
            current = following
        if not blocks:
            return None, current
        block = _stack(self._style, blocks) if len(blocks) > 1 else blocks[0]
        return block, current

    def _branch(self, start: FlowNode, target: str | None) -> _Block:
        """Lay out one branch beginning at node `start`, up to `target`."""

        self._seen.add(start.number)
        head = self._card(start)
        rest, _ = self._series_from(start.output, target)
        return _stack(self._style, [head, rest]) if rest is not None else head

    def _region(self, fork_line: str, rejoin: str) -> _Block:
        """Draw everything from a fork on `fork_line` down to where it rejoins.

        `rejoin` is the merge where the branches come back together (or the end,
        if they never do). Its inputs are the fork's real branches (one per
        incoming line) which `_subregion` draws, laid out side by side.
        """

        if rejoin == _END or not self._is_merge(self._nodes[rejoin]):
            branches = [self._branch(node, None) for node in self._children[fork_line]]
            return _parallel(self._style, branches, None)
        merge = self._nodes[rejoin]
        self._seen.add(merge.number)
        branches = [self._subregion(fork_line, head) for head in merge.inputs]
        names = [self._labels.get(head, "") for head in merge.inputs]
        return _parallel(self._style, branches, self._card(merge), names)

    def _subregion(self, fork_line: str, branch_end: str) -> _Block:
        """Draw the one branch of a fork whose last line is `branch_end`.

        Usually that is a plain chain of steps. But the branch may itself fork and
        merge before it ends (several of the fork's outgoing lines lead to it); in
        that case it is a smaller fork, drawn the same way by recursing.
        """

        end_node = self._nodes[branch_end]
        leading = [
            node
            for node in self._children[fork_line]
            if self._reaches(node.output, branch_end)
        ]
        if len(leading) == 1:
            return self._branch(leading[0], branch_end)
        self._seen.add(end_node.number)
        inner = [self._subregion(fork_line, head) for head in end_node.inputs]
        names = [self._labels.get(head, "") for head in end_node.inputs]
        return _parallel(self._style, inner, self._card(end_node), names)
