"""The 2-D flow graph: branches side by side, forks split, staged merges join."""

from __future__ import annotations

import re

from csiphon.graphics.style import Styler
from csiphon.graphics.tree import FlowNode, render_flow

# A no-color styler renders plain text, so tests can match glyphs and layout directly.
PLAIN = Styler(enabled=False)


def _node(number: int, name: str, inputs: tuple[str, ...], output: str) -> FlowNode:
    """Shorthand for one node (no layout detail)."""

    return FlowNode(
        number=number, name=name, inputs=inputs, output=output, axes="", kinds=""
    )


def _draw(nodes, labels, outlets) -> list[str] | None:
    """Render to plain lines, or None if the wiring is not drawable."""

    return render_flow(PLAIN, nodes, labels=labels, outlets=outlets)


def test_linear_is_a_single_column() -> None:
    """A straight run draws as one column with no fork or merge glyphs."""

    nodes = [_node(1, "magnitude", ("_in",), "a"), _node(2, "power", ("a",), "b")]
    lines = _draw(nodes, labels={}, outlets={})

    assert lines is not None
    text = "\n".join(lines)
    assert "magnitude" in text and "power" in text
    assert "┴" not in text and "┬" not in text  # no fork / merge


def test_fork_draws_branches_side_by_side() -> None:
    """Two branches share a row (parallel on the page); a merge, named, joins them."""

    nodes = [
        _node(1, "magnitude", ("_in",), "a"),
        _node(2, "windowed-variance", ("a",), "fast"),
        _node(3, "windowed-variance", ("a",), "slow"),
        _node(4, "stack", ("fast", "slow"), "out"),
    ]
    lines = _draw(
        nodes, labels={"fast": "fast", "slow": "slow"}, outlets={"out": "result"}
    )

    assert lines is not None
    text = "\n".join(lines)
    # The branch aliases appear together on one line: they are parallel, not sequential.
    assert any("fast" in line and "slow" in line for line in lines)
    assert "┴" in text and "┬" in text  # a fork bar and a merge bar
    # The merge is named by its operation, not "merge ◀ fast, slow".
    assert "stack" in text and "◀" not in text
    assert "▶ result" in text  # the outlet spout points out to its name


def test_three_way_merge_joins_all_branches() -> None:
    """Three branches, each with its own step, join at one three-input merge."""

    nodes = [
        _node(1, "magnitude", ("_in",), "m"),
        _node(2, "op-a", ("m",), "a"),
        _node(3, "op-b", ("m",), "b"),
        _node(4, "op-c", ("m",), "c"),
        _node(5, "mean", ("a", "b", "c"), "out"),
    ]
    lines = _draw(nodes, labels={"a": "a", "b": "b", "c": "c"}, outlets={"out": "out"})

    assert lines is not None
    assert any("op-a" in line and "op-b" in line and "op-c" in line for line in lines)
    assert "mean" in "\n".join(lines)


def test_staged_dual_merge_lays_out_every_node() -> None:
    """Merge a and b first, then merge that with c; every node is drawn."""

    nodes = [
        _node(1, "magnitude", ("_in",), "m"),
        _node(2, "op-a", ("m",), "a"),
        _node(3, "op-b", ("m",), "b"),
        _node(4, "op-c", ("m",), "c"),
        _node(5, "mean", ("a", "b"), "ab"),  # a + b first
        _node(6, "stack", ("ab", "c"), "out"),  # then with c
    ]
    lines = _draw(
        nodes,
        labels={"a": "a", "b": "b", "c": "c", "ab": "ab"},
        outlets={"out": "out"},
    )

    assert lines is not None  # the staged shape is drawable, not a fallback
    text = "\n".join(lines)
    assert "mean" in text and "stack" in text  # both merges are drawn and named
    assert "op-c" in text


def test_nested_branch_has_its_own_fork() -> None:
    """A branch that itself forks and merges lays out inside its own column."""

    nodes = [
        _node(1, "magnitude", ("_in",), "m"),
        # branch "combo": a step, then its own fork into x / y, then an inner merge
        _node(2, "pre", ("m",), "p"),
        _node(3, "x", ("p",), "px"),
        _node(4, "y", ("p",), "py"),
        _node(5, "mean", ("px", "py"), "combo"),
        # branch "plain": a single step
        _node(6, "plain-op", ("m",), "plain"),
        # outer merge of the two branches
        _node(7, "stack", ("combo", "plain"), "out"),
    ]
    lines = _draw(
        nodes,
        labels={"combo": "combo", "plain": "plain"},
        outlets={"out": "out"},
    )

    assert lines is not None
    text = "\n".join(lines)
    # Both the inner and outer merges are drawn and named by their operation.
    assert text.count("mean") >= 1 and "stack" in text
    assert "x" in text and "y" in text and "plain-op" in text


def test_fork_without_merge_stays_separate() -> None:
    """A fork whose branches never merge just draws the branches, each an outlet."""

    nodes = [
        _node(1, "magnitude", ("_in",), "a"),
        _node(2, "left-op", ("a",), "left"),
        _node(3, "right-op", ("a",), "right"),
    ]
    lines = _draw(nodes, labels={}, outlets={"left": "l", "right": "r"})

    assert lines is not None
    text = "\n".join(lines)
    assert "▶ l" in text and "▶ r" in text  # each branch taps out as an outlet
    assert any("left-op" in line and "right-op" in line for line in lines)


def test_unexpected_shape_falls_back() -> None:
    """Two disconnected roots are not a drawable graph, so drawing declines (None)."""

    nodes = [_node(1, "a", ("_in",), "x"), _node(2, "b", ("_other",), "y")]
    assert _draw(nodes, labels={}, outlets={}) is None


def test_empty_pipeline_draws_just_the_inlet() -> None:
    """No steps still draws the inlet marker (not a failure)."""

    lines = render_flow(PLAIN, [], labels={}, outlets={})
    assert lines is not None
    assert any("inlet" in line for line in lines)


# --- edge cases ---


def test_cards_show_axes_and_value_kind() -> None:
    """A node's card shows its axes and its representation:value detail lines."""

    node = FlowNode(1, "magnitude", ("_in",), "a", "(T, SC[4])", "CFR: magnitude")
    lines = render_flow(
        PLAIN, [node], labels={}, outlets={}, inlet=("(T, SC[4])", "CFR: complex")
    )
    text = "\n".join(lines)
    assert "(T, SC[4])" in text and "CFR: magnitude" in text


def test_outlet_spout_marks_terminal_and_mid_pipeline_differently() -> None:
    """A kept line that keeps flowing taps with `├─▶`; a terminal outlet with `└─▶`."""

    # `a` is probed but consumed by `power`; `b` is the pipeline's end.
    nodes = [_node(1, "magnitude", ("_in",), "a"), _node(2, "power", ("a",), "b")]
    text = "\n".join(_draw(nodes, labels={}, outlets={"a": "amp", "b": "out"}) or [])
    assert "├─▶ amp" in text  # keeps flowing
    assert "└─▶ out" in text  # terminal


def test_colour_does_not_shift_columns() -> None:
    """Colour is cosmetic: stripping the ANSI codes yields exactly the plain drawing."""

    nodes = [
        _node(1, "magnitude", ("_in",), "a"),
        _node(2, "va", ("a",), "fast"),
        _node(3, "vb", ("a",), "slow"),
        _node(4, "stack", ("fast", "slow"), "out"),
    ]
    labels = {"fast": "fast", "slow": "slow"}
    outlets = {"out": "out"}
    plain = render_flow(Styler(enabled=False), nodes, labels=labels, outlets=outlets)
    coloured = render_flow(Styler(enabled=True), nodes, labels=labels, outlets=outlets)

    assert coloured is not None and plain is not None
    stripped = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in coloured]
    assert stripped == plain
