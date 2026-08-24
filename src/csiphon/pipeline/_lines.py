"""Internal line-id constants for the graph model (not public API)."""

from __future__ import annotations

INLET = "_inlet"  # the line id of a single-inlet pipeline's raw input


def inlet_id(name: str) -> str:
    """The line id of a named inlet (multi-inlet pipelines)."""

    return f"_inlet:{name}"
