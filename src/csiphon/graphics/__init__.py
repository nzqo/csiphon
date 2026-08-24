"""Styling and (later) light visualizations.

For now this holds the terminal styling used by `describe()`. Visualization
helpers will grow here alongside it.
"""

from __future__ import annotations

from csiphon.graphics.style import (
    DEFAULT_THEME,
    VIBRANT,
    Color,
    Styler,
    Theme,
    Vibrant,
    should_color,
)

__all__ = [
    "DEFAULT_THEME",
    "VIBRANT",
    "Color",
    "Styler",
    "Theme",
    "Vibrant",
    "should_color",
]
