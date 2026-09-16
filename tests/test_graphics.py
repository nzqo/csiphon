"""describe() rendering: color toggle and custom themes."""

import re

from csiphon import describe
from csiphon.graphics import Theme, Vibrant
from csiphon.steps import Magnitude

_ANSI = re.compile("\x1b\\[[0-9;]*m")


def test_plain_render_has_no_ansi() -> None:
    """color=False produces plain text."""

    assert "\x1b[" not in describe(Magnitude).render(color=False)


def test_color_render_strips_back_to_plain() -> None:
    """Color adds ANSI but doesn't change the visible layout (alignment holds)."""

    plain = describe(Magnitude).render(color=False)
    colored = describe(Magnitude).render(color=True)
    assert "\x1b[" in colored
    assert _ANSI.sub("", colored) == plain


def test_custom_theme_recolors_but_keeps_text() -> None:
    """A custom theme changes the colors, not the visible text."""

    theme = Theme(
        title=Vibrant.MAGENTA,
        subtitle=Vibrant.GREY,
        rule=Vibrant.GREY,
        ok=Vibrant.BLUE,
        warn=Vibrant.ORANGE,
        bad=Vibrant.RED,
        override=Vibrant.CYAN,
        alias=Vibrant.ORANGE,
    )
    default = describe(Magnitude).render(color=True)
    custom = describe(Magnitude).render(color=True, theme=theme)
    assert default != custom
    assert _ANSI.sub("", default) == _ANSI.sub("", custom)
