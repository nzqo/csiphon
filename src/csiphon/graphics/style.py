"""Palette, theme, and a tiny ANSI painter for terminal output.

Colors come from Paul Tol's *vibrant* qualitative palette, which is
colorblind-safe. A Theme maps semantic roles (title, rule, ok / warn / bad, ...)
onto palette colors; build your own Theme and pass it to
`describe(...).render(theme=...)`, or reassign `DEFAULT_THEME`, to recolor
everything. Terminal output uses 24-bit ("truecolor") ANSI.
"""

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Final, Self


@dataclass(frozen=True, slots=True)
class Color:
    """An immutable 8-bit RGB color."""

    red: int
    green: int
    blue: int

    def __post_init__(self) -> None:
        """Validate that all RGB channels are integers between 0 and 255."""

        for component in self.to_rgb():
            if isinstance(component, bool) or not isinstance(component, int):
                raise TypeError(f"RGB components must be integers, got {self.to_rgb()}")
            if not 0 <= component <= 255:
                raise ValueError(
                    f"RGB components must be between 0 and 255, got {self.to_rgb()}"
                )

    @classmethod
    def from_rgb(cls, red: int, green: int, blue: int) -> Self:
        """Create a color from 8-bit red, green, and blue channels."""

        return cls(red=red, green=green, blue=blue)

    @classmethod
    def from_hex(cls, value: str) -> Self:
        """Create a color from an 'RRGGBB' or '#RRGGBB' string."""

        value = value.removeprefix("#")

        if len(value) != 6 or any(
            character not in "0123456789abcdefABCDEF" for character in value
        ):
            raise ValueError(f"Expected a six-digit hex color, got {value!r}")

        return cls(
            red=int(value[0:2], 16),
            green=int(value[2:4], 16),
            blue=int(value[4:6], 16),
        )

    def to_rgb(self) -> tuple[int, int, int]:
        """Return the color as an 8-bit RGB tuple."""

        return self.red, self.green, self.blue

    def to_hex(self) -> str:
        """Return the color as an uppercase '#RRGGBB' string."""

        return f"#{self.red:02X}{self.green:02X}{self.blue:02X}"

    def to_mpl(self) -> tuple[float, float, float]:
        """Return the color as a Matplotlib-compatible RGB tuple."""

        return (
            self.red / 255,
            self.green / 255,
            self.blue / 255,
        )


class Status(Enum):
    """A traffic-light status a state can carry: good, caution, or blocked."""

    # fmt: off
    OK   = "ok"    # a good / safe state
    WARN = "warn"  # a caution state
    BAD  = "bad"   # a blocked state
    # fmt: on


class Vibrant:  # pylint: disable=too-few-public-methods  # a palette namespace
    """Paul Tol's 'vibrant' qualitative palette (colorblind-safe).
    See: https://sronpersonalpages.nl/~pault/
    """

    # fmt: off
    BLUE    : Final[Color] = Color.from_hex("#0077BB")
    CYAN    : Final[Color] = Color.from_hex("#33BBEE")
    TEAL    : Final[Color] = Color.from_hex("#009988")
    ORANGE  : Final[Color] = Color.from_hex("#EE7733")
    RED     : Final[Color] = Color.from_hex("#CC3311")
    MAGENTA : Final[Color] = Color.from_hex("#EE3377")
    GREY    : Final[Color] = Color.from_hex("#BBBBBB")
    # fmt: on


@dataclass(frozen=True, slots=True)
class Theme:  # pylint: disable=too-many-instance-attributes  # a palette-role record
    """Maps semantic roles to palette colors."""

    # fmt: off
    title    : Color  # block name
    subtitle : Color  # category and secondary labels
    rule     : Color  # dividers
    ok       : Color  # a good / safe state (batch-equivalent streaming)
    warn     : Color  # a caution state (batch-divergent streaming)
    bad      : Color  # a blocked state (unavailable streaming)
    override : Color  # a parameter changed from its default
    alias    : Color  # a branch / outlet name in the flow graph
    # fmt: on

    def status_color(self, status: Status) -> Color:
        """Return the color this theme uses for a traffic-light status."""

        match status:
            case Status.OK:
                return self.ok
            case Status.WARN:
                return self.warn
            case Status.BAD:
                return self.bad

        raise ValueError(f"Unsupported status: {status!r}")


VIBRANT = Theme(
    title=Vibrant.MAGENTA,
    subtitle=Vibrant.GREY,
    rule=Vibrant.GREY,
    ok=Vibrant.TEAL,
    warn=Vibrant.ORANGE,
    bad=Vibrant.RED,
    override=Vibrant.CYAN,
    alias=Vibrant.ORANGE,
)

# Reassign this to recolor describe() output everywhere by default.
DEFAULT_THEME = VIBRANT


def should_color(color: bool | None) -> bool:
    """Decide whether to emit ANSI: on for a TTY unless NO_COLOR is set."""

    if color is not None:
        return color
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


class Styler:  # pylint: disable=too-few-public-methods  # a tiny stateful painter
    """Paints text with 24-bit ANSI when enabled, else returns it unchanged."""

    def __init__(self, enabled: bool, theme: Theme | None = None) -> None:
        """Bind the on/off flag and the theme (defaults to DEFAULT_THEME)."""

        self.enabled = enabled
        self.theme = theme or DEFAULT_THEME

    def paint(
        self,
        text: str,
        color: Color | None = None,
        *,
        bold: bool = False,
        dim: bool = False,
        italic: bool = False,
    ) -> str:
        """Wrap `text` in the given color and attributes (a no-op when disabled)."""

        if not self.enabled:
            return text

        codes: list[str] = []

        if bold:
            codes.append("1")
        if dim:
            codes.append("2")
        if italic:
            codes.append("3")
        if color is not None:
            red, green, blue = color.to_rgb()
            codes.append(f"38;2;{red};{green};{blue}")

        if not codes:
            return text

        return f"\033[{';'.join(codes)}m{text}\033[0m"
