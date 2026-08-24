#!/usr/bin/env python3
"""Generate the csiphon icon as a transparent PNG and vector PDF.

Dependency:
    pip install matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath

from csiphon.graphics import Color, Vibrant

VIEWBOX = 1024.0

DEFAULT_BLUE = Vibrant.BLUE
DEFAULT_LIGHT_BLUE = Vibrant.CYAN
DEFAULT_TEAL = Vibrant.TEAL
DEFAULT_MAGENTA = Vibrant.MAGENTA

OUTER_WIDTH = 104.0
INNER_WIDTH = 62.0

# Matplotlib stores these constants as NumPy uint8 values. Converting them to
# built-in int keeps strict type checkers happy.
MOVE_TO = int(MplPath.MOVETO)
LINE_TO = int(MplPath.LINETO)
CURVE_TO = int(MplPath.CURVE4)

MAIN_VERTICES: list[tuple[float, float]] = [
    (176.0, 210.0),
    (176.0, 610.0),
    (176.0, 700.0),
    (246.0, 760.0),
    (336.0, 760.0),
    (426.0, 760.0),
    (496.0, 700.0),
    (496.0, 610.0),
    (496.0, 470.0),
    (496.0, 420.0),
    (536.0, 390.0),
    (586.0, 390.0),
    (914.0, 390.0),
]

MAIN_CODES: list[int] = [
    MOVE_TO,
    LINE_TO,
    CURVE_TO,
    CURVE_TO,
    CURVE_TO,
    CURVE_TO,
    CURVE_TO,
    CURVE_TO,
    LINE_TO,
    CURVE_TO,
    CURVE_TO,
    CURVE_TO,
    LINE_TO,
]

BRANCH_VERTICES: list[tuple[float, float]] = [
    (650.0, 390.0),
    (650.0, 604.0),
    (914.0, 604.0),
]

BRANCH_CODES: list[int] = [MOVE_TO, LINE_TO, LINE_TO]

MAIN_PATH = MplPath(MAIN_VERTICES, MAIN_CODES)
BRANCH_PATH = MplPath(BRANCH_VERTICES, BRANCH_CODES)


def parse_color(value: str) -> Color:
    """Parse a CLI color as 'RRGGBB' or '#RRGGBB'."""

    try:
        return Color.from_hex(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def add_stroked_path(ax: Axes, path: MplPath, color: Color, width: float) -> None:
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color.to_mpl(),
            linewidth=width,
            capstyle="round",
            joinstyle="round",
            antialiased=True,
        )
    )


def add_node(
    ax: Axes,
    cx: float,
    cy: float,
    outer_radius: float,
    inner_radius: float,
    outer_color: Color,
    inner_color: Color,
) -> None:
    ax.add_patch(
        Circle(
            (cx, cy),
            outer_radius,
            facecolor=outer_color.to_mpl(),
            edgecolor="none",
            antialiased=True,
        )
    )
    ax.add_patch(
        Circle(
            (cx, cy),
            inner_radius,
            facecolor=inner_color.to_mpl(),
            edgecolor="none",
            antialiased=True,
        )
    )


def build_figure(
    blue: Color,
    light_blue: Color,
    teal: Color,
    magenta: Color,
) -> Figure:
    # At this figure size, one Matplotlib point equals one view-box unit.
    figure = Figure(figsize=(VIEWBOX / 72.0, VIEWBOX / 72.0), dpi=72.0)
    ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))

    ax.set_xlim(0.0, VIEWBOX)
    ax.set_ylim(VIEWBOX, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    add_stroked_path(ax, MAIN_PATH, blue, OUTER_WIDTH)
    add_stroked_path(ax, BRANCH_PATH, blue, OUTER_WIDTH)
    add_stroked_path(ax, MAIN_PATH, light_blue, INNER_WIDTH)
    add_stroked_path(ax, BRANCH_PATH, light_blue, INNER_WIDTH)

    add_node(ax, 176.0, 210.0, 88.0, 29.0, teal, light_blue)
    add_node(ax, 914.0, 390.0, 76.0, 25.0, magenta, light_blue)
    add_node(ax, 914.0, 604.0, 76.0, 25.0, magenta, light_blue)

    return figure


def render_png(figure: Figure, path: Path, size: int) -> None:
    figure.savefig(
        str(path),
        format="png",
        dpi=72.0 * size / VIEWBOX,
        transparent=True,
        bbox_inches=None,
        pad_inches=0.0,
    )


def render_pdf(figure: Figure, path: Path) -> None:
    figure.savefig(
        str(path),
        format="pdf",
        transparent=True,
        bbox_inches=None,
        pad_inches=0.0,
    )


def generate(
    output_dir: Path,
    stem: str,
    size: int,
    blue: Color,
    light_blue: Color,
    teal: Color,
    magenta: Color,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    figure = build_figure(blue, light_blue, teal, magenta)
    try:
        render_png(figure, png_path, size)
        render_pdf(figure, pdf_path)
    finally:
        figure.clear()

    return [png_path, pdf_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("./assets/"))
    parser.add_argument("--stem", default="csiphon_icon")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--blue", type=parse_color, default=DEFAULT_BLUE)
    parser.add_argument("--light-blue", type=parse_color, default=DEFAULT_LIGHT_BLUE)
    parser.add_argument("--teal", type=parse_color, default=DEFAULT_TEAL)
    parser.add_argument("--magenta", type=parse_color, default=DEFAULT_MAGENTA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size = int(args.size)

    if size < 32:
        raise SystemExit("--size must be at least 32")

    generated_paths = generate(
        Path(args.output_dir),
        str(args.stem),
        size,
        args.blue,
        args.light_blue,
        args.teal,
        args.magenta,
    )

    for generated_path in generated_paths:
        print(generated_path)


if __name__ == "__main__":
    main()