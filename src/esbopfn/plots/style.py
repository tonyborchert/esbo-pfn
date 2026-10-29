"""Thesis figure style: fonts, sizes, colors, method display names and saving.

Charter through usetex when a TeX installation with ``mathdesign`` is found, mathtext otherwise.
Font size 9 pt, figure widths 6.3, 3.02 and 2.02 in for one, two and three figures per text row.
The black theme swaps only background and foreground colors.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

THEMES = ("white", "black")
FORMATS = ("pdf", "png")

FONT_SIZE = 9
WIDTHS: dict[int, float] = {1: 6.3, 2: 3.02, 3: 2.02}
DEFAULT_HEIGHT = 2.0
GRID_COLOR = "#B0B0B0"

#: RWTH palette for schematics.
RWTH_BLUE = "#00549F"
RWTH_ORANGE = "#F6A800"
RWTH_RED = "#CC071E"
RWTH_GREEN = "#57AB27"
RWTH_COLORS: dict[str, str] = {
    "blue": RWTH_BLUE, "orange": RWTH_ORANGE, "red": RWTH_RED, "green": RWTH_GREEN,
}

_PREAMBLE = (
    r"\usepackage[charter]{mathdesign}\usepackage[scaled=1.0]{zlmtt}"
    r"\usepackage{amsmath}\usepackage{bm}"
)

#: Method and model ids to the names used in the thesis.
DISPLAY_NAME: dict[str, str] = {
    "esrs": "ESRS",
    "bo": "BO",
    "esbo-gp": "ESBO-GP",
    "esbo-t-tabpfn": "ESBO-T-TabPFN",
    "esbo-oracle": "ESBO-Oracle",
    "decpfn": "DecPFN",
    "decpfn-h": "DecPFN-H",
    "decpfn-nostates": "DecPFN-no-St",
    "xdecpfn": "XDecPFN",
    "xdecpfn-pi": "XDecPFN-PI",
    "xdecpfn-ei": "XDecPFN-EI",
    "ftpfn": "FT-PFN",
    "ftpfn-ei": "FT-PFN-EI",
    "ftpfn-pi": "FT-PFN-PI",
    "lcpfn": "LC-PFN",
    "lcpfn-last": "LC-PFN-last",
    "segment-gp": "Segment-GP",
    "tabpfn": "Trajectory-TabPFN",
    "segment-tabpfn": "Segment-TabPFN",
}

#: Okabe-Ito based colors per method family, keyed by display name.
METHOD_COLORS: dict[str, str] = {
    "DecPFN": "#0072B2",
    "DecPFN-H": "#0B3C73",
    "LC-PFN": "#D55E00",
    "FT-PFN": "#009E73",
    "XDecPFN": "#CC79A7",
    "Trajectory-TabPFN": "#9B4FA8",
    "ESBO-T-TabPFN": "#9B4FA8",
    "Segment-TabPFN": "#882255",
    "Segment-GP": "#E69F00",
    "ESBO-GP": "#E69F00",
    "ESRS": "#BBBBBB",
    "BO": "#8C564B",
    "ESBO-Oracle": "#000000",
}

#: Display names whose color is the one of a family prefix.
_FAMILY_PREFIXES: tuple[str, ...] = ("XDecPFN", "FT-PFN", "LC-PFN", "DecPFN")


def display_name(method: str) -> str:
    """Thesis name of a method or model id (``decpfn/cartpole/binned/1M`` gives ``DecPFN``)."""
    if method in METHOD_COLORS or method in DISPLAY_NAME.values():
        return method
    head = method.split("/")[0].lower()
    return DISPLAY_NAME.get(head, method)


def method_color(method: str, theme: str = "white") -> str:
    """Color of a method. Variants share their family color. Black becomes white on black."""
    name = display_name(method)
    color = METHOD_COLORS.get(name)
    if color is None:
        prefix = next((p for p in _FAMILY_PREFIXES if name.startswith(p)), None)
        color = METHOD_COLORS[prefix] if prefix else "#555555"
    if theme == "black" and color == "#000000":
        return "#FFFFFF"
    return color


@functools.lru_cache(maxsize=1)
def usetex_available() -> bool:
    """True if ``latex`` and ``kpsewhich`` exist and ``mathdesign.sty`` is installed."""
    if not shutil.which("latex") or not shutil.which("kpsewhich"):
        return False
    try:
        out = subprocess.run(
            ["kpsewhich", "mathdesign.sty"], capture_output=True, text=True, timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.stdout.strip())


def figsize(per_row: int = 1, height: float = DEFAULT_HEIGHT) -> tuple[float, float]:
    """Figure size for ``per_row`` figures side by side in the text width."""
    if per_row not in WIDTHS:
        raise ValueError(f"per_row must be one of {sorted(WIDTHS)}, got {per_row}")
    return (WIDTHS[per_row], height)


def theme_colors(theme: str) -> tuple[str, str]:
    """``(background, foreground)`` of a theme."""
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {THEMES}, got {theme!r}")
    return ("white", "black") if theme == "white" else ("black", "white")


def rc_params(theme: str = "white", usetex: bool | None = None) -> dict[str, Any]:
    """The rcParams of the thesis style."""
    bg, fg = theme_colors(theme)
    tex = usetex_available() if usetex is None else usetex
    params: dict[str, Any] = {
        "font.size": FONT_SIZE,
        "figure.figsize": figsize(1),
        "lines.linewidth": 1.0,
        "axes.grid": False,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "legend.frameon": True,
        "axes.unicode_minus": False,
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "legend.facecolor": bg,
        "legend.edgecolor": fg,
        "text.color": fg,
        "axes.edgecolor": fg,
        "axes.labelcolor": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": tex,
    }
    if tex:
        params.update({
            "font.family": "serif",
            "font.serif": ["Charter"],
            "text.latex.preamble": _PREAMBLE,
        })
    else:
        params.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    return params


@contextmanager
def thesis_style(theme: str = "white", usetex: bool | None = None) -> Iterator[dict[str, Any]]:
    """Context manager applying the thesis rcParams. Yields the parameters."""
    params = rc_params(theme, usetex)
    with plt.rc_context(params):
        yield params


def save(fig: Figure, path: Path | str, fmt: str = "pdf") -> Path:
    """Save ``fig`` as ``path`` with the suffix of ``fmt`` and return the written path.

    The figure keeps the background of the theme it was drawn under (white by default).
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    out = Path(path).with_suffix(f".{fmt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format=fmt, facecolor=fig.get_facecolor(), bbox_inches="tight",
                dpi=300 if fmt == "png" else None)
    return out
