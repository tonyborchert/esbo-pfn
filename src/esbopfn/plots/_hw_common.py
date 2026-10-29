"""Helpers of the hardware and README figure groups: labels, layout, violins, sidecars.

Built only on :mod:`esbopfn.plots.style`. The layout follows the thesis figures: the full text
width of 6.3 in, panels in one row, one shared legend laid flat above the panels, a dashed light
grid behind the data and method names in typewriter type when LaTeX sets the text.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from esbopfn.plots import style

#: Hardware methods in the thesis draw and legend order.
HW_METHODS: tuple[str, ...] = ("esbo-gp", "bo", "esrs", "decpfn-h", "decpfn", "xdecpfn-pi")
#: Rig titles of the two thesis configurations (other registry entries use their name).
RIG_TITLE: dict[str, str] = {"without_weight": "without weight", "with_weight": "with weight"}

GRID_KW: dict[str, Any] = dict(which="major", linestyle="--", linewidth=0.5,
                               color=style.GRID_COLOR, alpha=0.8)
BAND_ALPHA = 0.13
LINE_WIDTH = 1.4
LEGEND_FONTSIZE = 7.0
TITLE_FONTSIZE = 8.0


# --------------------------------------------------------------------------- text
def usetex() -> bool:
    return bool(plt.rcParams.get("text.usetex"))


def tt(name: str) -> str:
    r"""A method name in typewriter type under LaTeX, the plain name under mathtext."""
    return r"\texttt{" + name + "}" if usetex() else name


def pct() -> str:
    return r"\%" if usetex() else "%"


def better(label: str, higher: bool) -> str:
    """An axis label with the better direction appended: ``LL (→)`` or ``regret (←)``."""
    return f"{label} ({r'$\rightarrow$' if higher else r'$\leftarrow$'})"


def method_label(method: str) -> str:
    return tt(style.display_name(method))


def rig_title(rig: str) -> str:
    return RIG_TITLE.get(rig, rig.replace("_", " "))


# --------------------------------------------------------------------------- style
@contextmanager
def figure_style(cfg: dict[str, Any], theme: str):
    """The thesis style, with ``cfg["usetex"]`` (None = auto) deciding the text engine."""
    with style.thesis_style(theme, usetex=cfg.get("usetex")) as rc:
        yield rc


def fg_color(theme: str) -> str:
    return style.theme_colors(theme)[1]


def color(method: str, theme: str) -> str:
    return style.method_color(method, theme)


def grid(ax: Any, axis: str = "both") -> None:
    ax.grid(True, axis=axis, **GRID_KW)
    ax.set_axisbelow(True)


def panel_row(n: int, *, height: float = 2.0, sharey: bool = False,
              box_aspect: float | None = None) -> tuple[Figure, list[Any]]:
    """``n`` panels across the text width. ``box_aspect`` (height / width) fixes the box shape,
    by default square boxes for three or more panels and wide ones for two."""
    fig, axes = plt.subplots(1, n, figsize=(style.WIDTHS[1], height), sharey=sharey,
                             squeeze=False)
    aspect = box_aspect if box_aspect is not None else (1.0 if n >= 3 else 0.55)
    for ax in axes.flat:
        ax.set_box_aspect(aspect)
    return fig, list(axes.flat)


def top_legend(fig: Figure, handles: Sequence[Any], labels: Sequence[str], *, ncol: int,
               pad_in: float = 0.08, **kw: Any) -> Any:
    """One shared legend laid flat above every axes (frameless, 7 pt)."""
    fig.tight_layout(w_pad=1.2)
    fig.canvas.draw()
    top = 0.0
    for ax in fig.axes:
        bb = ax.get_tightbbox()
        if bb is not None:
            top = max(top, fig.transFigure.inverted().transform((0, bb.y1))[1])
    anchor = top + pad_in / fig.get_size_inches()[1]
    opts: dict[str, Any] = dict(ncol=ncol, frameon=False, fontsize=LEGEND_FONTSIZE,
                                handlelength=1.6, handletextpad=0.4, columnspacing=1.0,
                                borderpad=0.0, labelspacing=0.3, borderaxespad=0.0,
                                loc="lower center", bbox_to_anchor=(0.5, anchor))
    opts.update(kw)
    return fig.legend(list(handles), list(labels), **opts)


def finish(ax: Any, *, xlabel: str | None = None, ylabel: str | None = None,
           grid_axis: str = "both") -> None:
    grid(ax, grid_axis)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def save_figure(fig: Figure, out_dir: Path, name: str, fmt: str) -> Path:
    path = style.save(fig, Path(out_dir) / name, fmt)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- violins
def violin_bins(populations: Sequence[np.ndarray], ylim: tuple[float, float]) -> np.ndarray:
    """One bin grid for a whole figure (Freedman-Diaconis at the size of one violin, clamped to
    1/40 .. 1/6 of the drawn range)."""
    pooled = np.concatenate([np.asarray(v, float) for v in populations if len(v)])
    span = float(ylim[1] - ylim[0])
    lo_w, hi_w = span / 40.0, span / 6.0
    q1, q3 = np.percentile(pooled, [25, 75])
    n = max(int(np.median([len(v) for v in populations if len(v)])), 2)
    fd = 2.0 * (q3 - q1) / n ** (1.0 / 3.0) if q3 > q1 else hi_w
    width = float(np.clip(fd, lo_w, hi_w))
    lo = math.floor(ylim[0] / width) * width
    hi = math.ceil(ylim[1] / width) * width + width * 0.5
    return np.arange(lo, hi, width)


def violin_scale(populations: Sequence[np.ndarray], edges: np.ndarray,
                 cap: float = 0.5) -> float:
    """The bin share drawn at full width: the largest single-bin share of any non-degenerate
    violin, capped at ``cap``."""
    best = 0.0
    for values in populations:
        if len(values) < 2 or float(np.ptp(values)) < 1e-9:
            continue
        counts, _ = np.histogram(values, bins=edges)
        best = max(best, counts.max() / len(values))
    return min(best, cap) if best > 0 else cap


def violin_body(ax: Any, values: np.ndarray, x: float, color: str, *, edges: np.ndarray,
                scale: float, half_width: float, alpha: float) -> None:
    """A mirrored histogram drawn smooth (monotone PCHIP through the bin centers), no KDE."""
    from scipy.interpolate import PchipInterpolator

    counts, _ = np.histogram(values, bins=edges)
    occupied = np.flatnonzero(counts)
    if not occupied.size:
        return
    first, last = int(occupied[0]), int(occupied[-1])
    half = np.clip(counts / len(values) / scale, 0.0, 1.0) * half_width
    centers = 0.5 * (edges[:-1] + edges[1:])
    ky = np.concatenate(([edges[first]], centers[first:last + 1], [edges[last + 1]]))
    kw = np.concatenate(([0.0], half[first:last + 1], [0.0]))
    ys = np.linspace(ky[0], ky[-1], 240)
    ws = np.clip(PchipInterpolator(ky, kw)(ys), 0.0, None)
    ax.fill_betweenx(ys, x - ws, x + ws, facecolor=color, edgecolor="none", alpha=alpha,
                     zorder=3)


def violin_points(ax: Any, values: np.ndarray, x: float, color: str, *, half_width: float,
                  size: float = 3.0) -> None:
    n = len(values)
    offs = (np.linspace(-0.72, 0.72, n) if n > 1 else np.zeros(1)) * half_width
    ax.scatter(x + offs, values, s=size, c=color, alpha=0.9, linewidths=0.0, zorder=5)


def violin_median(ax: Any, values: np.ndarray, x: float, color: str, *,
                  half_width: float) -> float:
    med = float(np.median(values))
    ax.plot([x - half_width, x + half_width], [med, med], color=color, lw=1.3, zorder=6,
            solid_capstyle="butt")
    return med


# --------------------------------------------------------------------------- sidecars
def fmt_num(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "n/a"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{digits}f}"


def fmt_list(xs: Iterable[Any], digits: int = 3) -> str:
    return "[" + ", ".join(fmt_num(x, digits) for x in xs) + "]"


def write_sidecar(out_dir: Path, name: str, *, shown: Sequence[str], metric: Sequence[str],
                  data: Sequence[str], numbers: Sequence[str]) -> Path:
    """``<name>.md`` beside the figure: what is shown, the metric, the data sources and the key
    numbers the figure is drawn from."""
    parts = [f"# {name}", ""]
    for title, lines in (("What is shown", shown), ("Metric", metric), ("Data", data),
                         ("Numbers", numbers)):
        parts.append(f"## {title}")
        parts += [line if line.startswith(("|", "  ")) else f"- {line}" for line in lines]
        parts.append("")
    path = Path(out_dir) / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    from esbopfn.plots.common import home_relative

    path.write_text(home_relative("\n".join(parts)))
    return path
