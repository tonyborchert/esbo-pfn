"""Shared helpers of the plot groups, on top of :mod:`esbopfn.plots.style`.

- Rendering: :func:`render_group` runs a group's :class:`FigureSpec` list under the thesis style,
  saves every figure through :func:`esbopfn.plots.style.save` (``--format pdf|png``,
  ``--theme white|black``) and writes a sidecar ``<name>.md`` next to it. A figure whose data is
  missing raises :class:`MissingData` and is skipped with a one-line reason (listed in
  ``skipped.txt`` of the output directory), the rest of the group still renders.
- Layout: :func:`panel_sheet` places equal axes boxes with the thesis panel geometry (box height
  98.3 pt, gap 61.5 pt, a row spanning 396.7 pt of the 6.3 in text width), :func:`finish` adds the
  dashed grid and insets the outermost ticks from the frame.
- Data: :func:`thesis_series` reads imported aggregates, :func:`prediction_frame` the rows of
  ``prediction`` units, :func:`timing_table` the median inference seconds. :class:`Provenance`
  records which units a figure read so the sidecar lists them, with a visible note when a figure
  falls back to the thesis import.
- Drawing: :func:`draw_front` (Pareto fronts over context sizes with numbered glyphs).
"""

from __future__ import annotations

import logging
import math
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from esbopfn.plots import style

log = logging.getLogger(__name__)

#: Thesis root of the generated assets (named in the sidecars, never read).
THESIS_ASSETS = "writing/thesis/assets_generated"

# ------------------------------------------------------------------------------- names
PROBLEMS: tuple[str, ...] = ("boiler", "threetank_pi", "pt2_deadtime", "cartpole", "furuta")
CRASH_PROBLEMS: tuple[str, ...] = ("pt2_deadtime", "cartpole", "furuta")
PROBLEM_LABEL: dict[str, str] = {
    "boiler": "boiler",
    "threetank_pi": "three-tank",
    "pt2_deadtime": "PT2 dead-time",
    "cartpole": "cartpole",
    "furuta": "furuta",
    "pooled": "all problems",
}
#: Figure labels of the model families (the thesis figures write the no-states arm DecPFN-no-St).
MODEL_LABEL: dict[str, str] = {
    "decpfn": "DecPFN",
    "decpfn-nostates": "DecPFN-no-St",
    "decpfn-nodeadtime": "DecPFN-no-dt",
    "lcpfn": "LC-PFN",
    "lcpfn-last": "LC-PFN-last",
    "xdecpfn": "XDecPFN",
    "ftpfn": "FT-PFN",
    "tabpfn": "Trajectory-TabPFN",
    "segment-tabpfn": "Segment-TabPFN",
    "segment-gp": "Segment-GP",
}
#: Second color of Segment-TabPFN (the ``rate`` mode, the ``split`` mode has the family color).
SEGMENT_TABPFN_RATE_COLOR = "#C0506A"
#: The red of crash probabilities and crash markers (RWTH red).
CRASH_COLOR = style.RWTH_RED


def problem_label(problem: str) -> str:
    return PROBLEM_LABEL.get(problem, problem)


def model_family(model: str) -> str:
    return model.split("/")[0]


def model_label(model: str) -> str:
    """Figure label of a model id (``decpfn-nostates/cartpole/binned/1M`` -> ``DecPFN-no-St``)."""
    return MODEL_LABEL.get(model_family(model), style.display_name(model))


def model_color(model: str, theme: str = "white") -> str:
    fam = model_family(model)
    if fam == "segment-tabpfn" and model.endswith("rate"):
        return SEGMENT_TABPFN_RATE_COLOR
    if fam == "decpfn-nostates":
        return style.method_color("decpfn", theme)
    return style.method_color(MODEL_LABEL.get(fam, model), theme)


def fg_color(theme: str) -> str:
    return style.theme_colors(theme)[1]


def bg_color(theme: str) -> str:
    return style.theme_colors(theme)[0]


def usetex_on() -> bool:
    return bool(plt.rcParams.get("text.usetex"))


def pct() -> str:
    r"""A percent sign for the active text engine (``\%`` under usetex)."""
    return r"\%" if usetex_on() else "%"


def tt(name: str) -> str:
    r"""A method name in typewriter face under usetex, plain under mathtext."""
    return r"\texttt{" + name + "}" if usetex_on() else name


def better(label: str, higher: bool) -> str:
    """Axis label with the better direction (``LL (→)``, on a y label the arrow points up)."""
    arrow = r"$\rightarrow$" if higher else r"$\leftarrow$"
    return f"{label} ({arrow})"


# ------------------------------------------------------------------------------- geometry
PT = 1.0 / 72.0
TEXT_WIDTH_IN = 6.3
BOX_H_PT = 98.3
GAP_PT = 61.5
ROW_SPAN_PT = 396.7
SAVE_PAD_PT = 7.2
GRID_KW = dict(which="major", linestyle="--", linewidth=0.5, color=style.GRID_COLOR, alpha=0.8)
TICK_INSET = 0.06


def box_width_pt(ncols: int, gap_pt: float = GAP_PT) -> float:
    """Width of one axes box when ``ncols`` boxes share the row."""
    return (ROW_SPAN_PT - (ncols - 1) * gap_pt) / max(ncols, 1)


def panel_sheet(nrows: int, ncols: int, *, as_cols: int | None = None, n: int | None = None,
                box_h_pt: float = BOX_H_PT, gap_pt: float = GAP_PT,
                row_gap_pt: float = 46.0, top_pt: float = 16.0, bottom_pt: float = 30.0,
                left_pt: float | None = None, sharex: bool = False, sharey: bool = False,
                box_w_pt: float | None = None) -> tuple[Figure, np.ndarray]:
    """``nrows x ncols`` axes on the thesis panel geometry. Returns ``(fig, axes[nrows, ncols])``.

    The row spans :data:`ROW_SPAN_PT` with a gap of ``gap_pt`` between boxes (box width from
    :func:`box_width_pt` unless ``box_w_pt`` is given). ``as_cols`` sizes the boxes as if the row
    had that many columns and centers the ``ncols`` boxes. ``n`` removes the slots after the first
    ``n`` (row-major), those entries of the returned array are None.
    """
    layout_cols = max(ncols, as_cols or 0)
    bw = box_w_pt if box_w_pt is not None else box_width_pt(layout_cols, gap_pt)
    span = ncols * bw + (ncols - 1) * gap_pt
    fig_w = TEXT_WIDTH_IN / PT
    left0 = left_pt if left_pt is not None else fig_w - SAVE_PAD_PT - ROW_SPAN_PT - 6.0
    left0 += (ROW_SPAN_PT - span) / 2 if span < ROW_SPAN_PT - 1e-6 else 0.0
    fig_h = top_pt + nrows * box_h_pt + (nrows - 1) * row_gap_pt + bottom_pt
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, fig_h * PT))
    axes = np.empty((nrows, ncols), dtype=object)
    first: Axes | None = None
    for r in range(nrows):
        for c in range(ncols):
            if n is not None and r * ncols + c >= n:
                axes[r, c] = None
                continue
            x = left0 + c * (bw + gap_pt)
            y = fig_h - top_pt - (r + 1) * box_h_pt - r * row_gap_pt
            kw: dict[str, Any] = {}
            if first is not None and sharex:
                kw["sharex"] = first
            if first is not None and sharey:
                kw["sharey"] = first
            ax = fig.add_axes((x / fig_w, y / fig_h, bw / fig_w, box_h_pt / fig_h), **kw)
            axes[r, c] = ax
            first = first or ax
    return fig, axes


def grid(ax: Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, **GRID_KW)
    ax.set_axisbelow(True)


def pad_ticks(ax: Axes, which: str = "both", frac: float = TICK_INSET) -> None:
    """Widen the limits so the outermost gridline sits inside the frame (linear axes only)."""
    for name in (("x", "y") if which == "both" else (which,)):
        scale = ax.get_xscale() if name == "x" else ax.get_yscale()
        if scale != "linear":
            continue
        get_lim, set_lim = ((ax.get_xlim, ax.set_xlim) if name == "x"
                            else (ax.get_ylim, ax.set_ylim))
        ticks = ax.get_xticks() if name == "x" else ax.get_yticks()
        lo, hi = get_lim()
        if not lo < hi:
            continue
        inside = [t for t in ticks if lo - 1e-12 <= t <= hi + 1e-12]
        if not inside:
            continue
        pad = frac * (hi - lo)
        set_lim(min(lo, min(inside) - pad), max(hi, max(inside) + pad))


def finish(ax: Axes, *, xlabel: str | None = None, ylabel: str | None = None,
           pad: str = "both", grid_axis: str = "both") -> None:
    """Grid, labels and tick inset of one panel. Call it last."""
    grid(ax, grid_axis)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if pad != "none":
        pad_ticks(ax, pad)


def zoom_x(ax: Axes, xmax: float, *, xmin: float = 0.0, xticks: Sequence[float] | None = None,
           frac: float = 0.08) -> None:
    """Show ``[xmin, xmax]`` and fit y to the visible part of the drawn lines (the boundary
    values interpolated)."""
    ys: list[np.ndarray] = []
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size != y.size or not x.size:
            continue
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if not x.size:
            continue
        inside = (x >= xmin) & (x <= xmax)
        if inside.any():
            ys.append(y[inside])
        order = np.argsort(x, kind="stable")
        for edge in (xmin, xmax):
            if x[order][0] <= edge <= x[order][-1]:
                ys.append(np.array([float(np.interp(edge, x[order], y[order]))]))
    ax.set_xlim(xmin, xmax)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if ys:
        v = np.concatenate(ys)
        lo, hi = float(v.min()), float(v.max())
        span = max(hi - lo, 1e-12)
        ax.set_ylim(lo - frac * span, hi + frac * span)


def legend_above(fig: Figure, handles: list[Any], labels: list[str], *, ncol: int,
                 y_pt: float = 4.0, fontsize: float = 7.0, **kw: Any) -> Any:
    """A frameless legend centered above the top row of axes."""
    top = max(ax.get_position().y1 for ax in fig.axes)
    fig_h_pt = fig.get_size_inches()[1] / PT
    lefts = [ax.get_position().x0 for ax in fig.axes]
    rights = [ax.get_position().x1 for ax in fig.axes]
    xc = 0.5 * (min(lefts) + max(rights))
    opts = dict(loc="lower center", bbox_to_anchor=(xc, top + y_pt / fig_h_pt), ncol=ncol,
                frameon=False, fontsize=fontsize, handlelength=1.6, handletextpad=0.4,
                columnspacing=1.0, labelspacing=0.25, borderaxespad=0.0)
    opts.update(kw)
    return fig.legend(handles, labels, **opts)


def inner_legend(ax: Axes, handles: list[Any] | None = None, labels: list[str] | None = None,
                 **kw: Any) -> Any:
    """A small framed legend inside the axes."""
    opts = dict(frameon=True, fontsize=6, loc="best", handlelength=1.2, handletextpad=0.4,
                columnspacing=0.9, labelspacing=0.3, borderpad=0.3, framealpha=0.85)
    opts.update(kw)
    if handles is None:
        return ax.legend(**opts)
    return ax.legend(handles, labels, **opts)


def log_ticks_decades(ax: Axes, axis: str = "x") -> None:
    """Plain decimal labels (0.001, 0.01, ...) on a log axis."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def fmt(v: float, _: Any) -> str:
        if v <= 0:
            return ""
        e = math.log10(v)
        if abs(e - round(e)) > 1e-6:
            return ""
        if v >= 1:
            return f"{v:.0f}"
        digits = int(round(-e))
        return f"{v:.{digits}f}"

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(LogLocator(base=10, numticks=12))
    target.set_major_formatter(FuncFormatter(fmt))
    target.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------------------- fronts
MARKER_AREA = 130.0
COLLAPSED_MARKER_AREA = 85.0


def draw_glyph(ax: Axes, x: float, y: float, color: str, *, text: str | None = None,
               hollow: bool = False, area: float = MARKER_AREA, theme: str = "white",
               zorder: float = 4.0, fontsize: float = 4.8) -> None:
    """One circular glyph, optionally with a number inside (white on the filled circle)."""
    bg = bg_color(theme)
    ax.scatter([x], [y], s=area, marker="o", facecolor=bg if hollow else color,
               edgecolor=color, linewidths=1.0, zorder=zorder)
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=color if hollow else bg, zorder=zorder + 0.1, fontweight="bold")


def draw_front(ax: Axes, pts: pd.DataFrame, color: str, *, x: str = "seconds",
               y: str = "median", n: str = "n_ctx", numbers: bool = True,
               theme: str = "white", lw: float = 1.3, area: float = MARKER_AREA,
               linestyle: str = "-", glyphs: bool = True, front_only: bool = True) -> pd.DataFrame:
    """Draw the Pareto front of ``pts`` over context sizes as a staircase (a point holds its score
    until the next faster-to-better point) with one glyph per front point carrying its ``n``.
    Returns the front."""
    from esbopfn.benchmarks.metrics import pareto_front

    pts = pts[np.isfinite(pts[x]) & np.isfinite(pts[y])]
    if pts.empty:
        return pts
    front = pareto_front(pts, x=x, y=y) if front_only else pts.sort_values(x)
    xs, ys = front[x].to_numpy(float), front[y].to_numpy(float)
    if len(front) > 1:
        ax.plot(xs, ys, drawstyle="steps-post", color=color, lw=lw, ls=linestyle, zorder=2)
    if glyphs:
        for xv, yv, nv in zip(xs, ys, front[n]):
            draw_glyph(ax, xv, yv, color, text=str(int(nv)) if numbers else None, area=area,
                       theme=theme)
    return front


# ------------------------------------------------------------------------------- provenance
@dataclass
class Provenance:
    """Which DB units and files a figure read."""

    units: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    thesis_import: bool = False
    notes: list[str] = field(default_factory=list)

    def add_units(self, kind: str, units: Iterable[Any]) -> None:
        for u in units:
            self.units.append((kind, u.key_hash, dict(u.key)))
            if u.key.get("provenance") == "thesis":
                self.thesis_import = True

    def add_file(self, path: Path | str) -> None:
        self.files.append(str(path))

    def lines(self) -> list[str]:
        out: list[str] = []
        if self.thesis_import:
            out.append("**source: thesis import** (units imported with `pfn import "
                       "thesis-results`, `provenance: thesis`)")
        by_kind: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for kind, h, key in self.units:
            by_kind.setdefault(kind, []).append((h, key))
        for kind, items in sorted(by_kind.items()):
            prov = sorted({str(k.get("provenance", "native")) for _, k in items})
            out.append(f"`{kind}`: {len(items)} units ({', '.join(prov)}), hashes "
                       + ", ".join(h[:10] for h, _ in items[:6])
                       + (", ..." if len(items) > 6 else ""))
        out += [f"file `{f}`" for f in self.files]
        out += self.notes
        return out


# ------------------------------------------------------------------------------- sidecars
@dataclass
class Sidecar:
    """Content of ``<name>.md``. Every string is built from the numbers the figure draws."""

    thesis_asset: str
    label: str = ""
    shown: list[str] = field(default_factory=list)
    metric: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    data: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    def text(self, name: str, files: Sequence[Path] = ()) -> str:
        lines = [f"# {name}", ""]
        ref = f"`{THESIS_ASSETS}/{self.thesis_asset}`"
        lines += [f"Thesis asset: {ref}" + (f" ({self.label})" if self.label else ""), ""]
        if files:
            lines += ["Files: " + ", ".join(f"`{Path(f).name}`" for f in files), ""]
        sections = (("What is shown", self.shown), ("Metric", self.metric),
                    ("Key numbers", self.numbers), ("Data", self.data),
                    ("Sources", self.provenance.lines()))
        for heading, items in sections:
            if not items:
                continue
            lines.append(f"## {heading}")
            lines += [f"- {s}" for s in items]
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def home_relative(text: str, home: Path | str | None = None) -> str:
    """``text`` with the absolute output root written as ``$PFN_HOME`` (default root:
    :func:`esbopfn.config.pfn_home`), so the sidecars carry no machine-specific paths."""
    from esbopfn.config import pfn_home

    root = Path(home) if home is not None else pfn_home()
    for form in sorted({str(root), str(root.resolve())}, key=len, reverse=True):
        text = text.replace(form, "$PFN_HOME")
    return text


def db_home(db: Any) -> Path | None:
    """The output root of a database at ``<home>/benchmarks/db.sqlite``, else None."""
    path = getattr(db, "path", None)
    if path is None or Path(path).parent.name != "benchmarks":
        return None
    return Path(path).parent.parent


def write_sidecar(out_dir: Path, name: str, sidecar: Sidecar, files: Sequence[Path] = (),
                  home: Path | str | None = None) -> Path:
    path = Path(out_dir) / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(home_relative(sidecar.text(name, files), home), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------- rendering
class MissingData(Exception):
    """The data of a figure is not in the database. ``missing`` names the absent units."""

    def __init__(self, missing: Sequence[str] | str, hint: str = ""):
        self.missing = [missing] if isinstance(missing, str) else list(missing)
        self.hint = hint
        text = "missing " + "; ".join(self.missing)
        super().__init__(text + (f" ({hint})" if hint else ""))


@dataclass
class Context:
    """What a figure function gets."""

    cfg: dict[str, Any]
    db: Any
    out_dir: Path
    fmt: str
    theme: str
    group: str = ""


@dataclass
class Output:
    """A figure function's result: a matplotlib figure and/or text files, plus the sidecar."""

    sidecar: Sidecar
    fig: Figure | None = None
    texts: dict[str, str] = field(default_factory=dict)   # file name (with suffix) -> content


@dataclass(frozen=True)
class FigureSpec:
    name: str
    draw: Callable[[Context], Output]


@dataclass
class RenderReport:
    written: dict[str, list[Path]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)


def _usetex(cfg: dict[str, Any]) -> bool | None:
    st = cfg.get("style") if isinstance(cfg.get("style"), dict) else {}
    value = st.get("usetex", cfg.get("usetex"))
    return None if value is None else bool(value)


def render_group(group: str, figures: Sequence[FigureSpec], cfg: dict[str, Any], db: Any,
                 out_dir: Path, fmt: str, theme: str, only: Iterable[str] | None = None
                 ) -> RenderReport:
    """Render ``figures`` into ``out_dir``. Missing data skips a figure (logged, listed in
    ``skipped.txt``), an error in one figure does not stop the others but is raised at the end."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    home = db_home(db)
    wanted = set(only or cfg.get("only") or [f.name for f in figures])
    report = RenderReport()
    for spec in figures:
        if spec.name not in wanted:
            continue
        ctx = Context(cfg=cfg, db=db, out_dir=out_dir, fmt=fmt, theme=theme, group=group)
        try:
            with style.thesis_style(theme, usetex=_usetex(cfg)):
                result = spec.draw(ctx)
                files: list[Path] = []
                if result.fig is not None:
                    files.append(style.save(result.fig, out_dir / spec.name, fmt))
                    plt.close(result.fig)
            own_md = f"{spec.name}.md"
            for fname, text in result.texts.items():
                if fname == own_md:     # a markdown table is the body of the sidecar
                    continue
                p = out_dir / fname
                p.write_text(text, encoding="utf-8")
                files.append(p)
            side = write_sidecar(out_dir, spec.name, result.sidecar, files, home)
            if own_md in result.texts:
                side.write_text(side.read_text(encoding="utf-8") + "\n## Table\n\n"
                                + result.texts[own_md].strip() + "\n", encoding="utf-8")
            files.append(side)
            report.written[spec.name] = files
            log.info("%s/%s: %s", group, spec.name, ", ".join(p.name for p in files))
        except MissingData as exc:
            plt.close("all")
            report.skipped[spec.name] = str(exc)
            log.warning("%s/%s skipped: %s", group, spec.name, exc)
        except Exception as exc:  # noqa: BLE001 -- the other figures still render
            plt.close("all")
            report.failed[spec.name] = f"{type(exc).__name__}: {exc}"
            log.error("%s/%s failed:\n%s", group, spec.name, traceback.format_exc())
    skipped_file = out_dir / "skipped.txt"
    if report.skipped:
        skipped_file.write_text(home_relative(
            "".join(f"{k}: {v}\n" for k, v in report.skipped.items()), home))
    elif skipped_file.exists():
        skipped_file.unlink()
    if report.failed:
        raise RuntimeError(f"plot group {group}: {len(report.failed)} figures failed: "
                           + "; ".join(f"{k} ({v})" for k, v in report.failed.items()))
    return report


def make_render(group: str, figures: Sequence[FigureSpec]
                ) -> Callable[[dict[str, Any], Any, Path, str, str], None]:
    """``PlotGroup.render`` for a list of figures."""

    def render(cfg: dict[str, Any], db: Any, out_dir: Path, fmt: str, theme: str) -> None:
        render_group(group, figures, cfg or {}, db, out_dir, fmt, theme)

    return render


def spec_needs(group: str) -> Callable[[dict[str, Any], Any], list[tuple[str, dict[str, Any]]]]:
    """``PlotGroup.needs`` from the group's benchmark spec."""

    def needs(cfg: dict[str, Any], db: Any) -> list[tuple[str, dict[str, Any]]]:
        from esbopfn.benchmarks.groups import load, units

        spec = dict(cfg) if cfg and cfg.get("kinds") is not None else load(group)
        spec.setdefault("name", group)
        return units(spec)

    return needs


# ------------------------------------------------------------------------------- data
@dataclass
class SeriesBlob:
    """One ``thesis_series`` unit."""

    figure: str
    problem: str
    series: str
    variant: str
    frame: pd.DataFrame
    meta: dict[str, Any]
    source: str
    unit: Any


def thesis_series(db: Any, figure: str, *, variant: str = "thesis", problem: str | None = None,
                  series: str | None = None, prov: Provenance | None = None) -> list[SeriesBlob]:
    """Imported aggregates of one thesis figure (``thesis_series`` units), possibly empty."""
    import json

    filters: dict[str, Any] = {"figure": figure, "variant": variant}
    if problem is not None:
        filters["problem"] = problem
    if series is not None:
        filters["series"] = series
    units = db.query("thesis_series", **filters)
    out = []
    for u in units:
        blob = json.loads((Path(u.blob_dir) / "series.json").read_text())
        data = blob.get("data", {})
        frame = pd.DataFrame({c: data.get(c, []) for c in blob.get("columns", list(data))})
        out.append(SeriesBlob(figure, u.key["problem"], u.key["series"], u.key["variant"],
                              frame, blob.get("meta", {}), str(blob.get("source", "")), u))
    if prov is not None and units:
        prov.add_units("thesis_series", units)
    return out


def thesis_model_id(model: str) -> str:
    """The imported id of a model template: cross-trajectory ``default/default`` weights are
    ``thesis/default`` in the thesis import."""
    fam = model_family(model)
    if fam in ("xdecpfn", "ftpfn"):
        return model.replace("/default/default", "/thesis/default")
    return model


def is_thesis_unit(unit: Any) -> bool:
    """A unit written by the thesis importers (``provenance: thesis``)."""
    return unit.key.get("provenance") == "thesis"


def thesis_tasks(problem: str, n_ctx: int | None = None) -> pd.DataFrame | None:
    """``task_id, crashed, crash_step, n_ctx`` of the thesis task table (all tasks, or those at
    ``n_ctx``): the stop-curve denominator of the imported thesis predictions. As in the thesis
    (``predplay.aggregate.stop_curve``) every task counts, a query without rows (a crash before
    the first predictable step, or a run an old store did not score) as never stopped."""
    from esbopfn.benchmarks.tasks import load_tasks

    loaded = load_tasks(problem, "tasks-thesis")
    if loaded is None:
        return None
    t = loaded[0]
    if n_ctx is not None:
        t = t[t.n_ctx == n_ctx]
    return t[["task_id", "crashed", "crash_step", "n_ctx"]].reset_index(drop=True)


def prediction_frame(db: Any, kind: str, models: Iterable[str], problems: Iterable[str],
                     prov: Provenance | None = None, native_only: bool = False,
                     thesis_raw: bool = False) -> pd.DataFrame:
    """Rows of the finished ``prediction`` / ``prediction_ladder`` units of the given model
    templates (``{p}`` replaced) on ``problems``, with ``problem`` and ``model`` columns. A model
    template is also matched by its thesis ids (:func:`thesis_model_id`, then the imported
    checkpoint ids of :func:`esbopfn.training.registry.thesis_ids`) unless ``native_only``. The
    ``model`` column holds the template's resolved id either way. ``native_only`` also leaves out
    the imported thesis predictions (``provenance: thesis``, every task the old store scored),
    ``thesis_raw`` reads only those."""
    from esbopfn.benchmarks.kinds.prediction import load_predictions, resolve_model
    from esbopfn.training.registry import thesis_ids

    frames = []
    used: list[Any] = []
    for p in problems:
        for tmpl in models:
            mid = resolve_model(tmpl, p)
            units = db.query(kind, problem=p, model=mid)
            if native_only:
                units = [u for u in units if not is_thesis_unit(u)]
            if thesis_raw:
                units = [u for u in units if is_thesis_unit(u)]
            if not units and not native_only:
                for tid in dict.fromkeys([thesis_model_id(mid), *thesis_ids(mid)]):
                    units = db.query(kind, problem=p, model=tid) if tid != mid else []
                    if thesis_raw:
                        units = [u for u in units if is_thesis_unit(u)]
                    if units:
                        break
            if not units:
                continue
            df = load_predictions(units)
            df["model"] = mid
            frames.append(df)
            used += units
    if prov is not None:
        prov.add_units(kind, used)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def timing_table(db: Any, prov: Provenance | None = None) -> pd.DataFrame:
    """Median inference seconds per ``(problem, model, n_ctx)`` from ``prediction_timing`` units.

    Native units come first. Thesis records are read under their thesis id (``.../thesis/default``
    and the ``/prezscore`` LC-PFN records under the plain LC-PFN id) where no native unit exists.
    Columns ``problem, model, n_ctx, seconds, provenance``.
    """
    import json

    rows = []
    units = db.query("prediction_timing")
    for u in units:
        try:
            sec = json.loads((Path(u.blob_dir) / "timing.json").read_text())["seconds_median"]
        except (OSError, KeyError, ValueError):
            continue
        model = str(u.key["model"])
        prov_tag = str(u.key.get("provenance", "native"))
        if prov_tag == "thesis":
            model = model.replace("/thesis/default", "/default/default")
            if model.endswith("/prezscore"):
                model = model[: -len("/prezscore")]
        rows.append({"problem": u.key["problem"], "model": model, "n_ctx": int(u.key["n_ctx"]),
                     "seconds": float(sec), "provenance": prov_tag, "unit": u})
    if not rows:
        return pd.DataFrame(columns=["problem", "model", "n_ctx", "seconds", "provenance"])
    df = pd.DataFrame(rows)
    df["rank"] = (df.provenance != "native").astype(int)
    df = df.sort_values("rank").drop_duplicates(["problem", "model", "n_ctx"], keep="first")
    if prov is not None:
        prov.add_units("prediction_timing", list(df.unit))
    return df.drop(columns=["rank", "unit"]).reset_index(drop=True)


def seconds_for(times: pd.DataFrame, model: str, problem: str | None, n_ctx: int
                ) -> float:
    """Median seconds of ``model`` (resolved id, or a ``{p}`` template with ``problem=None`` for
    the cross-problem median) at ``n_ctx``. Context-blind models use their ``n = 0`` time. NaN if
    nothing is recorded."""
    from esbopfn.benchmarks.metrics import is_context_blind

    n = 0 if is_context_blind(model) else int(n_ctx)
    if problem is None:
        vals = []
        for p in PROBLEMS:
            v = seconds_for(times, model.replace("{p}", p), p, n)
            if np.isfinite(v):
                vals.append(v)
        return float(np.median(vals)) if vals else float("nan")
    sel = times[(times.problem == problem) & (times.model == model) & (times.n_ctx == n)]
    if sel.empty and is_context_blind(model):
        sel = times[(times.problem == problem) & (times.model == model)]
    return float(sel.seconds.iloc[0]) if len(sel) else float("nan")


def missing_units_message(kind: str, what: Iterable[str], group: str) -> str:
    items = list(what)
    head = ", ".join(items[:4]) + (f" (+{len(items) - 4} more)" if len(items) > 4 else "")
    return f"{kind} units of {head}: run `pfn bench run {group}`"


def fmt_num(x: float, digits: int = 3) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"
