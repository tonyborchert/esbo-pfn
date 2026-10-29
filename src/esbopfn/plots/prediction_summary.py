"""Plot group ``prediction_summary``: the representative predictors of every group.

- ``fig3_cross`` (fig:sim:fig3-cross): median normalized log-likelihood pooled over the five
  problems against the median inference time per prediction. Context-blind models are single
  points (DecPFN-no-St hollow), context-aware models a Pareto front over the context sizes with
  the size written in the glyph.
- ``fig3_sheet`` (fig:app:fig3-sheet): the same per problem plus the pooled panel, 3 x 2, lines
  and points only, one legend above.
- ``fig2_sheet`` (fig:sim:fig2-sheet): mean steps saved on crashed queries against the false-stop
  share on non-crashed queries (window 0 to 0.1), XDecPFN and FT-PFN at ``n = 0`` and ``64``, the
  context-blind models pooled over ``n``.

Data: native ``prediction`` units of the spec's models first (normalization ``med_D`` over the
spec's ``participants``). Without them the thesis import is used (``thesis_series`` variant
``thesis``: ``fig3_cross``, ``fig3_per_problem`` minus ``fig3_med_D``, ``fig2_stop_curves``).
Times come from ``prediction_timing`` units. The fig3 machinery is shared with
:mod:`esbopfn.plots.tabpfn`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from esbopfn.benchmarks import metrics as M
from esbopfn.plots import PlotGroup
from esbopfn.plots import common as C

GROUP_NAME = "prediction_summary"
ASSET_DIR = "simulation/prediction/at_the_end"
POOLED = "pooled"
YLABEL = "median norm. LL"
#: y range per panel (the thesis fixes them by hand).
PANEL_YLIM: dict[str, tuple[float, float]] = {
    POOLED: (-1.0, 2.0), "boiler": (-1.0, 2.0), "threetank_pi": (-1.0, 2.0),
    "pt2_deadtime": (-2.0, 4.0), "cartpole": (-1.0, 2.0), "furuta": (-1.0, 3.0),
}
SHEET_PANELS: tuple[str, ...] = (POOLED, "boiler", "threetank_pi", "pt2_deadtime", "cartpole",
                                 "furuta")
DEFAULT_PARTICIPANTS: tuple[str, ...] = (
    "decpfn/{p}/binned/1M", "decpfn-nostates/{p}/binned/1M", "lcpfn/{p}/binned/1M",
    "ftpfn/{p}/default/default", "xdecpfn/{p}/default/default", "tabpfn/50",
    "segment-tabpfn/split", "segment-gp",
)
#: Legend / draw order of the fig3 series (thesis order).
SERIES_ORDER: tuple[str, ...] = DEFAULT_PARTICIPANTS
#: Direct-label placement on fig3_cross: (dx pt, dy pt, ha, va) relative to the top visible point.
LABEL_OFFSETS: dict[str, tuple[float, float, str, str]] = {
    "decpfn": (-8, 0, "right", "center"),
    "decpfn-nostates": (-7, -13, "left", "top"),
    "lcpfn": (8, 0, "left", "center"),
    "ftpfn": (-8, 0, "right", "center"),
    "xdecpfn": (9, 0, "left", "center"),
    "tabpfn": (-4, 9, "right", "bottom"),
    "segment-tabpfn": (-4, 9, "right", "bottom"),
    "segment-gp": (-2, -16, "center", "top"),
}


# ================================================================================ fig3 data
def participants(cfg: dict[str, Any]) -> list[str]:
    spec = cfg.get("participants") or list(DEFAULT_PARTICIPANTS)
    if isinstance(spec, dict):
        spec = spec.get("default", list(DEFAULT_PARTICIPANTS))
    return list(spec)


def problems_of(cfg: dict[str, Any]) -> list[str]:
    return list(cfg.get("problems") or C.PROBLEMS)


def template_of(model: str, problem: str | None = None) -> str:
    """Problem-agnostic id with the cross-trajectory thesis ids mapped to the spec's ids."""
    tmpl = M.series_label(model, problem) if problem else model
    if C.model_family(tmpl) in ("xdecpfn", "ftpfn"):
        tmpl = tmpl.replace("/thesis/default", "/default/default")
    return tmpl


@dataclass
class Board:
    """Normalized medians per (panel, series, n_ctx) with their inference seconds."""

    points: pd.DataFrame          # panel, series, n_ctx, y, seconds
    source: str                   # native | thesis
    prov: C.Provenance
    med: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def series(self, panel: str, series: str) -> pd.DataFrame:
        p = self.points
        return p[(p.panel == panel) & (p.series == series)].sort_values("n_ctx")


def _native_scores(db: Any, cfg: dict[str, Any], templates: list[str], prov: C.Provenance
                   ) -> tuple[pd.DataFrame, list[str]] | None:
    """Per-trial scores of the native units, or None when the participants are incomplete."""
    probs = problems_of(cfg)
    parts = participants(cfg)
    wanted = list(dict.fromkeys(parts + templates))
    rows = C.prediction_frame(db, "prediction", wanted, probs, native_only=True)
    if rows.empty:
        return None
    have = {(p, m) for p, m in zip(rows.problem, rows.model)}
    missing = [f"{t.replace('{p}', p)}" for p in probs for t in parts
               if (p, t.replace("{p}", p)) not in have]
    if missing:
        return None
    prov.add_units("prediction", [
        u for p in probs for t in wanted
        for u in db.query("prediction", problem=p, model=t.replace("{p}", p))])
    scores = M.collapse_context_blind(M.trial_scores(rows))
    return scores, missing


def fig3_board(ctx: C.Context, templates: list[str]) -> Board:
    """Pooled and per-problem normalized medians of ``templates`` (native, else thesis import).

    Raises :class:`common.MissingData` when neither exists."""
    db, cfg = ctx.db, ctx.cfg
    prov = C.Provenance()
    times = C.timing_table(db, prov)
    probs = problems_of(cfg)
    native = _native_scores(db, cfg, templates, prov)
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    if native is not None:
        scores, _ = native
        part_ids = [t.replace("{p}", p) for p in probs for t in participants(cfg)]
        med = M.med_d(scores, part_ids)
        normed = M.ll_norm(scores, med)
        per = (normed.groupby(["problem", "model", "n_ctx"]).ll_norm.median().reset_index())
        for r in per.itertuples(index=False):
            rows.append(dict(panel=r.problem, series=template_of(r.model, r.problem),
                             n_ctx=int(r.n_ctx), y=float(r.ll_norm),
                             seconds=C.seconds_for(times, r.model, r.problem, int(r.n_ctx))))
        for r in M.pooled_median(normed).itertuples(index=False):
            rows.append(dict(panel=POOLED, series=template_of(r.series), n_ctx=int(r.n_ctx),
                             y=float(r.median),
                             seconds=C.seconds_for(times, r.series, None, int(r.n_ctx))))
        source = "native"
        med_map = {str(k): float(v) for k, v in med.items()}
        notes.append("med_D: median of every per-trial score of the spec's participants "
                     f"({', '.join(participants(cfg))}) per problem")
    else:
        cross = C.thesis_series(db, "fig3_cross", prov=prov)
        per = C.thesis_series(db, "fig3_per_problem", prov=prov)
        medb = C.thesis_series(db, "fig3_med_D", prov=prov)
        if not cross:
            raise C.MissingData(
                [C.missing_units_message("prediction", templates, ctx.cfg.get("name", GROUP_NAME)),
                 "or thesis_series fig3_cross (pfn import thesis-results)"])
        med_map = ({str(p): float(v) for p, v in zip(medb[0].frame.problem, medb[0].frame.med_D)}
                   if medb else {})
        for s in cross:
            tmpl = template_of(s.series)
            for n, y in zip(s.frame.n_ctx, s.frame.y):
                rows.append(dict(panel=POOLED, series=tmpl, n_ctx=int(n), y=float(y),
                                 seconds=C.seconds_for(times, tmpl, None, int(n))))
        for s in per:
            if s.problem not in med_map:
                continue
            tmpl = template_of(s.series, s.problem)
            resolved = tmpl.replace("{p}", s.problem)
            for n, y in zip(s.frame.n_ctx, s.frame.y):
                rows.append(dict(panel=s.problem, series=tmpl, n_ctx=int(n),
                                 y=float(y) - med_map[s.problem],
                                 seconds=C.seconds_for(times, resolved, s.problem, int(n))))
        source = "thesis"
        notes.append("thesis import, variant `thesis` (the numbers sim_results.tex quotes): pooled "
                     "values are `fig3_cross` as imported (already normalized), per-problem values "
                     "are `fig3_per_problem` minus the imported `fig3_med_D` (median over the "
                     "thesis's full participant set of 38 old models, not the spec's)")
    pts = pd.DataFrame(rows, columns=["panel", "series", "n_ctx", "y", "seconds"])
    pts = pts[pts.series.isin(templates)].reset_index(drop=True)
    filled = []
    for i in np.flatnonzero(~np.isfinite(pts.seconds.to_numpy(float))):
        r = pts.iloc[i]
        if r.panel == POOLED:
            continue
        sec = C.seconds_for(times, r.series, None, int(r.n_ctx))
        if np.isfinite(sec):
            pts.loc[i, "seconds"] = sec
            filled.append(f"{r.panel}/{C.model_label(r.series)}"
                          + ("" if int(r.n_ctx) == M.COLLAPSED_N else f" n={int(r.n_ctx)}"))
    if filled:
        notes.append("no timing record on the problem, the cross-problem median time of the same "
                     "model and context size is used: " + ", ".join(sorted(filled)))
    no_time = pts[~np.isfinite(pts.seconds)]
    if len(no_time):
        notes.append("no inference time recorded (not drawn): " + ", ".join(sorted(
            {f"{a}/{C.model_label(b)} n={c}" for a, b, c in
             zip(no_time.panel, no_time.series, no_time.n_ctx)}))[:600])
    if not prov.thesis_import and source == "native" and times.empty:
        notes.append("no prediction_timing units")
    return Board(points=pts.reset_index(drop=True), source=source, prov=prov, med=med_map,
                 notes=notes)


def fronts(board: Board, panel: str, templates: list[str], floors: dict[str, int] | None = None
           ) -> dict[str, pd.DataFrame]:
    """Pareto front per series on one panel (context-blind series: their pooled point)."""
    out: dict[str, pd.DataFrame] = {}
    for tmpl in templates:
        pts = board.series(panel, tmpl)
        pts = pts[np.isfinite(pts.seconds) & np.isfinite(pts.y)]
        if floors:
            pts = pts[(pts.n_ctx == M.COLLAPSED_N) | (pts.n_ctx >= floors.get(tmpl, 0))]
        if pts.empty:
            continue
        out[tmpl] = M.pareto_front(pts.rename(columns={"y": "median"}), x="seconds",
                                   y="median").rename(columns={"median": "y"})
    return out


def x_limits(fr: dict[str, pd.DataFrame], ylim: tuple[float, float], decades: bool = False,
             per_series: bool = False) -> tuple[float, float]:
    """``(min / 2, max * 2)`` over the visible points (``per_series``: over every point of a
    series that is visible anywhere). ``decades`` widens to the next decade tick on both ends."""
    if per_series:
        xs = [x for f in fr.values() if ((f.y >= ylim[0]) & (f.y <= ylim[1])).any()
              for x in f.seconds]
    else:
        xs = [x for f in fr.values() for x, y in zip(f.seconds, f.y) if ylim[0] <= y <= ylim[1]]
    if not xs:
        xs = [x for f in fr.values() for x in f.seconds]
    lo, hi = min(xs) / 2, max(xs) * 2
    if decades:
        lo = min(lo, 0.5 * 10 ** np.floor(np.log10(lo)))
        hi = max(hi, 2.0 * 10 ** np.ceil(np.log10(hi)))
    else:
        hi = hi * (hi / lo) ** 0.04
    return lo, hi


def log_axis(ax: Axes, axis: str = "x", mult: tuple[int, ...] | None = None) -> None:
    """Numbered log ticks only (no minor ticks): 1-2-5 per decade on axes spanning under 1.5
    decades, whole decades otherwise, plain decimals down to 0.0001 and powers of ten below."""
    import math

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    lo, hi = sorted(ax.get_xlim() if axis == "x" else ax.get_ylim())
    span = math.log10(hi / lo)
    mult = mult or ((1, 2, 5) if span < 1.5 else (1,))
    ticks = [m * 10.0 ** e for e in range(math.floor(math.log10(lo)) - 1,
                                          math.ceil(math.log10(hi)) + 1)
             for m in mult if lo <= m * 10.0 ** e <= hi]

    def fmt(v: float, _: Any = None) -> str:
        e = math.floor(math.log10(v) + 1e-9)
        if e < -4:
            return f"$10^{{{e}}}$"
        return f"{v:.10f}".rstrip("0").rstrip(".") if e < 0 else f"{v:g}"

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(fmt))
    target.set_minor_locator(NullLocator())


def bold(text: str) -> str:
    return r"\textbf{" + text + "}" if C.usetex_on() else text


def spines_on_top(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_zorder(10)


def is_blind(tmpl: str) -> bool:
    return M.is_context_blind(tmpl)


def is_hollow(tmpl: str) -> bool:
    return C.model_family(tmpl) == "decpfn-nostates"


def front_lines(ax: Axes, fr: dict[str, pd.DataFrame], theme: str, *, lw: float = 1.4,
                styles: dict[str, dict[str, Any]] | None = None) -> None:
    for tmpl, f in fr.items():
        if is_blind(tmpl) or len(f) < 2:
            continue
        kw = dict(color=C.model_color(tmpl, theme), lw=lw, alpha=0.9, zorder=2)
        kw.update((styles or {}).get(tmpl, {}))
        ax.plot(f.seconds.to_numpy(), f.y.to_numpy(), drawstyle="steps-post", **kw)


def summary_lines(board: Board, panel: str, fr: dict[str, pd.DataFrame]) -> list[str]:
    """One line per drawn series: every context size with score and seconds, points off the
    Pareto front (or below the display floor) marked."""
    out = []
    for tmpl, f in fr.items():
        on = set(int(n) for n in f.n_ctx)
        allp = board.series(panel, tmpl)
        pts = ", ".join(("pooled" if int(n) == M.COLLAPSED_N else f"n={int(n)}")
                        + f" {y:.3f} @ {s:.4g} s" + ("" if int(n) in on else " (not drawn)")
                        for n, y, s in zip(allp.n_ctx, allp.y, allp.seconds))
        out.append(f"{C.problem_label(panel)}, {C.model_label(tmpl)} ({tmpl}): {pts}")
    return out


# ================================================================================ fig3_cross
def _glyphs(ax: Axes, fr: dict[str, pd.DataFrame], theme: str, ylim: tuple[float, float]
            ) -> list[tuple[str, int]]:
    """Numbered glyphs of every front point. A glyph covered by a higher-context one is not
    drawn, except the top point of its series. Returns the dropped (series, n)."""
    cands = []
    for tmpl, f in fr.items():
        top = f.y.idxmax()
        for idx, (n, x, y) in zip(f.index, zip(f.n_ctx, f.seconds, f.y)):
            if not ylim[0] <= y <= ylim[1]:
                continue
            cands.append(dict(tmpl=tmpl, n=int(n), x=float(x), y=float(y), top=idx == top))
    trans = ax.transData
    radius = (C.MARKER_AREA / np.pi) ** 0.5 * ax.figure.dpi / 72.0
    for c in cands:
        c["px"] = trans.transform((c["x"], c["y"]))
    order = sorted(cands, key=lambda c: (not c["top"], -c["n"]))
    kept, dropped = [], []
    for c in order:
        if not c["top"] and any(np.hypot(*(c["px"] - k["px"])) < 1.8 * radius for k in kept):
            dropped.append((c["tmpl"], c["n"]))
            continue
        kept.append(c)
    for c in kept:
        blind = c["n"] == M.COLLAPSED_N
        C.draw_glyph(ax, c["x"], c["y"], C.model_color(c["tmpl"], theme),
                     text=None if blind else bold(str(c["n"])), hollow=is_hollow(c["tmpl"]),
                     area=C.COLLAPSED_MARKER_AREA if blind else C.MARKER_AREA, theme=theme,
                     zorder=3 + (0 if blind else c["n"] / 1000), fontsize=5.5)
    return dropped


def _direct_labels(ax: Axes, fr: dict[str, pd.DataFrame], theme: str,
                   ylim: tuple[float, float]) -> None:
    for tmpl, f in fr.items():
        vis = f[(f.y >= ylim[0]) & (f.y <= ylim[1])]
        if vis.empty:
            continue
        top = vis.loc[vis.y.idxmax()]
        dx, dy, ha, va = LABEL_OFFSETS.get(C.model_family(tmpl), (8, 0, "left", "center"))
        ax.annotate(bold(C.model_label(tmpl)), (top.seconds, top.y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=7.5, fontweight="bold",
                    color=C.model_color(tmpl, theme), zorder=6, annotation_clip=False)


def draw_fig3_cross(ctx: C.Context) -> C.Output:
    templates = participants(ctx.cfg)
    board = fig3_board(ctx, templates)
    ylim = PANEL_YLIM[POOLED]
    fr = fronts(board, POOLED, templates)
    if not fr:
        raise C.MissingData("pooled scores with inference times")
    fig, axes = C.panel_sheet(1, 1)
    ax = axes[0, 0]
    ax.set_xscale("log")
    ax.set_xlim(*x_limits(fr, ylim, decades=True))
    ax.set_ylim(*ylim)
    log_axis(ax)
    front_lines(ax, fr, ctx.theme)
    dropped = _glyphs(ax, fr, ctx.theme, ylim)
    _direct_labels(ax, fr, ctx.theme, ylim)
    C.finish(ax, xlabel=C.better("inference time / s", higher=False),
             ylabel=C.better(YLABEL, higher=True), pad="none")
    ax.set_yticks([-1, 0, 1, 2])
    spines_on_top(ax)
    sc = _fig3_sidecar(board, "fig3_cross.pdf", "fig:sim:fig3-cross", [POOLED], fr={POOLED: fr})
    if dropped:
        sc.shown.append("glyphs hidden behind a higher-context glyph (line still drawn): "
                        + ", ".join(f"{C.model_label(t)} n={n}" for t, n in dropped))
    return C.Output(sc, fig=fig)


def _fig3_sidecar(board: Board, asset: str, label: str, panels: list[str],
                  fr: dict[str, dict[str, pd.DataFrame]]) -> C.Sidecar:
    sc = C.Sidecar(f"{ASSET_DIR}/{asset}", label=label)
    sc.shown = [
        "one glyph per (model, context size) on the Pareto front over the context sizes "
        "(staircase line, steps-post), context-blind models (DecPFN, DecPFN-no-St hollow, LC-PFN) "
        "as one point pooled over every context size",
        "series: " + ", ".join(f"{C.model_label(t)} = `{t}`" for t in SERIES_ORDER
                               if any(t in f for f in fr.values())),
        "y ranges: " + ", ".join(f"{C.problem_label(p)} [{PANEL_YLIM[p][0]:g}, "
                                 f"{PANEL_YLIM[p][1]:g}]" for p in panels),
    ]
    sc.metric = [
        "y: median over non-crashed queries of the per-trial conditioned LL (mean over the five "
        "cutoffs 10..90 % of T) minus the per-problem median med_D of the participants, pooled "
        "across problems on the pooled panel (higher is better)",
        "x: median inference seconds per prediction (log), the pooled panel takes the median "
        "across problems, context-blind models use their n = 0 time (lower is better)",
    ]
    for p in panels:
        sc.numbers += summary_lines(board, p, fr[p])
    if board.med:
        sc.numbers.append("med_D: " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(board.med.items())))
    sc.data = [f"source: {board.source}"] + board.notes
    sc.provenance = board.prov
    return sc


# ================================================================================ fig3_sheet
def sheet_dot(ax: Axes, x: float, y: float, tmpl: str, theme: str) -> None:
    color = C.model_color(tmpl, theme)
    hollow = is_hollow(tmpl)
    ax.scatter([x], [y], s=16 if hollow else 22, facecolor=C.bg_color(theme) if hollow else color,
               edgecolor=color, linewidths=1.0 if hollow else 0.5, zorder=3)


def legend_handles(templates: list[str], theme: str) -> tuple[list[Any], list[str]]:
    handles, labels = [], []
    for tmpl in templates:
        color = C.model_color(tmpl, theme)
        if is_blind(tmpl):
            handles.append(Line2D([], [], lw=0, marker="o", markersize=5,
                                  markerfacecolor=C.bg_color(theme) if is_hollow(tmpl) else color,
                                  markeredgecolor=color, markeredgewidth=1.0))
        else:
            handles.append(Line2D([], [], color=color, lw=1.4))
        labels.append(C.model_label(tmpl))
    return handles, labels


def draw_fig3_sheet(ctx: C.Context) -> C.Output:
    templates = participants(ctx.cfg)
    board = fig3_board(ctx, templates)
    panels = [p for p in SHEET_PANELS if p == POOLED or p in problems_of(ctx.cfg)]
    fig, axes = C.panel_sheet(3, 2, n=len(panels), row_gap_pt=40.0, top_pt=24.0)
    flat = [a for a in axes.flat if a is not None]
    all_fr: dict[str, dict[str, pd.DataFrame]] = {}
    drawn: set[str] = set()
    for i, (ax, panel) in enumerate(zip(flat, panels)):
        ylim = PANEL_YLIM.get(panel, (-1.0, 3.0))
        fr = fronts(board, panel, templates)
        all_fr[panel] = fr
        drawn |= set(fr)
        ax.set_xscale("log")
        if fr:
            ax.set_xlim(*x_limits(fr, ylim, per_series=True))
        ax.set_ylim(*ylim)
        log_axis(ax)
        spines_on_top(ax)
        front_lines(ax, fr, ctx.theme)
        for tmpl, f in fr.items():
            if is_blind(tmpl):
                for x, y in zip(f.seconds, f.y):
                    sheet_dot(ax, x, y, tmpl, ctx.theme)
            elif len(f) == 1:
                ax.plot(f.seconds, f.y, "o", ms=2.5, color=C.model_color(tmpl, ctx.theme))
        ax.set_title(C.problem_label(panel), fontsize=7.5)
        last_row = i >= len(panels) - 2
        C.finish(ax, xlabel=C.better("inference time / s", higher=False) if last_row else None,
                 ylabel=C.better(YLABEL, higher=True) if i % 2 == 0 else None, pad="none")
    if not drawn:
        raise C.MissingData("fig3 scores with inference times")
    order = [t for t in SERIES_ORDER if t in drawn] + [t for t in templates
                                                       if t in drawn and t not in SERIES_ORDER]
    h, lab = legend_handles(order, ctx.theme)
    C.legend_above(fig, h, lab, ncol=len(lab), y_pt=14.0, fontsize=6.5, columnspacing=0.9)
    sc = _fig3_sidecar(board, "fig3_sheet.pdf", "fig:app:fig3-sheet", panels, all_fr)
    return C.Output(sc, fig=fig)


# ================================================================================ fig2_sheet
#: alpha of the shown context sizes of XDecPFN and FT-PFN (n = 0 faint).
SHOWN_ALPHA = {0: 0.40, 64: 0.95}


@dataclass
class StopCurve:
    problem: str
    model: str            # template
    n_ctx: int            # -1 pooled
    curve: pd.DataFrame   # false_stop_rate, mean_steps_saved


def fig2_curves(ctx: C.Context, problems: list[str], aware: list[str], ns: list[int],
                pooled: list[str], prov: C.Provenance, notes: list[str]
                ) -> tuple[list[StopCurve], str]:
    """Stop curves per (problem, model, n): native first, else the thesis import."""
    db = ctx.db
    out: list[StopCurve] = []
    rows = C.prediction_frame(db, "prediction", aware + pooled, problems, prov=prov,
                              native_only=True)
    if not rows.empty:
        for p in problems:
            for tmpl in aware + pooled:
                mid = tmpl.replace("{p}", p)
                sel = rows[(rows.problem == p) & (rows.model == mid)]
                if sel.empty:
                    continue
                groups = [(n, sel[sel.n_ctx == n]) for n in ns] if tmpl in aware \
                    else [(M.COLLAPSED_N, sel)]
                for n, g in groups:
                    if g.empty:
                        continue
                    g = g.assign(task_id=g.task_id.astype(str) + "@" + g.n_ctx.astype(str))
                    out.append(StopCurve(p, tmpl, int(n), M.stop_curve(g)))
        return out, "native"
    series = C.thesis_series(db, "fig2_stop_curves", prov=prov)
    if not series:
        raise C.MissingData([C.missing_units_message("prediction", aware + pooled, GROUP_NAME),
                             "or thesis_series fig2_stop_curves (pfn import thesis-results)"])
    for s in series:
        if s.problem not in problems:
            continue
        tmpl = template_of(s.series, s.problem)
        if tmpl not in aware + pooled:
            continue
        for n, g in s.frame.groupby("n_ctx"):
            if tmpl in aware and int(n) not in ns:
                continue
            out.append(StopCurve(s.problem, tmpl, int(n), g.reset_index(drop=True)))
    have = {c.model for c in out}
    lacking = [t for t in aware + pooled if t not in have]
    if lacking:
        notes.append("not in the thesis import, not drawn: "
                     + ", ".join(f"{C.model_label(t)} (`{t}`)" for t in lacking)
                     + " (the thesis curve came from cluster prediction files)")
    return out, "thesis"


def draw_fig2_sheet(ctx: C.Context) -> C.Output:
    fcfg = (ctx.cfg.get("figures") or {}).get("fig2_sheet") or {}
    problems = list(fcfg.get("problems") or C.CRASH_PROBLEMS)
    ca = fcfg.get("context_aware") or {}
    aware = list(ca.get("models") or ["xdecpfn/{p}/default/default", "ftpfn/{p}/default/default"])
    ns = [int(n) for n in ca.get("n", [0, 64])]
    pooled = list(fcfg.get("pooled") or ["decpfn/{p}/binned/1M", "decpfn-nostates/{p}/binned/1M",
                                         "lcpfn/{p}/binned/1M"])
    xmax = float(fcfg.get("max_false_stop", 0.1))
    prov, notes = C.Provenance(), []
    curves, source = fig2_curves(ctx, problems, aware, ns, pooled, prov, notes)
    if not curves:
        raise C.MissingData("stop curves of " + ", ".join(aware + pooled))
    fig, axes = C.panel_sheet(1, len(problems), top_pt=26.0)
    entries: dict[str, tuple[Any, int]] = {}
    numbers = []
    order = pooled + [f"{a}@{n}" for a in aware for n in ns]
    for j, (ax, p) in enumerate(zip(axes[0], problems)):
        for c in sorted(curves, key=lambda c: order.index(c.model if c.model in pooled
                                                          else f"{c.model}@{c.n_ctx}")):
            if c.problem != p:
                continue
            color = C.model_color(c.model, ctx.theme)
            ls = "--" if is_hollow(c.model) else "-"
            if c.model in pooled:
                alpha, label = 0.95, C.model_label(c.model)
            else:
                alpha, label = SHOWN_ALPHA.get(c.n_ctx, 0.95), f"{C.model_label(c.model)} $N={c.n_ctx}$"
            cv = c.curve.sort_values(["false_stop_rate", "mean_steps_saved"])
            ax.plot(cv.false_stop_rate.to_numpy(float), cv.mean_steps_saved.to_numpy(float),
                    color=color, ls=ls, lw=1.3, alpha=alpha, zorder=3 if c.model in pooled else 2)
            key = c.model if c.model in pooled else f"{c.model}@{c.n_ctx}"
            entries.setdefault(label, (Line2D([], [], color=color, ls=ls, lw=1.3, alpha=alpha),
                                       order.index(key)))
            saved = M.steps_saved_at(c.curve, xmax)
            numbers.append(f"{C.problem_label(p)}, {label.replace('$', '')}: {saved:.1f} steps "
                           f"saved at false-stop share <= {xmax:g}")
        C.zoom_x(ax, xmax, xticks=(0.0, xmax / 2, xmax))
        ax.set_title(C.problem_label(p), fontsize=7.5)
        C.finish(ax, xlabel=C.better("false-stop rate", higher=False),
                 ylabel=C.better("mean steps saved", higher=True) if j == 0 else None, pad="both")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    items = sorted(entries.items(), key=lambda kv: kv[1][1])
    C.legend_above(fig, [v[0] for _, v in items], [k for k, _ in items], ncol=len(items),
                   y_pt=16.0, fontsize=6.5, columnspacing=0.9)
    sc = C.Sidecar(f"{ASSET_DIR}/fig2_sheet.pdf", label="fig:sim:fig2-sheet")
    sc.shown = [f"panels: {', '.join(C.problem_label(p) for p in problems)}",
                "XDecPFN and FT-PFN at " + ", ".join(f"n = {n}" for n in ns)
                + " (n = 0 faint), DecPFN, DecPFN-no-St (dashed) and LC-PFN pooled over n",
                "models: " + ", ".join(f"`{t}`" for t in pooled + aware)]
    sc.metric = ["x: share of non-crashed queries the rule 'stop at the first step with "
                 "p_crash >= tau' stops, tau swept over every emitted probability, window "
                 f"[0, {xmax:g}]",
                 "y: mean steps saved on crashed queries (crash step minus stop step, 0 when "
                 "never stopped), each panel its own range (higher is better)"]
    sc.numbers = numbers
    sc.data = [f"source: {source}"] + notes
    sc.provenance = prov
    return C.Output(sc, fig=fig)


FIGURES: list[C.FigureSpec] = [
    C.FigureSpec("fig3_cross", draw_fig3_cross),
    C.FigureSpec("fig3_sheet", draw_fig3_sheet),
    C.FigureSpec("fig2_sheet", draw_fig2_sheet),
]

GROUP = PlotGroup(name=GROUP_NAME, needs=C.spec_needs(GROUP_NAME),
                  render=C.make_render(GROUP_NAME, FIGURES))
