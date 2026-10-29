"""Plot group ``single_trajectory``: LC-PFN, DecPFN and DecPFN-no-St.

Figures (names = the thesis asset basenames):

- ``lcpfn_decpfn_heads_benchmark`` (fig:sim:lcpfn-decpfn-heads): one dot per (model, head,
  training size) and problem, the median conditioned LL of the non-crashed queries pooled over the
  context sizes and the five cutoffs, centered per problem on the drawn arms.
- ``lcpfn_decpfn_stop_sheet`` (fig:sim:lcpfn-decpfn-stop-sheet): mean steps saved on crashed
  queries against the false-stop rate (shown up to 0.1), binned head, 100k fainter, 1M solid.
- ``gridsearch_panels`` (fig:app:hpo-gridsearch): stage-2 HPO grid cells on Cartpole, training
  wall-clock against validation LL, marker area growing with the parameter count.
- ``lcpfn_decpfn_example`` (fig:sim:lcpfn-decpfn-example): live inference, from
  :mod:`esbopfn.plots.examples`.

Data: native ``prediction`` units first, else the thesis import (``thesis_series``
``fig3_per_problem`` and ``fig2_stop_curves``), the HPO tree ``$PFN_HOME/hpo`` for the grid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from esbopfn.benchmarks import metrics
from esbopfn.benchmarks.kinds.prediction import resolve_model
from esbopfn.plots import PlotGroup, common
from esbopfn.plots.common import Context, MissingData, Output, Provenance, Sidecar

log = logging.getLogger(__name__)

GROUP_NAME = "single_trajectory"
ASSET_DIR = "simulation/prediction/lcpfn_decpfn"

#: (family, label, x offset in a problem's slot, hollow)
SERIES: tuple[tuple[str, str, float, bool], ...] = (
    ("decpfn", "DecPFN", -0.22, False),
    ("decpfn-nostates", "DecPFN-no-St", 0.0, True),
    ("lcpfn", "LC-PFN", 0.22, False),
)
HEADS: tuple[str, ...] = ("binned", "gaussian", "linear")
SIZES: tuple[str, ...] = ("100k", "1M")
HEAD_MARKERS = {"binned": "s", "gaussian": "^", "linear": "o"}
SIZE_AREA = {"100k": 14.0, "1M": 42.0}
SIZE_DX = {"100k": -0.032, "1M": 0.032}
SIZE_ALPHA = {"100k": 0.40, "1M": 0.95}
FALSE_STOP_XMAX = 0.1
FALSE_STOP_XTICKS = (0.0, 0.05, 0.1)


def _fig_cfg(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    figs = cfg.get("figures") or {}
    sec = figs.get(name)
    return dict(sec) if isinstance(sec, dict) else {}


def _problems(cfg: dict[str, Any], default: tuple[str, ...]) -> list[str]:
    return list(cfg.get("problems") or default)


def series_color(family: str, theme: str) -> str:
    return common.model_color(family, theme)


# ============================================================================ heads sheet
def arm_id(family: str, head: str, size: str) -> str:
    return f"{family}/{{p}}/{head}/{size}"


def heads_table(db: Any, problems: list[str], prov: Provenance) -> tuple[pd.DataFrame, list[str]]:
    """Raw median LL per (problem, family, head, size): native ``prediction`` units first, else the
    thesis ``fig3_per_problem`` series. Returns the table and the arms found nowhere."""
    rows, missing = [], []
    native_used, thesis_used = [], []
    thesis = {(b.problem, b.series): b for b in common.thesis_series(db, "fig3_per_problem")}
    for p in problems:
        for fam, _, _, _ in SERIES:
            for head in HEADS:
                for size in SIZES:
                    mid = resolve_model(arm_id(fam, head, size), p)
                    units = [u for u in db.query("prediction", problem=p, model=mid)
                             if not common.is_thesis_unit(u)]
                    value, n, source = float("nan"), 0, ""
                    if units:
                        from esbopfn.benchmarks.kinds.prediction import load_predictions

                        df = load_predictions(units)
                        sc = metrics.collapse_context_blind(metrics.trial_scores(df))
                        if len(sc):
                            value, n, source = float(sc.score.median()), len(sc), "native"
                            native_used += units
                    if not source and (p, mid) in thesis:
                        b = thesis[(p, mid)]
                        f = b.frame[b.frame.n_ctx == -1] if "n_ctx" in b.frame else b.frame
                        if len(f):
                            value, n, source = float(f.y.iloc[0]), int(f.n.iloc[0]), "thesis"
                            thesis_used.append(b.unit)
                    if not source:
                        missing.append(f"{p}/{mid}")
                        continue
                    rows.append({"problem": p, "family": fam, "head": head, "size": size,
                                 "model": mid, "ll": value, "n": n, "source": source})
    prov.add_units("prediction", native_used)
    prov.add_units("thesis_series", thesis_used)
    return pd.DataFrame(rows), missing


def draw_heads(ctx: Context) -> Output:
    fc = _fig_cfg(ctx.cfg, "heads")
    problems = _problems(ctx.cfg, common.PROBLEMS)
    center_how = str(fc.get("center", "mean"))
    prov = Provenance()
    table, missing = heads_table(ctx.db, problems, prov)
    if table.empty:
        raise MissingData([common.missing_units_message(
            "prediction", [arm_id(f, h, s) for f, *_ in SERIES for h in HEADS for s in SIZES],
            GROUP_NAME)], "or import the thesis results with `pfn import thesis-results`")
    drawn = [p for p in problems if p in set(table.problem)]
    centers = {p: float(getattr(table[table.problem == p].ll, center_how)()) for p in drawn}
    table["y"] = table.ll - table.problem.map(centers)

    fig, axes = common.panel_sheet(1, 1, top_pt=40.0)
    ax = axes[0, 0]
    bg = common.bg_color(ctx.theme)
    for slot, p in enumerate(drawn):
        sub = table[table.problem == p]
        for fam, _, dx, hollow in SERIES:
            color = series_color(fam, ctx.theme)
            for r in sub[sub.family == fam].itertuples():
                ax.scatter([slot + dx + SIZE_DX[r.size]], [r.y], marker=HEAD_MARKERS[r.head],
                           s=SIZE_AREA[r.size], facecolors=bg if hollow else color,
                           edgecolors=color, linewidths=0.9, zorder=3)
        if slot:
            ax.axvline(slot - 0.5, color="#DDDDDD" if ctx.theme == "white" else "#444444", lw=0.6,
                       zorder=1)
    ax.set_xticks(range(len(drawn)))
    ax.set_xticklabels([common.problem_label(p) for p in drawn], fontsize=7)
    ax.set_xlim(-0.55, len(drawn) - 0.45)
    common.finish(ax, ylabel=common.better("LL", True), pad="y", grid_axis="y")

    gray = "#444444" if ctx.theme == "white" else "#BBBBBB"
    opts = {"frameon": False, "fontsize": 5.8, "handlelength": 1.1, "handletextpad": 0.4,
                "labelspacing": 0.32, "borderpad": 0.0, "borderaxespad": 0.0, "loc": "lower left"}
    pos = ax.get_position()
    top = pos.y1 + 6.0 / (fig.get_size_inches()[1] * 72.0)
    series_h = [Line2D([], [], color=series_color(f, ctx.theme), marker="o", lw=0, markersize=5,
                       markerfacecolor=bg if h else series_color(f, ctx.theme),
                       markeredgewidth=0.9) for f, _, _, h in SERIES]
    head_h = [Line2D([], [], color=gray, marker=HEAD_MARKERS[h], lw=0, markersize=5,
                     markerfacecolor="none", markeredgewidth=0.9) for h in HEADS]
    size_h = [Line2D([], [], color=gray, marker="o", lw=0, markersize=SIZE_AREA[s] ** 0.5,
                     markerfacecolor=gray) for s in SIZES]
    for x, (h, lab) in zip((0.10, 0.40, 0.66), ((series_h, [s[1] for s in SERIES]),
                                                (head_h, list(HEADS)), (size_h, list(SIZES)))):
        fig.legend(h, lab, bbox_to_anchor=(pos.x0 + x * pos.width, top), **opts)

    def raw(p: str, fam: str, head: str = "binned", size: str = "1M") -> float:
        s = table[(table.problem == p) & (table.family == fam) & (table["head"] == head)
                  & (table["size"] == size)]
        return float(s.ll.iloc[0]) if len(s) else float("nan")

    numbers = [f"per-problem center ({center_how} over the drawn arms, add back to a dot for the "
               "raw median LL): " + ", ".join(f"{common.problem_label(p)} {c:.3f}"
                                              for p, c in centers.items())]
    for p in drawn:
        d, ns, lc = raw(p, "decpfn"), raw(p, "decpfn-nostates"), raw(p, "lcpfn")
        numbers.append(f"{common.problem_label(p)} 1M binned raw median LL: DecPFN {d:.4f}, "
                       f"DecPFN-no-St {ns:.4f}, LC-PFN {lc:.4f} (DecPFN - LC-PFN {d - lc:.2f}, "
                       f"DecPFN-no-St - LC-PFN {ns - lc:.2f}, DecPFN - DecPFN-no-St {d - ns:.2f})")
    sources = sorted(set(table.source))
    sidecar = Sidecar(
        thesis_asset=f"{ASSET_DIR}/lcpfn_decpfn_heads_benchmark.pdf",
        label="fig:sim:lcpfn-decpfn-heads",
        shown=["One panel, one x slot per problem. Color and fill = model (DecPFN filled, "
               "DecPFN-no-St hollow in the DecPFN color, LC-PFN filled), marker = output head "
               "(binned square, gaussian triangle, linear circle), marker size = training size "
               "(100k small, 1M large, nudged left and right inside the series slot).",
               f"Arms not drawn (no data): {', '.join(missing) if missing else 'none'}."],
        metric=["y: median over the non-crashed queries of the per-query mean conditioned LL at "
                "the five cutoffs (10/30/50/70/90 % of T), pooled over every context size "
                "(context-blind models), higher is better.",
                f"Each problem centered on the {center_how} of its drawn arms (the thesis figure "
                "centers on the mean, the caption says median), 0 is not a baseline model. "
                "`figures.heads.center` in single_trajectory.yaml switches it."],
        numbers=numbers,
        data=[f"values from: {', '.join(sources)} "
              "(thesis = `thesis_series` fig3_per_problem, the median the thesis computed over "
              "every cutoff row)"],
        provenance=prov)
    return Output(sidecar, fig=fig)


# ============================================================================ stop sheet
def native_stop_curves(db: Any, kind: str, models: list[str], problems: list[str],
                       prov: Provenance, pooled: bool, thesis_raw: bool = False
                       ) -> dict[tuple[str, str, int], pd.DataFrame]:
    """``{(problem, model template, n_ctx): stop curve}`` from native stop-grid rows (the
    imported thesis predictions with ``thesis_raw``). ``pooled`` gives one curve over every ``n``
    (``n_ctx = -1``)."""
    out: dict[tuple[str, str, int], pd.DataFrame] = {}
    for tmpl in models:
        rows = common.prediction_frame(db, kind, [tmpl], problems, prov,
                                       native_only=not thesis_raw, thesis_raw=thesis_raw)
        if rows.empty:
            continue
        for p, g in rows.groupby("problem"):
            groups = [(-1, g)] if pooled else list(g.groupby("n_ctx"))
            for n, gn in groups:
                tasks = (common.thesis_tasks(str(p), None if n == -1 else int(n))
                         if thesis_raw else None)
                curve = metrics.stop_curve(gn, tasks)
                if len(curve):
                    out[(str(p), tmpl, int(n))] = curve.sort_values("false_stop_rate",
                                                                    kind="stable")
    return out


def thesis_stop_curves(db: Any, models: list[str], problems: list[str], prov: Provenance
                       ) -> dict[tuple[str, str, int], pd.DataFrame]:
    out: dict[tuple[str, str, int], pd.DataFrame] = {}
    used = []
    for b in common.thesis_series(db, "fig2_stop_curves"):
        if b.problem not in problems:
            continue
        for tmpl in models:
            if common.thesis_model_id(resolve_model(tmpl, b.problem)) != b.series:
                continue
            for n, g in b.frame.groupby("n_ctx"):
                out[(b.problem, tmpl, int(n))] = g.sort_values("false_stop_rate", kind="stable")
            used.append(b.unit)
    prov.add_units("thesis_series", used)
    return out


def stop_curves(db: Any, kind: str, models: list[str], problems: list[str], prov: Provenance,
                pooled: bool) -> tuple[dict[tuple[str, str, int], pd.DataFrame], str]:
    """Native curves where a model has them, the thesis import for the others: the
    ``fig2_stop_curves`` aggregates, then the imported thesis predictions (every task the old
    store scored, as the thesis pooled them)."""
    native = native_stop_curves(db, kind, models, problems, prov, pooled)
    have = {(p, m) for p, m, _ in native}
    thesis = {k: v for k, v in thesis_stop_curves(db, models, problems, prov).items()
              if (k[0], k[1]) not in have}
    have |= {(p, m) for p, m, _ in thesis}
    lacking = [m for m in models if any((p, m) not in have for p in problems)]
    raw = {k: v for k, v in native_stop_curves(db, kind, lacking, problems, prov, pooled,
                                               thesis_raw=True).items()
           if (k[0], k[1]) not in have} if lacking else {}
    src = "+".join(s for s, d in (("native", native), ("thesis import", thesis),
                                  ("thesis predictions", raw)) if d)
    return {**native, **thesis, **raw}, src


def stop_axes(ax: Any, problem: str) -> None:
    ax.set_title(common.problem_label(problem), fontsize=7.5)
    common.zoom_x(ax, FALSE_STOP_XMAX, xticks=FALSE_STOP_XTICKS)
    common.finish(ax, xlabel=common.better("false-stop rate", False))


def draw_stop_sheet(ctx: Context) -> Output:
    fc = _fig_cfg(ctx.cfg, "stop_sheet")
    problems = list(fc.get("problems") or common.CRASH_PROBLEMS)
    models = list(fc.get("models") or [arm_id(f, "binned", s) for f, *_ in SERIES for s in SIZES])
    xmax = float(fc.get("max_false_stop", FALSE_STOP_XMAX))
    prov = Provenance()
    curves, src = stop_curves(ctx.db, "prediction", models, problems, prov, pooled=True)
    if not curves:
        raise MissingData([common.missing_units_message("prediction", models, GROUP_NAME)])
    fig, axes = common.panel_sheet(1, len(problems), top_pt=28.0)
    numbers, missing = [], []
    for ax, p in zip(axes[0], problems):
        for tmpl in models:
            fam = common.model_family(tmpl)
            size = tmpl.rsplit("/", 1)[-1]
            c = curves.get((p, tmpl, -1))
            if c is None:
                missing.append(resolve_model(tmpl, p))
                continue
            ax.plot(c.false_stop_rate, c.mean_steps_saved, color=series_color(fam, ctx.theme),
                    ls="--" if fam == "decpfn-nostates" else "-", lw=1.3,
                    alpha=SIZE_ALPHA.get(size, 0.95), zorder=2 + (size == "1M"))
            numbers.append(f"{common.problem_label(p)} {common.model_label(tmpl)} {size}: "
                           f"{metrics.steps_saved_at(c, xmax):.1f} steps saved at false-stop "
                           f"<= {xmax:g} ({int(c.n_crashed.iloc[0])} crashed / "
                           f"{int(c.n_ok.iloc[0])} ok queries)")
        stop_axes(ax, p)
        ax.set_xlim(0, xmax)
    axes[0, 0].set_ylabel(common.better("mean steps saved", True))
    gray = "#555555" if ctx.theme == "white" else "#BBBBBB"
    handles = [Line2D([], [], color=series_color(f, ctx.theme), lw=1.3,
                      ls="--" if f == "decpfn-nostates" else "-") for f, *_ in SERIES]
    labels = [lab for _, lab, *_ in SERIES]
    for s in SIZES:
        handles.append(Line2D([], [], color=gray, lw=1.3, alpha=SIZE_ALPHA[s]))
        labels.append(s)
    common.legend_above(fig, handles, labels, ncol=len(labels), y_pt=14.0)
    if missing:
        log.warning("lcpfn_decpfn_stop_sheet: not drawn (no data): %s", ", ".join(missing))
    sidecar = Sidecar(
        thesis_asset=f"{ASSET_DIR}/lcpfn_decpfn_stop_sheet.pdf",
        label="fig:sim:lcpfn-decpfn-stop-sheet",
        shown=["Three panels (PT2 dead-time, Cartpole, Furuta), DecPFN, DecPFN-no-St (dashed) "
               "and LC-PFN with the binned head, 100k fainter, 1M full strength.",
               "Series not drawn (no data): " + (", ".join(missing) if missing else "none")
               + ". The thesis aggregates hold only the DecPFN and LC-PFN 1M curves, the other "
               "series come from native `prediction` units or the imported thesis predictions "
               "(every task the old store scored)."],
        metric=["Stop at the first predictable step with crash probability >= tau, tau swept "
                "over every emitted probability. x: share of non-crashed queries stopped "
                f"(shown up to {xmax:g}, y fitted to that window), y: mean steps saved on crashed "
                "queries (crash step - stop step, 0 if never stopped). Context-blind models "
                "pooled over every context size. Each panel has its own y range."],
        numbers=numbers,
        data=[f"curves from: {src}"],
        provenance=prov)
    return Output(sidecar, fig=fig)


# ============================================================================ gridsearch
GRID_MODELS: tuple[str, ...] = ("decpfn", "lcpfn")
GRID_CAPS = {"100k": 25.0, "1M": 250.0}


def hpo_trials(model: str, problem: str, size: str, home: Path | None = None) -> Path:
    from esbopfn.config import paths

    return paths(home).root / "hpo" / model / problem / size / "trials.csv"


def _size_area(n_params: Any, lo: float, hi: float) -> np.ndarray:
    v = np.log10(np.maximum(np.asarray(n_params, dtype=float), 1.0))
    return 8.0 + 62.0 * np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def draw_gridsearch(ctx: Context) -> Output:
    fc = _fig_cfg(ctx.cfg, "gridsearch")
    problem = str(fc.get("problem", "cartpole"))
    sizes = list(fc.get("sizes", SIZES))
    caps = {**GRID_CAPS, **(fc.get("caps") or {})}
    prov = Provenance()
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    missing = []
    for size in sizes:
        for m in GRID_MODELS:
            path = hpo_trials(m, problem, size)
            if not path.is_file():
                missing.append(str(path))
                continue
            df = pd.read_csv(path)
            if "stage" in df:
                df = df[df.stage == "stage2"]
            df = df[np.isfinite(df.val_nll.astype(float)) & np.isfinite(df.train_time_s)]
            if len(df):
                frames[(size, m)] = df
                prov.add_file(path)
    if not frames:
        raise MissingData(["HPO results " + ", ".join(missing)],
                          f"run `pfn hpo <decpfn|lcpfn> --problem {problem} --size <100k|1M>`")
    allp = np.log10(pd.concat([f.n_params.astype(float) for f in frames.values()]))
    lo, hi = float(allp.min()), float(allp.max())
    fig, axes = common.panel_sheet(1, len(sizes), top_pt=34.0)
    edge = "black" if ctx.theme == "white" else "white"
    numbers = []
    for ax, size in zip(axes[0], sizes):
        for m in GRID_MODELS:
            df = frames.get((size, m))
            if df is None:
                continue
            for head, g in df.groupby("head"):
                ax.scatter(g.train_time_s, -g.val_nll.astype(float),
                           color=series_color(m, ctx.theme), marker=HEAD_MARKERS.get(head, "o"),
                           s=_size_area(g.n_params, lo, hi), linewidths=0.4, edgecolors=edge,
                           alpha=0.88, zorder=3)
            best = df.loc[df.val_nll.astype(float).idxmin()]
            numbers.append(f"{size} {common.model_label(m)}: best cell {best['head']} "
                           f"d{int(best.d_model)}/l{int(best.n_layers)}, LL "
                           f"{-float(best.val_nll):.3f} after {float(best.train_time_s):.1f} s, "
                           f"{len(df)} cells")
        ax.set_title(size, fontsize=9)
        ax.set_xscale("log")
        common.log_ticks_decades(ax)
        from matplotlib.ticker import FuncFormatter, LogLocator

        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        common.finish(ax, xlabel="training time / s", pad="y")
        if size in caps:
            ax.set_xlim(right=float(caps[size]))
    axes[0, 0].set_ylabel(common.better("LL", True))
    gray = "#888888"
    handles = [Line2D([0], [0], marker="o", color="none",
                      markerfacecolor=series_color(m, ctx.theme), markeredgecolor=edge,
                      markeredgewidth=0.4, markersize=5) for m in GRID_MODELS]
    labels = [common.model_label(m) for m in GRID_MODELS]
    for h in ("linear", "binned", "gaussian"):
        handles.append(Line2D([0], [0], marker=HEAD_MARKERS[h], color="none",
                              markerfacecolor=gray, markeredgecolor=edge, markeredgewidth=0.4,
                              markersize=5))
        labels.append(h)
    for v, lab in ((1e6, "1M"), (5e6, "5M"), (3e7, "30M")):
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=gray,
                              markeredgecolor=edge, markeredgewidth=0.4,
                              markersize=float(_size_area([v], lo, hi)[0]) ** 0.5))
        labels.append(lab)
    common.legend_above(fig, handles, labels, ncol=len(labels), y_pt=16.0)
    sidecar = Sidecar(
        thesis_asset=f"{ASSET_DIR}/gridsearch_panels.pdf",
        label="fig:app:hpo-gridsearch",
        shown=["Two panels (training size 100k and 1M), one marker per stage-2 grid cell of the "
               f"HPO on {problem}: color = model, shape = output head, area = parameter count "
               "(log-linear between the smallest and the largest model of both panels).",
               "HPO results not found: " + (", ".join(missing) if missing else "none")],
        metric=["x: measured training wall-clock (log), view capped at "
                + ", ".join(f"{s} {caps[s]:g} s" for s in sizes if s in caps)
                + ". y: minus the validation NLL of the horizon prediction at the grid cutoffs "
                  "between 20 % and 99 % (`val_nll` of trials.csv), higher is better."],
        numbers=numbers,
        data=["`$PFN_HOME/hpo/<model>/<problem>/<size>/trials.csv` written by `pfn hpo`, rows "
              "with `stage = stage2`."],
        provenance=prov)
    return Output(sidecar, fig=fig)


# ============================================================================ group
def _figures() -> list[common.FigureSpec]:
    figs = [common.FigureSpec("lcpfn_decpfn_heads_benchmark", draw_heads),
            common.FigureSpec("lcpfn_decpfn_stop_sheet", draw_stop_sheet),
            common.FigureSpec("gridsearch_panels", draw_gridsearch)]
    import importlib.util

    if importlib.util.find_spec("esbopfn.plots.examples") is None:
        log.info("%s: the live-inference example figures are not available", GROUP_NAME)
        return figs
    from esbopfn.plots import examples

    spec = getattr(examples, "LCPFN_DECPFN_EXAMPLE", None)
    if spec is not None:
        figs.append(spec)
    return figs


FIGURES = _figures()

GROUP = PlotGroup(name=GROUP_NAME, needs=common.spec_needs(GROUP_NAME),
                  render=common.make_render(GROUP_NAME, FIGURES))
