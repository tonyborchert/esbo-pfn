"""Plot group ``cross_trajectory``: XDecPFN and FT-PFN over context sizes.

Figures (names = the thesis asset basenames):

- ``context_ladder`` (fig:sim:context-sweep): mean conditioned LL over the fixed-query context
  ladder (query = last trial of the ``oracle181`` runs, context = the first ``n`` trials), PT2
  dead-time, Cartpole and Furuta, the rungs placed equidistantly.
- ``ftpfn_xdecpfn_stop_sheet`` (fig:sim:ftpfn-xdecpfn-stop-sheet): steps saved on crashed queries
  against the false-stop rate, one curve per context size, fainter for less context.
- ``ftpfn_xdecpfn_example`` and ``landscape_furuta``: live inference, from
  :mod:`esbopfn.plots.examples`.

Data: native ``prediction_ladder`` / ``prediction`` units first, else the thesis import (the
imported ``prediction_ladder`` units, ``thesis_series`` ``fig2_stop_curves``).
"""

from __future__ import annotations

import logging
from typing import Any

from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from esbopfn.benchmarks import metrics
from esbopfn.benchmarks.kinds.prediction import load_predictions, resolve_model
from esbopfn.plots import PlotGroup, common
from esbopfn.plots.common import Context, MissingData, Output, Provenance, Sidecar
from esbopfn.plots.single_trajectory import FALSE_STOP_XMAX, stop_axes, stop_curves

log = logging.getLogger(__name__)

GROUP_NAME = "cross_trajectory"
ASSET_DIR = "simulation/prediction/ftpfn_xdecpfn"
MODELS: tuple[str, ...] = ("xdecpfn/{p}/default/default", "ftpfn/{p}/default/default")
#: Rungs that carry a tick label (every rung is drawn).
LADDER_TICKS: dict[str, tuple[int, ...]] = {
    "pt2_deadtime": (0, 2, 8, 33, 134),
    "cartpole": (0, 2, 11, 45, 180),
    "furuta": (0, 2, 11, 45, 180),
}
STOP_CONTEXTS: tuple[int, ...] = (0, 2, 4, 8, 16, 32, 64)
SHADE_KEY = (0.30, 0.62, 0.95)


def _fig_cfg(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    sec = (cfg.get("figures") or {}).get(name)
    return dict(sec) if isinstance(sec, dict) else {}


def _models(cfg: dict[str, Any], sec: str) -> list[str]:
    s = cfg.get(sec) or {}
    out = []
    for m in s.get("models") or MODELS:
        out.append(m["id"] if isinstance(m, dict) else str(m))
    return out


# ============================================================================ ladder
def ladder_rows(db: Any, models: list[str], problems: list[str], prov: Provenance):
    """Rows of the ladder units per (problem, model template): native units first, else the
    imported thesis units (``.../thesis/default``, the thesis cell)."""
    import pandas as pd

    frames, used = [], []
    for p in problems:
        for tmpl in models:
            mid = resolve_model(tmpl, p)
            units = [u for u in db.query("prediction_ladder", problem=p, model=mid)]
            if not units:
                units = db.query("prediction_ladder", problem=p,
                                 model=common.thesis_model_id(mid))
            if not units:
                continue
            df = load_predictions(units)
            df["problem"] = p
            df["model"] = tmpl
            frames.append(df)
            used += units
    prov.add_units("prediction_ladder", used)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def draw_context_ladder(ctx: Context) -> Output:
    sec = ctx.cfg.get("prediction_ladder") or {}
    problems = list(sec.get("problems") or common.CRASH_PROBLEMS)
    models = _models(ctx.cfg, "prediction_ladder")
    prov = Provenance()
    rows = ladder_rows(ctx.db, models, problems, prov)
    if rows.empty:
        raise MissingData([common.missing_units_message("prediction_ladder", models,
                                                        GROUP_NAME)])
    curve = metrics.ladder_curve(rows)
    fig, axes = common.panel_sheet(1, len(problems), top_pt=28.0)
    numbers = []
    for ax, p in zip(axes[0], problems):
        sub = curve[curve.problem == p]
        rungs = sorted(int(n) for n in sub.n_ctx.unique())
        slot = {n: i for i, n in enumerate(rungs)}
        for tmpl in models:
            c = sub[sub.model == tmpl].sort_values("n_ctx")
            if c.empty:
                continue
            ax.plot([slot[int(n)] for n in c.n_ctx], c["mean"], color=common.model_color(tmpl,
                    ctx.theme), marker="o", markersize=3.2, lw=1.1, zorder=3)
            numbers.append(f"{common.problem_label(p)} {common.model_label(tmpl)}: "
                           + ", ".join(f"n={int(n)} {v:.3f}" for n, v in zip(c.n_ctx, c["mean"]))
                           + f" ({int(c.n.min())}-{int(c.n.max())} queries per rung)")
        ticks = [n for n in LADDER_TICKS.get(p, tuple(rungs)) if n in slot] or rungs
        ax.set_xticks([slot[n] for n in ticks])
        ax.set_xticklabels([str(n) for n in ticks], fontsize=6.5)
        if rungs:
            ax.set_xlim(-0.35, len(rungs) - 1 + 0.35)
        ax.set_title(common.problem_label(p), fontsize=7.5)
        common.finish(ax, xlabel="context", pad="y", grid_axis="y")
    axes[0, 0].set_ylabel(common.better("LL", True))
    handles = [Line2D([0], [0], color=common.model_color(m, ctx.theme), marker="o",
                      markersize=4.0, lw=1.1) for m in models]
    common.legend_above(fig, handles, [common.model_label(m) for m in models], ncol=len(models),
                        y_pt=14.0)
    sidecar = Sidecar(
        thesis_asset=f"{ASSET_DIR}/context_ladder.pdf",
        label="fig:sim:context-sweep",
        shown=["One panel per problem (PT2 dead-time, Cartpole, Furuta), XDecPFN and FT-PFN, "
               "every rung drawn at equal spacing, a subset of rungs labeled.",
               "The query is fixed (last trial of each `oracle181` run, the first 50 non-crashed "
               "in seed-major order), the context is the first n trials of the same run. "
               "The top Cartpole/Furuta rung is labeled with its stored value 180 (the thesis "
               "axis printed 179)."],
        metric=["y: conditioned LL, mean over the five cutoffs per query, then the mean over the "
                "queries (not normalized, a mean rather than the medians used elsewhere)."],
        numbers=numbers,
        data=["XDecPFN on PT2 dead-time is the thesis cell `xdecpfn/pt2_deadtime/thesis/default`"
              " when read from the thesis import (the `/lnjw128` twin is not drawn)."],
        provenance=prov)
    return Output(sidecar, fig=fig)


# ============================================================================ stop sheet
def draw_stop_sheet(ctx: Context) -> Output:
    fc = _fig_cfg(ctx.cfg, "stop_sheet")
    problems = list(fc.get("problems") or common.CRASH_PROBLEMS)
    xmax = float(fc.get("max_false_stop", FALSE_STOP_XMAX))
    models = _models(ctx.cfg, "prediction")
    contexts = [int(n) for n in fc.get("n", STOP_CONTEXTS)]
    prov = Provenance()
    curves, src = stop_curves(ctx.db, "prediction", models, problems, prov, pooled=False)
    if not curves:
        raise MissingData([common.missing_units_message("prediction", models, GROUP_NAME)])
    fig, axes = common.panel_sheet(1, len(problems), top_pt=28.0)
    numbers = []
    for ax, p in zip(axes[0], problems):
        for tmpl in models:
            color = common.model_color(tmpl, ctx.theme)
            saved = []
            for rank, n in enumerate(contexts):
                c = curves.get((p, tmpl, n))
                if c is None:
                    continue
                alpha = 0.30 + 0.65 * rank / max(1, len(contexts) - 1)
                ax.plot(c.false_stop_rate, c.mean_steps_saved, color=color, lw=1.0,
                        alpha=alpha, zorder=2)
                saved.append(f"n={n} {metrics.steps_saved_at(c, xmax):.1f}")
            if saved:
                numbers.append(f"{common.problem_label(p)} {common.model_label(tmpl)} steps saved "
                               f"at false-stop <= {xmax:g}: " + ", ".join(saved))
        stop_axes(ax, p)
        ax.set_xlim(0, xmax)
    axes[0, 0].set_ylabel(common.better("mean steps saved", True))
    gray = "#555555" if ctx.theme == "white" else "#BBBBBB"
    handles: list[Any] = [tuple(Line2D([], [], color=common.model_color(m, ctx.theme), lw=1.0,
                                       alpha=a) for a in SHADE_KEY) for m in models]
    handles.append(tuple(Line2D([], [], color=gray, lw=1.0, alpha=a) for a in SHADE_KEY))
    labels = [common.model_label(m) for m in models] + ["fainter = less context"]
    common.legend_above(fig, handles, labels, ncol=len(labels), y_pt=14.0,
                        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
                        handlelength=2.2)
    n_info = []
    for (p, tmpl, n), c in sorted(curves.items()):
        if n in contexts and tmpl == models[0]:
            n_info.append(f"{p} n={n}: {int(c.n_crashed.iloc[0])} crashed / "
                          f"{int(c.n_ok.iloc[0])} ok queries")
    sidecar = Sidecar(
        thesis_asset=f"{ASSET_DIR}/ftpfn_xdecpfn_stop_sheet.pdf",
        label="fig:sim:ftpfn-xdecpfn-stop-sheet",
        shown=["Three panels (PT2 dead-time, Cartpole, Furuta), XDecPFN and FT-PFN, one curve "
               f"per context size n in {contexts}, alpha rising linearly from 0.30 (n = "
               f"{contexts[0]}) to 0.95 (n = {contexts[-1]})."],
        metric=["Stop at the first predictable step with crash probability >= tau, tau swept "
                "over every emitted probability. x: share of non-crashed queries stopped "
                f"(shown up to {xmax:g}, y fitted to that window), y: mean steps saved on the 50 "
                "crashed queries per context size. Each panel has its own y range."],
        numbers=numbers,
        data=[f"curves from: {src}"] + n_info,
        provenance=prov)
    return Output(sidecar, fig=fig)


# ============================================================================ group
def _figures() -> list[common.FigureSpec]:
    figs = [common.FigureSpec("context_ladder", draw_context_ladder),
            common.FigureSpec("ftpfn_xdecpfn_stop_sheet", draw_stop_sheet)]
    import importlib.util

    if importlib.util.find_spec("esbopfn.plots.examples") is None:
        log.info("%s: the live-inference example figures are not available", GROUP_NAME)
        return figs
    from esbopfn.plots import examples

    for attr in ("FTPFN_XDECPFN_EXAMPLE", "LANDSCAPE_FURUTA"):
        spec = getattr(examples, attr, None)
        if spec is not None:
            figs.append(spec)
    return figs


FIGURES = _figures()

GROUP = PlotGroup(name=GROUP_NAME, needs=common.spec_needs(GROUP_NAME),
                  render=common.make_render(GROUP_NAME, FIGURES))
