"""Plot group ``hardware``: the rig figures of the thesis hardware chapter and appendix.

Reads the ``hardware_optimization`` runs (rig logs) and the ``hardware_prediction`` scores of the
rigs in ``configs/benchmarks/hardware.yaml`` from the benchmark database, computes every number
through :mod:`esbopfn.hardware.metrics` and writes one figure plus one sidecar ``.md`` (sources,
rig, methods, seeds, key numbers) per name:

- ``hwplay_optimization__<rig>``: (a) median log10 regret over reset-normalized wall-clock and
  (b) over the first trials, with 25/75 % seed bands, (c) mean crashed count over wall-clock.
- ``hwplay_regret_violin``: per-seed log10 regret at 50 % and 100 % of the wall-clock budget,
  every rig, the seed that defines ``J_best`` left out.
- ``hwplay_cost_1000``: median best successful raw cost over the first 1000 s, 25/75 % band.
- ``hwplay_best_controller__500s__<rig>``: arm and unwrapped pendulum angle of the incumbent at
  500 s of the median seed (ESBO-GP and XDecPFN-PI), one rig execution.
- ``hwplay_acq_wait``: mean acquisition wait per trial, bars over seeds plus per-seed dots.
- ``hwpred_benchmark``: median conditioned LL against inference time with Pareto fronts over the
  context size, medians below -7 omitted.
- ``hwpred_ablation_bars``: the four DecPFN prior and channel variants, median LL on the
  non-crashed queries with their full history as context.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from esbopfn.benchmarks.db import Db
from esbopfn.hardware import metrics as M
from esbopfn.hardware.records import HwRun
from esbopfn.plots import PlotGroup
from esbopfn.plots import _hw_common as C

log = logging.getLogger(__name__)

KIND_OPT = "hardware_optimization"
KIND_PRED = "hardware_prediction"

#: Defaults of the ``figures`` section of ``configs/benchmarks/hardware.yaml``.
DEFAULTS: dict[str, Any] = {
    "acq_wait_methods": ["esbo-gp", "bo", "decpfn-h", "decpfn", "xdecpfn-pi"],
    "violin_alphas": [0.22, 0.45],
    "pred_y_cutoff": -7.0,
    "ladder_metric": "median_ll_all",
    "ablation_metric": "median_ll",
    "context_blind": ["decpfn"],
    "ladder_predictors": ["segment-gp", "decpfn", "xdecpfn"],
    "ablation_arms": [
        {"predictor": "decpfn", "label": "dead-time, states", "dead_time": True, "states": True},
        {"predictor": "decpfn-nostates", "label": "dead-time, no states", "dead_time": True,
         "states": False},
        {"predictor": "decpfn-nodeadtime", "label": "no dead-time, states", "dead_time": False,
         "states": True},
        {"predictor": "decpfn-nodeadtime-nostates", "label": "no dead-time, no states",
         "dead_time": False, "states": False},
    ],
    "deadtime_colors": ["#0072B2", "#56B4E9"],
    "legend_on_first_rig_only": True,
}


# =========================================================================== configuration
def _metrics_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    m = dict(cfg.get("metrics") or {})
    m.setdefault("trials_axis", 60)
    m.setdefault("budget_fracs", [0.5, 1.0])
    m.setdefault("cost_window_s", 1000)
    m.setdefault("incumbent_at_s", 500)
    m.setdefault("incumbent_methods", ["esbo-gp", "xdecpfn-pi"])
    return m


def _fig_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULTS, **(cfg.get("figures") or {})}


def _rigs(cfg: dict[str, Any]) -> list[str]:
    from esbopfn.hardware.rigs import load_rigs

    return list(cfg.get("rigs") or load_rigs())


def _methods(cfg: dict[str, Any]) -> list[str]:
    ms = list((cfg.get("optimization") or {}).get("methods") or C.HW_METHODS)
    order = {m: i for i, m in enumerate(C.HW_METHODS)}
    return sorted(ms, key=lambda m: order.get(m, len(order)))


def _opt_provenance(cfg: dict[str, Any]) -> str:
    return str((cfg.get("optimization") or {}).get("provenance", "thesis"))


def _pred_provenance(cfg: dict[str, Any]) -> str:
    pred = cfg.get("prediction") or {}
    return str(pred.get("provenance", pred.get("logs", "thesis")))


def _reset_method(rig: str) -> str:
    from esbopfn.hardware.rigs import load_rigs

    r = load_rigs().get(rig)
    return r.reset_reference_method if r is not None else "decpfn-h"


# =========================================================================== data
def load_rig(db: Db, rig: str, cfg: dict[str, Any]) -> list[HwRun]:
    from esbopfn.hardware.store import load_runs

    runs = load_runs(rig, db, provenance=_opt_provenance(cfg), methods=_methods(cfg))
    if not runs:
        raise RuntimeError(
            f"no {_opt_provenance(cfg)} hardware_optimization runs of rig {rig!r} in {db.path} "
            "(import the thesis logs with `pfn hw import-logs` or run `pfn hw run`)")
    return runs


def _unit_lines(runs: Sequence[HwRun], db: Db) -> list[str]:
    seeds: dict[str, list[int]] = {}
    for r in runs:
        seeds.setdefault(r.method, []).append(r.seed)
    rig = runs[0].rig if runs else "?"
    prov = runs[0].key.get("provenance", "?") if runs else "?"
    return [f"rig `{rig}`: {len(runs)} `{KIND_OPT}` units (provenance `{prov}`) in "
            f"`{db.path}`, methods and seeds "
            + ", ".join(f"{m} {sorted(s)}" for m, s in seeds.items())]


def _pred_units(db: Db, cfg: dict[str, Any], rig: str, bench: str) -> list[Any]:
    prov = _pred_provenance(cfg)
    units = db.query(KIND_PRED, rig=rig, bench=bench)
    if prov == "thesis":
        units = [u for u in units if u.key.get("provenance") == "thesis"]
    else:
        units = [u for u in units if u.key.get("provenance") == "native"
                 and u.key.get("logs") == prov]
    return units


def _summary(unit: Any) -> dict[str, Any]:
    return json.loads((Path(unit.blob_dir) / "summary.json").read_text())


def _scores(unit: Any) -> pd.DataFrame:
    return pd.read_parquet(Path(unit.blob_dir) / "scores.parquet")


# =========================================================================== optimization curves
def _regret_ylim(curves: dict[str, Any]) -> tuple[float, float] | None:
    vals = [m["per_seed"][np.isfinite(m["per_seed"])] for m in curves["methods"].values()]
    vals = [v for v in vals if v.size]
    if not vals:
        return None
    pooled = np.concatenate(vals)
    above = pooled[pooled > math.log10(M.REGRET_FLOOR) + 1.0]
    return (float(above.min() - 0.5), float(above.max() + 0.5)) if above.size else None


def _draw_band_lines(ax: Any, x: np.ndarray, series: dict[str, Any], center: str,
                     theme: str) -> None:
    for m, s in series.items():
        col = C.color(m, theme)
        ax.fill_between(x, s["lo"], s["hi"], color=col, alpha=C.BAND_ALPHA, lw=0, step="post",
                        zorder=1)
        ax.plot(x, s[center], color=col, lw=C.LINE_WIDTH, drawstyle="steps-post", zorder=3,
                label=C.method_label(m))


def fig_optimization(db: Db, cfg: dict[str, Any], rig: str, runs: list[HwRun], out_dir: Path,
                     fmt: str, theme: str, legend: bool) -> dict[str, Any]:
    mc = _metrics_cfg(cfg)
    methods = _methods(cfg)
    reset = _reset_method(rig)
    wall = M.regret_curves(runs, "wall", reset_method=reset, methods=methods)
    trials = M.regret_curves(runs, "trials", reset_method=reset, methods=methods,
                             max_trials=int(mc["trials_axis"]))
    crash = M.crash_curves(runs, reset_method=reset, methods=methods)
    order = [m for m in methods if m in wall["methods"]]
    name = f"hwplay_optimization__{rig}"
    with C.figure_style(cfg, theme):
        fig, axes = C.panel_row(3)
        panels = (
            (wall, "median", "wall-clock / s", C.better(r"median $\log_{10}$ regret", False)),
            (trials, "median", "trials", C.better(r"median $\log_{10}$ regret", False)),
            (crash, "mean", "wall-clock / s", C.better("mean crashes", False)),
        )
        for ax, (curves, center, xlabel, ylabel) in zip(axes, panels):
            _draw_band_lines(ax, curves["x"], {m: curves["methods"][m] for m in order},
                             center, theme)
            if center == "median":
                ylim = _regret_ylim(curves)
                if ylim:
                    ax.set_ylim(*ylim)
            C.finish(ax, xlabel=xlabel, ylabel=ylabel)
        if legend:
            h, lab = axes[0].get_legend_handles_labels()
            C.top_legend(fig, h, lab, ncol=len(order))
        else:
            fig.tight_layout(w_pad=1.2)
        C.save_figure(fig, out_dir, name, fmt)

    counts = M.crash_counts(runs)
    ends = {m: {r.seed: float(M.wall_axis(r, wall["reset_ref_s"])[-1]) for r in g if r.trials}
            for m, g in M.by_method(runs).items()}
    numbers = [f"J_best = {C.fmt_num(wall['J_best'])} (method `{wall['J_best_method']}`, seed "
               f"{wall['J_best_seed']}), reset normalization {C.fmt_num(wall['reset_ref_s'])} s "
               f"(mean reset of `{reset}`)"]
    for m in order:
        w = wall["methods"][m]
        final = w["per_seed"][:, -1]
        numbers.append(
            f"{m}: seeds {w['seeds']}, final log10 regret per seed {C.fmt_list(final)}, "
            f"median {C.fmt_num(w['median'][-1])}, crashed trials per seed "
            f"{[counts[m][s] for s in w['seeds']]} (mean {C.fmt_num(np.mean(list(counts[m].values())), 1)}), "
            f"trials per seed {[len(r.trials) for r in M.by_method(runs)[m]]}, run end "
            f"{C.fmt_list([ends[m][s] for s in w['seeds']], 0)} s")
        t = trials["methods"][m]
        numbers.append(f"{m}: median log10 regret after {int(mc['trials_axis'])} trials "
                       f"{C.fmt_num(t['median'][-1])}")
    C.write_sidecar(
        out_dir, name,
        shown=[f"rig `{rig}` ({C.rig_title(rig)}), methods {order}",
               "(a) median over seeds of the unscaled log10 regret against reset-normalized "
               "wall-clock, 25/75 % seed band",
               f"(b) the same over the first {int(mc['trials_axis'])} trials",
               "(c) mean over seeds of the cumulative crashed-trial count against wall-clock, "
               "25/75 % band",
               "legend above the panels" if legend else "no legend (same palette as the first rig)"],
        metric=["regret = log10(J^- - J_best), J^- the best successful cost so far, J_best the "
                "best successful cost on the rig over the shown methods",
                "wall-clock = cumulative trial time with every reset replaced by the mean reset "
                "of the reference method (`hardware.metrics.wall_axis`)",
                "a seed without a success yet is left out of the median at that x"],
        data=_unit_lines(runs, db),
        numbers=numbers)
    return {"J_best": wall["J_best"], "reset_ref_s": wall["reset_ref_s"], "crash_counts": counts}


# =========================================================================== violins
def fig_regret_violin(db: Db, cfg: dict[str, Any], rig_runs: dict[str, list[HwRun]],
                      out_dir: Path, fmt: str, theme: str) -> dict[str, Any]:
    mc, fc = _metrics_cfg(cfg), _fig_cfg(cfg)
    fracs = [float(f) for f in mc["budget_fracs"]]
    methods = _methods(cfg)
    alphas = list(fc["violin_alphas"])
    res = {rig: M.regret_at_fractions(runs, fracs, reset_method=_reset_method(rig),
                                      methods=methods)
           for rig, runs in rig_runs.items()}
    pops = [np.asarray(res[r]["methods"][m][f]["values"]) for r in res
            for m in res[r]["methods"] for f in fracs]
    pops = [p for p in pops if p.size]
    name = "hwplay_regret_violin"
    half, off = 0.20, 0.21
    with C.figure_style(cfg, theme):
        fg = C.fg_color(theme)
        fig, axes = C.panel_row(len(res), sharey=True)
        ylim = None
        if pops:
            lo, hi = float(min(p.min() for p in pops)), float(max(p.max() for p in pops))
            pad = 0.12 * (hi - lo) + 0.05
            ylim = (lo - pad, hi + pad)
            edges = C.violin_bins(pops, ylim)
            scale = C.violin_scale(pops, edges)
        for i, (ax, rig) in enumerate(zip(axes, res)):
            ms = [m for m in methods if m in res[rig]["methods"]]
            for pos, m in enumerate(ms):
                col = C.color(m, theme)
                for frac, side, alpha in zip(fracs, (-1.0, 1.0), alphas):
                    vals = np.asarray(res[rig]["methods"][m][frac]["values"])
                    if not vals.size:
                        continue
                    x = pos + side * off
                    C.violin_body(ax, vals, x, col, edges=edges, scale=scale, half_width=half,
                                  alpha=alpha)
                    C.violin_points(ax, vals, x, col, half_width=half)
                    C.violin_median(ax, vals, x, col, half_width=half)
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([C.method_label(m) for m in ms], rotation=35, fontsize=6.0,
                               ha="right", rotation_mode="anchor")
            ax.set_xlim(-0.6, len(ms) - 0.4)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(C.rig_title(rig), fontsize=C.TITLE_FONTSIZE)
            C.finish(ax, ylabel=C.better(r"$\log_{10}$ regret", False) if i == 0 else None,
                     grid_axis="y")
        handles: list[Any] = [Patch(facecolor=fg, edgecolor="none", alpha=a) for a in alphas]
        labels = [f"{f * 100:.0f} {C.pct()} budget" for f in fracs]
        handles.append(Line2D([], [], color=fg, lw=1.3))
        labels.append("median")
        C.top_legend(fig, handles, labels, ncol=len(handles))
        C.save_figure(fig, out_dir, name, fmt)

    numbers = []
    for rig, r in res.items():
        numbers.append(f"{rig}: J_best = {C.fmt_num(r['J_best'])}")
        for m, per in r["methods"].items():
            for f in fracs:
                p = per[f]
                med = C.fmt_num(np.median(p["values"])) if p["values"] else "n/a"
                numbers.append(f"{rig} {m} at {f:.0%} (x = {C.fmt_num(p['x'], 0)} s): seeds "
                               f"{p['seeds']} values {C.fmt_list(p['values'])} median {med}"
                               + (f", excluded (defines J_best) {p['excluded_seeds']}"
                                  if p["excluded_seeds"] else ""))
    C.write_sidecar(
        out_dir, name,
        shown=[f"one panel per rig {list(res)}, per method two violins: left {fracs[0]:.0%}, "
               f"right {fracs[-1]:.0%} of the wall-clock budget (darker), seeds as dots, the "
               "line is the median",
               "violin body = smoothed mirrored histogram on one bin grid for the whole figure"],
        metric=["per-seed unscaled log10 regret of the incumbent at the fraction of the longest "
                "reset-normalized run on the rig",
                "the seed whose incumbent defines J_best sits on the regret floor and is left "
                "out of its violin"],
        data=[line for runs in rig_runs.values() for line in _unit_lines(runs, db)],
        numbers=numbers)
    return res


# =========================================================================== cost 0..1000 s
def fig_cost_window(db: Db, cfg: dict[str, Any], rig_runs: dict[str, list[HwRun]],
                    out_dir: Path, fmt: str, theme: str) -> dict[str, Any]:
    mc = _metrics_cfg(cfg)
    t_max = float(mc["cost_window_s"])
    methods = _methods(cfg)
    res = {rig: M.cost_curves(runs, t_max, reset_method=_reset_method(rig), methods=methods,
                              n_points=1000)
           for rig, runs in rig_runs.items()}
    # one y range for all panels: from the lowest band edge to the highest median line (the
    # band of the first seconds, when only one or two seeds have a success, is clipped)
    lows = [s["lo"][np.isfinite(s["lo"])] for r in res.values() for s in r["methods"].values()]
    mids = [s["median"][np.isfinite(s["median"])] for r in res.values()
            for s in r["methods"].values()]
    lows = [v for v in lows if v.size] or [np.array([0.0])]
    mids = [v for v in mids if v.size] or [np.array([1.0])]
    lo = float(min(v.min() for v in lows))
    hi = float(max(v.max() for v in mids))
    ylim = (lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
    name = f"hwplay_cost_{int(t_max)}"
    with C.figure_style(cfg, theme):
        fig, axes = C.panel_row(len(res), sharey=True)
        for i, (ax, rig) in enumerate(zip(axes, res)):
            r = res[rig]
            order = [m for m in methods if m in r["methods"]]
            _draw_band_lines(ax, r["x"], {m: r["methods"][m] for m in order}, "median", theme)
            ax.set_xlim(0.0, t_max)
            ax.set_ylim(*ylim)
            ax.set_title(C.rig_title(rig), fontsize=C.TITLE_FONTSIZE)
            C.finish(ax, xlabel="wall-clock / s",
                     ylabel=C.better("median best cost", False) if i == 0 else None)
        h, lab = axes[0].get_legend_handles_labels()
        C.top_legend(fig, h, lab, ncol=3)
        C.save_figure(fig, out_dir, name, fmt)

    numbers = []
    checkpoints = np.arange(100.0, t_max + 1e-9, 100.0)
    for rig, r in res.items():
        numbers.append(f"{rig}: reset normalization {C.fmt_num(r['reset_ref_s'])} s")
        for t in checkpoints:
            col = int(np.searchsorted(r["x"], t, side="right") - 1)
            row = []
            for m, s in r["methods"].items():
                n_ok = int(np.isfinite(s["per_seed"][:, col]).sum())
                row.append(f"{m} {C.fmt_num(s['median'][col], 1)} ({n_ok})")
            numbers.append(f"{rig} t = {t:.0f} s: " + ", ".join(row))
    C.write_sidecar(
        out_dir, name,
        shown=[f"one panel per rig {list(res)}, median over seeds of the best successful raw "
               f"cost so far over the first {t_max:.0f} s of reset-normalized wall-clock, "
               "25/75 % seed band, one y range for all panels (lowest band edge to highest "
               "median line, band spikes of the first seconds clipped)",
               "numbers: median at 100 s checkpoints, seeds with a success in brackets"],
        metric=["raw cost J_T of the best successful trial so far (not regret), a seed without "
                "a success is left out at that x"],
        data=[line for runs in rig_runs.values() for line in _unit_lines(runs, db)],
        numbers=numbers)
    return res


# =========================================================================== incumbents at 500 s
def fig_best_controller(db: Db, cfg: dict[str, Any], rig: str, runs: list[HwRun],
                        out_dir: Path, fmt: str, theme: str) -> dict[str, Any]:
    mc = _metrics_cfg(cfg)
    wall_s = float(mc["incumbent_at_s"])
    methods = [m for m in mc["incumbent_methods"] if m in {r.method for r in runs}]
    # the thesis draws the PFN loop first
    methods = sorted(methods, key=lambda m: 0 if m.startswith("xdecpfn") else 1)
    picked = {m: M.incumbent_at(runs, m, wall_s, reset_method=_reset_method(rig))
              for m in methods}
    picked = {m: p for m, p in picked.items() if p is not None}
    name = f"hwplay_best_controller__{int(wall_s)}s__{rig}"
    with C.figure_style(cfg, theme):
        fig, axes = C.panel_row(2)
        t_end = 0.0
        for m, p in picked.items():
            col = C.color(m, theme)
            pend = np.asarray(p["pend_deg"], float)
            pend = pend - 360.0 * np.round(pend[-1] / 360.0)   # end at upright = 0 deg
            axes[0].plot(p["t_s"], p["arm_deg"], color=col, lw=1.1, ls="-",
                         label=C.method_label(m))
            axes[1].plot(p["t_s"], pend, color=col, lw=1.1, ls="--")
            t_end = max(t_end, float(p["t_s"][-1]))
        for ax, lab in zip(axes, (r"$\varphi_\mathrm{arm}$ / deg",
                                  r"$\varphi_\mathrm{pend}$ / deg")):
            ax.set_xlim(-0.2, t_end)
            C.finish(ax, xlabel="time / s", ylabel=lab)
        h, lab = axes[0].get_legend_handles_labels()
        C.top_legend(fig, h, lab, ncol=len(h))
        C.save_figure(fig, out_dir, name, fmt)

    numbers = []
    for m, p in picked.items():
        ranking = ", ".join(f"seed {s}: {c:.1f}" for s, c in p["candidates"])
        numbers.append(f"{m}: median seed {p['seed']}, incumbent trial index {p['trial_index']}, "
                       f"J = {C.fmt_num(p['J'], 1)}, kick sign {p['kick_sign']:+.0f}, theta "
                       f"{C.fmt_list(p['theta'], 3)}, incumbent cost per seed at {wall_s:.0f} s "
                       f"(ranked) {ranking}")
    C.write_sidecar(
        out_dir, name,
        shown=[f"rig `{rig}` ({C.rig_title(rig)}), methods {list(picked)}",
               "left: arm angle, right: pendulum angle (dashed, unwrapped, 0 deg upright, the "
               "trace shifted by a multiple of 360 deg so that it ends upright), degrees over "
               "time within the trial (120 Hz)"],
        metric=[f"controller = the incumbent (best successful trial so far) at {wall_s:.0f} s of "
                "reset-normalized wall-clock of the seed with the median incumbent cost "
                "(index (n - 1) // 2 of the ranked seeds), one rig execution, traces from the "
                "stored x channels of that trial"],
        data=_unit_lines(runs, db),
        numbers=numbers)
    return picked


# =========================================================================== acquisition wait
def fig_acq_wait(db: Db, cfg: dict[str, Any], rig_runs: dict[str, list[HwRun]], out_dir: Path,
                 fmt: str, theme: str) -> dict[str, Any]:
    fc = _fig_cfg(cfg)
    methods = [m for m in fc["acq_wait_methods"] if m in _methods(cfg)]
    res = {rig: M.acquisition_wait(runs, methods) for rig, runs in rig_runs.items()}
    name = "hwplay_acq_wait"
    with C.figure_style(cfg, theme):
        fig, axes = C.panel_row(len(res), sharey=True)
        for i, (ax, rig) in enumerate(zip(axes, res)):
            ms = [m for m in methods if m in res[rig]]
            for pos, m in enumerate(ms):
                col = C.color(m, theme)
                ax.bar(pos, res[rig][m]["mean"], width=0.62, color=col, alpha=0.35,
                       edgecolor=col, linewidth=0.8, zorder=3)
                per = list(res[rig][m]["per_seed"].values())
                offs = np.linspace(-0.18, 0.18, len(per)) if len(per) > 1 else [0.0]
                ax.scatter(pos + np.asarray(offs), per, s=3.0, color=col, zorder=5,
                           linewidths=0.0)
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([C.method_label(m) for m in ms], rotation=35, fontsize=6.0,
                               ha="right", rotation_mode="anchor")
            ax.set_xlim(-0.6, len(ms) - 0.4)
            ax.set_ylim(bottom=0.0)
            ax.set_title(C.rig_title(rig), fontsize=C.TITLE_FONTSIZE)
            C.finish(ax, ylabel=C.better("avg. acquisition wait / s", False) if i == 0 else None,
                     grid_axis="y")
        fig.tight_layout(w_pad=1.2)
        C.save_figure(fig, out_dir, name, fmt)

    numbers = []
    for rig, r in res.items():
        for m, v in r.items():
            numbers.append(f"{rig} {m}: mean {C.fmt_num(v['mean'])} s per trial, per seed "
                           + ", ".join(f"{s}: {w:.3f}" for s, w in v["per_seed"].items()))
    C.write_sidecar(
        out_dir, name,
        shown=[f"one panel per rig {list(res)}, bar = mean over seeds of the seed's mean wait "
               f"per trial, dots = seeds, methods {methods} (ESRS has no acquisition and is "
               "left out)"],
        metric=["wait = propose time per trial (refit and acquisition optimization, "
                "`timings.propose_s`), the early stopping decisions during a trial are not "
                "included, resets are not counted"],
        data=[line for runs in rig_runs.values() for line in _unit_lines(runs, db)],
        numbers=numbers)
    return res


# =========================================================================== prediction ladder
def _label_points(ax: Any, labels: list[tuple[str, str, tuple[float, float]]],
                  obstacles: list[tuple[tuple[float, float], float]], fontsize: float) -> None:
    """Direct series labels placed off every glyph and inside the axes."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    obst_px = [(ax.transData.transform(xy), r * fig.dpi / 72.0) for xy, r in obstacles]
    cands = [((0, 11), "center", "bottom"), ((-9, 7), "right", "bottom"),
             ((9, 7), "left", "bottom"), ((10, 0), "left", "center"),
             ((-10, 0), "right", "center"), ((0, -11), "center", "top"),
             ((-9, -7), "right", "top"), ((9, -7), "left", "top"), ((0, 20), "center", "bottom"),
             ((-14, 14), "right", "bottom"), ((14, 14), "left", "bottom")]
    placed: list[Any] = []
    for text, colour, xy in labels:
        best = None
        for (dx, dy), ha, va in cands:
            t = ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=colour, zorder=12)
            bb = t.get_window_extent(renderer).expanded(1.05, 1.1)
            inside = (bb.x0 >= box.x0 and bb.x1 <= box.x1 and bb.y0 >= box.y0
                      and bb.y1 <= box.y1)
            clash = any(bb.x0 - r < c[0] < bb.x1 + r and bb.y0 - r < c[1] < bb.y1 + r
                        for c, r in obst_px)
            clash = clash or any(bb.overlaps(p) for p in placed)
            if inside and not clash:
                best = t
                placed.append(bb)
                break
            t.remove()
        if best is None:
            t = ax.annotate(text, xy, xytext=(0, 11), textcoords="offset points",
                            ha="center", va="bottom", fontsize=fontsize, color=colour, zorder=12)
            placed.append(t.get_window_extent(renderer))


def _ladder_points(cfg: dict[str, Any], unit: Any) -> dict[str, Any]:
    """``[(n, seconds, median LL, queries)]`` of one ladder unit, context-blind predictors
    pooled to one point (every query once, n = None)."""
    fc = _fig_cfg(cfg)
    s = _summary(unit)
    metric = fc["ladder_metric"]
    med = s.get(metric) or {}
    secs = {str(k): float(v) for k, v in (s.get("seconds_per_prediction") or {}).items()}
    counts = s.get("n_scored_all" if metric.endswith("_all") else "n_scored") or {}
    pred = unit.key["predictor"]
    if pred in fc["context_blind"]:
        df = _scores(unit)
        if metric == "median_ll":
            df = df[~df.crashed]
        per_query = df.groupby(["method", "seed", "query_index"]).ll.mean()
        t = float(np.median(list(secs.values()))) if secs else math.nan
        return {"points": [(None, t, float(per_query.median()), int(len(per_query)))],
                "raw": med, "seconds": secs}
    pts = []
    for k, y in med.items():
        if k == "full" or k not in secs or y is None or not math.isfinite(float(y)):
            continue
        pts.append((int(k), secs[k], float(y), int(counts.get(k, 0))))
    return {"points": sorted(pts, key=lambda p: p[1]), "raw": med, "seconds": secs}


def _pareto(pts: list[tuple[Any, float, float, int]]) -> list[tuple[Any, float, float, int]]:
    """Keep the rungs that are better than every faster rung (ascending time)."""
    kept, best = [], -math.inf
    for p in sorted(pts, key=lambda p: p[1]):
        if p[2] > best + 1e-12:
            kept.append(p)
            best = p[2]
    return kept


MARKER_AREA = 120.0
POOLED_AREA = MARKER_AREA * 85.0 / 130.0


def _resolve_overlaps(ax: Any, glyphs: list[dict[str, Any]]) -> set[int]:
    """Indices of the glyphs to draw: among circles closer than one diameter only the higher
    context size stays, each series keeps its best point."""
    dia = 2.0 * math.sqrt(MARKER_AREA / math.pi) * ax.figure.dpi / 72.0
    disp = [ax.transData.transform((g["t"], g["y"])) for g in glyphs]
    protected = {i for i, g in enumerate(glyphs) if g["best"]}
    order = sorted((i for i in range(len(glyphs)) if i not in protected),
                   key=lambda i: (-(glyphs[i]["n"] if glyphs[i]["n"] is not None else -1),
                                  -glyphs[i]["y"]))
    kept: list[int] = list(protected)
    for i in order:
        if all(np.hypot(*(disp[i] - disp[k])) >= dia for k in kept):
            kept.append(i)
    return set(kept)


def fig_pred_benchmark(db: Db, cfg: dict[str, Any], rigs: list[str], out_dir: Path, fmt: str,
                       theme: str) -> dict[str, Any]:
    fc = _fig_cfg(cfg)
    cutoff = float(fc["pred_y_cutoff"])
    per_rig: dict[str, dict[str, Any]] = {}
    for rig in rigs:
        units = {u.key["predictor"]: u for u in _pred_units(db, cfg, rig, "ladder")}
        per_rig[rig] = {p: _ladder_points(cfg, units[p]) | {"unit": units[p]}
                        for p in fc["ladder_predictors"] if p in units}
    if not any(per_rig.values()):
        raise RuntimeError("no hardware_prediction ladder units in the database (import them "
                           "with `pfn hw import-pred` or run `pfn hw bench-pred`)")
    drawn: dict[str, dict[str, Any]] = {}
    tops = [p[2] for r in per_rig.values() for d in r.values() for p in d["points"]
            if p[2] >= cutoff]
    ytop = max(tops) + 0.5 if tops else 0.0
    name = "hwpred_benchmark"
    with C.figure_style(cfg, theme):
        fig, axes = C.panel_row(len(rigs), sharey=False)
        for i, (ax, rig) in enumerate(zip(axes, rigs)):
            series = {}
            for pred, d in per_rig[rig].items():
                keep = [p for p in d["points"] if p[2] >= cutoff and math.isfinite(p[1])]
                series[pred] = _pareto(keep)
            xs = [p[1] for s in series.values() for p in s]
            ax.set_xscale("log")
            if xs:
                ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
            ax.set_ylim(cutoff, ytop)
            ax.set_title(C.rig_title(rig), fontsize=C.TITLE_FONTSIZE)
            C.finish(ax, xlabel=C.better("inference time / s", False),
                     ylabel=C.better("median LL", True) if i == 0 else None)
            fig.tight_layout(w_pad=1.2)
            fig.canvas.draw()
            glyphs = []
            for pred, pts in series.items():
                best = max(pts, key=lambda p: p[2]) if pts else None
                for p in pts:
                    glyphs.append({"pred": pred, "n": p[0], "t": p[1], "y": p[2],
                                   "best": p is best})
            kept = _resolve_overlaps(ax, glyphs)
            labels, obstacles = [], []
            for pred, pts in series.items():
                if not pts:
                    continue
                col = C.color(pred, theme)
                ax.plot([p[1] for p in pts], [p[2] for p in pts], color=col, lw=1.4, alpha=0.9,
                        drawstyle="steps-post", zorder=2)
                best = max(pts, key=lambda p: p[2])
                labels.append((C.method_label(pred), col, (best[1], best[2])))
            for j, g in enumerate(glyphs):
                if j not in kept:
                    continue
                col = C.color(g["pred"], theme)
                pooled = g["n"] is None
                n = -1 if pooled else g["n"]
                ax.scatter([g["t"]], [g["y"]], s=POOLED_AREA if pooled else MARKER_AREA,
                           color=col, zorder=5 + n / 1000.0, edgecolors="white",
                           linewidths=0.7)
                if not pooled:
                    ax.annotate(str(g["n"]), (g["t"], g["y"]), ha="center", va="center",
                                fontsize=4.6, fontweight="bold", color="white",
                                zorder=10 + n / 1000.0)
                obstacles.append(((g["t"], g["y"]),
                                  math.sqrt((POOLED_AREA if pooled else MARKER_AREA) / math.pi)))
            _label_points(ax, labels, obstacles, fontsize=6.5)
            drawn[rig] = {"series": series,
                          "hidden": [(g["pred"], g["n"]) for j, g in enumerate(glyphs)
                                     if j not in kept]}
        C.save_figure(fig, out_dir, name, fmt)

    numbers = []
    for rig in rigs:
        for pred, d in per_rig[rig].items():
            raw = ", ".join(f"{k}: {C.fmt_num(v)}" for k, v in d["raw"].items())
            secs = ", ".join(f"{k}: {v:.4g}" for k, v in d["seconds"].items())
            numbers.append(f"{rig} {pred} ({fc['ladder_metric']} per n) {raw}")
            numbers.append(f"{rig} {pred} seconds per prediction per n: {secs}")
            if pred in fc["context_blind"]:
                p = d["points"][0]
                numbers.append(f"{rig} {pred} pooled (every query once): median LL "
                               f"{C.fmt_num(p[2])} over {p[3]} queries at {p[1]:.4g} s")
        dr = drawn.get(rig, {})
        for pred, pts in dr.get("series", {}).items():
            numbers.append(f"{rig} {pred} drawn front: "
                           + ", ".join(f"n={'pooled' if n is None else n} ({C.fmt_num(y)}, "
                                       f"{t:.4g} s)" for n, t, y, _ in pts))
        if dr.get("hidden"):
            numbers.append(f"{rig} glyphs hidden by overlap (line still drawn): "
                           + ", ".join(f"{p} n={n}" for p, n in dr["hidden"]))
    data = []
    for rig in rigs:
        for pred, d in per_rig[rig].items():
            u = d["unit"]
            data.append(f"{rig} {pred}: `{KIND_PRED}` unit {u.key_hash[:12]} (model "
                        f"`{u.key['model']}`, provenance `{u.key['provenance']}`, logs "
                        f"`{u.key['logs']}`)")
    C.write_sidecar(
        out_dir, name,
        shown=["one panel per rig, one circle per (predictor, context size n) with n written "
               "inside, lines are Pareto staircases over n (dominated rungs dropped), medians "
               f"below {cutoff:g} omitted, overlapping circles keep the higher n",
               f"context-blind predictors {fc['context_blind']} pooled to one smaller unnumbered "
               "point (every query counted once, x = median of the per-rung timings)"],
        metric=[f"y = `{fc['ladder_metric']}`: median over queries of the per-query mean "
                "conditioned LL over the cutoffs (crashed queries included for `_all`)",
                "x = seconds per prediction (median per context size)"],
        data=data, numbers=numbers)
    return {"per_rig": per_rig, "drawn": drawn}


# =========================================================================== ablation bars
def fig_ablation_bars(db: Db, cfg: dict[str, Any], rigs: list[str], out_dir: Path, fmt: str,
                      theme: str) -> dict[str, Any]:
    fc = _fig_cfg(cfg)
    arms = fc["ablation_arms"]
    metric = fc["ablation_metric"]
    values: dict[str, dict[str, tuple[float, int, Any]]] = {}
    for rig in rigs:
        units = {u.key["predictor"]: u for u in _pred_units(db, cfg, rig, "ablation")}
        values[rig] = {}
        for arm in arms:
            u = units.get(arm["predictor"])
            if u is None:
                continue
            s = _summary(u)
            v = (s.get(metric) or {}).get("full")
            n = (s.get("n_scored" if metric == "median_ll" else "n_scored_all") or {}).get("full")
            values[rig][arm["predictor"]] = (float(v) if v is not None else math.nan,
                                             int(n or 0), u)
    finite = [v[0] for r in values.values() for v in r.values() if math.isfinite(v[0])]
    if not finite:
        raise RuntimeError("no hardware_prediction ablation units in the database")
    lo, hi = min(finite), max(finite)
    span = max(hi - lo, 0.05)
    ylo, yhi = lo - 0.2 * span, hi + 0.2 * span
    dt_col, nd_col = fc["deadtime_colors"]
    name = "hwpred_ablation_bars"
    with C.figure_style(cfg, theme):
        bg = "white" if theme == "white" else "black"
        fig, axes = C.panel_row(len(rigs), sharey=True)
        for i, (ax, rig) in enumerate(zip(axes, rigs)):
            for x, arm in enumerate(arms):
                v = values[rig].get(arm["predictor"])
                if v is None or not math.isfinite(v[0]):
                    continue
                col = dt_col if arm["dead_time"] else nd_col
                ax.bar(x, v[0] - ylo, bottom=ylo, width=0.62, zorder=3,
                       color=col if arm["states"] else bg, edgecolor=col,
                       linewidth=0 if arm["states"] else 1.2)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([a["label"].replace(", ", ",\n") for a in arms], fontsize=6.0,
                               linespacing=1.0)
            ax.set_xlim(-0.6, len(arms) - 0.4)
            ax.set_ylim(ylo, yhi)
            ax.set_title(C.rig_title(rig), fontsize=C.TITLE_FONTSIZE)
            C.finish(ax, ylabel=C.better("median LL", True) if i == 0 else None, grid_axis="y")
        fig.tight_layout(w_pad=1.2)
        C.save_figure(fig, out_dir, name, fmt)

    numbers = [f"y range [{ylo:.3f}, {yhi:.3f}] (not from 0)"]
    data = []
    for rig, r in values.items():
        numbers.append(f"{rig}: " + " | ".join(
            f"{a['label']} {C.fmt_num(r[a['predictor']][0])} (n={r[a['predictor']][1]})"
            for a in arms if a["predictor"] in r))
        for p, (_, _, u) in r.items():
            data.append(f"{rig} {p}: `{KIND_PRED}` unit {u.key_hash[:12]} (model "
                        f"`{u.key['model']}`, provenance `{u.key['provenance']}`)")
    C.write_sidecar(
        out_dir, name,
        shown=["one panel per rig, four DecPFN variants: dead-time in the prior or not (dark or "
               "light blue) x state channels or not (filled or outlined)",
               "the y axis does not start at 0, every bar grows from the shared axis floor"],
        metric=[f"y = `{metric}`: median over the non-crashed queries of the per-query mean "
                "conditioned LL over the cutoffs, every query with its full run history as "
                "context"],
        data=data, numbers=numbers)
    return values


# =========================================================================== group
def needs(cfg: dict[str, Any], db: Db) -> list[tuple[str, dict[str, Any]]]:
    from esbopfn.benchmarks.kinds import hardware_optimization, hardware_prediction

    # rig runs cannot be computed by `pfn bench run` (they need the rig or the thesis import),
    # so only the ones already in the database are listed
    out = [(KIND_OPT, k) for k in hardware_optimization.enumerate_keys(cfg)
           if db.has(KIND_OPT, k)]
    if _pred_provenance(cfg) == "thesis":
        out += [(KIND_PRED, u.key) for rig in _rigs(cfg)
                for bench in ("ladder", "ablation") for u in _pred_units(db, cfg, rig, bench)]
    else:
        out += [(KIND_PRED, k) for k in hardware_prediction.enumerate_keys(cfg)]
    return out


def render(cfg: dict[str, Any], db: Db, out_dir: Path, fmt: str, theme: str) -> dict[str, Any]:
    """Render every hardware figure into ``out_dir``. Returns the computed numbers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rigs = _rigs(cfg)
    try:
        rig_runs = {rig: load_rig(db, rig, cfg) for rig in rigs}
    except RuntimeError as exc:
        log.warning("hardware figures skipped: %s", exc)
        return {}
    fc = _fig_cfg(cfg)
    mc = _metrics_cfg(cfg)
    result: dict[str, Any] = {"optimization": {}, "best_controller": {}}
    for i, rig in enumerate(rigs):
        legend = i == 0 or not fc["legend_on_first_rig_only"]
        result["optimization"][rig] = fig_optimization(db, cfg, rig, rig_runs[rig], out_dir,
                                                       fmt, theme, legend)
    result["regret_violin"] = fig_regret_violin(db, cfg, rig_runs, out_dir, fmt, theme)
    result["cost_window"] = fig_cost_window(db, cfg, rig_runs, out_dir, fmt, theme)
    for rig in rigs:
        result["best_controller"][rig] = fig_best_controller(db, cfg, rig, rig_runs[rig],
                                                             out_dir, fmt, theme)
    result["acq_wait"] = fig_acq_wait(db, cfg, rig_runs, out_dir, fmt, theme)
    for key, fn in (("pred_benchmark", fig_pred_benchmark), ("ablation", fig_ablation_bars)):
        try:
            result[key] = fn(db, cfg, rigs, out_dir, fmt, theme)
        except RuntimeError as exc:
            log.warning("%s skipped: %s", key, exc)
    log.info("hardware figures (incumbent at %s s) written to %s", mc["incumbent_at_s"], out_dir)
    return result


def _render(cfg: dict[str, Any], db: Db, out_dir: Path, fmt: str, theme: str) -> None:
    render(cfg, db, out_dir, fmt, theme)


GROUP = PlotGroup(name="hardware", needs=needs, render=_render)
