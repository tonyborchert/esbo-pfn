"""Plot group ``priors``: fig:setup:prior-samples and fig:app:prior-landscapes.

- ``priors_samples``: ten cumulative-cost trajectories ``J_t`` per problem on screened instance 0,
  at most five of ten unsuccessful, five flush panels, linear ``J_t`` without tick values, a red
  cross where a trajectory crashes. Read from ``landscapes`` units ``{unit: prior_samples}``,
  computed on the fly (a few hundred rollouts) when the unit is absent.
- ``priors_landscapes``: five problems (columns) x up to three screened instances (rows), the true
  landscape on one slice through Theta. Blue rises with the cost rank among a panel's cells
  (brighter = cheaper), red with the crash probability, the two add. Boiler (``d = 1``) is a line.
  Read from ``landscapes`` units ``{unit: slice}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D

from esbopfn.plots import PlotGroup
from esbopfn.plots import common as c

GROUP_NAME = "priors"
#: Panel geometry of the samples row, inches.
_S_LEFT, _S_BOTTOM, _S_TITLE, _S_LEGEND = 0.30, 0.28, 0.22, 0.24
#: Landscape sheet geometry, inches.
_L_LEFT, _L_TITLE, _L_BOTTOM = 0.30, 0.20, 0.28
_L_KEY_GAP, _L_KEY_LABEL, _L_KEY = 0.34, 0.44, 0.62
#: Additive landscape colors (objective rank, crash probability).
BLUE_RGB = np.array([0.20, 0.55, 0.98])
RED_RGB = np.array([0.98, 0.20, 0.20])
LINE_BLUE = "#0072B2"


# ------------------------------------------------------------------------------- samples
class _EndMarker(HandlerLine2D):
    """Legend handle with the marker at the right end of the line (where a crash ends a curve)."""

    def create_artists(self, legend: Any, orig_handle: Any, xdescent: float, ydescent: float,
                       width: float, height: float, fontsize: float, trans: Any) -> list[Any]:
        y = height / 2.0 - ydescent
        x0, x1 = -xdescent, width - xdescent
        line = Line2D([x0, x1], [y, y])
        line.update_from(orig_handle)
        line.set_marker("none")
        mark = Line2D([x1], [y])
        mark.update_from(orig_handle)
        mark.set_linestyle("none")
        for artist in (line, mark):
            artist.set_transform(trans)
        return [line, mark]


def _crash_marker(**kw: Any) -> dict[str, Any]:
    out = dict(marker="x", color=c.CRASH_COLOR, ms=4.5, mew=1.2)
    out.update(kw)
    return out


def _samples_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    sec = cfg.get("landscapes") if isinstance(cfg.get("landscapes"), dict) else {}
    ps = sec.get("prior_samples") or {}
    return {"instance": int(ps.get("instance", 0)), "n": int(ps.get("n", 10)),
            "max_unsuccessful": int(ps.get("max_unsuccessful", 5))}


def _instances_seed(cfg: dict[str, Any]) -> Any:
    from esbopfn.benchmarks.groups import instances_seed

    return instances_seed(cfg)


def load_samples(ctx: c.Context, problem: str, prov: c.Provenance) -> dict[str, Any]:
    """``{J (n, T), success, crash_step, theta, source}`` of the prior samples of ``problem``."""
    from esbopfn.benchmarks.instances import BENCH_SEED, load_instances
    from esbopfn.benchmarks.kinds.landscapes import prior_samples
    from esbopfn.problems import get_problem
    from esbopfn.util.seeds import substream

    sc = _samples_cfg(ctx.cfg)
    units = ctx.db.query("landscapes", unit="prior_samples", problem=problem,
                         instance=sc["instance"], n=sc["n"])
    units = sorted(units, key=lambda u: u.key.get("instances_seed") != _instances_seed(ctx.cfg))
    if units:
        prov.add_units("landscapes", units[:1])
        with np.load(Path(units[0].blob_dir) / "samples.npz") as z:
            data = {k: z[k] for k in z.files}
        data["source"] = f"`landscapes` unit {units[0].key_hash[:10]}"
        return data
    inst, source = None, ""
    for seed in (_instances_seed(ctx.cfg), "thesis"):
        try:
            inst = load_instances(problem, seed, db=ctx.db)[sc["instance"]]
            source = ("native instance list" if seed != "thesis"
                      else "thesis-imported instance list")
            if seed == "thesis":
                prov.thesis_import = True
            break
        except (LookupError, IndexError):
            continue
    if inst is None:
        raise c.MissingData(f"instances of {problem}",
                            "run `pfn bench run priors --kind instances`")
    prob = get_problem(problem)
    rng = substream(BENCH_SEED, "prior_samples", problem, sc["instance"])
    trajs = prior_samples(prob, inst, sc["n"], sc["max_unsuccessful"], rng)
    return {"theta": np.stack([t.theta for t in trajs]),
            "J": np.stack([np.asarray(t.J, np.float32) for t in trajs]),
            "success": np.array([t.success for t in trajs]),
            "crash_step": np.array([t.crash_step or 0 for t in trajs], np.int32),
            "source": f"computed for this figure on instance {sc['instance']} of the {source} "
                      f"(seed `substream(BENCH_SEED, 'prior_samples', problem, instance)`, as the "
                      f"`landscapes` kind)"}


def _draw_samples_panel(ax: Any, data: dict[str, Any], problem: str, d: int, T: int,
                        fg: str) -> None:
    for J, cs in zip(data["J"], data["crash_step"]):
        J = np.asarray(J, float)
        end = int(cs) - 1 if int(cs) > 0 else J.size
        end = max(end, 1)
        steps = np.arange(1, end + 1)
        y = J[:end]
        ok = np.isfinite(y)
        ax.plot(steps[ok], y[ok], color=fg, lw=0.7, zorder=3 if cs else 2)
        if int(cs) > 0 and ok.any():
            last = np.flatnonzero(ok)[-1]
            ax.plot(steps[last], y[last], ls="none", zorder=4, **_crash_marker())
    ax.set_xlim(0, T)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.set_title(f"{c.problem_label(problem)} ($d_\\theta={d}$)", fontsize=7.5)


def draw_samples(ctx: c.Context) -> c.Output:
    from esbopfn.problems.base import cached_spec

    prov = c.Provenance()
    fg = c.fg_color(ctx.theme)
    problems = list(ctx.cfg.get("problems") or c.PROBLEMS)
    datas = {p: load_samples(ctx, p, prov) for p in problems}

    W = c.TEXT_WIDTH_IN
    n = len(problems)
    cell_w = (W - _S_LEFT) / n
    cell_h = c.BOX_H_PT * c.PT
    H = _S_LEGEND + _S_TITLE + cell_h + _S_BOTTOM
    fig = plt.figure(figsize=(W, H))
    axes = []
    numbers, data_lines = [], []
    for i, p in enumerate(problems):
        spec = cached_spec(p)
        ax = fig.add_axes(((_S_LEFT + i * cell_w) / W, _S_BOTTOM / H, cell_w / W, cell_h / H))
        _draw_samples_panel(ax, datas[p], p, spec.d, spec.T, fg)
        axes.append(ax)
        dd = datas[p]
        J = np.asarray(dd["J"], float)
        n_crash = int((np.asarray(dd["crash_step"]) > 0).sum())
        n_bad = int((~np.asarray(dd["success"], bool)).sum())
        fin = J[np.isfinite(J)]
        numbers.append(f"{p}: {len(J)} trajectories, {n_crash} crashed, {n_bad} unsuccessful, "
                       f"T = {spec.T}, J in [{fin.min():.4g}, {fin.max():.4g}]")
        data_lines.append(f"{p}: {dd['source']}")
    axes[0].set_ylabel(r"$J_t$")
    axes[n // 2].set_xlabel("step")
    cm = _crash_marker()
    handles = [Line2D([], [], color=fg, lw=1.0),
               Line2D([], [], color=fg, lw=1.0, marker=cm["marker"], ms=cm["ms"], mew=cm["mew"],
                      markeredgecolor=cm["color"], markerfacecolor=cm["color"])]
    fig.legend(handles, ["ran to horizon", "crashed"], loc="lower center",
               bbox_to_anchor=((_S_LEFT + 0.5 * n * cell_w) / W,
                               (_S_BOTTOM + cell_h + _S_TITLE) / H),
               ncol=2, frameon=False, fontsize=7, handlelength=1.8, handletextpad=0.7,
               columnspacing=1.5, borderaxespad=0.0, handler_map={handles[1]: _EndMarker()})
    sc = _samples_cfg(ctx.cfg)
    side = c.Sidecar(
        thesis_asset="priors/priors_samples.pdf", label="fig:setup:prior-samples",
        shown=[f"Five flush panels, one per problem, {sc['n']} cumulative-cost trajectories each "
               "on screened instance " f"{sc['instance']}.",
               "Every trajectory is drawn in the foreground color, a red cross marks the crash "
               "step of a crashed trajectory, a curve without a cross ran to the horizon.",
               "No tick values on either axis, each panel has its own linear y range."],
        metric=["y: cumulative cost J_t, linear scale, own limits per panel.",
                "x: simulation step from 0 to the problem's horizon T."],
        numbers=numbers,
        data=[f"Controllers theta ~ U(Theta), an unsuccessful trajectory is kept only while fewer "
              f"than {sc['max_unsuccessful']} of the kept ones are unsuccessful."] + data_lines,
        provenance=prov)
    return c.Output(side, fig=fig)


# ------------------------------------------------------------------------------- landscapes
def landscape_rgb(cost: np.ndarray, p_crash: np.ndarray) -> np.ndarray:
    """Additive RGB: blue scaled by the cost rank (1 = cheapest), red by the crash probability."""
    from esbopfn.benchmarks.metrics import cost_rank

    rank = np.nan_to_num(cost_rank(np.asarray(cost, float)), nan=0.0)
    crash = np.clip(np.nan_to_num(np.asarray(p_crash, float), nan=0.0), 0.0, 1.0)
    return np.clip(BLUE_RGB * rank[..., None] + RED_RGB * crash[..., None], 0.0, 1.0)


def _panel_cost(z: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(z["cost_mean_nocrash"] if "cost_mean_nocrash" in z else z["cost_mean"],
                      float)


def load_slices(ctx: c.Context, prov: c.Provenance
                ) -> dict[str, list[tuple[int, dict[str, np.ndarray]]]]:
    """Per problem the stored slices ``[(instance, arrays)]``, sorted by instance (lowest slice
    seed per instance, native instance list first). Without such units the imported thesis
    panels are used in their thesis row order."""
    sec = ctx.cfg.get("landscapes") if isinstance(ctx.cfg.get("landscapes"), dict) else {}
    wanted = [int(i) for i in sec.get("instances", [0, 1, 2])]
    out: dict[str, list[tuple[int, dict[str, np.ndarray]]]] = {}
    for p in list(ctx.cfg.get("problems") or c.PROBLEMS):
        units = ctx.db.query("landscapes", unit="slice", problem=p)
        best: dict[int, Any] = {}
        for u in sorted(units, key=lambda u: (u.key.get("instances_seed")
                                              != _instances_seed(ctx.cfg),
                                              int(u.key["slice_seed"]))):
            i = int(u.key["instance"])
            if i in wanted and i not in best and "row" not in u.key:
                best[i] = u
        if not best:
            # The imported thesis panels (key ``row``) show other instances in some columns,
            # so they are taken by their row in the thesis figure.
            thesis = sorted((u for u in units if "row" in u.key), key=lambda u: int(u.key["row"]))
            best = dict(enumerate(thesis[:len(wanted)]))
        if not best:
            continue
        prov.add_units("landscapes", best.values())
        rows = []
        for r in sorted(best):
            with np.load(Path(best[r].blob_dir) / "landscape.npz") as z:
                rows.append((int(best[r].key["instance"]), {k: z[k] for k in z.files}))
        out[p] = rows
    return out


def _draw_image(ax: Any, z: dict[str, np.ndarray]) -> str:
    cost = _panel_cost(z)
    p_crash = np.asarray(z["p_crash"], float)
    ext = float(z["extent"])
    ax.imshow(landscape_rgb(cost, p_crash), origin="lower", extent=(-ext, ext, -ext, ext),
              aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    fin = cost[np.isfinite(cost)]
    rng = f"J in [{fin.min():.4g}, {fin.max():.4g}]" if fin.size else "no successful cell"
    return f"mean crash probability {p_crash.mean():.2f}, {rng}"


def _draw_line(ax: Any, z: dict[str, np.ndarray]) -> str:
    from matplotlib.ticker import NullLocator

    thetas = np.asarray(z["thetas"], float).reshape(-1)
    cost = np.asarray(z["cost_mean"], float)
    p_crash = np.asarray(z["p_crash"], float)
    ax.plot(thetas, cost, color=LINE_BLUE, lw=1.0, zorder=3)
    if np.nanmin(cost) > 0:
        ax.set_yscale("log")
    ax.set_xlim(thetas.min(), thetas.max())
    twin = ax.twinx()
    twin.plot(thetas, p_crash, color=c.CRASH_COLOR, lw=0.8, alpha=0.8, zorder=2)
    twin.set_ylim(0, 1)
    for a in (ax, twin):
        a.xaxis.set_major_locator(NullLocator())
        a.yaxis.set_major_locator(NullLocator())
        a.yaxis.set_minor_locator(NullLocator())
        for spine in a.spines.values():
            spine.set_linewidth(0.5)
    fin = cost[np.isfinite(cost)]
    return (f"mean crash probability {p_crash.mean():.2f}, "
            f"J in [{fin.min():.4g}, {fin.max():.4g}]" if fin.size else "no successful theta")


def _color_key(ax: Any) -> None:
    n = 48
    q = np.linspace(0.0, 1.0, n)
    cost = np.tile(q[:, None], (1, n))           # rank rises with the row
    crash = np.tile(q[None, :], (n, 1))
    rgb = np.clip(BLUE_RGB * cost[..., None] + RED_RGB * crash[..., None], 0, 1)
    ax.imshow(rgb, origin="lower", extent=(0, 1, 0, 1), aspect="auto", interpolation="bilinear")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=5.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["worst", "best"], fontsize=5.5)
    ax.set_xlabel("crash prob.", fontsize=5.8, labelpad=1)
    ax.set_ylabel("objective", fontsize=5.8, labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def draw_landscapes(ctx: c.Context) -> c.Output:
    prov = c.Provenance()
    cols = load_slices(ctx, prov)
    if not cols:
        raise c.MissingData("`landscapes` slice units of every problem",
                            "run `pfn bench run priors`")
    problems = list(cols)
    n_rows = max(len(v) for v in cols.values())
    n_cols = len(problems)
    W = c.TEXT_WIDTH_IN
    cell = (W - _L_LEFT - _L_KEY_GAP - _L_KEY_LABEL - _L_KEY) / 5
    H = _L_TITLE + n_rows * cell + _L_BOTTOM
    fig = plt.figure(figsize=(W, H))
    notes = []
    row_instances: list[set[int]] = [set() for _ in range(n_rows)]
    for ci, p in enumerate(problems):
        for ri, (inst, z) in enumerate(cols[p]):
            row_instances[ri].add(inst)
            ax = fig.add_axes(((_L_LEFT + ci * cell) / W,
                               (_L_BOTTOM + (n_rows - 1 - ri) * cell) / H, cell / W, cell / H))
            note = _draw_line(ax, z) if np.asarray(z["thetas"]).ndim == 2 else _draw_image(ax, z)
            notes.append(f"{p} instance {inst}: {note}")
            if ri == 0:
                ax.annotate(c.problem_label(p), xy=(0.5, 1), xycoords="axes fraction",
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=6.3)
    for ri in range(n_rows):
        insts = sorted(row_instances[ri])
        label = f"instance {insts[0]}" if len(insts) == 1 else f"row {ri}"
        y = (_L_BOTTOM + (n_rows - 1 - ri) * cell + 0.5 * cell) / H
        fig.text((_L_LEFT - 0.06) / W, y, label, rotation=90, ha="right", va="center",
                 fontsize=6.0)
    side = min(_L_KEY, n_rows * cell)
    _color_key(fig.add_axes(((_L_LEFT + 5 * cell + _L_KEY_GAP + _L_KEY_LABEL) / W,
                              (_L_BOTTOM + (n_rows * cell - side) / 2) / H, side / W, side / H)))
    missing = [p for p in (ctx.cfg.get("problems") or c.PROBLEMS) if p not in cols]
    sidecar = c.Sidecar(
        thesis_asset="priors/priors_landscapes.pdf", label="fig:app:prior-landscapes",
        shown=[f"{n_cols} columns (problems) x {n_rows} rows (screened instances), touching "
               "cells without ticks.",
               "Each 2-D cell is the additive blend of blue (cost rank among the cell's own "
               "panel, brighter = cheaper) and red (crash probability), so cheap and safe is "
               "bright blue, always crashing is red, cheap but risky is magenta and expensive "
               "and safe is black. The key on the right is drawn with the same function.",
               "Boiler (d = 1) is a line: mean cost (blue, log scale) and crash probability "
               "(red, 0 to 1) over the whole interval."]
        + ([f"Columns without stored slices are left out: {', '.join(missing)}."]
           if missing else []),
        metric=["Cost: mean J_T of the non-crashed repeats per grid point (the `cost_mean_"
                "nocrash` array, so a Furuta controller that never crashes but misses the "
                "swing-up keeps its cost), colored by percentile rank.",
                "Crash probability: share of crashed repeats per grid point."],
        numbers=notes,
        data=["`landscapes` slice units: a slice through the center of Theta (oblique random "
              "plane for d > 2, the plane for d = 2, the interval for d = 1), every grid point "
              "simulated `repeats` times."],
        provenance=prov)
    return c.Output(sidecar, fig=fig)


FIGURES = [c.FigureSpec("priors_samples", draw_samples),
           c.FigureSpec("priors_landscapes", draw_landscapes)]

GROUP = PlotGroup(name=GROUP_NAME, needs=c.spec_needs(GROUP_NAME),
                  render=c.make_render(GROUP_NAME, FIGURES))
