"""Plot group ``readme``: the images of the repository README (light and dark variants).

- ``hero_3d``: early stopping and cross-trajectory prediction in one 3D view over time ``t``,
  cumulative cost ``J_t`` and one controller coordinate ``theta``. Three real rig trials of the
  rig without weight are the only lines (blue): one ran to the horizon ``T``, two are shown
  stopped early, with dotted drop lines. On the back wall ``t = T`` the thesis Furuta XDecPFN
  predicts the total cost of unrun controllers along a 1-D slice through the box (nested fills
  mu +- sigma and mu +- 2 sigma), a star marks the next controller (the argmax of the probability
  of improvement over the slice) and a flat red graph above the wall the predicted crash
  probability. The slice runs along the box dimension in which the prediction varies most, with
  the other coordinates at the incumbent, and every trial sits at its own coordinate on it.
- ``problems_strip``: one set of cost trajectories per benchmark problem (live rollouts of random
  controllers on the nominal plant).
- ``results_cards``: the headline numbers, read from the benchmark database.

Images are written as PNG at twice the screen resolution and as SVG, ``<name>.png`` (light) and
``<name>_dark.png`` (dark). ``python -m esbopfn.plots.readme_art <dir>`` writes them into
``<dir>`` (the repository's ``assets/``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from esbopfn.benchmarks.db import Db
from esbopfn.core import Trial
from esbopfn.plots import PlotGroup, style
from esbopfn.plots import _hw_common as C
from esbopfn.predictors.base import get_predictor
from esbopfn.util.seeds import substream

log = logging.getLogger(__name__)

PNG_DPI = 200          # 2x of a 100 dpi screen rendering
VARIANTS = ("light", "dark")

#: README defaults (``configs/benchmarks/readme.yaml`` may override any of them).
DEFAULTS: dict[str, Any] = {
    "hero": {
        "rig": "without_weight",
        "method": "bo",
        "provenance": "thesis",
        # (seed, trial index, shown stop as a fraction of T), the thesis slide's three trials
        "trials": [[4, 25, 1.0], [1, 12, 0.9 / 2.6], [0, 3, 0.6 / 2.6]],
        "extra_context": [4, [19, 20, 21, 22, 23, 24]],   # more trials of seed 4 as context
        "model": "xdecpfn/furuta/thesis/default",
        "n_slice": 41,
    },
    "strip": {"n_traj": 12, "seed": 20261029,
              "problems": ["boiler", "threetank_pi", "pt2_deadtime", "cartpole", "furuta"]},
    "cards": {"variant": "thesis", "n_ctx": 64},
}

PROBLEM_TITLE = {"boiler": "Boiler", "threetank_pi": "Three-Tank", "pt2_deadtime": "PT2",
                 "cartpole": "Cartpole", "furuta": "Furuta"}


@dataclass(frozen=True)
class Palette:
    bg: str
    fg: str
    muted: str
    grid: str
    blue: str
    orange: str
    red: str
    card: str


#: Wall view of the slide: theta along the front, t into the depth, the plane t = T at the back.
HERO_VIEW = (24.0, -72.0)                 # (elevation, azimuth) in degrees
HERO_BOX = (1.8, 1.2, 1.2)                # theta : t : J_t
HERO_POSITION = (-0.12, -0.10, 1.24, 1.2)

PALETTES = {
    "light": Palette(bg="#FFFFFF", fg="#1F2328", muted="#57606A", grid="#D0D7DE",
                     blue=style.RWTH_BLUE, orange=style.RWTH_ORANGE, red=style.RWTH_RED,
                     card="#F6F8FA"),
    "dark": Palette(bg="#0D1117", fg="#E6EDF3", muted="#9198A1", grid="#3D444D",
                    blue="#58A6FF", orange=style.RWTH_ORANGE, red="#FF6B6B", card="#161B22"),
}


def _cfg(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    return {**DEFAULTS[section], **((cfg or {}).get(section) or {})}


def _save(fig: Figure, out_dir: Path, name: str, tight: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, format=ext, dpi=PNG_DPI if ext == "png" else None,
                    facecolor=fig.get_facecolor(), bbox_inches="tight" if tight else None,
                    pad_inches=0.05)
        paths.append(p)
    plt.close(fig)
    return paths


def _rc(pal: Palette) -> dict[str, Any]:
    return {"font.family": "sans-serif", "font.size": 9, "text.usetex": False,
            "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
            "figure.facecolor": pal.bg, "axes.facecolor": pal.bg, "savefig.facecolor": pal.bg,
            "text.color": pal.fg, "axes.edgecolor": pal.muted, "axes.labelcolor": pal.fg,
            "xtick.color": pal.muted, "ytick.color": pal.muted, "svg.fonttype": "path"}


def _suffix(variant: str) -> str:
    return "" if variant == "light" else "_dark"


# =========================================================================== hero data
@dataclass
class HeroData:
    theta_name: str
    dim: int
    slice_x: np.ndarray            # (n,) values of the slice coordinate
    mu: np.ndarray
    sd: np.ndarray
    p_crash: np.ndarray
    pi: np.ndarray
    next_index: int
    trials: list[dict[str, Any]]   # x, t (s), J, stopped, label
    T_s: float
    incumbent_J: float
    variation: dict[str, float]


def hero_data(db: Db, cfg: dict[str, Any]) -> HeroData:
    """Rig trials from the database and the XDecPFN slice prediction."""
    from esbopfn.hardware.store import load_runs
    from esbopfn.problems import get_problem

    hc = _cfg(cfg, "hero")
    spec = get_problem("furuta").spec
    runs = {r.seed: r for r in load_runs(hc["rig"], db, provenance=hc["provenance"],
                                         methods=[hc["method"]])}
    if not runs:
        raise RuntimeError(f"no {hc['method']} runs of rig {hc['rig']!r} in the database "
                           "(import the thesis rig logs with `pfn hw import-logs`)")
    shown: list[tuple[Any, int]] = []
    for seed, idx, frac in hc["trials"]:
        tr = runs[int(seed)].trials[int(idx)]
        t_obs = tr.t_obs if frac >= 1.0 else max(1, min(tr.t_obs, round(frac * tr.T)))
        shown.append((tr, t_obs))
    context = [Trial(traj=tr.trajectory(), t_obs=t) for tr, t in shown]
    seed_x, idxs = hc["extra_context"]
    context += [runs[int(seed_x)].trials[int(i)].trial() for i in idxs]
    done = [c for c in context if c.complete and c.traj.success]
    if not done:
        raise RuntimeError("the hero context has no successful complete trial")
    inc = min(done, key=lambda c: c.J_obs)
    theta_inc = np.asarray(inc.theta, float)

    predictor = get_predictor(hc["model"], spec)
    n = int(hc["n_slice"])

    def slice_preds(d: int) -> tuple[np.ndarray, list[Any]]:
        th = np.tile(theta_inc, (n, 1))
        xs = np.linspace(spec.theta_lo[d], spec.theta_hi[d], n)
        th[:, d] = xs
        return xs, predictor.predict_unrun(th, context)

    variation, preds = {}, {}
    for d in range(spec.d):
        xs, pr = slice_preds(d)
        mu = np.array([p.mean_var()[0] for p in pr])
        ok = mu[np.isfinite(mu)]
        variation[spec.theta_names[d] if spec.theta_names else str(d)] = (
            float(ok.max() - ok.min()) if ok.size else 0.0)
        preds[d] = (xs, pr)
    names = list(variation)
    dim = int(np.argmax([variation[k] for k in names]))
    xs, pr = preds[dim]
    mv = np.array([p.mean_var() for p in pr], float)
    pi = np.array([float(p.cdf(inc.J_obs)) for p in pr])
    trials = []
    for (tr, t_obs), label in zip(shown, "ABC"):
        J = np.concatenate([[0.0], np.asarray(tr.J[:t_obs], float)])
        trials.append({"x": float(tr.theta[dim]), "t": np.arange(t_obs + 1) / 120.0, "J": J,
                       "stopped": t_obs < tr.t_obs or tr.early_stopped, "label": label,
                       "index": tr.index, "t_obs": t_obs, "J_T": float(tr.J_obs)})
    return HeroData(theta_name=names[dim], dim=dim, slice_x=xs, mu=mv[:, 0],
                    sd=np.sqrt(np.maximum(mv[:, 1], 0.0)),
                    p_crash=np.array([p.p_crash for p in pr]), pi=pi,
                    next_index=int(np.argmax(pi)), trials=trials, T_s=spec.T / 120.0,
                    incumbent_J=float(inc.J_obs), variation=variation)


# =========================================================================== hero drawing
def _poly(ax: Any, xs: np.ndarray, y: float, lo: np.ndarray, hi: np.ndarray, color: str,
          alpha: float, zorder: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = [(x, y, z) for x, z in zip(xs, hi)] + [(x, y, z) for x, z in zip(xs[::-1], lo[::-1])]
    pc = Poly3DCollection([verts], facecolors=color, edgecolors="none", alpha=alpha)
    pc.set_zorder(zorder)
    ax.add_collection3d(pc)


def draw_hero(h: HeroData, variant: str) -> Figure:
    pal = PALETTES[variant]
    with plt.rc_context(_rc(pal)):
        fig = plt.figure(figsize=(6.4, 4.0))
        ax = fig.add_subplot(projection="3d", computed_zorder=False)
        ax.set_proj_type("ortho")
        x0, x1 = float(h.slice_x[0]), float(h.slice_x[-1])
        T = h.T_s
        j_traj = max(float(np.max(t["J"])) for t in h.trials)
        zmax = float(min(np.nanmax(h.mu + 2 * h.sd), 1.9 * j_traj))
        zmax = max(zmax, 1.15 * j_traj)
        crash_base, crash_h = 1.06 * zmax, 0.28 * zmax

        # light grid on the wall (t = T) and the floor (J = 0), open ended
        gkw = dict(color=pal.grid, lw=0.45, zorder=0)
        xt = np.linspace(x0, x1, 6)
        zt = np.linspace(0, zmax, 5)[1:]
        tt_ = np.linspace(0, T, 5)[1:]
        for x in xt:
            ax.plot([x, x], [T, T], [0, zmax * 0.98], **gkw)
            ax.plot([x, x], [0, T], [0, 0], **gkw)
        for z in zt[:-1]:
            ax.plot([x0, x1 + 0.02 * (x1 - x0)], [T, T], [z, z], **gkw)
        for t in tt_:
            ax.plot([x0, x1 + 0.02 * (x1 - x0)], [t, t], [0, 0], **gkw)

        # predicted total cost at T: nested fills, no edge lines
        lo2, hi2 = np.clip(h.mu - 2 * h.sd, 0, zmax), np.clip(h.mu + 2 * h.sd, 0, zmax)
        lo1, hi1 = np.clip(h.mu - h.sd, 0, zmax), np.clip(h.mu + h.sd, 0, zmax)
        _poly(ax, h.slice_x, T, lo2, hi2, pal.orange, 0.22, 1)
        _poly(ax, h.slice_x, T, lo1, hi1, pal.orange, 0.45, 1.1)

        # crash probability: a flat graph above the back wall
        pc = np.clip(h.p_crash, 0, 1)
        _poly(ax, h.slice_x, T, np.full_like(pc, crash_base), crash_base + crash_h * pc,
              pal.red, 0.16, 1)
        line = np.where(pc > 0.01, crash_base + crash_h * pc, np.nan)
        ax.plot(h.slice_x, np.full_like(pc, T), line, color=pal.red, lw=1.0, zorder=1.2)
        ax.plot([x0, x1], [T, T], [crash_base, crash_base], color=pal.muted, lw=0.6,
                zorder=1.2)
        ax.text(x0, T, crash_base + 0.5 * crash_h, "crash %  ", color=pal.red, fontsize=7,
                ha="right", va="center", zorder=5)

        # axes: theta along the front, t into the depth, J_t up
        akw = dict(color=pal.muted, lw=0.9, zorder=2)
        ax.plot([x0, x1 + 0.04 * (x1 - x0)], [0, 0], [0, 0], **akw)
        ax.plot([x0, x0], [0, T], [0, 0], **akw)
        ax.plot([x0, x0], [0, 0], [0, 0.9 * zmax], **akw)
        ax.text(0.5 * (x0 + x1), 0, -0.07 * zmax, r"$\theta$", ha="center", va="top",
                fontsize=11, color=pal.fg)
        ax.text(x0 - 0.03 * (x1 - x0), 0.5 * T, -0.04 * zmax, r"$t$", ha="right", va="top",
                fontsize=11, color=pal.fg)
        ax.text(x0 - 0.03 * (x1 - x0), 0, 0.45 * zmax, r"$J_t$", ha="right", va="center",
                fontsize=11, color=pal.fg)
        ax.text(x0, 1.04 * T, 0.0, r"$T$", ha="left", va="bottom", fontsize=9,
                color=pal.muted)

        # dotted drop lines of the trials stopped early
        for tr in h.trials:
            if tr["stopped"]:
                ax.plot([tr["x"]] * 2, [tr["t"][-1]] * 2, [0, tr["J"][-1]], color=pal.blue,
                        lw=0.9, ls=(0, (1, 1.6)), alpha=0.75, zorder=2.5)
        # the trials: the only lines inside the space
        for tr in h.trials:
            ax.plot(np.full_like(tr["t"], tr["x"]), tr["t"], tr["J"], color=pal.blue, lw=1.8,
                    solid_capstyle="round", zorder=3)
            if not tr["stopped"]:
                ax.scatter([tr["x"]], [T], [tr["J"][-1]], s=16, color=pal.blue, zorder=3.5,
                           depthshade=False, linewidths=0)
        # next controller: star on the predicted cost at T
        k = h.next_index
        ax.scatter([h.slice_x[k]], [T], [min(h.mu[k], zmax)], marker="*", s=170,
                   color=pal.orange, edgecolors="#B07800", linewidths=0.5, zorder=4,
                   depthshade=False)

        ax.set_xlim(x0, x1)
        ax.set_ylim(0, T)
        ax.set_zlim(0, crash_base + crash_h)
        ax.set_box_aspect(HERO_BOX)
        ax.view_init(elev=HERO_VIEW[0], azim=HERO_VIEW[1])
        ax.set_axis_off()
        ax.set_position(HERO_POSITION)
    return fig


def hero_sidecar(h: HeroData, cfg: dict[str, Any]) -> list[str]:
    hc = _cfg(cfg, "hero")
    lines = [f"model `{hc['model']}` (`predict_unrun` over {len(h.slice_x)} slice points), rig "
             f"`{hc['rig']}`, method `{hc['method']}`, provenance `{hc['provenance']}`",
             f"slice dimension `{h.theta_name}` (index {h.dim}), range of the predicted mean per "
             "dimension: " + ", ".join(f"{k} {v:.1f}" for k, v in h.variation.items()),
             f"incumbent J = {h.incumbent_J:.2f}, next controller at {h.theta_name} = "
             f"{h.slice_x[h.next_index]:.3f} (PI {h.pi[h.next_index]:.3f}, mu "
             f"{h.mu[h.next_index]:.1f}, p_crash {h.p_crash[h.next_index]:.3f})"]
    for tr in h.trials:
        lines.append(f"trial {tr['label']}: index {tr['index']}, {h.theta_name} = {tr['x']:.3f}, "
                     f"shown to step {tr['t_obs']} ({'stopped' if tr['stopped'] else 'complete'}),"
                     f" J at the end of the shown part {tr['J'][-1]:.1f}, recorded J_T "
                     f"{tr['J_T']:.1f}")
    return lines


# =========================================================================== problems strip
def strip_data(cfg: dict[str, Any]) -> dict[str, list[Any]]:
    from esbopfn.problems import get_problem
    from esbopfn.problems.prior import sample_theta

    sc = _cfg(cfg, "strip")
    out = {}
    for p in sc["problems"]:
        prob = get_problem(p)
        inst = prob.nominal_instance(noise=True)
        rng = substream(int(sc["seed"]), "readme-strip", p)
        out[p] = [prob.simulate(sample_theta(prob.spec, rng), inst, rng)
                  for _ in range(int(sc["n_traj"]))]
    return out


def draw_strip(data: dict[str, list[Any]], variant: str) -> Figure:
    pal = PALETTES[variant]
    with plt.rc_context(_rc(pal)):
        fig, axes = plt.subplots(1, len(data), figsize=(8.0, 1.75))
        for ax, (p, trajs) in zip(np.atleast_1d(axes), data.items()):
            peaks = [float(np.max(t.J[:(t.crash_step - 1) if t.crash_step else len(t.J)],
                                  initial=0.0)) for t in trajs]
            top = float(np.quantile(peaks, 0.85)) * 1.25 if peaks else 1.0
            top = top if top > 0 else 1.0
            for t in trajs:
                T = len(t.J)
                end = (t.crash_step - 1) if t.crash_step else T
                x = np.arange(1, end + 1) / T
                ax.plot(x, np.minimum(t.J[:end], top * 1.5), color=pal.blue, lw=0.9,
                        alpha=0.65)
                if t.crash_step and end > 0:
                    ax.scatter([x[-1]], [min(float(t.J[end - 1]), top)], marker="x", s=14,
                               color=pal.red, linewidths=1.0, zorder=4)
            ax.set_ylim(0, top)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 1], ["0", "T"])
            ax.set_yticks([])
            ax.set_title(PROBLEM_TITLE.get(p, p), fontsize=9, color=pal.fg)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        np.atleast_1d(axes)[0].set_ylabel(r"$J_t$")
        fig.tight_layout(w_pad=0.8)
    return fig


# =========================================================================== results cards
def _series(db: Db, figure: str, variant: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for u in db.query("thesis_series", figure=figure, variant=variant):
        out[u.key["series"]][u.key["problem"]] = json.loads(
            (Path(u.blob_dir) / "series.json").read_text())
    return out


PRED_ROWS: list[tuple[str, list[str]]] = [
    ("FT-PFN", ["ftpfn/{p}/thesis/default", "ftpfn/{p}/default/default"]),
    ("XDecPFN", ["xdecpfn/{p}/thesis/default", "xdecpfn/{p}/default/default"]),
    ("DecPFN", ["decpfn/{p}/binned/1M"]),
    ("Trajectory-TabPFN", ["tabpfn/25", "tabpfn/50", "tabpfn/100", "tabpfn/250"]),
    ("Segment-TabPFN", ["segment-tabpfn/split", "segment-tabpfn/rate"]),
    ("Segment-GP", ["segment-gp"]),
    ("LC-PFN", ["lcpfn/{p}/binned/1M"]),
]
OPT_ROWS = ["xdecpfn-pi", "ftpfn-ei", "decpfn", "decpfn-h", "esbo-gp", "esbo-t-tabpfn",
            "esbo-oracle", "bo", "esrs"]


def cards_data(db: Db, cfg: dict[str, Any], hw_cfg: dict[str, Any] | None = None
               ) -> dict[str, Any]:
    """Every number of the cards, from the database only."""
    cc = _cfg(cfg, "cards")
    variant, n_ctx = cc["variant"], int(cc["n_ctx"])
    out: dict[str, Any] = {"prediction": [], "optimization": [], "hardware": {},
                           "missing": []}
    cross = _series(db, "fig3_cross", variant)
    for label, ids in PRED_ROWS:
        best = None
        for sid in ids:
            s = cross.get(sid, {}).get("pooled")
            if not s:
                continue
            d = s["data"]
            pairs = dict(zip(d["n_ctx"], d["y"]))
            y = pairs.get(n_ctx, pairs.get(-1))
            if y is not None and (best is None or y > best[1]):
                best = (sid, float(y))
        if best:
            out["prediction"].append({"label": label, "series": best[0], "value": best[1]})
        else:
            out["missing"].append(f"prediction {label}")

    finals, ref = _series(db, "optplay_finals", variant), _series(db, "optplay_reference",
                                                                  variant)
    jb = ref.get("J_best_all_cells", {})
    probs = sorted(jb)
    for m in OPT_ROWS:
        meds = []
        for p in probs:
            if p not in finals.get(m, {}) or p not in finals.get("esrs", {}):
                break
            d, e, r = finals[m][p]["data"], finals["esrs"][p]["data"], jb[p]["data"]
            jbest = dict(zip(r["instance"], r["J_best"]))
            jesrs = dict(zip(e["instance"], e["incumbent_time"]))
            sc = [(J - jbest[i]) / (jesrs[i] - jbest[i])
                  for i, J in zip(d["instance"], d["incumbent_time"])
                  if i in jesrs and i in jbest and jesrs[i] != jbest[i]]
            meds.append(float(np.median(sc)))
        if probs and len(meds) == len(probs):
            out["optimization"].append({"method": m, "value": float(np.log10(np.mean(meds))),
                                        "per_problem": dict(zip(probs, meds))})
        else:
            out["missing"].append(f"optimization {m}")

    from esbopfn.hardware import metrics as M
    from esbopfn.hardware.store import load_runs
    from esbopfn.plots import hardware as H

    hw = hw_cfg if hw_cfg is not None else _hardware_cfg()
    for rig in H._rigs(hw):
        runs = load_runs(rig, db, provenance=H._opt_provenance(hw), methods=H._methods(hw))
        if not runs:
            out["missing"].append(f"hardware {rig}")
            continue
        reg = M.regret_curves(runs, "wall", reset_method=H._reset_method(rig),
                              methods=H._methods(hw))
        crashes = M.crash_counts(runs)
        rows = []
        for m in H._methods(hw):
            if m not in reg["methods"]:
                continue
            final = reg["methods"][m]["per_seed"][:, -1]
            final = final[np.isfinite(final)]
            rows.append({"method": m,
                         "regret": float(np.median(final)) if final.size else math.nan,
                         "crashes": float(np.mean(list(crashes[m].values())))})
        out["hardware"][rig] = rows
    return out


def _hardware_cfg() -> dict[str, Any]:
    from esbopfn.plots import group_config

    return group_config("hardware")


def _card(ax: Any, pal: Palette, title: str, subtitle: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                transform=ax.transAxes, facecolor=pal.card,
                                edgecolor=pal.grid, linewidth=0.8, zorder=0, clip_on=False))
    ax.text(0.05, 0.93, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=pal.fg, va="top")
    ax.text(0.05, 0.84, subtitle, transform=ax.transAxes, fontsize=7.5, color=pal.muted,
            va="top")


def _bars(fig: Figure, ax: Any, pal: Palette, rows: Sequence[tuple[str, float, str]],
          fmt: str, variant: str) -> None:
    """Horizontal bars inside a card, labels left, values at the bar ends."""
    pos = ax.get_position()
    inner = fig.add_axes([pos.x0 + 0.40 * pos.width, pos.y0 + 0.06 * pos.height,
                          0.50 * pos.width, 0.70 * pos.height])
    inner.set_facecolor("none")
    vals = np.array([v for _, v, _ in rows], float)
    lo, hi = min(0.0, float(np.nanmin(vals))), max(0.0, float(np.nanmax(vals)))
    span = (hi - lo) or 1.0
    y = np.arange(len(rows))[::-1]
    label_tf = inner.get_yaxis_transform()        # x in axes fraction, y in data
    for yi, (label, v, col) in zip(y, rows):
        c = col if not (variant == "dark" and col == "#000000") else "#FFFFFF"
        inner.barh(yi, v, height=0.62, color=c, alpha=0.9)
        inner.text(-0.04, yi, label, ha="right", va="center", fontsize=7.5, color=pal.fg,
                   transform=label_tf)
        if hi > 0:      # values right of the bar end
            inner.text(max(v, 0.0) + 0.03 * span, yi, fmt.format(v), ha="left",
                       va="center", fontsize=7, color=pal.muted)
        else:           # all bars point left: values right of the zero line
            inner.text(0.03 * span, yi, fmt.format(v), ha="left", va="center", fontsize=7,
                       color=pal.muted)
    inner.axvline(0, color=pal.muted, lw=0.6)
    inner.set_xlim(lo, hi + 0.22 * span)
    inner.set_ylim(-0.6, len(rows) - 0.4)
    inner.set_axis_off()


def draw_cards(data: dict[str, Any], variant: str) -> Figure:
    pal = PALETTES[variant]
    theme = "white" if variant == "light" else "black"
    with plt.rc_context(_rc(pal)):
        fig = plt.figure(figsize=(8.0, 6.4))
        gs = fig.add_gridspec(2, 2, wspace=0.06, hspace=0.08, left=0.01, right=0.99,
                              bottom=0.01, top=0.99)
        cards = []
        pred = sorted(data["prediction"], key=lambda r: -r["value"])
        cards.append(("Prediction", "pooled normalized LL at 64 context trials, higher is "
                      "better", [(r["label"], r["value"], style.method_color(r["label"], theme))
                                 for r in pred], "{:+.2f}"))
        opt = sorted(data["optimization"], key=lambda r: r["value"])
        cards.append(("Optimization (simulation)", "log10 scaled regret at 100 % time, 5 "
                      "problems, lower is better",
                      [(style.display_name(r["method"]), r["value"],
                        style.method_color(r["method"], theme)) for r in opt], "{:+.2f}"))
        for rig, rows in list(data["hardware"].items())[:1]:
            rows = sorted(rows, key=lambda r: r["regret"])
            cards.append((f"Hardware rig, {C.rig_title(rig)}",
                          "median log10 regret at the end of the budget, lower is better",
                          [(style.display_name(r["method"]), r["regret"],
                            style.method_color(r["method"], theme)) for r in rows], "{:.2f}"))
        crash_rows = []
        for rig, rows in data["hardware"].items():
            for r in rows:
                crash_rows.append((rig, r))
        if data["hardware"]:
            rigs = list(data["hardware"])
            per_m: dict[str, list[float]] = defaultdict(list)
            for _, r in crash_rows:
                per_m[r["method"]].append(r["crashes"])
            rows2 = sorted(((m, float(np.mean(v))) for m, v in per_m.items()),
                           key=lambda r: r[1])
            cards.append(("Hardware crashes", f"mean crashed trials per run, {len(rigs)} rig "
                          "configurations, lower is better",
                          [(style.display_name(m), v, style.method_color(m, theme))
                           for m, v in rows2], "{:.1f}"))
        for k, (title, sub, rows, fmt) in enumerate(cards[:4]):
            ax = fig.add_subplot(gs[k // 2, k % 2])
            _card(ax, pal, title, sub)
            if rows:
                _bars(fig, ax, pal, rows, fmt, variant)
    return fig


def cards_sidecar(data: dict[str, Any]) -> list[str]:
    lines = [f"prediction {r['label']} (`{r['series']}`): {r['value']:+.3f}"
             for r in data["prediction"]]
    lines += [f"optimization {r['method']}: {r['value']:+.3f} (per-problem median scaled regret "
              + ", ".join(f"{p} {v:.3f}" for p, v in r["per_problem"].items()) + ")"
              for r in data["optimization"]]
    for rig, rows in data["hardware"].items():
        lines += [f"hardware {rig} {r['method']}: median log10 regret {r['regret']:.3f}, mean "
                  f"crashes {r['crashes']:.1f}" for r in rows]
    if data["missing"]:
        lines.append("not in the database (not drawn): " + ", ".join(data["missing"]))
    return lines


# =========================================================================== group
def needs(cfg: dict[str, Any], db: Db) -> list[tuple[str, dict[str, Any]]]:
    from esbopfn.plots import hardware as H

    hw = _hardware_cfg()
    return [(k, key) for k, key in H.needs(hw, db) if k == "hardware_optimization"]


def render_all(cfg: dict[str, Any], db: Db, out_dir: Path,
               variants: Sequence[str] = VARIANTS) -> dict[str, Any]:
    """Write every README image (both variants) and a sidecar per image into ``out_dir``."""
    out_dir = Path(out_dir)
    out: dict[str, Any] = {}
    try:
        hero = hero_data(db, cfg)
    except (RuntimeError, KeyError, IndexError, FileNotFoundError) as exc:
        log.warning("README images skipped (rig runs or the XDecPFN checkpoint missing): %s",
                    exc)
        return out
    strip = strip_data(cfg)
    cards = cards_data(db, cfg)
    for v in variants:
        _save(draw_hero(hero, v), out_dir, "hero_3d" + _suffix(v), tight=False)
        _save(draw_strip(strip, v), out_dir, "problems_strip" + _suffix(v))
        _save(draw_cards(cards, v), out_dir, "results_cards" + _suffix(v))
    C.write_sidecar(out_dir, "hero_3d",
                    shown=["axes t (seconds), J_t (raw cost), theta (one controller coordinate);"
                           " blue lines = real rig trials, orange fills = predicted total cost "
                           "mu +- sigma and mu +- 2 sigma at t = T, star = next controller "
                           "(argmax PI over the slice), red graph above the wall = predicted "
                           "crash probability, dotted drop lines = trials shown stopped early"],
                    metric=["mean and standard deviation of the predicted bar distribution "
                            "without the crash atom, PI = P(J_T < incumbent J) with the crash "
                            "atom counted as no improvement"],
                    data=["`hardware_optimization` units of the rig and the model checkpoint "
                          "named below"],
                    numbers=hero_sidecar(hero, cfg))
    sc = _cfg(cfg, "strip")
    C.write_sidecar(out_dir, "problems_strip",
                    shown=[f"{sc['n_traj']} random controllers per problem on the nominal "
                           "plant (with measurement noise), cumulative cost over t / T, red "
                           "crosses mark crashes"],
                    metric=["raw cumulative cost J_t (up to the crash for a crashed trial), y "
                            "axis per panel up to 1.25 x the 85 % quantile of the last values"],
                    data=[f"live rollouts, seed {sc['seed']}"],
                    numbers=[f"{p}: {sum(t.crash_step is not None for t in ts)} of {len(ts)} "
                             "crashed" for p, ts in strip.items()])
    C.write_sidecar(out_dir, "results_cards",
                    shown=["four cards: pooled prediction score, simulation optimization regret,"
                           " hardware regret on the first rig, hardware crashes (mean over rigs)"],
                    metric=["prediction: pooled median ll_norm at the context size (context-"
                            "blind models at their single value), best variant per family",
                            "optimization: log10 of the mean over problems of the per-problem "
                            "median scaled regret at 100 % of the simulated-time budget",
                            "hardware: median over all seeds of the unscaled log10 regret at "
                            "the end of the wall-clock budget (as the thesis text quotes it)"],
                    data=["`thesis_series` (figures fig3_cross, optplay_finals, "
                          f"optplay_reference, variant `{_cfg(cfg, 'cards')['variant']}`) and "
                          "`hardware_optimization` units"],
                    numbers=cards_sidecar(cards))
    out.update(hero=hero, strip=strip, cards=cards)
    return out


def _render(cfg: dict[str, Any], db: Db, out_dir: Path, fmt: str, theme: str) -> None:
    render_all(cfg, db, out_dir)


GROUP = PlotGroup(name="readme", needs=needs, render=_render)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="write the README images")
    ap.add_argument("out", type=Path, help="output directory (the repository's assets/)")
    args = ap.parse_args(argv)
    from esbopfn.plots import group_config

    render_all(group_config("readme"), Db(), args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
