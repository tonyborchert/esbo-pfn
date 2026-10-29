"""Plot group ``readme``: the images of the repository README (light and dark variants).

All images share one GitHub-like look (sans-serif text, GitHub's Primer colors, no LaTeX) and one
color language: blue for measured cost trajectories, orange for the model's forecast, red for
crashes, stops and forecasts worse than the best trial so far, green for a new best.

- ``stop_demo`` (animated GIF): real rig trials replayed through the Furuta XDecPFN. For every
  shown trial of one rig run the forecast of the final cost is recomputed from the trial's prefix
  and all earlier trials of the run, at the decision steps of the pure model loop, and the stop
  rules of ``xdecpfn-pi`` (``exceed`` and ``naive`` at ``tau``) are applied. A stopped trial then
  reveals the rest of its logged trajectory (the source runs never stop a trial). Left: the
  pendulum drawn from the logged arm and pendulum angles.
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

Static images are written as SVG and as PNG at twice the screen resolution, ``<name>.svg|png``
(light) and ``<name>_dark.svg|png`` (dark), the demo as ``stop_demo.gif`` and
``stop_demo_dark.gif``. Every image gets a sidecar ``<name>.md`` with the numbers it is drawn
from. ``python -m esbopfn.plots.readme_art <dir>`` writes them into ``<dir>`` (the repository's
``assets/``).
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
#: System sans-serif fonts, the first one installed is used (Helvetica on macOS).
FONTS = ("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")
#: Display smoothing of the forecast density (standard deviation in cost units).
DENSITY_SMOOTH = 2.5

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
    "demo": {
        "rig": "without_weight",
        "method": "bo",          # a run without early stopping, so every trial is logged in full
        "seed": 0,
        "provenance": "thesis",
        "trials": [9, 11, 19],   # kept and new best, stopped (ends far worse), stopped (crashes)
        "model": "xdecpfn/furuta/thesis/default",
        "tau": 0.99,             # the naive rule of xdecpfn-pi
        "fps": 20,
        "fast_after_s": 1.6,     # play faster after this many seconds of a trial
        "fast_speed": 4.0,
        "y_max": 420.0,
    },
    "strip": {"n_traj": 12, "seed": 20261029,
              "problems": ["boiler", "threetank_pi", "pt2_deadtime", "cartpole", "furuta"]},
    "cards": {"variant": "thesis", "n_ctx": 64},
}

PROBLEM_TITLE = {"boiler": "Boiler", "threetank_pi": "Three-Tank", "pt2_deadtime": "PT2 dead-time",
                 "cartpole": "Cartpole", "furuta": "Furuta pendulum"}


@dataclass(frozen=True)
class Palette:
    bg: str
    fg: str
    muted: str
    faint: str
    grid: str
    card: str
    bar: str        # neutral bars
    blue: str
    orange: str
    red: str
    green: str


#: GitHub Primer colors (light and dark default themes).
PALETTES = {
    "light": Palette(bg="#FFFFFF", fg="#1F2328", muted="#59636E", faint="#818B98",
                     grid="#D1D9E0", card="#F6F8FA", bar="#C8D1DA", blue="#0969DA",
                     orange="#F08A24", red="#CF222E", green="#1A7F37"),
    "dark": Palette(bg="#0D1117", fg="#F0F6FC", muted="#9198A1", faint="#656C76",
                    grid="#3D444D", card="#151B23", bar="#3D444D", blue="#4493F8",
                    orange="#F0883E", red="#F85149", green="#3FB950"),
}

#: Wall view of the slide: theta along the front, t into the depth, the plane t = T at the back.
HERO_VIEW = (24.0, -72.0)                 # (elevation, azimuth) in degrees
HERO_BOX = (1.8, 1.2, 1.2)                # theta : t : J_t
HERO_POSITION = (-0.10, -0.12, 1.20, 1.22)


def _cfg(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    return {**DEFAULTS[section], **((cfg or {}).get(section) or {})}


def _save(fig: Figure, out_dir: Path, name: str, tight: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, format=ext, dpi=PNG_DPI if ext == "png" else None,
                    facecolor=fig.get_facecolor(), bbox_inches="tight" if tight else None,
                    pad_inches=0.08, metadata={"Date": None} if ext == "svg" else None)
        paths.append(p)
    plt.close(fig)
    return paths


def _font() -> str:
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONTS if f in names), "DejaVu Sans")


def _rc(pal: Palette) -> dict[str, Any]:
    return {"font.family": _font(), "font.size": 9, "text.usetex": False,
            "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
            "figure.facecolor": pal.bg, "axes.facecolor": pal.bg, "savefig.facecolor": pal.bg,
            "text.color": pal.fg, "axes.edgecolor": pal.grid, "axes.labelcolor": pal.muted,
            "xtick.color": pal.muted, "ytick.color": pal.muted, "xtick.major.size": 0,
            "ytick.major.size": 0, "axes.linewidth": 0.8, "svg.fonttype": "path",
            "svg.hashsalt": "esbo-pfn", "lines.solid_capstyle": "round"}


def _suffix(variant: str) -> str:
    return "" if variant == "light" else "_dark"


def _predictor(cache: dict[str, Any] | None, model: str, spec: Any) -> Any:
    """One predictor instance per model id and render call."""
    if cache is None:
        return get_predictor(model, spec)
    if model not in cache:
        cache[model] = get_predictor(model, spec)
    return cache[model]


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


def hero_data(db: Db, cfg: dict[str, Any], predictors: dict[str, Any] | None = None
              ) -> HeroData:
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

    predictor = _predictor(predictors, hc["model"], spec)
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
        _poly(ax, h.slice_x, T, lo1, hi1, pal.orange, 0.50, 1.1)

        # crash probability: a flat graph above the back wall
        pc = np.clip(h.p_crash, 0, 1)
        _poly(ax, h.slice_x, T, np.full_like(pc, crash_base), crash_base + crash_h * pc,
              pal.red, 0.16, 1)
        line = np.where(pc > 0.01, crash_base + crash_h * pc, np.nan)
        ax.plot(h.slice_x, np.full_like(pc, T), line, color=pal.red, lw=1.0, zorder=1.2)
        ax.plot([x0, x1], [T, T], [crash_base, crash_base], color=pal.faint, lw=0.6,
                zorder=1.2)

        # axes: theta along the front, t into the depth, J_t up
        akw = dict(color=pal.faint, lw=0.9, zorder=2)
        ax.plot([x0, x1 + 0.04 * (x1 - x0)], [0, 0], [0, 0], **akw)
        ax.plot([x0, x0], [0, T], [0, 0], **akw)
        ax.plot([x0, x0], [0, 0], [0, 0.9 * zmax], **akw)
        lab = dict(fontsize=10.5, color=pal.muted)
        ax.text(0.5 * (x0 + x1), 0, -0.08 * zmax, "controller parameter", ha="center",
                va="top", **lab)
        ax.text(x0 - 0.03 * (x1 - x0), 0.55 * T, -0.05 * zmax, "time", ha="right", va="top",
                **lab)
        ax.text(x0 - 0.03 * (x1 - x0), 0, 0.45 * zmax, "cost", ha="right", va="center", **lab)

        # annotations on the wall
        ann = dict(fontsize=10, zorder=6)
        ax.text(x0, T, crash_base + 0.62 * crash_h, "crash risk  ", color=pal.red, ha="right",
                va="center", **ann)
        k_lab = int(0.06 * (len(h.slice_x) - 1))
        ax.text(h.slice_x[k_lab], T, lo2[k_lab] - 0.05 * zmax, "forecast of the final cost",
                color=pal.orange, ha="left", va="top", **ann)

        # dotted drop lines of the trials stopped early
        for tr in h.trials:
            if tr["stopped"]:
                ax.plot([tr["x"]] * 2, [tr["t"][-1]] * 2, [0, tr["J"][-1]], color=pal.blue,
                        lw=0.9, ls=(0, (1, 1.6)), alpha=0.75, zorder=2.5)
        # the trials: the only lines inside the space
        for tr in h.trials:
            ax.plot(np.full_like(tr["t"], tr["x"]), tr["t"], tr["J"], color=pal.blue, lw=2.0,
                    solid_capstyle="round", zorder=3)
            if not tr["stopped"]:
                ax.scatter([tr["x"]], [T], [tr["J"][-1]], s=18, color=pal.blue, zorder=3.5,
                           depthshade=False, linewidths=0)
            else:
                ax.scatter([tr["x"]], [tr["t"][-1]], [tr["J"][-1]], s=22, marker="o",
                           color=pal.bg, edgecolors=pal.red, linewidths=1.4, zorder=3.6,
                           depthshade=False)
        # next controller: star on the predicted cost at T
        k = h.next_index
        ax.scatter([h.slice_x[k]], [T], [min(h.mu[k], zmax)], marker="*", s=190,
                   color=pal.orange, edgecolors=pal.fg, linewidths=0.5, zorder=4,
                   depthshade=False)
        ax.text(h.slice_x[k], T, min(h.mu[k], zmax) - 0.10 * zmax, "next try", ha="center",
                va="top", color=pal.fg, fontsize=10, zorder=6)

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


# =========================================================================== stop demo data
def _smooth(rows: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along the last axis (``sigma`` in samples)."""
    if sigma <= 0:
        return rows
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.array([np.convolve(row, k, mode="same") for row in rows])


@dataclass
class DemoAct:
    """One replayed rig trial."""

    index: int
    J: np.ndarray                  # (t_obs,) logged cumulative cost
    x: np.ndarray                  # (t_obs, 2) logged [phi_arm, phi_pend] in rad
    t_obs: int
    T: int
    crashed: bool
    J_final: float                 # logged J_T, inf if crashed
    J_best: float                  # incumbent before the trial
    stop_step: int | None          # first step at which the replayed rules stop
    stop_rule: str | None          # "forecast" or "exceed"
    steps: np.ndarray              # forecast steps (0 = before the trial starts)
    p_worse: np.ndarray            # Pr[J_T >= J_best] (crash counted as worse) per step
    p_crash: np.ndarray
    dens: np.ndarray               # (n_steps, n_y) density of the final cost on DemoData.y
    median: np.ndarray             # median of the final cost (inf if beyond the value mass)

    @property
    def end_step(self) -> int:
        """Last step drawn live (the stop step, else the logged end)."""
        return self.stop_step if self.stop_step is not None else self.t_obs

    def forecast_index(self, step: int) -> int:
        return int(np.searchsorted(self.steps, step, side="right") - 1)


@dataclass
class DemoData:
    acts: list[DemoAct]
    y: np.ndarray
    dt: float
    tau: float
    rig: str
    method: str
    seed: int
    model: str


def demo_data(db: Db, cfg: dict[str, Any], predictors: dict[str, Any] | None = None
              ) -> DemoData:
    """Replay the configured rig trials through the model and the ``xdecpfn-pi`` stop rules."""
    from esbopfn.grids import decoder_grid
    from esbopfn.hardware.store import load_runs
    from esbopfn.problems import get_problem
    from esbopfn.strategies import rules

    dc = _cfg(cfg, "demo")
    spec = get_problem("furuta").spec
    runs = [r for r in load_runs(dc["rig"], db, provenance=dc["provenance"],
                                 methods=[dc["method"]]) if r.seed == int(dc["seed"])]
    if not runs:
        raise RuntimeError(f"no {dc['method']} run with seed {dc['seed']} of rig {dc['rig']!r}")
    run = runs[0]
    predictor = _predictor(predictors, dc["model"], spec)
    tau = float(dc["tau"])
    grid = decoder_grid(spec)
    grid = grid[grid < spec.T]
    history = [t.trial() for t in run.trials]
    y = np.linspace(0.0, float(dc["y_max"]), 421)
    acts = []
    for k in dc["trials"]:
        k = int(k)
        tr = run.trials[k]
        if k == 0 or tr.t_obs < 2:
            raise IndexError(f"trial {k} of the demo run has no predecessor or no steps")
        J_best = float(run.trials[k - 1].incumbent_J)
        live = tr.trajectory(live=True)
        J = np.asarray(tr.J, float)
        steps = np.concatenate([[0], grid[grid <= tr.t_obs]]).astype(int)
        preds = [predictor.predict(Trial(traj=live, t_obs=int(s)), history[:k]) for s in steps]
        p_worse = np.array([rules.prob_at_least(p, J_best) for p in preds])
        p_crash = np.array([float(p.p_crash) for p in preds])
        cdf = np.array([np.asarray(p.cdf(y), float) for p in preds])
        dens = np.clip(np.diff(cdf, axis=1), 0.0, None) / np.diff(y)[None]
        dens = _smooth(dens, DENSITY_SMOOTH / float(y[1] - y[0]))
        median = np.array([float(p.quantile(0.5)) for p in preds])
        stop_step, stop_rule = None, None
        decide = {int(s): pw for s, pw in zip(steps, p_worse) if s > 0}
        for s in range(1, tr.t_obs + 1):
            if s >= spec.T:
                break
            if math.isfinite(J_best) and J[s - 1] >= J_best:
                stop_step, stop_rule = s, "exceed"
                break
            if s in decide and decide[s] > tau:
                stop_step, stop_rule = s, "forecast"
                break
        acts.append(DemoAct(index=k, J=J, x=np.asarray(tr.x, float).reshape(-1, 2),
                            t_obs=int(tr.t_obs), T=int(tr.T), crashed=bool(tr.crashed),
                            J_final=float(tr.J_obs), J_best=J_best, stop_step=stop_step,
                            stop_rule=stop_rule, steps=steps, p_worse=p_worse, p_crash=p_crash,
                            dens=dens, median=median))
    return DemoData(acts=acts, y=y, dt=1.0 / 120.0, tau=tau, rig=dc["rig"], method=dc["method"],
                    seed=int(dc["seed"]), model=dc["model"])


# =========================================================================== stop demo frames
#: QUBE-Servo 2 geometry in meters (arm and pendulum length, base half width and height).
ARM_L, PEND_L, BASE_W, BASE_H = 0.085, 0.129, 0.05, 0.16
#: Oblique view: depth (toward the viewer) moves a point down by this factor.
DEPTH_DROP = 0.32


def _proj(p: np.ndarray) -> tuple[float, float]:
    return float(p[0]), float(p[2] - DEPTH_DROP * p[1])


def draw_pendulum(ax: Any, phi_arm: float, phi_pend: float, pal: Palette,
                  dim: bool = False) -> None:
    """Front view of the rig: base, rotary arm (pointing at the viewer at 0) and pendulum
    (``phi_pend`` from upright)."""
    from matplotlib.patches import Ellipse, FancyBboxPatch

    a = 0.45 if dim else 1.0
    base_col = pal.faint if dim else pal.muted
    ax.add_patch(FancyBboxPatch((-BASE_W, -BASE_H), 2 * BASE_W, BASE_H,
                                boxstyle="round,pad=0,rounding_size=0.008", facecolor=pal.card,
                                edgecolor=base_col, lw=1.2, alpha=a, zorder=1))
    ax.add_patch(Ellipse((0, 0), 0.05, 0.05 * DEPTH_DROP * 1.6, facecolor=pal.grid,
                         edgecolor=base_col, lw=1.0, alpha=a, zorder=2))
    ax.plot([-0.19, 0.19], [-BASE_H, -BASE_H], color=pal.grid, lw=1.0, zorder=0)
    arm_dir = np.array([math.sin(phi_arm), math.cos(phi_arm), 0.0])
    tangent = np.array([math.cos(phi_arm), -math.sin(phi_arm), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    pivot = np.zeros(3)
    tip = pivot + ARM_L * arm_dir
    bob = tip + PEND_L * (math.sin(phi_pend) * tangent + math.cos(phi_pend) * up)
    p0, p1, p2 = _proj(pivot), _proj(tip), _proj(bob)
    behind = arm_dir[1] < 0     # arm pointing away: draw it behind the hub
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=pal.fg if not dim else pal.faint, lw=5,
            alpha=a, zorder=1.5 if behind else 3, solid_capstyle="round")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=pal.red, lw=4, alpha=0.55 if dim else 1.0,
            zorder=4, solid_capstyle="round")
    ax.scatter([p1[0]], [p1[1]], s=22, color=pal.fg if not dim else pal.faint, zorder=4.5,
               linewidths=0, alpha=a)


@dataclass
class _Frame:
    act: int
    step: int                  # live step (cost and pendulum)
    ghost: int = 0             # revealed logged steps after the stop
    phase: str = "run"         # run | stopped | final
    speed: float = 1.0


def demo_schedule(data: DemoData, cfg: dict[str, Any]) -> list[_Frame]:
    """Frames of the animation: each trial runs (real time, faster after ``fast_after_s``), a
    stopped trial pauses, then reveals the rest of its logged trajectory, then holds."""
    dc = _cfg(cfg, "demo")
    fps = float(dc["fps"])
    per_frame = 1.0 / (fps * data.dt)          # logged steps per frame at real time
    fast_after = float(dc["fast_after_s"]) / data.dt
    fast = float(dc["fast_speed"])
    out: list[_Frame] = []

    def hold(seconds: float, **kw: Any) -> None:
        out.extend(_Frame(**kw) for _ in range(int(round(seconds * fps))))

    for i, a in enumerate(data.acts):
        hold(0.5, act=i, step=0)
        s = 0.0
        while s < a.end_step:
            speed = fast if s >= fast_after else 1.0
            out.append(_Frame(act=i, step=int(s), speed=speed))
            s += per_frame * speed
        if a.stop_step is None:
            hold(2.6, act=i, step=a.end_step, phase="final")
            continue
        hold(1.2, act=i, step=a.end_step, phase="stopped")
        rest = a.t_obs - a.end_step
        n_rev = max(1, int(round(1.2 * fps)))
        for j in range(1, n_rev + 1):
            out.append(_Frame(act=i, step=a.end_step, ghost=int(round(rest * j / n_rev)),
                              phase="stopped"))
        hold(2.6, act=i, step=a.end_step, ghost=rest, phase="final")
    return out


def _pct(p: float) -> str:
    if p >= 0.9995:
        return ">99.9%"
    if p >= 0.99 or p <= 0.01:
        return f"{100 * p:.1f}%"
    return f"{100 * p:.0f}%"


def _headline(a: DemoAct, f: _Frame, dt: float, tau: float) -> tuple[str, str, str]:
    """(title, subtitle, color key) of a frame."""
    if f.phase == "run":
        return f"Trial {a.index}: running", "the model updates its forecast every 75 ms", "fg"
    if a.stop_step is None:
        if a.crashed:
            return f"Trial {a.index}: crashed", "the forecast did not reach the stop level", "red"
        new_best = a.J_final < a.J_best
        return (f"Trial {a.index}: kept to the end",
                f"final cost {a.J_final:.0f}" + (", a new best" if new_best else ""),
                "green" if new_best else "fg")
    t_stop = a.stop_step * dt
    title = f"Trial {a.index}: stopped after {t_stop:.2f} s"
    if f.phase == "stopped" and f.ghost == 0:
        return title, f"{100 * tau:.0f}% sure it ends worse than the best so far", "red"
    if a.crashed:
        return title, (f"the logged trial crashed at {a.t_obs * dt:.2f} s, "
                       f"{(a.t_obs - a.stop_step) * dt:.2f} s later"), "red"
    return title, f"the logged trial ended at a cost of {a.J_final:.0f}", "red"


def draw_demo_frame(data: DemoData, f: _Frame, pal: Palette) -> Figure:
    from matplotlib.patches import FancyBboxPatch, Rectangle

    a = data.acts[f.act]
    dt = data.dt
    T_s = a.T * dt
    y = data.y
    ymax = float(y[-1])
    fi = a.forecast_index(f.step)
    fig = Figure(figsize=(8.4, 3.5))     # no pyplot manager: no HiDPI scaling of the frames
    col = {"fg": pal.fg, "red": pal.red, "green": pal.green}

    # header
    title, sub, ck = _headline(a, f, dt, data.tau)
    fig.text(0.025, 0.915, title, fontsize=13.5, fontweight="bold", color=col[ck])
    fig.text(0.025, 0.835, sub, fontsize=9.5, color=pal.muted)
    for j in range(len(data.acts)):
        fig.patches.append(FancyBboxPatch(
            (0.905 + 0.026 * j, 0.915), 0.016, 0.03, boxstyle="round,pad=0,rounding_size=0.008",
            transform=fig.transFigure, facecolor=pal.blue if j == f.act else pal.grid,
            edgecolor="none"))

    # pendulum
    axp = fig.add_axes((0.0, 0.10, 0.25, 0.68))
    axp.set_xlim(-0.2, 0.2)
    axp.set_ylim(-BASE_H - 0.01, 0.16)
    axp.set_aspect("equal")
    axp.set_axis_off()
    k = min(max(f.step, 1), a.t_obs) - 1
    stopped = f.phase != "run" and a.stop_step is not None
    draw_pendulum(axp, float(a.x[k, 0]), float(a.x[k, 1]), pal, dim=stopped)
    tlab = f"t = {f.step * dt:.2f} s"
    speed = "" if f.phase != "run" else ("   real time" if f.speed == 1.0 else
                                         f"   {f.speed:.0f}x speed")
    axp.text(0.0, -BASE_H - 0.03, tlab + speed, ha="center", va="top", fontsize=8.5, color=pal.muted,
             transform=axp.transData)

    # cost so far
    ax = fig.add_axes((0.30, 0.17, 0.47, 0.58))
    ax.set_xlim(0, T_s + 1.05)
    ax.set_ylim(0, ymax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_bounds(0, T_s)
    ax.set_xticks(range(int(T_s) + 1))
    ax.set_xticklabels([f"{t} s" for t in range(int(T_s) + 1)], fontsize=8)
    yt = [v for v in (0, 100, 200, 300, 400) if v <= ymax]
    ax.set_yticks(yt)
    ax.set_yticklabels([str(v) for v in yt], fontsize=8)
    ax.set_ylabel("cost so far", fontsize=9)
    for v in yt[1:]:
        ax.plot([0, T_s], [v, v], color=pal.grid, lw=0.5, alpha=0.6, zorder=0)
    if math.isfinite(a.J_best):
        ax.plot([0, T_s + 1.0], [a.J_best, a.J_best], color=pal.muted, lw=1.0,
                ls=(0, (4, 3)), zorder=1)
        ax.text(0.06, a.J_best + 0.02 * ymax, f"best so far  {a.J_best:.0f}", fontsize=8,
                color=pal.muted, va="bottom")
    t = np.arange(f.step + 1) * dt
    Jl = np.concatenate([[0.0], a.J[:f.step]])
    if f.ghost > 0:
        g0, g1 = a.end_step, a.end_step + f.ghost
        tg = np.arange(g0, g1 + 1) * dt
        Jg = a.J[g0 - 1:g1]
        ax.plot(tg, Jg, color=pal.faint, lw=1.6, ls=(0, (1.2, 2.0)), zorder=2)
        if f.phase == "final":
            if a.crashed:
                ax.scatter([tg[-1]], [Jg[-1]], marker="x", s=46, color=pal.red, lw=2, zorder=5)
                ax.text(tg[-1] + 0.12, Jg[-1], "crash", color=pal.red, fontsize=8.5,
                        va="center")
            else:
                ax.scatter([tg[-1]], [Jg[-1]], s=20, color=pal.faint, zorder=5, lw=0)
                ax.text(tg[-1] - 0.08, Jg[-1] + 0.03 * ymax, f"{a.J_final:.0f}",
                        color=pal.muted, fontsize=8.5, ha="right", va="bottom")
    ax.plot(t, Jl, color=pal.blue, lw=2.2, zorder=3)
    end_col = pal.red if stopped else pal.blue
    ax.scatter([t[-1]], [Jl[-1]], s=30 if not stopped else 44, color=end_col, zorder=6, lw=0)
    if a.stop_step is None and f.phase == "final" and not a.crashed:
        ax.scatter([t[-1]], [Jl[-1]], s=46, color=pal.green, zorder=6, lw=0)
    if stopped:
        ax.text(t[-1], Jl[-1] - 0.05 * ymax, "stop", color=pal.red, fontsize=9,
                fontweight="bold", ha="center", va="top", zorder=6)

    # forecast of the final cost, a sideways density right of the horizon
    x0, w = T_s + 0.12, 0.85
    d = a.dens[fi]
    d = d / d.max() if d.max() > 0 else d
    yc = 0.5 * (y[:-1] + y[1:])
    below = yc <= a.J_best
    ax.fill_betweenx(yc, x0, x0 + w * np.where(below, d, 0.0), color=pal.orange, alpha=0.75,
                     lw=0, zorder=2)
    ax.fill_betweenx(yc, x0, x0 + w * np.where(~below, d, 0.0), color=pal.red, alpha=0.55,
                     lw=0, zorder=2)
    ax.plot([x0, x0], [0, ymax], color=pal.grid, lw=0.8, zorder=1)
    ax.text(x0, ymax * 1.03, "forecast of\nthe final cost", fontsize=8, color=pal.muted,
            va="bottom", ha="left", linespacing=1.15, clip_on=False)

    # gauge: probability of ending worse than the best so far
    p = float(a.p_worse[fi])
    over = p > data.tau
    gx = 0.805
    fig.text(gx, 0.70, "chance it ends worse\nthan the best so far", fontsize=9,
             color=pal.muted, va="top", linespacing=1.2)
    fig.text(gx, 0.42, _pct(p), fontsize=27, fontweight="bold",
             color=pal.red if over else pal.fg, va="bottom")
    bar = fig.add_axes((gx, 0.355, 0.165, 0.032))
    bar.set_xlim(0, 1)
    bar.set_ylim(0, 1)
    bar.set_axis_off()
    bar.add_patch(Rectangle((0, 0), 1, 1, facecolor=pal.card, edgecolor=pal.grid, lw=0.8))
    if p > 0.005:
        bar.add_patch(Rectangle((0, 0), p, 1, facecolor=pal.red if over else pal.orange,
                                edgecolor="none"))
    bar.plot([data.tau, data.tau], [-0.5, 1.5], color=pal.fg, lw=1.2, clip_on=False)
    fig.text(gx + 0.165 * data.tau, 0.32, f"stop above {100 * data.tau:.0f}%", fontsize=8,
             color=pal.muted, ha="right", va="top")
    fig.text(gx, 0.17, f"crash risk  {_pct(float(a.p_crash[fi]))}", fontsize=9,
             color=pal.muted, va="bottom")
    return fig


def _frame_image(fig: Figure) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    canvas = FigureCanvasAgg(fig)       # fixed pixel size, whatever the interactive backend
    canvas.draw()
    w, h = canvas.get_width_height()
    img = Image.frombuffer("RGBA", (w, h), canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
    out = img.convert("RGB")
    plt.close(fig)
    return out


def _write_gif(frames: list[Any], path: Path, fps: float, colors: int = 96) -> Path:
    """One shared palette with the exact background color, exact nearest-color mapping (no
    dithering), consecutive equal frames merged by Pillow."""
    from PIL import Image

    pick = sorted({int(i) for i in np.linspace(0, len(frames) - 1, 12)})
    w, h = frames[0].size
    sheet = Image.new("RGB", (w, h * len(pick)))
    for j, i in enumerate(pick):
        sheet.paste(frames[i], (0, j * h))
    pal_img = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    entries = np.asarray(pal_img.getpalette()[:3 * colors], int).reshape(-1, 3)
    bg = np.asarray(frames[0].getpixel((0, 0)), int)
    entries[int(np.argmin(np.abs(entries - bg).sum(axis=1)))] = bg
    flat = entries.astype(np.uint8).ravel().tolist()
    q = []
    for fr in frames:
        rgb = np.asarray(fr, np.int32)
        packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        uniq, inv = np.unique(packed, return_inverse=True)
        u = np.stack([uniq >> 16, (uniq >> 8) & 255, uniq & 255], axis=1)
        near = np.argmin(((u[:, None, :] - entries[None]) ** 2).sum(axis=2), axis=1)
        img = Image.fromarray(near[inv].reshape(packed.shape).astype(np.uint8), "P")
        img.putpalette(flat)
        q.append(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(round(1000 / fps)),
              loop=0, optimize=False, disposal=1)
    return path


GIF_DPI = 200          # 1680 px wide, 2x of the README display width


def render_demo(data: DemoData, cfg: dict[str, Any], out_dir: Path, variant: str) -> Path:
    pal = PALETTES[variant]
    frames = []
    with plt.rc_context({**_rc(pal), "figure.dpi": GIF_DPI}):
        for f in demo_schedule(data, cfg):
            frames.append(_frame_image(draw_demo_frame(data, f, pal)))
    return _write_gif(frames, Path(out_dir) / f"stop_demo{_suffix(variant)}.gif",
                      float(_cfg(cfg, "demo")["fps"]))


def demo_sidecar(data: DemoData) -> list[str]:
    dt = data.dt
    lines = [f"rig `{data.rig}`, method `{data.method}` seed {data.seed} (provenance thesis), "
             f"model `{data.model}`, rules `exceed` + `naive` at tau = {data.tau}, forecasts at "
             "step 0 and at the decoder grid steps"]
    for a in data.acts:
        fi = a.forecast_index(a.end_step)
        stop = ("not stopped" if a.stop_step is None else
                f"stopped by `{a.stop_rule}` at step {a.stop_step} ({a.stop_step * dt:.3f} s), "
                f"Pr[worse] {a.p_worse[fi]:.4f}, p_crash {a.p_crash[fi]:.3f}")
        end = (f"crashed at step {a.t_obs} ({a.t_obs * dt:.3f} s)" if a.crashed else
               f"logged J_T {a.J_final:.2f} at step {a.t_obs}")
        lines.append(f"trial {a.index}: best so far {a.J_best:.2f}, {stop}, {end}, Pr[worse] "
                     f"at step 0 {a.p_worse[0]:.3f}")
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
    from esbopfn.problems import get_problem

    pal = PALETTES[variant]
    with plt.rc_context(_rc(pal)):
        fig, axes = plt.subplots(1, len(data), figsize=(7.2, 1.8))
        for ax, (p, trajs) in zip(np.atleast_1d(axes), data.items()):
            peaks = [float(np.max(t.J[:(t.crash_step - 1) if t.crash_step else len(t.J)],
                                  initial=0.0)) for t in trajs]
            top = float(np.quantile(peaks, 0.85)) * 1.25 if peaks else 1.0
            top = top if top > 0 else 1.0
            for t in trajs:
                T = len(t.J)
                end = (t.crash_step - 1) if t.crash_step else T
                x = np.arange(1, end + 1) / T
                ax.plot(x, np.minimum(t.J[:end], top * 1.5), color=pal.blue, lw=1.0,
                        alpha=0.7)
                if t.crash_step and end > 0:
                    ax.scatter([x[-1]], [min(float(t.J[end - 1]), top)], marker="x", s=18,
                               color=pal.red, linewidths=1.3, zorder=4)
            ax.set_ylim(0, top)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            d = get_problem(p).spec.d
            ax.set_title(PROBLEM_TITLE.get(p, p), fontsize=9.5, color=pal.fg, loc="left",
                         pad=14, fontweight="bold")
            ax.text(0.0, 1.02, f"{d} controller parameter{'s' if d > 1 else ''}",
                    transform=ax.transAxes, fontsize=8, color=pal.muted, va="bottom")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.tight_layout(w_pad=1.6)
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
                           "curves": {}, "missing": []}
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
        out["curves"][rig] = {"x_min": np.asarray(reg["x"], float) / 60.0,
                              "median": {m: np.asarray(v["median"], float)
                                         for m, v in reg["methods"].items()}}
    return out


def _hardware_cfg() -> dict[str, Any]:
    from esbopfn.plots import group_config

    return group_config("hardware")


#: Rows drawn in the highlight color on the cards (the cross-trajectory predictor).
HIGHLIGHT = {"XDecPFN", "xdecpfn-pi"}


def _card(ax: Any, pal: Palette, title: str, subtitle: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.035",
                                transform=ax.transAxes, facecolor=pal.card,
                                edgecolor=pal.grid, linewidth=0.9, zorder=0, clip_on=False,
                                mutation_aspect=1.25))
    ax.text(0.055, 0.915, title, transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=pal.fg, va="top")
    ax.text(0.055, 0.825, subtitle, transform=ax.transAxes, fontsize=8, color=pal.muted,
            va="top")


def _bars(fig: Figure, ax: Any, pal: Palette, rows: Sequence[tuple[str, float, bool]],
          fmt: str) -> None:
    """Horizontal bars inside a card, labels left, values at the bar ends, highlighted rows in
    blue and bold."""
    pos = ax.get_position()
    inner = fig.add_axes((pos.x0 + 0.42 * pos.width, pos.y0 + 0.07 * pos.height,
                          0.46 * pos.width, 0.66 * pos.height))
    inner.set_facecolor("none")
    vals = np.array([v for _, v, _ in rows], float)
    lo, hi = min(0.0, float(np.nanmin(vals))), max(0.0, float(np.nanmax(vals)))
    span = (hi - lo) or 1.0
    y = np.arange(len(rows))[::-1]
    label_tf = inner.get_yaxis_transform()        # x in axes fraction, y in data
    for yi, (label, v, hl) in zip(y, rows):
        inner.barh(yi, v, height=0.64, color=pal.blue if hl else pal.bar, zorder=2)
        kw = dict(fontweight="bold") if hl else {}
        inner.text(-0.05, yi, label, ha="right", va="center", fontsize=8.5,
                   color=pal.fg if hl else pal.muted, transform=label_tf, **kw)
        xv = max(v, 0.0) + 0.03 * span if hi > 0 else 0.03 * span
        inner.text(xv, yi, fmt.format(v), ha="left", va="center", fontsize=8,
                   color=pal.fg if hl else pal.muted, **kw)
    inner.axvline(0, color=pal.faint, lw=0.7, zorder=3)
    inner.set_xlim(lo, hi + 0.24 * span)
    inner.set_ylim(-0.6, len(rows) - 0.4)
    inner.set_axis_off()


def draw_cards(data: dict[str, Any], variant: str) -> Figure:
    pal = PALETTES[variant]
    with plt.rc_context(_rc(pal)):
        fig = plt.figure(figsize=(7.2, 5.9))
        gs = fig.add_gridspec(2, 2, wspace=0.05, hspace=0.07, left=0.01, right=0.99,
                              bottom=0.01, top=0.99)
        cards = []
        pred = sorted(data["prediction"], key=lambda r: -r["value"])
        cards.append(("Forecast quality", "log-likelihood of the final cost with 64 earlier "
                      "trials,\n5 simulated problems, higher is better",
                      [(r["label"], r["value"], r["label"] in HIGHLIGHT) for r in pred],
                      "{:.2f}"))
        opt = sorted(data["optimization"], key=lambda r: r["value"])
        cards.append(("Tuning in simulation", "log10 regret relative to random search, 100 "
                      "runs,\nend of the time budget, lower is better",
                      [(style.display_name(r["method"]), r["value"], r["method"] in HIGHLIGHT)
                       for r in opt], "{:.2f}"))
        for rig, rows in list(data["hardware"].items())[:1]:
            rows = sorted(rows, key=lambda r: r["regret"])
            cards.append(("Tuning the real pendulum", f"log10 regret, {C.rig_title(rig)}, median"
                          " of 5 runs,\nend of the time budget, lower is better",
                          [(style.display_name(r["method"]), r["regret"],
                            r["method"] in HIGHLIGHT) for r in rows], "{:.2f}"))
        if data["hardware"]:
            per_m: dict[str, list[float]] = defaultdict(list)
            for rows in data["hardware"].values():
                for r in rows:
                    per_m[r["method"]].append(r["crashes"])
            rows2 = sorted(((m, float(np.mean(v))) for m, v in per_m.items()),
                           key=lambda r: r[1])
            cards.append(("Crashes on the real pendulum", "crashed trials per run, mean over "
                          f"{len(data['hardware'])} rig setups,\nlower is better",
                          [(style.display_name(m), v, m in HIGHLIGHT) for m, v in rows2],
                          "{:.1f}"))
        for k, (title, sub, rows, fmt) in enumerate(cards[:4]):
            ax = fig.add_subplot(gs[k // 2, k % 2])
            _card(ax, pal, title, sub)
            if rows:
                _bars(fig, ax, pal, rows, fmt)
    return fig


# =========================================================================== rig progress
def _spread(ys: list[float], gap: float) -> list[float]:
    """Push sorted label positions apart to at least ``gap``, keeping their mean."""
    order = np.argsort(ys)
    pos = np.asarray(ys, float)[order]
    for _ in range(50):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap:
                mid = 0.5 * (pos[i] + pos[i - 1])
                pos[i - 1], pos[i] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return out.tolist()


def draw_rig(data: dict[str, Any], variant: str) -> Figure:
    """Median log10 regret over reset-normalized wall-clock per rig, the cross-trajectory loop
    in blue, every other method in gray, labels at the line ends."""
    pal = PALETTES[variant]
    curves = data["curves"]
    with plt.rc_context(_rc(pal)):
        fig, axes = plt.subplots(1, len(curves), figsize=(7.2, 2.7), sharey=True)
        axes = np.atleast_1d(axes)
        finite = [v[np.isfinite(v)] for c in curves.values() for v in c["median"].values()]
        top = max((float(v.max()) for v in finite if v.size), default=1.0)
        for ax, (rig, c) in zip(axes, curves.items()):
            x = c["x_min"]
            ends = []
            for m, v in sorted(c["median"].items(), key=lambda kv: kv[0] in HIGHLIGHT):
                hl = m in HIGHLIGHT
                ok = np.isfinite(v)
                ax.plot(x[ok], v[ok], color=pal.blue if hl else pal.faint, lw=2.4 if hl else 1.2,
                        alpha=1.0 if hl else 0.8, zorder=3 if hl else 2)
                if ok.any():
                    ends.append((m, float(v[ok][-1]), float(x[ok][-1]), hl))
            ys = _spread([e[1] for e in ends], 0.075 * top)
            for (m, _, _, hl), yl in zip(ends, ys):
                ax.text(x[-1] * 1.02, yl, style.display_name(m), fontsize=8, va="center",
                        color=pal.blue if hl else pal.muted,
                        fontweight="bold" if hl else "normal")
            ax.set_xlim(0, x[-1])
            ax.set_ylim(min(0.0, ax.get_ylim()[0]), top * 1.05)
            ax.set_title(f"rig {C.rig_title(rig)}", loc="left", fontsize=9.5, color=pal.fg,
                         fontweight="bold", pad=8)
            ax.set_xlabel("wall-clock time (min)", fontsize=8.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", color=pal.grid, lw=0.5, alpha=0.6)
        axes[0].set_ylabel("log10 regret", fontsize=8.5)
        fig.tight_layout(w_pad=6.0)
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
    predictors: dict[str, Any] = {}
    try:
        hero = hero_data(db, cfg, predictors)
    except (RuntimeError, KeyError, IndexError, FileNotFoundError) as exc:
        log.warning("README images skipped (rig runs or the XDecPFN checkpoint missing): %s",
                    exc)
        return out
    try:
        demo: DemoData | None = demo_data(db, cfg, predictors)
    except (RuntimeError, KeyError, IndexError, FileNotFoundError, AttributeError) as exc:
        log.warning("README stop demo skipped (demo run or predictor missing): %s", exc)
        demo = None
    strip = strip_data(cfg)
    cards = cards_data(db, cfg)
    for v in variants:
        _save(draw_hero(hero, v), out_dir, "hero_3d" + _suffix(v), tight=False)
        _save(draw_strip(strip, v), out_dir, "problems_strip" + _suffix(v))
        _save(draw_cards(cards, v), out_dir, "results_cards" + _suffix(v))
        if cards["curves"]:
            _save(draw_rig(cards, v), out_dir, "rig_progress" + _suffix(v))
        if demo is not None:
            render_demo(demo, cfg, out_dir, v)
    C.write_sidecar(out_dir, "hero_3d",
                    shown=["axes t (seconds), J_t (raw cost), theta (one controller coordinate);"
                           " blue lines = real rig trials, orange fills = predicted total cost "
                           "mu +- sigma and mu +- 2 sigma at t = T, star = next controller "
                           "(argmax PI over the slice), red graph above the wall = predicted "
                           "crash probability, dotted drop lines and red rings = trials shown "
                           "stopped early"],
                    metric=["mean and standard deviation of the predicted bar distribution "
                            "without the crash atom, PI = P(J_T < incumbent J) with the crash "
                            "atom counted as no improvement"],
                    data=["`hardware_optimization` units of the rig and the model checkpoint "
                          "named below"],
                    numbers=hero_sidecar(hero, cfg))
    if demo is not None:
        C.write_sidecar(out_dir, "stop_demo",
                        shown=["animated replay of logged rig trials: pendulum drawn from the "
                               "logged angles, cumulative cost (blue) against the best final "
                               "cost so far (dashed), the forecast density of the final cost "
                               "(orange below, red above the best so far), Pr[final cost >= best"
                               " so far] with the stop level, the predicted crash probability; "
                               "after a stop the rest of the logged trial is revealed (dotted)"],
                        metric=["forecast = `predict` of the running prefix with all earlier "
                                "trials of the run as context, updated at the decision steps",
                                "Pr[worse] = survival at the best so far, crash mass included "
                                "(the `naive` rule of `xdecpfn-pi`)"],
                        data=["`hardware_optimization` units of the rig and the checkpoint "
                              "named below"],
                        numbers=demo_sidecar(demo))
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
    if cards["curves"]:
        C.write_sidecar(out_dir, "rig_progress",
                        shown=["median over seeds of the log10 regret of the best successful "
                               "trial so far against reset-normalized wall-clock time, one "
                               "panel per rig configuration"],
                        metric=["log10(J- - J_best), J_best the best cost recorded on that rig"],
                        data=["`hardware_optimization` units (the figure group `hardware`)"],
                        numbers=[f"{rig} {m}: final median {v[np.isfinite(v)][-1]:.3f}"
                                 for rig, c in cards["curves"].items()
                                 for m, v in c["median"].items() if np.isfinite(v).any()])
    out.update(hero=hero, strip=strip, cards=cards, demo=demo)
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
