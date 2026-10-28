"""Metrics of the hardware figures, pure functions over the trial records of the rig runs.

Conventions (thesis hardware results and appendix):

- Wall-clock is reset-normalized: every trial's reset time is replaced by the mean reset time of
  the reference method (``decpfn-h``) over all its trials on that rig. A trial's wall-clock is
  ``total_s - reset_s + reset_ref`` and the run's axis is the cumulative sum.
- ``J_best`` is the best (lowest) successful cost recorded on that rig across the declared
  methods. Regret is the unscaled ``log10(max(J^- - J_best, floor))`` of the incumbent ``J^-``
  (best successful cost so far), undefined (NaN) before a run's first success.
- A run is a right-continuous step function over its axis: at ``x`` it holds the value after the
  last trial that ended at or before ``x``.
- Bands are the 25 % and 75 % quantiles over seeds, centers the median (regret, raw cost) or the
  mean (crash counts).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from esbopfn.hardware.records import HwRun, HwTrial

REGRET_FLOOR = 1e-12
BAND_Q = (0.25, 0.75)
N_POINTS = 400


def by_method(runs: Iterable[HwRun]) -> dict[str, list[HwRun]]:
    """Runs grouped by method, each group sorted by seed."""
    out: dict[str, list[HwRun]] = {}
    for r in runs:
        out.setdefault(r.method, []).append(r)
    return {m: sorted(g, key=lambda r: r.seed) for m, g in out.items()}


# --------------------------------------------------------------------------- time axes
def reset_reference(runs: Iterable[HwRun], method: str = "decpfn-h") -> float:
    """Mean reset time over every trial of ``method`` (NaN if none recorded)."""
    vals = [float(t.timings.get("reset_s", math.nan)) for r in runs if r.method == method
            for t in r.trials]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def trial_wall(trial: HwTrial, reset_ref: float = math.nan) -> float:
    """Wall-clock of one trial, with its reset time replaced by ``reset_ref`` when finite."""
    total = float(trial.timings.get("total_s", math.nan))
    reset = float(trial.timings.get("reset_s", math.nan))
    if math.isfinite(reset_ref) and math.isfinite(reset):
        return total - reset + reset_ref
    return total


def wall_axis(run: HwRun, reset_ref: float = math.nan) -> np.ndarray:
    """Cumulative (reset-normalized) wall-clock after each trial."""
    return np.cumsum([trial_wall(t, reset_ref) for t in run.trials], dtype=float)


def trial_axis(run: HwRun) -> np.ndarray:
    return np.arange(1, len(run.trials) + 1, dtype=float)


def step_axis(run: HwRun) -> np.ndarray:
    return np.cumsum([t.t_obs for t in run.trials], dtype=float)


def axis(run: HwRun, name: str, reset_ref: float = math.nan) -> np.ndarray:
    if name == "wall":
        return wall_axis(run, reset_ref)
    if name == "trials":
        return trial_axis(run)
    if name == "steps":
        return step_axis(run)
    raise ValueError(f"unknown axis {name!r} (wall, trials, steps)")


def lookup(x: np.ndarray, values: np.ndarray, grid: np.ndarray, before: float) -> np.ndarray:
    """Right-continuous step function ``values`` over ``x`` evaluated on ``grid``."""
    idx = np.searchsorted(x, grid, side="right") - 1
    out = np.full(len(grid), before, dtype=float)
    inside = idx >= 0
    out[inside] = np.asarray(values, dtype=float)[idx[inside]]
    return out


def wall_grid(runs: Sequence[HwRun], reset_ref: float, n: int = N_POINTS,
              x_max: float | None = None) -> np.ndarray:
    """Shared wall-clock grid from 0 to the longest run (or ``x_max``)."""
    if x_max is None:
        x_max = max((float(wall_axis(r, reset_ref)[-1]) for r in runs if r.trials), default=1.0)
    return np.linspace(0.0, float(x_max), n)


# --------------------------------------------------------------------------- incumbents
def best_so_far(run: HwRun) -> np.ndarray:
    """Best successful cost after each trial (inf before the first success)."""
    best, out = math.inf, []
    for t in run.trials:
        if t.success and math.isfinite(t.J_obs):
            best = min(best, float(t.J_obs))
        out.append(best)
    return np.asarray(out, dtype=float)


def incumbent_indices(run: HwRun) -> np.ndarray:
    """Index of the incumbent after each trial (-1 before the first success)."""
    best, idx, out = math.inf, -1, []
    for i, t in enumerate(run.trials):
        if t.success and math.isfinite(t.J_obs) and t.J_obs < best:
            best, idx = float(t.J_obs), i
        out.append(idx)
    return np.asarray(out, dtype=int)


@dataclass(frozen=True)
class BestCost:
    J_best: float
    method: str | None
    seed: int | None


def j_best(runs: Iterable[HwRun], methods: Sequence[str] | None = None) -> BestCost:
    """Best successful cost recorded on the rig across ``methods`` (all if None)."""
    best = BestCost(math.inf, None, None)
    for r in runs:
        if methods is not None and r.method not in methods:
            continue
        b = best_so_far(r)
        if b.size and b[-1] < best.J_best:
            best = BestCost(float(b[-1]), r.method, r.seed)
    return best


def log10_regret(costs: np.ndarray, J_best: float, floor: float = REGRET_FLOOR) -> np.ndarray:
    """``log10(max(J - J_best, floor))``, NaN where ``J`` is inf."""
    costs = np.asarray(costs, dtype=float)
    out = np.full(costs.shape, np.nan)
    ok = np.isfinite(costs)
    out[ok] = np.log10(np.maximum(costs[ok] - J_best, floor))
    return out


# --------------------------------------------------------------------------- curves
def band(matrix: np.ndarray, center: str = "median",
         q: tuple[float, float] = BAND_Q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center (``median`` NaN-aware, or ``mean``) and quantile band down the seed axis."""
    m = np.asarray(matrix, dtype=float)
    mid = np.full(m.shape[1], np.nan)
    lo = np.full(m.shape[1], np.nan)
    hi = np.full(m.shape[1], np.nan)
    cols = np.isfinite(m).any(axis=0)
    if cols.any():
        sub = m[:, cols]
        with np.errstate(invalid="ignore"):
            mid[cols] = np.nanmedian(sub, axis=0) if center == "median" else np.nanmean(sub, 0)
            lo[cols] = np.nanquantile(sub, q[0], axis=0)
            hi[cols] = np.nanquantile(sub, q[1], axis=0)
    return mid, lo, hi


def regret_matrix(runs: Sequence[HwRun], grid: np.ndarray, axis_name: str, J_best: float,
                  reset_ref: float = math.nan) -> np.ndarray:
    """``(n_seeds, n_grid)`` log10 regret, NaN where a seed has no success yet."""
    out = np.full((len(runs), len(grid)), np.nan)
    for i, r in enumerate(runs):
        if not r.trials:
            continue
        best = lookup(axis(r, axis_name, reset_ref), best_so_far(r), grid, math.inf)
        out[i] = log10_regret(best, J_best)
    return out


def crash_matrix(runs: Sequence[HwRun], grid: np.ndarray, axis_name: str,
                 reset_ref: float = math.nan) -> np.ndarray:
    """``(n_seeds, n_grid)`` cumulative number of crashed trials."""
    out = np.zeros((len(runs), len(grid)))
    for i, r in enumerate(runs):
        if r.trials:
            cum = np.cumsum([t.crashed for t in r.trials])
            out[i] = lookup(axis(r, axis_name, reset_ref), cum, grid, 0.0)
    return out


def regret_curves(runs: Sequence[HwRun], axis_name: str = "wall", *,
                  reset_method: str = "decpfn-h", methods: Sequence[str] | None = None,
                  max_trials: int = 60, n_points: int = N_POINTS) -> dict[str, Any]:
    """Median log10 regret with the seed band per method, over reset-normalized wall-clock
    (``axis_name="wall"``) or over the first ``max_trials`` trials (``"trials"``)."""
    ref = reset_reference(runs, reset_method)
    best = j_best(runs, methods)
    groups = {m: g for m, g in by_method(runs).items() if methods is None or m in methods}
    if axis_name == "trials":
        grid = np.arange(1, max_trials + 1, dtype=float)
    else:
        grid = wall_grid([r for g in groups.values() for r in g], ref, n_points)
    out: dict[str, Any] = {"x": grid, "J_best": best.J_best, "J_best_method": best.method,
                           "J_best_seed": best.seed, "reset_ref_s": ref, "methods": {}}
    for m, g in groups.items():
        mat = regret_matrix(g, grid, axis_name, best.J_best, ref)
        mid, lo, hi = band(mat)
        out["methods"][m] = {"median": mid, "lo": lo, "hi": hi, "per_seed": mat,
                             "seeds": [r.seed for r in g]}
    return out


def crash_curves(runs: Sequence[HwRun], *, reset_method: str = "decpfn-h",
                 methods: Sequence[str] | None = None,
                 n_points: int = N_POINTS) -> dict[str, Any]:
    """Mean cumulative crash count with the seed band per method, over reset-normalized time."""
    ref = reset_reference(runs, reset_method)
    groups = {m: g for m, g in by_method(runs).items() if methods is None or m in methods}
    grid = wall_grid([r for g in groups.values() for r in g], ref, n_points)
    out: dict[str, Any] = {"x": grid, "reset_ref_s": ref, "methods": {}}
    for m, g in groups.items():
        mat = crash_matrix(g, grid, "wall", ref)
        mid, lo, hi = band(mat, center="mean")
        out["methods"][m] = {"mean": mid, "lo": lo, "hi": hi, "per_seed": mat,
                             "seeds": [r.seed for r in g]}
    return out


def crash_counts(runs: Iterable[HwRun]) -> dict[str, dict[int, int]]:
    """Total crashed trials per method and seed."""
    out: dict[str, dict[int, int]] = {}
    for r in runs:
        out.setdefault(r.method, {})[r.seed] = int(sum(t.crashed for t in r.trials))
    return out


def regret_at_fractions(runs: Sequence[HwRun], fracs: Sequence[float] = (0.5, 1.0), *,
                        reset_method: str = "decpfn-h", methods: Sequence[str] | None = None,
                        n_points: int = N_POINTS) -> dict[str, Any]:
    """Per-seed log10 regret at fractions of the wall-clock budget (the longest run on the rig).

    A seed whose incumbent defines ``J_best`` sits on the regret floor and is excluded from
    ``values`` (it is listed in ``excluded_seeds``). Seeds without a success are left out too.
    """
    curves = regret_curves(runs, "wall", reset_method=reset_method, methods=methods,
                           n_points=n_points)
    x = curves["x"]
    floor = math.log10(REGRET_FLOOR) + 1e-9
    out: dict[str, Any] = {"J_best": curves["J_best"], "fracs": list(fracs), "methods": {}}
    for m, c in curves["methods"].items():
        per = {}
        for f in fracs:
            col = int(np.searchsorted(x, float(x[-1]) * f, side="right") - 1)
            vals = c["per_seed"][:, col]
            keep = np.isfinite(vals) & (vals > floor)
            per[f] = {"values": vals[keep].tolist(),
                      "seeds": [s for s, k in zip(c["seeds"], keep) if k],
                      "excluded_seeds": [s for s, v in zip(c["seeds"], vals)
                                         if np.isfinite(v) and v <= floor],
                      "x": float(x[col])}
        out["methods"][m] = per
    return out


def cost_curves(runs: Sequence[HwRun], t_max: float = 1000.0, *,
                reset_method: str = "decpfn-h", methods: Sequence[str] | None = None,
                n_points: int = N_POINTS) -> dict[str, Any]:
    """Median best successful raw cost with the seed band over the first ``t_max`` seconds."""
    ref = reset_reference(runs, reset_method)
    grid = np.linspace(0.0, float(t_max), n_points)
    out: dict[str, Any] = {"x": grid, "reset_ref_s": ref, "methods": {}}
    for m, g in by_method(runs).items():
        if methods is not None and m not in methods:
            continue
        mat = np.full((len(g), len(grid)), np.nan)
        for i, r in enumerate(g):
            if r.trials:
                v = lookup(wall_axis(r, ref), best_so_far(r), grid, math.inf)
                mat[i] = np.where(np.isfinite(v), v, np.nan)
        mid, lo, hi = band(mat)
        out["methods"][m] = {"median": mid, "lo": lo, "hi": hi, "per_seed": mat,
                             "seeds": [r.seed for r in g]}
    return out


def incumbent_at(runs: Sequence[HwRun], method: str, wall_s: float = 500.0, *,
                 reset_method: str = "decpfn-h") -> dict[str, Any] | None:
    """The incumbent at ``wall_s`` of the seed with the median incumbent cost there.

    Returns the seed, trial index, cost, controller and its traces: arm angle and unwrapped
    pendulum angle (0 deg upright) in degrees over time in seconds. With an even number of
    seeds that have an incumbent, the lower of the two middle seeds is taken.
    """
    ref = reset_reference(runs, reset_method)
    cands = []
    for r in by_method(runs).get(method, []):
        if not r.trials:
            continue
        x = wall_axis(r, ref)
        k = int(np.searchsorted(x, wall_s, side="right") - 1)
        if k < 0:
            continue
        idx = int(incumbent_indices(r)[k])
        if idx >= 0:
            cands.append((float(r.trials[idx].J_obs), r.seed, r, idx))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[1]))
    cost, seed, run, idx = cands[(len(cands) - 1) // 2]
    tr = run.trials[idx]
    xs = np.asarray(tr.x, dtype=float).reshape(-1, 2)
    t = np.arange(1, len(xs) + 1) / 120.0
    return {"method": method, "seed": seed, "trial_index": idx, "J": cost,
            "theta": list(tr.theta), "t_s": t, "arm_deg": np.degrees(xs[:, 0]),
            "pend_deg": np.degrees(np.unwrap(xs[:, 1])), "kick_sign": tr.kick_sign,
            "candidates": [(c[1], c[0]) for c in cands]}


def acquisition_wait(runs: Iterable[HwRun],
                     methods: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """Mean propose time per trial: per seed (mean over its trials) and the mean over seeds."""
    out: dict[str, dict[str, Any]] = {}
    for m, g in by_method(runs).items():
        if methods is not None and m not in methods:
            continue
        per = {r.seed: float(np.mean([t.timings.get("propose_s", 0.0) for t in r.trials]))
               for r in g if r.trials}
        out[m] = {"per_seed": per, "mean": float(np.mean(list(per.values()))) if per else
                  math.nan}
    return out
