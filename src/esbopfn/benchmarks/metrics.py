"""Benchmark metrics of the thesis (setup, prediction metrics S:542-576 and optimization metrics
S:630-681), as pure functions over data frames, arrays and run records.

Prediction (rows of the ``prediction`` kind, see :mod:`esbopfn.benchmarks.kinds.prediction`):

- :func:`trial_scores`: per non-crashed query the mean of the five cutoff log-likelihoods.
- :func:`collapse_context_blind`: context-blind models pooled over ``n`` (``n_ctx = -1``).
- :func:`median_scores`: median score per (problem, model, n).
- :func:`med_d`: per-problem median of every score of a declared participant set.
- :func:`ll_norm`, :func:`pooled_median`: scores minus ``med_D``, median across problems.
- :func:`pareto_front`: non-dominated points over (inference time, score).
- :func:`stop_curve`, :func:`steps_saved_at`: the crash early stopping trade-off.
- :func:`ladder_curve`: context ladder, mean over cutoffs then mean over queries.

Optimization (:class:`RunRecord` objects):

- :func:`incumbent_vs_trials`, :func:`incumbent_vs_time`, :func:`time_grid`.
- :func:`j_best`, :func:`j_esrs`, :func:`scaled_regret`, :func:`problem_median`,
  :func:`cross_problem_mean`, :func:`log10_display`, :func:`regret_at_fraction`.
- :func:`crash_count_vs_time`, :func:`mean_crash_count`, :func:`timing_split`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

#: ``n_ctx`` of a context-blind model pooled over the context sizes.
COLLAPSED_N = -1
#: Families that ignore the context.
CONTEXT_BLIND_FAMILIES = frozenset({"decpfn", "decpfn-nostates", "decpfn-nodeadtime",
                                    "decpfn-nodeadtime-nostates", "lcpfn", "lcpfn-last"})
#: Display floor of the scaled regret before log10.
REGRET_FLOOR = 1e-4
#: Denominators ``J_ESRS - J_best`` at or below this leave the instance undefined (NaN).
DENOM_EPS = 1e-9
#: Strategy timing categories -> segments of the decision-time figure.
TIMING_SEGMENTS: dict[str, str] = {
    "gp_acq": "GP acquisition",
    "model_acq": "Transformer acquisition",
    "segment_gp": "Segment-GP forecast",
    "model_forecast": "Transformer forecast",
    "tabpfn_forecast": "Transformer forecast",
}


# ============================================================================ prediction
def series_label(model: str, problem: str) -> str:
    """Problem-agnostic id of a model: ``decpfn/cartpole/binned/1M`` -> ``decpfn/{p}/binned/1M``."""
    parts = model.split("/")
    return "/".join("{p}" if part == problem else part for part in parts)


def is_context_blind(model: str) -> bool:
    return model.split("/")[0] in CONTEXT_BLIND_FAMILIES


def trial_scores(rows: pd.DataFrame) -> pd.DataFrame:
    """Per non-crashed query the mean conditioned LL over its cutoff rows (thesis: five cutoffs).

    Columns ``problem, model, task_id, instance, n_ctx, score, n_cutoffs``.
    """
    ok = rows[~rows.crashed.astype(bool) & rows.cutoff_frac.notna()]
    keys = [c for c in ("problem", "model", "task_id", "instance", "n_ctx") if c in ok.columns]
    out = ok.groupby(keys, sort=True).ll.agg(score="mean", n_cutoffs="size").reset_index()
    return out


def collapse_context_blind(scores: pd.DataFrame, blind: Iterable[str] | None = None
                           ) -> pd.DataFrame:
    """Set ``n_ctx = -1`` for context-blind models (all of :data:`CONTEXT_BLIND_FAMILIES` unless
    ``blind`` lists the model ids)."""
    out = scores.copy()
    mask = (out.model.isin(set(blind)) if blind is not None
            else out.model.map(is_context_blind).astype(bool))
    out.loc[mask, "n_ctx"] = COLLAPSED_N
    return out


def median_scores(scores: pd.DataFrame, value: str = "score") -> pd.DataFrame:
    """Median per (problem, model, n_ctx) with the number of scores."""
    return (scores.groupby(["problem", "model", "n_ctx"], sort=True)[value]
            .agg(median="median", n="size").reset_index())


def med_d(scores: pd.DataFrame, participants: Iterable[str]) -> pd.Series:
    """``med_D`` per problem: the median of every score of the participating models."""
    part = scores[scores.model.isin(set(participants))]
    return part.groupby("problem").score.median().rename("med_D")


def ll_norm(scores: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    """Scores with ``med_D`` and ``ll_norm = score - med_D`` of their problem."""
    out = scores.join(med, on="problem")
    out["ll_norm"] = out["score"] - out["med_D"]
    return out


def pooled_median(normed: pd.DataFrame) -> pd.DataFrame:
    """Median of ``ll_norm`` pooled over problems per (series, n_ctx), ``series`` being the
    problem-agnostic model id."""
    df = normed.copy()
    df["series"] = [series_label(m, p) for m, p in zip(df.model, df.problem)]
    return (df.groupby(["series", "n_ctx"], sort=True)
            .agg(median=("ll_norm", "median"), n=("ll_norm", "size"),
                 n_problems=("problem", "nunique")).reset_index())


def pareto_front(points: pd.DataFrame, x: str = "seconds", y: str = "median") -> pd.DataFrame:
    """Points not dominated by a faster-or-equal point with a higher-or-equal score (one of them
    strictly), sorted by ``x``."""
    df = points.sort_values([x, y], ascending=[True, False], kind="stable")
    keep, best = [], -math.inf
    for idx, val in zip(df.index, df[y]):
        if val > best:
            keep.append(idx)
            best = val
    return df.loc[keep]


def stop_curve(rows: pd.DataFrame, tasks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Steps saved vs false stops when a trial stops at the first step with ``p_crash >= tau``.

    ``rows``: one predictor's rows on a group of queries with columns ``task_id, crashed,
    crash_step, cutoff_step, p_crash`` (duplicate steps are allowed). ``tasks`` (``task_id,
    crashed, crash_step``) is the denominator, queries without rows count as never stopped.
    ``tau`` sweeps every emitted probability. Returns ``tau, false_stop_rate, mean_steps_saved,
    stop_rate_crashed, n_crashed, n_ok`` with ``tau`` descending (both rates non-decreasing).
    """
    if tasks is None:
        tasks = rows[["task_id", "crashed", "crash_step"]].drop_duplicates("task_id")
    rows = rows.drop_duplicates(["task_id", "cutoff_step"]).sort_values(["task_id",
                                                                          "cutoff_step"])
    groups = {k: g for k, g in rows.groupby("task_id", sort=False)}
    taus = np.unique(rows.p_crash.to_numpy(dtype=np.float64))[::-1]
    n_crash = int(tasks.crashed.astype(bool).sum())
    n_ok = int(len(tasks) - n_crash)
    saved = np.zeros(taus.size)
    caught = np.zeros(taus.size)
    false = np.zeros(taus.size)
    for t in tasks.itertuples(index=False):
        g = groups.get(t.task_id)
        if g is None or g.empty:
            continue
        prefmax = np.maximum.accumulate(g.p_crash.to_numpy(dtype=np.float64))
        steps = g.cutoff_step.to_numpy(dtype=np.int64)
        idx = np.searchsorted(prefmax, taus, side="left")   # first step with p >= tau
        stopped = idx < prefmax.size
        if bool(t.crashed):
            stop_step = steps[np.minimum(idx, prefmax.size - 1)]
            saved += np.where(stopped, int(t.crash_step) - stop_step, 0)
            caught += stopped
        else:
            false += stopped
    return pd.DataFrame({
        "tau": taus,
        "false_stop_rate": false / n_ok if n_ok else np.full(taus.size, np.nan),
        "mean_steps_saved": saved / n_crash if n_crash else np.full(taus.size, np.nan),
        "stop_rate_crashed": caught / n_crash if n_crash else np.full(taus.size, np.nan),
        "n_crashed": n_crash, "n_ok": n_ok})


def steps_saved_at(curve: pd.DataFrame, max_false_stop: float = 0.1) -> float:
    """Largest mean steps saved over the curve points with false-stop share ``<= max_false_stop``
    (0 if none)."""
    ok = curve[curve.false_stop_rate <= max_false_stop + 1e-12]
    return float(ok.mean_steps_saved.max()) if len(ok) else 0.0


def ladder_curve(rows: pd.DataFrame) -> pd.DataFrame:
    """Context ladder: mean LL over the cutoffs per query, then the mean over queries per
    (problem, model, n_ctx)."""
    per_q = (rows[rows.cutoff_frac.notna()]
             .groupby(["problem", "model", "task_id", "n_ctx"]).ll.mean().reset_index())
    return (per_q.groupby(["problem", "model", "n_ctx"]).ll
            .agg(mean="mean", n="size").reset_index())


def cost_rank(cost: np.ndarray) -> np.ndarray:
    """Percentile rank of the finite costs, 1 = cheapest, 0 = most expensive, NaN elsewhere."""
    out = np.full(cost.shape, np.nan)
    valid = np.isfinite(cost)
    n = int(valid.sum())
    if n:
        rank = np.argsort(np.argsort(cost[valid])) / max(n - 1, 1)
        out[valid] = 1.0 - rank
    return out


# ============================================================================ optimization
def _incumbents(record: Any) -> np.ndarray:
    return np.asarray(record.incumbent_J, dtype=np.float64)


def incumbent_vs_trials(record: Any, n: int) -> np.ndarray:
    """Incumbent cost after ``1 .. n`` trials (``inf`` before the first success), the last value
    carried past the end of the run."""
    inc = _incumbents(record)
    out = np.full(int(n), math.inf)
    if inc.size:
        k = min(inc.size, int(n))
        out[:k] = inc[:k]
        out[k:] = inc[-1]
    return out


def time_grid(t_budget: float, n: int = 300) -> np.ndarray:
    """``n`` points from 0 to ``T_budget`` (simulated steps)."""
    return np.linspace(0.0, float(t_budget), int(n))


def incumbent_vs_time(record: Any, grid: np.ndarray) -> np.ndarray:
    """Incumbent cost at every simulated time of ``grid``: the incumbent after the last trial that
    finished at or before that time (``inf`` before)."""
    cum = np.cumsum([t.t_obs for t in record.trials]).astype(np.float64)
    inc = _incumbents(record)
    idx = np.searchsorted(cum, grid, side="right") - 1
    return np.where(idx >= 0, inc[np.maximum(idx, 0)] if inc.size else math.inf, math.inf)


def final_incumbent(record: Any) -> float:
    inc = _incumbents(record)
    return float(inc[-1]) if inc.size else math.inf


def j_best(finals: pd.DataFrame, strategies: Iterable[str] | None = None) -> pd.Series:
    """``J_best`` per instance: the best finite final incumbent of the declared strategies.

    ``finals``: columns ``instance, strategy, J_final`` (one row per run).
    """
    df = finals if strategies is None else finals[finals.strategy.isin(set(strategies))]
    df = df[np.isfinite(df.J_final)]
    return df.groupby("instance").J_final.min().rename("J_best")


def j_esrs(finals: pd.DataFrame, esrs: str = "esrs") -> pd.Series:
    """``J_ESRS`` per instance: ESRS's final incumbent (median over its runs if several)."""
    return finals[finals.strategy == esrs].groupby("instance").J_final.median().rename("J_ESRS")


def scaled_regret(J: np.ndarray, J_best: np.ndarray | float, J_esrs: np.ndarray | float
                  ) -> np.ndarray:
    """``(J - J_best) / (J_ESRS - J_best)``. ``inf`` where ``J`` is ``inf`` (no incumbent yet),
    NaN where the denominator is ``<= 1e-9`` (ESRS tied the best, undefined)."""
    J = np.asarray(J, dtype=np.float64)
    best = np.asarray(J_best, dtype=np.float64)
    denom = np.asarray(J_esrs, dtype=np.float64) - best
    if J.ndim == 2 and best.ndim == 1:
        best, denom = best[:, None], denom[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.isinf(J), np.inf, (J - best) / np.where(denom > DENOM_EPS, denom, 1.0))
    return np.where(np.broadcast_to(denom > DENOM_EPS, out.shape), out, np.nan)


def problem_median(scaled: np.ndarray) -> np.ndarray:
    """Median over instances (axis 0) of a scaled-regret matrix. Undefined instances (NaN rows)
    are left out, unsolved ones (``inf``) count as the worst values."""
    scaled = np.atleast_2d(np.asarray(scaled, dtype=np.float64))
    rows = scaled[~np.isnan(scaled).all(axis=1)]
    if rows.size == 0:
        return np.full(scaled.shape[1], np.nan)
    return np.median(rows, axis=0)


def cross_problem_mean(medians: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of the per-problem median curves."""
    return np.mean(np.stack([np.asarray(m, dtype=np.float64) for m in medians]), axis=0)


def log10_display(x: np.ndarray | float, floor: float = REGRET_FLOOR) -> np.ndarray:
    """``log10(max(x, floor))`` (``inf`` stays ``inf``)."""
    return np.log10(np.maximum(np.asarray(x, dtype=np.float64), floor))


def regret_at_fraction(record: Any, J_best: float, J_esrs: float, t_budget: float,
                       fraction: float) -> float:
    """Scaled regret of one run at ``fraction`` of the simulated-time budget (violins)."""
    J = incumbent_vs_time(record, np.array([fraction * t_budget]))
    return float(scaled_regret(J, J_best, J_esrs)[0])


def crash_count_vs_time(record: Any, grid: np.ndarray) -> np.ndarray:
    """Crashed trials among the trials finished by each simulated time (NaN before the first
    trial finished)."""
    cum = np.cumsum([t.t_obs for t in record.trials]).astype(np.float64)
    count = np.cumsum([1.0 if t.crashed else 0.0 for t in record.trials])
    idx = np.searchsorted(cum, grid, side="right") - 1
    return np.where(idx >= 0, count[np.maximum(idx, 0)] if count.size else np.nan, np.nan)


def mean_crash_count(records: Sequence[Any], grid: np.ndarray) -> np.ndarray:
    """Mean over runs of :func:`crash_count_vs_time` (runs that started)."""
    import warnings

    stack = np.stack([crash_count_vs_time(r, grid) for r in records])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


def timing_split(records: Sequence[Any], segments: dict[str, str] | None = None
                 ) -> dict[str, float]:
    """Optimizer decision time per segment, summed per run and averaged over runs."""
    segments = TIMING_SEGMENTS if segments is None else segments
    totals: dict[str, float] = {}
    for rec in records:
        for cat, sec in rec.timing.items():
            name = segments.get(cat, cat)
            totals[name] = totals.get(name, 0.0) + float(sec)
    return {k: v / len(records) for k, v in totals.items()} if records else {}
