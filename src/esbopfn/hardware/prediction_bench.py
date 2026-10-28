"""The hardware prediction benchmark (thesis hardware results, appendix query counts).

Queries are the rig trials that were not stopped early (complete or crashed), except the first
trial of each run. A query is scored at context size ``n`` only if it has at least ``n``
predecessors in its run, and then gets exactly the ``n`` trials immediately preceding it,
whatever their outcome (``n`` in ``LADDER``). The ablation instead gives every query its full
history. Cutoffs are 10, 30, 50, 70 and 90 % of the query's own length ``t``
(``min(max(1, round(f t)), t - 1)``), each snapped to the closest step at or before it at which
the predictor is defined (a cutoff with no such step is skipped for that predictor). The score is
the conditioned log-likelihood of the true total cost (``log p_crash`` for a crashed query),
averaged over the query's cutoffs, and the summary is the median over queries per context size,
on the non-crashed queries as in the figures (all queries are reported too). Inference time is
the median wall-clock of one prediction per context size.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from esbopfn.core import ProblemSpec, Trial
from esbopfn.hardware import records
from esbopfn.hardware.records import HwRun
from esbopfn.predictors.base import conditioned_log_likelihood

log = logging.getLogger(__name__)

LADDER: tuple[int, ...] = (0, 1, 2, 5, 11, 22, 45, 90)
CUTOFF_FRACS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
FULL = -1            # n_ctx of the full-history context (ablation)

SCORES_FILE = "scores.parquet"
SUMMARY_FILE = "summary.json"

#: columns of ``scores.parquet``
SCORE_COLUMNS = ("rig", "method", "seed", "query_index", "n_ctx", "n_ctx_used", "cutoff_frac",
                 "cutoff_step", "step", "predictor", "ll", "p_crash", "crashed", "target_J",
                 "query_t", "infer_s")


def raw_cutoffs(t: int, fracs: Sequence[float] = CUTOFF_FRACS) -> list[int]:
    """``min(max(1, round(f t)), t - 1)`` for every fraction."""
    return [min(max(1, round(f * t)), t - 1) for f in fracs]


def snap(c: int, steps: np.ndarray) -> int | None:
    """The largest predictable step ``<= c`` (None if there is none)."""
    steps = np.asarray(steps)
    below = steps[steps <= c]
    return int(below.max()) if below.size else None


def is_query(trial: records.HwTrial, index: int) -> bool:
    """Not the first trial of its run, not stopped early, at least two logged steps."""
    return index > 0 and trial.outcome in (records.COMPLETE, records.CRASHED) and trial.t_obs >= 2


def select_queries(runs: Sequence[HwRun]) -> list[tuple[HwRun, int]]:
    return [(r, i) for r in runs for i, t in enumerate(r.trials) if is_query(t, i)]


def eligible_counts(runs: Sequence[HwRun], ladder: Sequence[int] = LADDER) -> dict[int, int]:
    """Queries that have at least ``n`` predecessors, per ``n`` (thesis tab:app:hwpred-counts)."""
    qs = select_queries(runs)
    return {int(n): sum(1 for _, i in qs if i >= n) for n in ladder}


@dataclass
class _Cache:
    trials: dict[tuple[str, int, int], Trial]

    def get(self, run: HwRun, i: int) -> Trial:
        k = (run.method, run.seed, i)
        if k not in self.trials:
            self.trials[k] = run.trials[i].trial()
        return self.trials[k]


def score(runs: Sequence[HwRun], predictor: Any, spec: ProblemSpec, *,
          predictor_name: str, rig: str, ladder: Sequence[int] = LADDER,
          cutoff_fracs: Sequence[float] = CUTOFF_FRACS, full_history: bool = False
          ) -> pd.DataFrame:
    """One row per (query, context size, cutoff) the predictor answered."""
    steps = np.asarray(predictor.predictable_steps(spec), dtype=np.int64)
    aware = bool(getattr(predictor, "context_aware", True))
    cache = _Cache({})
    blind: dict[tuple, tuple[Any, float]] = {}
    rows: list[dict[str, Any]] = []
    failures: dict[int, int] = {}
    rungs = [FULL] if full_history else [int(n) for n in ladder]
    for run, i in select_queries(runs):
        rec = run.trials[i]
        t = int(rec.t_obs)
        traj = rec.trajectory()
        crashed = rec.crashed
        y_true = math.inf if crashed else float(rec.J_obs)
        for n in rungs:
            if n != FULL and i < n:
                continue
            ctx_idx = range(i) if n == FULL else range(i - n, i)
            context = [cache.get(run, j) for j in ctx_idx] if aware else []
            for frac, c in zip(cutoff_fracs, raw_cutoffs(t, cutoff_fracs)):
                s = snap(c, steps)
                if s is None:
                    continue
                query = Trial(traj=traj, t_obs=s)
                key = (run.method, run.seed, i, s)
                try:
                    if not aware and key in blind:
                        pred, dt = blind[key]
                    else:
                        t0 = time.perf_counter()
                        pred = predictor.predict(query, context)
                        dt = time.perf_counter() - t0
                        if not aware:
                            blind[key] = (pred, dt)
                except Exception as exc:  # noqa: BLE001 - e.g. Segment-GPs without context
                    failures[n] = failures.get(n, 0) + 1
                    if failures[n] == 1:
                        log.info("%s n=%d: %s", predictor_name, n, exc)
                    continue
                j_obs = float(traj.J[s - 1])
                rows.append({"rig": rig, "method": run.method, "seed": run.seed,
                             "query_index": i, "n_ctx": n, "n_ctx_used": len(ctx_idx),
                             "cutoff_frac": float(frac), "cutoff_step": int(c), "step": s,
                             "predictor": predictor_name,
                             "ll": conditioned_log_likelihood(pred, y_true, j_obs),
                             "p_crash": float(pred.p_crash), "crashed": crashed,
                             "target_J": y_true, "query_t": t, "infer_s": dt})
    df = pd.DataFrame(rows, columns=list(SCORE_COLUMNS))
    df.attrs["failures"] = failures
    return df


def summarize(df: pd.DataFrame, eligible: dict[int, int] | None = None) -> dict[str, Any]:
    """Medians per context size of the per-query mean LL (non-crashed and all queries), the
    number of scored queries and the median inference time per prediction."""
    out: dict[str, Any] = {"median_ll": {}, "median_ll_all": {}, "n_scored": {},
                           "n_scored_all": {}, "seconds_per_prediction": {}}
    if len(df):
        per_query = (df.groupby(["n_ctx", "method", "seed", "query_index"])
                     .agg(ll=("ll", "mean"), crashed=("crashed", "first")).reset_index())
        for n, g in per_query.groupby("n_ctx"):
            ok = g[~g.crashed]
            key = str(int(n)) if int(n) != FULL else "full"
            out["median_ll"][key] = float(ok.ll.median()) if len(ok) else math.nan
            out["median_ll_all"][key] = float(g.ll.median())
            out["n_scored"][key] = len(ok)
            out["n_scored_all"][key] = len(g)
        if df["infer_s"].notna().any():
            for n, g in df.groupby("n_ctx"):
                key = str(int(n)) if int(n) != FULL else "full"
                out["seconds_per_prediction"][key] = float(g.infer_s.median())
    if eligible is not None:
        out["n_queries"] = {str(k): int(v) for k, v in eligible.items()}
    return out


def write_blob(blob_dir: Path, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    blob_dir = Path(blob_dir)
    blob_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(blob_dir / SCORES_FILE, index=False)
    (blob_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=1, default=float))


def run_bench(runs: Sequence[HwRun], predictor: Any, spec: ProblemSpec, *, rig: str,
              predictor_name: str, model: str, bench: str = "ladder",
              ladder: Sequence[int] = LADDER, cutoff_fracs: Sequence[float] = CUTOFF_FRACS,
              blob_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one predictor on the rig runs and summarize (writes the blob if ``blob_dir``)."""
    full = bench == "ablation"
    df = score(runs, predictor, spec, predictor_name=predictor_name, rig=rig, ladder=ladder,
               cutoff_fracs=cutoff_fracs, full_history=full)
    elig = eligible_counts(runs, ladder) if not full else {FULL: len(select_queries(runs))}
    summary = {"rig": rig, "bench": bench, "predictor": predictor_name, "model": model,
               "ladder": list(ladder) if not full else ["full"],
               "cutoff_fracs": list(cutoff_fracs), "provenance": "native",
               "runs": [f"{r.method}/seed{r.seed}" for r in runs],
               "failures": {str(k): v for k, v in df.attrs.get("failures", {}).items()},
               **summarize(df, elig)}
    if full:
        summary["n_queries"] = {"full": len(select_queries(runs))}
    if blob_dir is not None:
        write_blob(blob_dir, df, summary)
    return df, summary
