"""Builders of a small synthetic benchmark database for the plot tests.

Units are written with the blob schemas of the benchmark kinds (docs/BENCHMARKS.md) into the
database of the active ``PFN_HOME``. Load this module with :func:`load` from a test file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from esbopfn.benchmarks.db import Db


def add_unit(db: Db, kind: str, key: dict[str, Any], files: dict[str, Any]) -> Path:
    """Store a finished unit whose blob holds ``files`` (name -> dict/list for JSON, DataFrame for
    parquet, dict of arrays for ``.npz``, str for text, or a callable ``fn(path)``)."""

    def write(_key: dict[str, Any], blob: Path) -> None:
        for name, content in files.items():
            path = Path(blob) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if callable(content):
                content(path)
            elif name.endswith(".parquet"):
                content.to_parquet(path)
            elif name.endswith(".npz"):
                np.savez_compressed(path, **content)
            elif isinstance(content, str):
                path.write_text(content)
            else:
                path.write_text(json.dumps(content))

    return db.run(kind, key, write)


def add_series(db: Db, figure: str, problem: str, series: str, data: dict[str, list[Any]],
               meta: dict[str, Any] | None = None, variant: str = "thesis") -> Path:
    """A ``thesis_series`` unit."""
    key = {"figure": figure, "problem": problem, "series": series, "variant": variant,
           "provenance": "thesis"}
    blob = {"figure": figure, "problem": problem, "series": series, "variant": variant,
            "source": "synthetic", "columns": list(data), "data": data, "meta": meta or {}}
    return add_unit(db, "thesis_series", key, {"series.json": blob})


def add_timing(db: Db, problem: str, model: str, n_ctx: int, seconds: float,
               provenance: str | None = "thesis") -> Path:
    """A ``prediction_timing`` unit (thesis-import key form by default)."""
    key: dict[str, Any] = {"problem": problem, "model": model, "n_ctx": int(n_ctx)}
    if provenance:
        key.update({"provenance": provenance, "source_model": "synthetic"})
    else:
        key.update({"checkpoint": "none", "tasks": "tasks-oracle65", "tasks_digest": "x",
                    "queries": 1, "repeats": 1, "warmup": 0, "cutoff_frac": 0.5})
    return add_unit(db, "prediction_timing", key,
                    {"timing.json": {"seconds_median": float(seconds), "seconds_mean": seconds,
                                     "seconds_p90": seconds, "n_calls": 1, "device": "cpu"}})


def prediction_rows(rng: np.random.Generator, n_ctx: list[int], n_tasks: int, ll_mean: float,
                    T: int = 100, crashed_share: float = 0.3, stop_grid: bool = True,
                    seconds: float = 0.01) -> pd.DataFrame:
    """Rows of a ``predictions.parquet``: five cutoff rows per non-crashed query (ll around
    ``ll_mean``), stop-grid rows with rising crash probabilities."""
    rows = []
    tid = 0
    for n in n_ctx:
        for q in range(n_tasks):
            tid += 1
            crashed = q < int(round(crashed_share * n_tasks))
            crash_step = int(rng.integers(T // 2, T)) if crashed else 0
            for f in (0.1, 0.3, 0.5, 0.7, 0.9):
                step = int(f * T)
                if crashed and step >= crash_step:
                    continue
                rows.append(dict(task_id=f"t{tid}", run_id=f"i{q:04d}_s00", instance=q, seed=0,
                                 n_ctx=n, query_index=n, crashed=crashed, crash_step=crash_step,
                                 target_J=float("nan") if crashed else 1.0, cutoff_frac=f,
                                 cutoff_fallback=False, cutoff_step=step, J_obs=0.5,
                                 ll=float(ll_mean + 0.1 * rng.standard_normal()),
                                 p_crash=0.9 if crashed else 0.05, dist="gaussian", edges=None,
                                 probs=None, mean=1.0, var=0.1, seconds=seconds))
            if stop_grid:
                end = crash_step if crashed else T
                for step in range(5, end, 10):
                    p = (step / max(end, 1)) if crashed else 0.3 * rng.random()
                    rows.append(dict(task_id=f"t{tid}", run_id=f"i{q:04d}_s00", instance=q,
                                     seed=0, n_ctx=n, query_index=n, crashed=crashed,
                                     crash_step=crash_step,
                                     target_J=float("nan") if crashed else 1.0,
                                     cutoff_frac=float("nan"), cutoff_fallback=False,
                                     cutoff_step=step, J_obs=0.5, ll=float("nan"), p_crash=p,
                                     dist="gaussian", edges=None, probs=None, mean=1.0, var=0.1,
                                     seconds=seconds))
    return pd.DataFrame(rows)


def add_prediction(db: Db, kind: str, problem: str, model: str, rows: pd.DataFrame,
                   n: list[int] | None = None) -> Path:
    """A native ``prediction`` or ``prediction_ladder`` unit."""
    key = {"problem": problem, "model": model, "checkpoint": "none", "tasks": "tasks-oracle65",
           "tasks_digest": "synthetic", "n": n or sorted(int(v) for v in rows.n_ctx.unique()),
           "part": "0/1"}
    return add_unit(db, kind, key, {"predictions.parquet": rows,
                                    "meta.json": {"key": key, "model": model}})


Builder = Callable[[Db], None]
