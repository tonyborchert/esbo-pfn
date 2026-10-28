"""Kind ``prediction_timing``: inference seconds per prediction of one predictor at one context
size (thesis setup, prediction metrics: median inference time per prediction on the workstation).

The first ``queries`` selected non-crashed tasks at context size ``n_ctx`` are predicted at the
cutoff ``cutoff_frac`` of ``T`` (snapped as in ``prediction``). Each query is predicted ``warmup``
times untimed, then ``repeats`` times timed (CUDA synchronized). The device is recorded, so the
unit should run on the machine whose times are wanted.

Key ``{problem, model, checkpoint, tasks, tasks_digest, n_ctx, queries, repeats, warmup,
cutoff_frac}``. Imported thesis records: ``{problem, model, n_ctx, provenance: "thesis",
source_model}``.
Blob ``timing.json``: ``{seconds_median, seconds_mean, seconds_p90, n_calls, device, host, cpu,
torch, torch_threads, measured_at}``.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.benchmarks.kinds import BenchmarkKind, Key
from esbopfn.benchmarks.kinds.prediction import (
    build_predictor,
    checkpoint_hash,
    cutoff_steps,
    df_name,
    environment,
    resolve_model,
)


def enumerate_units(cfg: dict[str, Any]) -> list[Key]:
    from esbopfn.benchmarks.groups import instances_seed, problems, section
    from esbopfn.benchmarks.tasks import ensure_tasks

    sec = section(cfg, "prediction_timing")
    pred_sec = section(cfg, "prediction")
    keys: list[Key] = []
    for p in problems(cfg, sec):
        found = ensure_tasks(p, "tasks", sec.get("field"), instances_seed(cfg))
        if found is None:
            continue
        df, h = found
        for m in sec.get("models", pred_sec.get("models", [])):
            m = {"id": m} if isinstance(m, str) else m
            if m.get("problems") and p not in m["problems"]:
                continue
            model = resolve_model(m["id"], p)
            for n in m.get("n", sec.get("n", pred_sec.get("n", sorted(df.n_ctx.unique())))):
                keys.append({"problem": p, "model": model, "checkpoint": checkpoint_hash(model),
                             "tasks": df_name("tasks", sec.get("field"), instances_seed(cfg)),
                             "tasks_digest": h, "n_ctx": int(n),
                             "queries": int(sec.get("queries", 5)),
                             "repeats": int(sec.get("repeats", 5)),
                             "warmup": int(sec.get("warmup", 1)),
                             "cutoff_frac": float(sec.get("cutoff_frac", 0.5))})
    return keys


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def time_predictor(predictor: Any, spec: Any, runs: list[Any], tasks: list[Any],
                   cutoff_frac: float, repeats: int, warmup: int) -> list[float]:
    """Seconds of every timed call."""
    steps = np.asarray(predictor.predictable_steps(spec), dtype=np.int64)
    step, _ = cutoff_steps(steps, spec.T, (cutoff_frac,))[float(cutoff_frac)]
    aware = bool(getattr(predictor, "context_aware", False))
    times: list[float] = []
    for run, task in zip(runs, tasks):
        query = run.trial(int(task.query_index), step)
        context = run.context(int(task.n_ctx)) if aware else []
        for _ in range(warmup):
            predictor.predict(query, context)
        for _ in range(repeats):
            _sync()
            t0 = time.perf_counter()
            predictor.predict(query, context)
            _sync()
            times.append(time.perf_counter() - t0)
    return times


def summarize(times: list[float]) -> dict[str, Any]:
    arr = np.asarray(times, dtype=np.float64)
    return {"seconds_median": float(statistics.median(arr)), "seconds_mean": float(arr.mean()),
            "seconds_p90": float(np.quantile(arr, 0.9)), "n_calls": int(arr.size)}


def run(key: Key, blob_dir: Path) -> None:
    from esbopfn.benchmarks.db import Db
    from esbopfn.benchmarks.kinds.fields import field_key, read_field
    from esbopfn.benchmarks.kinds.prediction import _instances_seed
    from esbopfn.benchmarks.tasks import load_tasks
    from esbopfn.problems.base import cached_spec

    problem = key["problem"]
    found = load_tasks(problem, key["tasks"])
    if found is None or found[1] != key["tasks_digest"]:
        raise LookupError(f"task table {problem}/{key['tasks']} not found")
    df = found[0]
    sel = df[df.selected & (df.n_ctx == int(key["n_ctx"]))].sort_values("run_order")
    sel = sel.head(int(key["queries"]))
    db = Db()
    seed = _instances_seed(key["tasks"])
    runs = []
    for t in sel.itertuples(index=False):
        fk = field_key(problem, str(t.field), int(t.instance), int(t.seed), seed)
        runs.append(read_field(db.blob_dir("fields", fk), fk))
    predictor = build_predictor(key["model"], problem)
    times = time_predictor(predictor, cached_spec(problem), runs, list(sel.itertuples(index=False)),
                           float(key["cutoff_frac"]), int(key["repeats"]), int(key["warmup"]))
    out = {**summarize(times), **environment(),
           "measured_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
    (Path(blob_dir) / "timing.json").write_text(json.dumps(out, indent=1))


def read_timing(blob_dir: Path) -> dict[str, Any]:
    return json.loads((Path(blob_dir) / "timing.json").read_text())


__all__ = ["KIND", "checkpoint_hash", "read_timing", "summarize", "time_predictor"]

KIND = BenchmarkKind(name="prediction_timing", enumerate=enumerate_units, run=run)
