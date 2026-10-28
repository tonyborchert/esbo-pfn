"""Kind ``prediction``: one predictor scored on the frozen context tasks of one problem (thesis
setup, prediction metrics).

For every task the query trial is cut at the five cutoffs 10/30/50/70/90 % of ``T``, each snapped
to the closest step of the predictor's ``predictable_steps`` at or before ``f * T`` (the first
predictable step if none lies before it, recorded in ``cutoff_fallback``). The prediction is
scored with :func:`conditioned_log_likelihood` (conditioned on ``J >= J_t``, the crash mass kept).
Crash-aware predictors are also evaluated at every predictable step before the end of the trial
(the crash step for crashed queries), which the stop curves replay. Context-blind predictors get no
context. Tasks: the ``selected`` non-crashed queries, plus the ``crash_selected`` crashed ones for
crash-aware predictors, at the requested context sizes.

Key ``{problem, model, checkpoint, tasks, tasks_digest, n, part}`` (``part = "i/k"``: every k-th
task starting at i, so a large unit can be split over Slurm array tasks).

Blob ``predictions.parquet``, one row per scored cutoff:
``task_id, run_id, instance, seed, n_ctx, query_index, crashed, crash_step, target_J,
cutoff_frac`` (NaN for a stop-grid row), ``cutoff_fallback, cutoff_step, J_obs, ll, p_crash, dist``
(``bar`` | ``gaussian``), ``edges, probs`` (float32 lists, bar rows at the five cutoffs only),
``mean, var, seconds`` (wall time of the predict call divided by the number of cutoffs of that
task). ``meta.json``: ``{key, model, context_aware, crash_aware, device, host, n_rows, n_tasks,
seconds}``.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from esbopfn.benchmarks.kinds import BenchmarkKind, Key

log = logging.getLogger(__name__)

#: Cutoff fractions of T (thesis setup, prediction metrics).
CUTOFF_FRACS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
#: Predictor families without a crash channel.
CRASH_BLIND_FAMILIES = frozenset({"segment-gp", "tabpfn", "segment-tabpfn"})
#: Untrained context-aware baselines (scored only where their context suffices).
PREDICTION_COLUMNS = ["task_id", "run_id", "instance", "seed", "n_ctx", "query_index", "crashed",
                      "crash_step", "target_J", "cutoff_frac", "cutoff_fallback", "cutoff_step",
                      "J_obs", "ll", "p_crash", "dist", "edges", "probs", "mean", "var", "seconds"]


# ------------------------------------------------------------------------------ models
def family(model_id: str) -> str:
    return model_id.split("/")[0]


def crash_aware(model_id: str) -> bool:
    return family(model_id) not in CRASH_BLIND_FAMILIES


def resolve_model(template: str, problem: str) -> str:
    """``decpfn/{p}/binned/1M`` -> ``decpfn/cartpole/binned/1M``."""
    return template.replace("{p}", problem)


def checkpoint_path(model_id: str, home: Path | str | None = None) -> Path | None:
    """The weights file a model id resolves to (None for models without weights)."""
    fam = family(model_id)
    if fam == "segment-gp":
        return None
    if fam in ("tabpfn", "segment-tabpfn"):
        from esbopfn.predictors.checkpoints import TABPFN_FILENAME, tabpfn_dir

        return tabpfn_dir(home) / TABPFN_FILENAME
    from esbopfn.training.registry import MODEL_FILE, ModelId, model_dir

    try:
        ModelId.parse(model_id)
    except ValueError:
        return None
    return model_dir(model_id, home) / MODEL_FILE


def checkpoint_hash(model_id: str, home: Path | str | None = None) -> str:
    """sha256 (16 hex digits) of the weights, ``none`` without weights, ``missing`` if absent.

    Hashes are cached in ``<PFN_HOME>/benchmarks/checkpoint_hashes.json`` by path, size and mtime.
    """
    from esbopfn.config import paths
    from esbopfn.predictors.checkpoints import sha256

    path = checkpoint_path(model_id, home)
    if path is None:
        return "none"
    if not path.is_file():
        return "missing"
    cache_file = paths(home).root / "benchmarks" / "checkpoint_hashes.json"
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cache = {}
    st = path.stat()
    entry = cache.get(str(path))
    if entry and entry["size"] == st.st_size and entry["mtime"] == st.st_mtime:
        return entry["sha16"]
    h = sha256(path)[:16]
    cache[str(path)] = {"size": st.st_size, "mtime": st.st_mtime, "sha16": h}
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1))
    except OSError:
        pass
    return h


def build_predictor(model_id: str, problem: str) -> Any:
    from esbopfn.predictors.base import get_predictor
    from esbopfn.training.registry import register_predictor_families

    register_predictor_families()
    return get_predictor(model_id, problem)


def device_tag() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def environment() -> dict[str, Any]:
    import torch

    return {"device": device_tag(), "host": socket.gethostname(), "cpu": platform.processor()
            or platform.machine(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads()}


# ------------------------------------------------------------------------------ cutoffs
def cutoff_steps(steps: np.ndarray, T: int, fracs: tuple[float, ...] = CUTOFF_FRACS
                 ) -> dict[float, tuple[int, bool]]:
    """``frac -> (step, fallback)``: the last step ``<= frac * T`` among ``steps`` (those in
    ``[1, T)``), or the first step with ``fallback=True`` if none lies at or before it."""
    live = np.unique(np.asarray(steps, dtype=np.int64))
    live = live[(live >= 1) & (live < T)]
    out: dict[float, tuple[int, bool]] = {}
    if live.size == 0:
        return out
    for f in fracs:
        want = math.floor(float(f) * T + 1e-9)
        below = live[live <= want]
        out[float(f)] = (int(below[-1]), False) if below.size else (int(live[0]), True)
    return out


# ------------------------------------------------------------------------------ scoring
def _dist_fields(pred: Any, full: bool) -> dict[str, Any]:
    from esbopfn.predictors.base import BarPrediction

    try:
        mean, var = pred.mean_var()
    except Exception:  # noqa: BLE001
        mean, var = float("nan"), float("nan")
    if isinstance(pred, BarPrediction):
        return {"dist": "bar", "mean": float(mean), "var": float(var),
                "edges": np.asarray(pred.edges, np.float32).tolist() if full else None,
                "probs": np.asarray(pred.probs, np.float32).tolist() if full else None}
    return {"dist": "gaussian", "mean": float(mean), "var": float(var), "edges": None,
            "probs": None}


def score_task(predictor: Any, spec: Any, run: Any, task: Any, *, context_aware: bool,
               stop_grid: bool, fracs: tuple[float, ...] = CUTOFF_FRACS) -> list[dict[str, Any]]:
    """Rows of one task: the five cutoffs and, with ``stop_grid``, every predictable step."""
    from esbopfn.predictors.base import conditioned_log_likelihood

    T = int(spec.T)
    crashed = bool(task.crashed)
    end = int(task.crash_step) if crashed else T
    steps_all = np.asarray(predictor.predictable_steps(spec), dtype=np.int64)
    live = np.unique(steps_all[(steps_all >= 1) & (steps_all < end)])
    marks = {} if crashed else cutoff_steps(live, T, fracs)
    if crashed or stop_grid:
        steps = live
    else:
        steps = np.unique(np.array([s for s, _ in marks.values()], dtype=np.int64))
    if steps.size == 0:
        return []
    qi = int(task.query_index)
    queries = [run.trial(qi, int(s)) for s in steps]
    context = run.context(int(task.n_ctx)) if context_aware else []
    t0 = time.perf_counter()
    preds = predictor.predict_many(queries, context)
    seconds = (time.perf_counter() - t0) / len(steps)
    target = math.inf if crashed else float(task.target_J)
    base = {"task_id": task.task_id, "run_id": task.run_id, "instance": int(task.instance),
            "seed": int(task.seed), "n_ctx": int(task.n_ctx), "query_index": qi,
            "crashed": crashed, "crash_step": int(task.crash_step),
            "target_J": float(task.target_J), "seconds": seconds}
    by_step: dict[int, list[tuple[float, bool]]] = {}
    for f, (s, fb) in marks.items():
        by_step.setdefault(s, []).append((f, fb))
    rows = []
    for s, q, pred in zip(steps, queries, preds):
        j_obs = float(q.J_obs)
        ll = conditioned_log_likelihood(pred, target, j_obs)
        common = {**base, "cutoff_step": int(s), "J_obs": j_obs, "ll": float(ll),
                  "p_crash": float(pred.p_crash)}
        for f, fb in by_step.get(int(s), []):
            rows.append({**common, "cutoff_frac": f, "cutoff_fallback": fb,
                         **_dist_fields(pred, True)})
        if int(s) not in by_step:
            rows.append({**common, "cutoff_frac": float("nan"), "cutoff_fallback": False,
                         **_dist_fields(pred, False)})
    return rows


def parse_part(part: str | None) -> tuple[int, int]:
    if not part:
        return 0, 1
    i, k = (int(v) for v in str(part).split("/"))
    if not 0 <= i < k:
        raise ValueError(f"invalid part {part!r}")
    return i, k


def select_tasks(df: pd.DataFrame, n_values: list[int] | None, crash: bool, part: str | None
                 ) -> pd.DataFrame:
    keep = df.selected | (df.crash_selected if crash else False)
    if n_values is not None:
        keep &= df.n_ctx.isin([int(n) for n in n_values])
    sel = df[keep].sort_values(["run_order", "n_ctx"], kind="stable").reset_index(drop=True)
    i, k = parse_part(part)
    return sel.iloc[i::k].reset_index(drop=True)


def run_scoring(key: Key, blob_dir: Path, table: str, stop_grid: bool) -> None:
    """Shared body of ``prediction`` and ``prediction_ladder``."""
    from esbopfn.benchmarks.db import Db
    from esbopfn.benchmarks.kinds.fields import read_field
    from esbopfn.benchmarks.tasks import load_tasks
    from esbopfn.problems.base import cached_spec

    problem, model = key["problem"], key["model"]
    found = load_tasks(problem, key["tasks"])
    if found is None or found[1] != key["tasks_digest"]:
        raise LookupError(f"task table {problem}/{key['tasks']} with digest "
                          f"{key['tasks_digest']} not found")
    tasks_df = found[0]
    if key.get("checkpoint") not in (None, "none") and checkpoint_hash(model) != key["checkpoint"]:
        raise RuntimeError(f"{model}: checkpoint hash {checkpoint_hash(model)} differs from the "
                           f"unit key {key['checkpoint']}")
    aware_crash = crash_aware(model)
    sel = select_tasks(tasks_df, key.get("n"), aware_crash and stop_grid, key.get("part"))
    spec = cached_spec(problem)
    predictor = build_predictor(model, problem)
    context_aware = bool(getattr(predictor, "context_aware", False))
    db = Db()
    field_name = str(tasks_df.field.iloc[0])
    inst_seed = _instances_seed(key["tasks"])
    runs: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for task in sel.itertuples(index=False):
        if task.run_id not in runs:
            from esbopfn.benchmarks.kinds.fields import field_key

            fk = field_key(problem, field_name, int(task.instance), int(task.seed), inst_seed)
            runs = {task.run_id: read_field(db.blob_dir("fields", fk), fk)}
        rows += score_task(predictor, spec, runs[task.run_id], task, context_aware=context_aware,
                           stop_grid=stop_grid and aware_crash)
    df = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    df.to_parquet(Path(blob_dir) / "predictions.parquet", index=False)
    meta = {"key": key, "model": model, "context_aware": context_aware,
            "crash_aware": aware_crash, "n_rows": int(len(df)), "n_tasks": int(len(sel)),
            "seconds": time.perf_counter() - t0, **environment()}
    (Path(blob_dir) / "meta.json").write_text(json.dumps(meta, indent=1))


def _instances_seed(table_name: str) -> Any:
    from esbopfn.benchmarks.instances import BENCH_SEED

    parts = table_name.split("-")
    return int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else BENCH_SEED


def scoring_keys(cfg: dict[str, Any], kind: str, table: str) -> list[Key]:
    """Unit keys for every (problem, model[, part]) of a group section whose tasks are frozen."""
    from esbopfn.benchmarks.groups import instances_seed, problems, section
    from esbopfn.benchmarks.tasks import ensure_tasks

    sec = section(cfg, kind)
    keys: list[Key] = []
    for p in problems(cfg, sec):
        found = ensure_tasks(p, table, sec.get("field"), instances_seed(cfg))
        if found is None:
            log.info("%s %s: field not complete, prediction units follow once it is", kind, p)
            continue
        df, h = found
        name = df_name(table, sec.get("field"), instances_seed(cfg))
        default_n = sorted(int(n) for n in df.n_ctx.unique())
        for m in sec.get("models", []):
            m = {"id": m} if isinstance(m, str) else m
            model = resolve_model(m["id"], p)
            if m.get("problems") and p not in m["problems"]:
                continue
            n_values = [int(n) for n in m.get("n", sec.get("n", default_n))]
            parts = int(m.get("parts", sec.get("parts", 1)))
            for i in range(parts):
                keys.append({"problem": p, "model": model, "checkpoint": checkpoint_hash(model),
                             "tasks": name, "tasks_digest": h, "n": n_values,
                             "part": f"{i}/{parts}"})
    return keys


def df_name(table: str, field: str | None, seed: Any) -> str:
    from esbopfn.benchmarks.tasks import table_name

    return table_name(table, field or ("oracle65" if table == "tasks" else "oracle181"), seed)


def enumerate_units(cfg: dict[str, Any]) -> list[Key]:
    return scoring_keys(cfg, "prediction", "tasks")


def run(key: Key, blob_dir: Path) -> None:
    run_scoring(key, blob_dir, "tasks", stop_grid=True)


def load_predictions(units: list[Any]) -> pd.DataFrame:
    """Concatenated ``predictions.parquet`` of DB units with ``problem`` and ``model`` columns."""
    frames = []
    for u in units:
        df = pd.read_parquet(Path(u.blob_dir) / "predictions.parquet")
        df.insert(0, "problem", u.key["problem"])
        df.insert(1, "model", u.key["model"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


KIND = BenchmarkKind(name="prediction", enumerate=enumerate_units, run=run)
