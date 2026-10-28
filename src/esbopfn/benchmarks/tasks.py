"""Frozen prediction task tables (thesis setup, prediction tasks).

**Context tasks** (field ``oracle65``): trial ``n + 1`` of a run is the query and trials
``1 .. n`` (each cut at the oracle's observed length) are its context, for
``n in (0, 2, 4, 8, 16, 32, 64)``, i.e. queries at trials 1, 3, 5, 9, 17, 33 and 65. A query is
*crashed* when its trajectory crashes when run to the end, its target is then the crash, otherwise
the total cost ``J_T`` of the hidden continuation. Per ``(problem, n)`` exactly 50 non-crashed
queries are ``selected`` and, on problems that can crash, 50 crashed queries are
``crash_selected``, both taken in seed-major run order (seed 0 of every instance, then seed 1, ...)
so a cell spans as many instances as possible.

**Ladder tasks** (field ``oracle181``): the query is the last trial of a run, the context the
first ``n`` trials for the rungs of :data:`LADDER_RUNGS`. The first 50 runs in seed-major order
whose last trial does not crash are used.

Tables are pure functions of the field. They are frozen to
``<PFN_HOME>/benchmarks/tasks/<problem>/<name>.parquet`` with a ``.json`` sidecar holding a
sha256 digest (first 16 hex digits) of the identity, target and selection columns. The digest is
part of every prediction unit key, so a regenerated field cannot silently move the tasks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Context sizes of the prediction benchmark.
CONTEXTS: tuple[int, ...] = (0, 2, 4, 8, 16, 32, 64)
#: Non-crashed and crashed queries per (problem, n).
K_PER_CELL = 50
K_CRASH_PER_CELL = 50
#: Context ladder rungs per problem (thesis fig:sim:context-sweep).
LADDER_RUNGS: dict[str, tuple[int, ...]] = {
    "pt2_deadtime": (0, 1, 2, 4, 8, 16, 33, 67, 134),
    "cartpole": (0, 1, 2, 5, 11, 22, 45, 90, 180),
    "furuta": (0, 1, 2, 5, 11, 22, 45, 90, 180),
}
LADDER_QUERIES = 50

TASK_COLUMNS = ["task_id", "problem", "field", "run_id", "instance", "seed", "run_order", "n_ctx",
                "query_index", "T", "t_obs", "crash_step", "crashed", "target_J", "selected",
                "crash_selected"]
TRIAL_COLUMNS = ["run_id", "instance", "seed", "trial_index", "T", "t_obs", "crash_step", "J_T"]


def trial_table(runs: list[Any]) -> pd.DataFrame:
    """One row per trial of the given field runs (:class:`FieldRun` objects)."""
    rows = []
    for run in runs:
        T = int(run.J.shape[1])
        for i in range(run.n_trials):
            cs = int(run.crash_step[i])
            rows.append({"run_id": run.run_id, "instance": int(run.key["instance"]),
                         "seed": int(run.key["seed"]), "trial_index": i, "T": T,
                         "t_obs": int(run.t_obs[i]), "crash_step": cs,
                         "J_T": float("nan") if cs > 0 else float(run.J[i, -1])})
    return pd.DataFrame(rows, columns=TRIAL_COLUMNS)


def _run_order(trials: pd.DataFrame) -> dict[str, int]:
    runs = trials[["run_id", "seed", "instance"]].drop_duplicates()
    runs = runs.sort_values(["seed", "instance"], kind="stable")
    return {rid: i for i, rid in enumerate(runs.run_id)}


def build_tasks(trials: pd.DataFrame, problem: str, field: str = "oracle65",
                contexts: tuple[int, ...] = CONTEXTS, k: int = K_PER_CELL,
                k_crash: int | None = K_CRASH_PER_CELL, can_crash: bool = True) -> pd.DataFrame:
    """The context task table of one problem from its trial table."""
    order = _run_order(trials)
    by_key = trials.set_index(["run_id", "trial_index"])
    rows = []
    for rid in sorted(order, key=order.get):
        for n in contexts:
            if (rid, n) not in by_key.index:
                raise ValueError(f"{problem} {rid}: no trial {n + 1}, the field needs at least "
                                 f"{max(contexts) + 1} trials")
            q = by_key.loc[(rid, n)]
            cs = int(q.crash_step)
            rows.append({"task_id": f"{problem}:{rid}:n{n}", "problem": problem, "field": field,
                         "run_id": rid, "instance": int(q.instance), "seed": int(q.seed),
                         "run_order": order[rid], "n_ctx": int(n), "query_index": int(n),
                         "T": int(q["T"]), "t_obs": int(q.t_obs), "crash_step": cs,
                         "crashed": cs > 0, "target_J": float("nan") if cs > 0 else float(q.J_T),
                         "selected": False, "crash_selected": False})
    df = pd.DataFrame(rows, columns=TASK_COLUMNS)
    for n in contexts:
        ok = df.index[(df.n_ctx == n) & ~df.crashed]
        if len(ok) < k:
            raise ValueError(f"{problem} n={n}: only {len(ok)} non-crashed queries, need {k}")
        df.loc[df.loc[ok].sort_values("run_order", kind="stable").index[:k], "selected"] = True
        if can_crash and k_crash:
            bad = df.index[(df.n_ctx == n) & df.crashed]
            if len(bad) < k_crash:
                raise ValueError(f"{problem} n={n}: only {len(bad)} crashed queries, need "
                                 f"{k_crash}")
            df.loc[df.loc[bad].sort_values("run_order", kind="stable").index[:k_crash],
                   "crash_selected"] = True
    return df


def build_ladder_tasks(trials: pd.DataFrame, problem: str, field: str = "oracle181",
                       rungs: tuple[int, ...] | None = None, k: int = LADDER_QUERIES
                       ) -> pd.DataFrame:
    """The context-ladder task table: the last trial of the first ``k`` seed-major runs whose last
    trial does not crash, with the first ``n`` trials as context for every rung ``n``."""
    rungs = rungs or LADDER_RUNGS[problem]
    order = _run_order(trials)
    rows = []
    chosen = 0
    for rid in sorted(order, key=order.get):
        run = trials[trials.run_id == rid]
        q = run.loc[run.trial_index.idxmax()]
        qi = int(q.trial_index)
        if int(q.crash_step) > 0:
            continue
        if max(rungs) > qi:
            raise ValueError(f"{problem} {rid}: rung {max(rungs)} exceeds the {qi} trials "
                             f"before the query")
        for n in rungs:
            rows.append({"task_id": f"{problem}:{rid}:n{n}", "problem": problem, "field": field,
                         "run_id": rid, "instance": int(q.instance), "seed": int(q.seed),
                         "run_order": order[rid], "n_ctx": int(n), "query_index": qi,
                         "T": int(q["T"]), "t_obs": int(q.t_obs), "crash_step": 0,
                         "crashed": False, "target_J": float(q.J_T), "selected": True,
                         "crash_selected": False})
        chosen += 1
        if chosen == k:
            break
    if chosen < k:
        raise ValueError(f"{problem}: only {chosen} runs with a non-crashed last trial, need {k}")
    return pd.DataFrame(rows, columns=TASK_COLUMNS)


def digest(df: pd.DataFrame) -> str:
    """sha256 (16 hex digits) over identity, target and selection columns, order independent."""
    cols = ["task_id", "crash_step", "crashed", "target_J", "selected", "crash_selected"]
    key = df.sort_values("task_id")[[c for c in cols if c in df.columns]].copy()
    key["target_J"] = key["target_J"].map(lambda v: "nan" if not np.isfinite(v) else repr(float(v)))
    return hashlib.sha256(key.to_csv(index=False).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------ storage
def tasks_dir(problem: str, home: Path | str | None = None) -> Path:
    from esbopfn.config import paths

    return paths(home).root / "benchmarks" / "tasks" / problem


def table_name(table: str, field: str, instances_seed: Any = None) -> str:
    from esbopfn.benchmarks.instances import BENCH_SEED

    name = f"{table}-{field}"
    if instances_seed is not None and instances_seed != BENCH_SEED:
        name += f"-{instances_seed}"
    return name


def save_tasks(df: pd.DataFrame, problem: str, name: str, meta: dict[str, Any] | None = None,
               home: Path | str | None = None) -> str:
    out = tasks_dir(problem, home)
    out.mkdir(parents=True, exist_ok=True)
    h = digest(df)
    df.to_parquet(out / f"{name}.parquet", index=False)
    info = {"problem": problem, "name": name, "digest": h, "n_tasks": len(df),
            "n_selected": int(df.selected.sum()), "n_crash_selected": int(df.crash_selected.sum()),
            "contexts": sorted(int(n) for n in df.n_ctx.unique()), **(meta or {})}
    (out / f"{name}.json").write_text(json.dumps(info, indent=1))
    return h


def load_tasks(problem: str, name: str, home: Path | str | None = None
               ) -> tuple[pd.DataFrame, str] | None:
    path = tasks_dir(problem, home) / f"{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    stored = json.loads(path.with_suffix(".json").read_text())["digest"]
    if digest(df) != stored:
        raise RuntimeError(f"task table {path} does not match its stored digest {stored}")
    return df, stored


def ensure_tasks(problem: str, table: str = "tasks", field: str | None = None,
                 instances_seed: Any = None, db: Any = None, home: Path | str | None = None
                 ) -> tuple[pd.DataFrame, str] | None:
    """The frozen table, built from the field on first use. None while the field is incomplete.

    ``table`` is ``"tasks"`` (context tasks, field ``oracle65``) or ``"ladder"`` (field
    ``oracle181``).
    """
    from esbopfn.benchmarks.instances import BENCH_SEED
    from esbopfn.benchmarks.kinds.fields import load_field_runs
    from esbopfn.problems.base import cached_spec

    field = field or ("oracle65" if table == "tasks" else "oracle181")
    seed = BENCH_SEED if instances_seed is None else instances_seed
    name = table_name(table, field, seed)
    found = load_tasks(problem, name, home)
    if found is not None:
        return found
    runs = load_field_runs(problem, field, seed, db)
    if runs is None:
        return None
    trials = trial_table(runs)
    if table == "tasks":
        df = build_tasks(trials, problem, field, can_crash=cached_spec(problem).can_crash)
    else:
        df = build_ladder_tasks(trials, problem, field)
    h = save_tasks(df, problem, name, {"field": field, "instances_seed": seed}, home)
    log.info("froze %s %s (%d tasks, digest %s)", problem, name, len(df), h)
    return df, h


def verify_tasks(problem: str, table: str = "tasks", field: str | None = None,
                 instances_seed: Any = None, db: Any = None) -> bool:
    """Rebuild a frozen table from its field and compare digests."""
    from esbopfn.benchmarks.instances import BENCH_SEED
    from esbopfn.benchmarks.kinds.fields import load_field_runs
    from esbopfn.problems.base import cached_spec

    field = field or ("oracle65" if table == "tasks" else "oracle181")
    seed = BENCH_SEED if instances_seed is None else instances_seed
    found = load_tasks(problem, table_name(table, field, seed))
    runs = load_field_runs(problem, field, seed, db)
    if found is None or runs is None:
        return False
    trials = trial_table(runs)
    df = (build_tasks(trials, problem, field, can_crash=cached_spec(problem).can_crash)
          if table == "tasks" else build_ladder_tasks(trials, problem, field))
    return digest(df) == found[1]
