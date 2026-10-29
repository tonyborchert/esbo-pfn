"""Import the thesis benchmark results into the benchmark database.

``import_thesis_results(root)`` reads the result files of the earlier implementation below the
``bachelorarbeit`` root and stores them with ``provenance: "thesis"`` in the unit key, in the blob
schema of the native kinds wherever the old data is raw enough:

==============================================  ==============================================
old file                                         stored as
==============================================  ==============================================
``impl/dynformer/optplay/instances/<p>.json``    ``instances`` (phi mapped by order to names)
``impl/dynformer/predplay/tasks/<p>.parquet``    task table ``tasks-thesis`` (tasks directory)
``experiments/ladder/results/*.parquet``         ``prediction_ladder`` (raw per-query scores)
``results_timing_bundle/predplay_timing.json``   ``prediction_timing`` (:mod:`.import_timing`)
``<p>@g0.05@bcapped/phi_NNNN/runs/<cell>.json``  ``optimization`` (raw runs, cluster copies)
``<p>@g0.05@bcapped@timing/...``                 ``optimization_timing``
``predplay/results/fig1_*.csv, fig2_*.csv``      ``thesis_series`` (aggregates only)
``writing/thesis/data/sources/fig1_cross_5prob_canonmed.csv``  ``thesis_series`` variant
                                                 ``canonmed`` (cluster recomputation before the
                                                 LC-PFN z-score retraining)
``optplay/results/<p>@g0.05@bcapped_*``          ``thesis_series`` (per-instance finals, runs,
                                                 crash counts, unscaled regret curves)
==============================================  ==============================================

Thesis names are mapped to the ids of this package (:func:`map_model`, :func:`map_cell`). Rows of
models outside the thesis registry (TabPFN ``ls``/``diff`` encodings, ``stengertpfn/modeminimal``,
the retired recipe arms, the plain ``furuta_lqr`` problem) are skipped.

Variants of the prediction aggregates: ``thesis`` = ``fig1_*.csv`` / ``fig2_<p>.csv`` (the build
whose pooled numbers sim_results.tex quotes, LC-PFN with z-scored inputs), ``pre_lcz`` = the
``*_pre_lcz`` build before the LC-PFN retraining, ``canonmed`` = the cluster recomputation
``fig1_cross_5prob_canonmed.csv`` (also before the retraining).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from esbopfn.benchmarks.db import DONE, PENDING, Db

log = logging.getLogger(__name__)

PROVENANCE = "thesis"
#: Old problem keys -> problem names (``furuta_lqr`` is not a thesis problem).
PROBLEM_MAP = {"boiler": "boiler", "threetank_pi": "threetank_pi", "pt2_deadtime": "pt2_deadtime",
               "cartpole": "cartpole", "furuta_lqr_rigreset": "furuta"}
#: Problems on which the XDecPFN cells ``nmatched`` and ``lnjw128`` are the same weights.
_LNJW128_IS_NMATCHED = ("cartpole", "furuta")


# ------------------------------------------------------------------------------ name maps
def map_problem(old: str) -> str | None:
    return PROBLEM_MAP.get(old)


def map_model(old: str, problem: str | None = None) -> str | None:
    """Old predictor id -> id of this package (``{p}`` kept when ``problem`` is None), or None
    for models outside the thesis registry."""
    p = problem or "{p}"
    size = {"10": "100k", "100": "1M"}
    m = re.fullmatch(r"decpfn/epochs(10|100)/head(binned|gaussian|linear)(/states0)?", old)
    if m:
        fam = "decpfn-nostates" if m.group(3) else "decpfn"
        return f"{fam}/{p}/{m.group(2)}/{size[m.group(1)]}"
    m = re.fullmatch(r"lcpfn_zscore/epochs(10|100)/head(binned|gaussian|linear)", old)
    if m:
        return f"lcpfn/{p}/{m.group(2)}/{size[m.group(1)]}"
    m = re.fullmatch(r"lcpfn/epochs(10|100)/head(binned|gaussian|linear)", old)
    if m:
        return f"lcpfn/{p}/{m.group(2)}/{size[m.group(1)]}/prezscore"
    if old in ("dynformer_c1r2/cellnmatched", "dynformer_c1r2/cellf_s20260931"):
        return f"xdecpfn/{p}/thesis/default"
    if old == "dynformer_c1r2/celllnjw128":
        if problem in _LNJW128_IS_NMATCHED:
            return f"xdecpfn/{p}/thesis/default"
        return f"xdecpfn/{p}/thesis/default/lnjw128"
    if old in ("dynformer_fr2", "dynformer_fr2/cellfuruta180sg"):
        return f"ftpfn/{p}/thesis/default"
    m = re.fullmatch(r"tabpfn/resolution(\d+)/typeraw", old)
    if m:
        return f"tabpfn/{m.group(1)}"
    m = re.fullmatch(r"stengertpfn/mode(split|rate)", old)
    if m:
        return f"segment-tabpfn/{m.group(1)}"
    if old == "stenger":
        return "segment-gp"
    return None


#: Old optplay cell -> (strategy id, parameters, series label). ``QC`` cells hold XDecPFN on
#: Cartpole and Furuta, ``QCn`` on the other problems (see :func:`map_cell`).
_CELLS: dict[str, tuple[str, dict[str, Any], str]] = {
    "Random uniform+cost": ("esrs", {}, "esrs"),
    "RandomESBO": ("esrs", {}, "esrs"),
    "Plain GP": ("bo", {}, "bo"),
    "Stenger ESBO": ("esbo-gp", {}, "esbo-gp"),
    "ESBO-TabPFN": ("esbo-t-tabpfn", {"L": 50}, "esbo-t-tabpfn"),
    "Oracle BO": ("esbo-oracle", {}, "esbo-oracle"),
    "MTr 0.9": ("decpfn", {"tau": 0.9, "crash_threshold": 0.9}, "decpfn"),
    "MTr 0.99": ("decpfn", {"tau": 0.99, "crash_threshold": 0.9}, "decpfn@tau=0.99"),
    "MTr 0.99 ft0.99": ("decpfn", {"tau": 0.99, "crash_threshold": 0.99},
                        "decpfn@crash_threshold=0.99,tau=0.99"),
    "HMTr 0.9": ("decpfn-h", {"tau": 0.9, "crash_threshold": 0.9}, "decpfn-h"),
    "HMTr 0.99": ("decpfn-h", {"tau": 0.99, "crash_threshold": 0.9}, "decpfn-h@tau=0.99"),
    "HMTr 0.99 ft0.99": ("decpfn-h", {"tau": 0.99, "crash_threshold": 0.99},
                         "decpfn-h@crash_threshold=0.99,tau=0.99"),
    "QF EI abs xgrid 0.99": ("ftpfn-ei", {"tau": 0.99}, "ftpfn-ei"),
    "QF PI abs xgrid 0.99": ("ftpfn-pi", {"tau": 0.99}, "ftpfn-pi"),
    "QF EI abs xgrid 0.9": ("ftpfn-ei", {"tau": 0.9}, "ftpfn-ei@tau=0.9"),
    "QF PI abs xgrid 0.9": ("ftpfn-pi", {"tau": 0.9}, "ftpfn-pi@tau=0.9"),
}
for _acq in ("PI", "EI"):
    for _tau, _suffix in ((0.99, ""), (0.9, "@tau=0.9")):
        _id = f"xdecpfn-{_acq.lower()}"
        for _prefix in ("QC", "QCn"):
            _CELLS[f"{_prefix} {_acq} abs {_tau}"] = (_id, {"tau": _tau}, _id + _suffix)


def map_cell(cell: str, problem: str) -> tuple[str, dict[str, Any], str] | None:
    """Old optplay cell on ``problem`` -> ``(strategy, params, label)`` or None."""
    if cell.startswith("QC ") and problem not in _LNJW128_IS_NMATCHED:
        return None
    if cell.startswith("QCn ") and problem in _LNJW128_IS_NMATCHED:
        return None
    entry = _CELLS.get(cell)
    if entry is None:
        return None
    sid, params, label = entry
    params = dict(params)
    if sid.startswith(("decpfn", "xdecpfn", "ftpfn")):
        family = {"decpfn": "decpfn", "decpfn-h": "decpfn", "xdecpfn-pi": "xdecpfn",
                  "xdecpfn-ei": "xdecpfn", "ftpfn-ei": "ftpfn", "ftpfn-pi": "ftpfn"}[sid]
        params["model"] = (f"decpfn/{problem}/binned/1M" if family == "decpfn"
                           else f"{family}/{problem}/thesis/default")
    return sid, params, label


#: Old performance categories -> timing categories, per strategy family.
def map_timing(strategy: str, category: str) -> str:
    pure = strategy.startswith(("xdecpfn", "ftpfn"))
    if category in ("fitting",):
        return "gp_acq"
    if category == "acquisition":
        return "model_acq" if pure else "gp_acq"
    if category == "stenger prediction":
        return "segment_gp"
    if category == "forecast":
        return "model_forecast"
    if category == "prediction":
        if strategy == "esbo-gp":
            return "segment_gp"
        if strategy == "esbo-t-tabpfn":
            return "tabpfn_forecast"
        return "model_forecast"
    return f"thesis_{category}"


def parse_run_id(run_id: str) -> tuple[int, int]:
    """``run_phi0012_s03`` -> ``(12, 3)``."""
    m = re.fullmatch(r"run_phi(\d+)_s(\d+)", run_id)
    if not m:
        raise ValueError(f"unexpected run id {run_id!r}")
    return int(m.group(1)), int(m.group(2))


# ------------------------------------------------------------------------------ writing
def store(db: Db, kind: str, key: dict[str, Any], write: Callable[[Path], None],
          overwrite: bool = False) -> bool:
    """Write one unit unless it exists. Returns True if written."""
    if db.has(kind, key):
        if not overwrite:
            return False
        db.mark(kind, key, PENDING)
    db.run(kind, key, lambda _k, d: write(d))
    return db.get(kind, key).status == DONE


def store_series(db: Db, figure: str, problem: str, series: str, variant: str, source: Path,
                 data: dict[str, list[Any]], meta: dict[str, Any] | None = None,
                 overwrite: bool = False) -> bool:
    key = {"figure": figure, "problem": problem, "series": series, "variant": variant,
           "provenance": PROVENANCE}

    def write(d: Path) -> None:
        payload = {**{k: key[k] for k in ("figure", "problem", "series", "variant")},
                   "source": str(source), "columns": list(data), "data": data,
                   "meta": meta or {}}
        (d / "series.json").write_text(json.dumps(payload, indent=1, default=_json_value))

    return store(db, "thesis_series", key, write, overwrite)


def _json_value(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    raise TypeError(type(v).__name__)


def _columns(df: pd.DataFrame) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for c in df.columns:
        vals = df[c].tolist()
        out[c] = [None if isinstance(v, float) and math.isnan(v) else v for v in vals]
    return out


# ------------------------------------------------------------------------------ predplay
def import_fig1_csv(db: Db, path: Path, figure: str, variant: str, per_problem: bool,
                    overwrite: bool = False) -> int:
    """fig1 cross / per-problem tables -> one ``thesis_series`` per (problem, model)."""
    df = pd.read_csv(path)
    n = 0
    groups = df.groupby(["problem", "model_id"] if per_problem else ["model_id"], sort=True)
    for keys, g in groups:
        if per_problem:
            old_problem, old_model = keys
            problem = map_problem(old_problem)
            if problem is None:
                continue
            series = map_model(old_model, problem)
        else:
            old_model = keys[0] if isinstance(keys, tuple) else keys
            problem, series = "pooled", map_model(old_model)
        if series is None:
            continue
        cols = g.drop(columns=[c for c in ("problem", "model_id", "group") if c in g.columns])
        n += store_series(db, figure, problem, series, variant, path,
                          _columns(cols.sort_values("n_ctx")), {"old_model": old_model},
                          overwrite)
    return n


def import_fig2_csv(db: Db, path: Path, variant: str, overwrite: bool = False) -> int:
    old_problem = path.stem.removeprefix("fig2_").removesuffix("_pre_lcz")
    problem = map_problem(old_problem)
    if problem is None:
        return 0
    df = pd.read_csv(path)
    n = 0
    for old_model, g in df.groupby("model_id", sort=True):
        series = map_model(old_model, problem)
        if series is None:
            continue
        cols = g.drop(columns=["model_id", "group"])
        n += store_series(db, "fig2_stop_curves", problem, series, variant, path, _columns(cols),
                          {"old_model": old_model}, overwrite)
    return n


def import_fig1_meta(db: Db, path: Path, variant: str, overwrite: bool = False) -> int:
    meta = json.loads(path.read_text())
    med = {map_problem(k): v for k, v in meta.get("median_per_dataset", {}).items()
           if map_problem(k)}
    return int(store_series(
        db, "fig3_med_D", "pooled", "med_D", variant, path,
        {"problem": list(med), "med_D": list(med.values())},
        {"participants": meta.get("participating_models", []), "K": meta.get("K")}, overwrite))


def import_task_table(path: Path, home: Path | str | None = None) -> str | None:
    """predplay ``tasks/<p>.parquet`` -> task table ``tasks-thesis`` of the mapped problem."""
    from esbopfn.benchmarks.tasks import TASK_COLUMNS, save_tasks

    problem = map_problem(path.stem)
    if problem is None:
        return None
    old = pd.read_parquet(path)
    ids = [parse_run_id(r) for r in old.run_id]
    df = pd.DataFrame({
        "task_id": [f"{problem}:i{i:04d}_s{s:02d}:n{n}" for (i, s), n in zip(ids, old.n_ctx)],
        "problem": problem, "field": "thesis_oracle65",
        "run_id": [f"i{i:04d}_s{s:02d}" for i, s in ids],
        "instance": [i for i, _ in ids], "seed": [s for _, s in ids],
        "run_order": old.run_order.astype(int), "n_ctx": old.n_ctx.astype(int),
        "query_index": old.query_index.astype(int), "T": old["T"].astype(int),
        "t_obs": old.t_obs.astype(int),
        "crash_step": np.where(old.crashed, old.natural_t, 0).astype(int),
        "crashed": old.crashed.astype(bool), "target_J": old.target_J.astype(float),
        "selected": old.selected.astype(bool), "crash_selected": old.crash_selected.astype(bool),
    })[TASK_COLUMNS]
    meta_path = path.with_suffix(".json")
    old_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return save_tasks(df, problem, "tasks-thesis",
                      {"provenance": PROVENANCE, "source": str(path),
                       "thesis_sha": old_meta.get("sha")}, home)


def import_ladder_parquet(db: Db, path: Path, overwrite: bool = False) -> bool:
    """Ladder results of one (problem, model) -> a native ``prediction_ladder`` unit."""
    from esbopfn.benchmarks.kinds.prediction import PREDICTION_COLUMNS
    from esbopfn.problems.base import cached_spec

    old = pd.read_parquet(path)
    if old.empty:
        return False
    problem = map_problem(str(old.problem.iloc[0]))
    old_model = str(old.model_id.iloc[0])
    model = map_model(old_model, problem) if problem else None
    if problem is None or model is None:
        return False
    T = cached_spec(problem).T
    ids = [parse_run_id(r) for r in old.run_id]
    run_ids = [f"i{i:04d}_s{s:02d}" for i, s in ids]
    df = pd.DataFrame({
        "task_id": [f"{problem}:{r}:n{n}" for r, n in zip(run_ids, old.n_ctx)],
        "run_id": run_ids, "instance": [i for i, _ in ids], "seed": [s for _, s in ids],
        "n_ctx": old.n_ctx.astype(int), "query_index": old.trial_index.astype(int),
        "crashed": False, "crash_step": 0, "target_J": old.target_J.astype(float),
        "cutoff_frac": old.cutoff_frac.astype(float),
        "cutoff_fallback": old.cutoff_step > np.round(old.cutoff_frac * T),
        "cutoff_step": old.cutoff_step.astype(int), "J_obs": np.nan,
        "ll": old.ll_conditioned.astype(float), "p_crash": old.p_fail.astype(float),
        "dist": None, "edges": None, "probs": None, "mean": old.pred_mean.astype(float),
        "var": old.pred_std.astype(float) ** 2, "seconds": np.nan,
    })[PREDICTION_COLUMNS]
    key = {"problem": problem, "model": model, "checkpoint": PROVENANCE, "tasks": "ladder-thesis",
           "tasks_digest": PROVENANCE, "n": sorted(int(n) for n in df.n_ctx.unique()),
           "part": "0/1", "provenance": PROVENANCE, "source_model": old_model}
    side = path.with_suffix(".json")

    def write(d: Path) -> None:
        df.to_parquet(d / "predictions.parquet", index=False)
        meta = {"key": key, "model": model, "source": str(path),
                "thesis_meta": json.loads(side.read_text()) if side.exists() else {},
                "context_aware": True, "crash_aware": True, "n_rows": len(df)}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))

    return store(db, "prediction_ladder", key, write, overwrite)


#: Old prediction-store ids that are not the thesis weights of the id they map to: the raw-J
#: XDecPFN twin and the default (non-``furuta180sg``) FT-PFN on Furuta.
_NOT_THESIS_PREDS = {"dynformer_c1r2/cellf_s20260931", "dynformer_c1r2/cellf_s20260931sg"}
#: Priority when two old ids of one problem map to the same model id (lower wins).
_PRED_PRIORITY = {"dynformer_c1r2/cellnmatched": 0, "dynformer_c1r2/celllnjw128": 1,
                  "dynformer_fr2/cellfuruta180sg": 0, "dynformer_fr2": 1}


def _predplay_model_dirs(preds: Path) -> list[tuple[str, str, Path]]:
    """``(problem, model id, dir)`` of the finished model directories of one predplay store
    ``preds/<old problem>/<old model>/`` (one per model id, see ``_PRED_PRIORITY``)."""
    best: dict[tuple[str, str], tuple[int, str, Path]] = {}
    for d in sorted(preds.glob("*/*")):
        problem = map_problem(d.parent.name)
        if problem is None or not d.is_dir() or not any(d.glob("shard-*.done.json")):
            continue
        meta = sorted(d.glob("part-*.json"))
        if not meta:
            continue
        old = str(json.loads(meta[0].read_text()).get("model_id", ""))
        if old in _NOT_THESIS_PREDS:
            continue
        if (problem == "furuta" and old == "dynformer_fr2"
                and (d.parent / "dynformer_fr2__cellfuruta180sg").is_dir()):
            continue
        model = map_model(old, problem)
        if model is None:
            continue
        rank = _PRED_PRIORITY.get(old, 0)
        if (problem, model) not in best or rank < best[(problem, model)][0]:
            best[(problem, model)] = (rank, old, d)
    return [(p, m, d) for (p, m), (_, _, d) in sorted(best.items())]


def import_predplay_dir(db: Db, path: Path, problem: str, model: str,
                        home: Path | str | None = None, overwrite: bool = False) -> bool:
    """One predplay model directory ``preds/<p>/<model>/part-*.parquet`` -> a ``prediction``
    unit on the thesis task table. Every task the old store scored is kept (the thesis stop
    curves of the context-blind models pool this full field, not only the ``selected`` tasks of
    a native unit), crash flags and crash steps come from the task table."""
    from esbopfn.benchmarks.kinds.prediction import PREDICTION_COLUMNS, crash_aware
    from esbopfn.benchmarks.tasks import load_tasks

    loaded = load_tasks(problem, "tasks-thesis", home)
    if loaded is None:
        raise FileNotFoundError(f"no thesis task table for {problem} (import the thesis "
                                "results without --raw first)")
    tasks, tasks_digest = loaded
    old = pd.concat([pd.read_parquet(f) for f in sorted(path.glob("part-*.parquet"))],
                    ignore_index=True)
    if old.empty:
        return False
    old_model = str(old.model_id.iloc[0])
    ids = [parse_run_id(r) for r in old.run_id]
    run_ids = [f"i{i:04d}_s{s:02d}" for i, s in ids]
    old["task_id"] = [f"{problem}:{r}:n{n}" for r, n in zip(run_ids, old.n_ctx)]
    t = tasks.set_index("task_id")
    unknown = ~old.task_id.isin(t.index)
    if unknown.any():
        log.warning("%s: %d rows of tasks outside the thesis task table, skipped", path,
                    int(unknown.sum()))
        old = old[~unknown]
    t = t.loc[old.task_id]
    if not (old.trial_index.to_numpy() == t.query_index.to_numpy()).all():
        raise ValueError(f"{path}: query index differs from the thesis task table")
    T = int(t["T"].iloc[0])
    frac = old.cutoff_frac.astype(float)
    df = pd.DataFrame({
        "task_id": old.task_id.to_numpy(), "run_id": t.run_id.to_numpy(),
        "instance": t.instance.to_numpy(), "seed": t.seed.to_numpy(),
        "n_ctx": old.n_ctx.astype(int).to_numpy(), "query_index": t.query_index.to_numpy(),
        "crashed": t.crashed.to_numpy(), "crash_step": t.crash_step.to_numpy(),
        "target_J": t.target_J.to_numpy(), "cutoff_frac": frac.to_numpy(),
        "cutoff_fallback": (frac.notna() & (old.cutoff_step > np.round(frac * T))).to_numpy(),
        "cutoff_step": old.cutoff_step.astype(int).to_numpy(),
        "J_obs": old.J_obs.astype(float).to_numpy(),
        "ll": old.ll_conditioned.astype(float).to_numpy(),
        "p_crash": old.p_fail.astype(float).to_numpy(), "dist": None, "edges": None,
        "probs": None, "mean": old.pred_mean.astype(float).to_numpy(),
        "var": (old.pred_std.astype(float) ** 2).to_numpy(), "seconds": np.nan,
    })[PREDICTION_COLUMNS]
    key = {"problem": problem, "model": model, "checkpoint": PROVENANCE,
           "tasks": "tasks-thesis-all",
           "tasks_digest": tasks_digest,
           "n": sorted(int(n) for n in df.n_ctx.unique()), "part": "0/1",
           "provenance": PROVENANCE, "source_model": old_model}
    metas = [json.loads(f.read_text()) for f in sorted(path.glob("part-*.json"))]

    def write(d: Path) -> None:
        df.to_parquet(d / "predictions.parquet", index=False)
        meta = {"key": key, "model": model, "source": str(path), "thesis_meta": metas,
                "context_aware": bool(metas and len(metas[0].get("contexts", [])) > 1),
                "crash_aware": crash_aware(model), "n_rows": len(df),
                "n_tasks": int(df.task_id.nunique()),
                "device": metas[0].get("device") if metas else None}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))

    return store(db, "prediction", key, write, overwrite)


def import_predplay_preds(db: Db, preds: Path, home: Path | str | None = None,
                          overwrite: bool = False) -> int:
    """Every thesis model directory of a copied predplay store (``predplay/preds``)."""
    n = 0
    for problem, model, d in _predplay_model_dirs(preds):
        try:
            n += import_predplay_dir(db, d, problem, model, home, overwrite)
        except FileNotFoundError as exc:
            log.warning("%s skipped: %s", d, exc)
    return n


# ------------------------------------------------------------------------------ optplay
def import_instances_json(db: Db, path: Path, overwrite: bool = False) -> bool:
    """optplay ``instances/<p>.json`` -> an ``instances`` unit (``{problem, provenance}``).

    The parameter vectors are mapped by position onto the prior's names (the old code used the
    same order). Furuta's tenth component is the actuation dead time and ``kick_voltage`` the
    kick amplitude, both stored as ``extras``. The raw vectors are kept under ``raw``.
    """
    from esbopfn.benchmarks.instances import write_blob
    from esbopfn.problems import get_problem

    data = json.loads(path.read_text())
    problem = map_problem(data.get("problem", path.stem))
    if problem is None:
        return False
    prior = get_problem(problem).prior
    phi_names, noise_names = prior.phi_names, prior.noise_names
    records = []
    for rec in data["instances"]:
        phi = list(rec["phi"])
        extras: dict[str, float] = {}
        if problem == "furuta" and len(phi) == len(phi_names) + 1:
            extras["dead_time"] = float(phi.pop())
        if rec.get("kick_voltage") is not None:
            extras["kick"] = float(np.ravel(rec["kick_voltage"])[0])
        phi_hat = list(rec.get("phi_hat") or [])[:len(phi_names)]
        noise = list(rec["noise_stds"])
        records.append({
            "index": int(rec["index"]), "attempt": int(rec.get("attempt", 0)),
            "attempts": int(rec.get("attempt", 0)) + 1,
            "phi": dict(zip(phi_names, map(float, phi))) if len(phi) == len(phi_names) else {},
            "phi_hat": dict(zip(phi_names, map(float, phi_hat))) if phi_hat else {},
            "noise_std": (dict(zip(noise_names, map(float, noise)))
                          if len(noise) == len(noise_names) else {}),
            "extras": extras, "seed": int(rec["index"]),
            "screen": {k: rec.get("screen", {}).get(k)
                       for k in ("pass_count", "theta_screened", "trials_used")},
            "rejected_trials": None,
            "raw": {"phi": rec["phi"], "noise_stds": rec["noise_stds"],
                    "phi_hat": rec.get("phi_hat"), "kick_voltage": rec.get("kick_voltage")},
        })
    key = {"problem": problem, "provenance": PROVENANCE}
    return store(db, "instances", key, lambda d: write_blob(
        d, problem, PROVENANCE, records, source=str(path), thesis_screen=data.get("screen")),
        overwrite)


def convert_run(record: dict[str, Any], problem: str, strategy: str, params: dict[str, Any],
                instance: int | None = None) -> Any:
    """One old optimizer run record (``OptimizerBenchmark`` checkpoint schema) -> RunRecord.
    ``instance`` overrides the record's ``phi_index``, which the optplay trees store per
    ``phi_NNNN`` directory (always 0 there)."""
    from esbopfn.problems.base import cached_spec
    from esbopfn.strategies import RunRecord, TrialRecord

    spec = cached_spec(problem)
    T = spec.T
    full_cost = {int(s["trial_index"]): float(s["full_cost"])
                 for s in record.get("optimizer_early_stops", []) if s.get("full_cost") is not None}
    trials = []
    best, best_i = math.inf, -1
    inc_index, inc_J = [], []
    for i, (tr, ok) in enumerate(zip(record["trials"], record["successful"])):
        t = int(tr["t"])
        failed = tr.get("failed") or []
        crashed = bool(failed[t - 1]) if len(failed) >= t else False
        J = tr.get("J") or []
        j_obs = math.inf if crashed else float(J[t - 1])
        complete = t == int(tr.get("T", T)) and not crashed
        j_true = j_obs if complete else (math.inf if crashed else full_cost.get(i, math.nan))
        trials.append(TrialRecord(
            theta=[float(v) for v in tr["theta"]], t_obs=t, J_obs=j_obs, crashed=crashed,
            success=bool(ok), complete=complete, J_true=j_true,
            source="init" if i < spec.n_init() else "strategy", propose_s=0.0, stop_rule_s=0.0,
            end_s=0.0))
        if ok and complete and j_obs < best:
            best, best_i = j_obs, i
        inc_index.append(best_i)
        inc_J.append(best)
    timing: dict[str, float] = {}
    for cat, sec in record.get("performance_times", {}).items():
        new = map_timing(strategy, cat)
        timing[new] = timing.get(new, 0.0) + float(sec)
    return RunRecord(problem=problem, strategy=strategy, params=params,
                     seed=int(record.get("seed", 0)),
                     instance_seed=int(record.get("phi_index", 0) if instance is None
                                       else instance),
                     T=T, n_init=spec.n_init(), n_max=spec.n_max(), t_budget=spec.t_budget(),
                     trials=trials, incumbent_index=inc_index, incumbent_J=inc_J, timing=timing,
                     wall_s=float(record.get("elapsed", 0.0)))


def import_optplay_runs(db: Db, path: Path, problem: str, instance: int, budget: str = "capped",
                        overwrite: bool = False) -> int:
    """One ``runs/<cell>.json`` -> one ``optimization`` (``optimization_timing`` for the
    ``@timing`` tree) unit per run record."""
    blob = json.loads(path.read_text())
    kind = "optimization_timing" if budget == "timing" else "optimization"
    n = 0
    for record in blob.get("runs", []):
        cell = str(record.get("optimizer", blob.get("optimizer", path.stem)))
        mapped = map_cell(cell, problem)
        if mapped is None:
            continue
        sid, params, label = mapped
        # The instance is the ``phi_NNNN`` directory: every run file of the thesis trees records
        # ``phi_index: 0`` (one instance per directory).
        rec = convert_run(record, problem, sid, params, instance)
        key = {"problem": problem, "instance": int(instance),
               "strategy": sid, "params": params, "seed": int(record.get("repetition", 0)),
               "budget": budget, "instances_seed": PROVENANCE, "provenance": PROVENANCE,
               "cell": cell, "label": label}
        n += store(db, kind, key, lambda d, r=rec: (d / "run.json").write_text(r.to_json()),
                   overwrite)
    return n


def import_optplay_tree(db: Db, tree: Path, overwrite: bool = False) -> int:
    """A raw optplay tree ``<problem>@g0.05@bcapped[@timing]/phi_NNNN/runs/*.json``."""
    parts = tree.name.split("@")
    problem = map_problem(parts[0])
    if problem is None:
        return 0
    budget = "timing" if "timing" in parts[1:] else "capped"
    n = 0
    for phi_dir in sorted(tree.glob("phi_*")):
        index = int(phi_dir.name.split("_")[1])
        for path in sorted((phi_dir / "runs").glob("*.json")):
            if path.stat().st_size:
                n += import_optplay_runs(db, path, problem, index, budget, overwrite)
    return n


def import_optplay_results(db: Db, results: Path, overwrite: bool = False) -> int:
    """Aggregated optplay results (``<p>@g0.05@bcapped_{curves,runs,crashes}.json``) ->
    ``thesis_series``."""
    n = 0
    for curves_path in sorted(results.glob("*@g0.05@bcapped_curves.json")):
        old_problem = curves_path.name.split("@")[0]
        problem = map_problem(old_problem)
        if problem is None:
            continue
        curves = json.loads(curves_path.read_text())
        ref = curves.get("reference", {})
        n += store_series(db, "optplay_reference", problem, "J_best_all_cells", "thesis",
                          curves_path, {"instance": [int(k) for k in ref],
                                        "J_best": list(ref.values())},
                          {"note": "minimum successful final cost over every old cell on disk"},
                          overwrite)
        for cell, block in curves.get("cells", {}).items():
            mapped = map_cell(cell, problem)
            if mapped is None:
                continue
            sid, params, label = mapped
            finals: dict[int, dict[str, Any]] = {}
            for axis in ("trials", "time", "capped"):
                for k, v in (block.get(axis, {}).get("final_by_instance") or {}).items():
                    finals.setdefault(int(k), {})[f"incumbent_{axis}"] = (
                        v.get("incumbent") if v else None)
            inst = sorted(finals)
            data = {"instance": inst, **{f"incumbent_{a}": [finals[i].get(f"incumbent_{a}")
                                                            for i in inst]
                                         for a in ("trials", "time", "capped")}}
            meta = {"cell": cell, "strategy": sid, "params": params,
                    "n_runs": block.get("n_runs"), "n_instances": block.get("n_instances")}
            n += store_series(db, "optplay_finals", problem, label, "thesis", curves_path, data,
                              meta, overwrite)
            for axis in ("trials", "time"):
                b = block.get(axis, {})
                n += store_series(db, f"optplay_log_regret_{axis}", problem, label, "thesis",
                                  curves_path,
                                  {"grid": curves["grids"][axis],
                                   **{c: b.get(c) for c in ("median_log_regret", "q25_log_regret",
                                                            "q75_log_regret", "solved_frac",
                                                            "n_contributing")}},
                                  {**meta, "note": "natural-log regret against J_best of every "
                                   "old cell, not the scaled thesis metric"}, overwrite)
            perf = curves.get("performance", {}).get(cell)
            if perf:
                cats: dict[str, float] = {}
                for cat, sec in perf.get("mean_categories", {}).items():
                    new = map_timing(sid, cat)
                    cats[new] = cats.get(new, 0.0) + float(sec)
                n += store_series(db, "optplay_campaign_timing", problem, label, "thesis",
                                  curves_path, {"category": list(cats), "seconds": list(cats.values())},
                                  {**meta, "mean_optimizer_time": perf.get("mean_optimizer_time"),
                                   "note": "contended campaign times, not the one-thread timing"
                                   " tree"}, overwrite)
        runs_path = results / curves_path.name.replace("_curves.json", "_runs.json")
        if runs_path.exists():
            runs = pd.DataFrame(json.loads(runs_path.read_text()))
            for cell, g in runs.groupby("cell"):
                mapped = map_cell(cell, problem)
                if mapped is None:
                    continue
                cols = g.drop(columns=["problem", "cell"]).sort_values("instance")
                n += store_series(db, "optplay_runs", problem, mapped[2], "thesis", runs_path,
                                  _columns(cols), {"cell": cell, "strategy": mapped[0],
                                                   "params": mapped[1]}, overwrite)
        crash_path = results / curves_path.name.replace("_curves.json", "_crashes.json")
        if crash_path.exists():
            crashes = json.loads(crash_path.read_text())
            for cell, block in crashes.get("cells", {}).items():
                mapped = map_cell(cell, problem)
                if mapped is None:
                    continue
                n += store_series(
                    db, "optplay_crashes", problem, mapped[2], "thesis", crash_path,
                    {"time": crashes["time_grid"], "mean_crash_count": block["mean_crash_count"],
                     "mean_crash_frac": block["mean_crash_frac"],
                     "n_contributing": block["n_contributing"]},
                    {"cell": cell, "n_runs": block.get("n_runs"),
                     "n_instances": block.get("n_instances"), "t_budget": crashes.get("t_budget")},
                    overwrite)
    return n


# ------------------------------------------------------------------------------ driver
def import_thesis_results(root: Path | str, db: Db | None = None, *, raw_dirs: list[Path] = (),
                          overwrite: bool = False, home: Path | str | None = None
                          ) -> dict[str, int]:
    """Import every thesis result file that exists below ``root`` (the ``bachelorarbeit``
    directory) plus raw optplay trees found under ``raw_dirs``. Returns units written per kind
    (``tasks`` counts frozen task tables)."""
    from esbopfn.benchmarks.import_timing import import_timing_bundle

    root = Path(root)
    db = db or Db()
    dyn = root / "impl" / "dynformer"
    counts: dict[str, int] = {}

    def add(kind: str, n: int) -> None:
        counts[kind] = counts.get(kind, 0) + int(n)

    for path in sorted((dyn / "optplay" / "instances").glob("*.json")):
        add("instances", import_instances_json(db, path, overwrite))
    for path in sorted((dyn / "predplay" / "tasks").glob("*.parquet")):
        add("tasks", import_task_table(path, home) is not None)
    for path in sorted((dyn / "experiments" / "ladder" / "results").glob("*.parquet")):
        add("prediction_ladder", import_ladder_parquet(db, path, overwrite))
    bundle = dyn / "results_timing_bundle"
    if bundle.is_dir():
        add("prediction_timing", import_timing_bundle(db, bundle, overwrite))
    results = dyn / "predplay" / "results"
    # fig1_cross.csv (2026-09-29, LC-PFN with z-scored inputs) holds the numbers sim_results.tex
    # quotes, so its variant is "thesis". The pre_lcz files and the cluster recomputation
    # fig1_cross_5prob_canonmed.csv predate the LC-PFN z-score retraining.
    for suffix, variant in (("", "thesis"), ("_pre_lcz", "pre_lcz")):
        p = results / f"fig1_cross{suffix}.csv"
        if p.exists():
            add("thesis_series", import_fig1_csv(db, p, "fig3_cross", variant, False, overwrite))
        p = results / f"fig1_per_problem{suffix}.csv"
        if p.exists():
            add("thesis_series", import_fig1_csv(db, p, "fig3_per_problem", variant, True,
                                                 overwrite))
        p = results / f"fig1_meta{suffix}.json"
        if p.exists():
            add("thesis_series", import_fig1_meta(db, p, variant, overwrite))
        for p in sorted(results.glob(f"fig2_*{suffix}.csv")):
            stem = p.stem.removeprefix("fig2_")
            if (suffix and stem.endswith(suffix)) or (not suffix and stem in PROBLEM_MAP):
                add("thesis_series", import_fig2_csv(db, p, variant, overwrite))
    canon = root / "writing" / "thesis" / "data" / "sources" / "fig1_cross_5prob_canonmed.csv"
    if canon.exists():
        add("thesis_series", import_fig1_csv(db, canon, "fig3_cross", "canonmed", False,
                                             overwrite))
    opt_results = dyn / "optplay" / "results"
    if opt_results.is_dir():
        add("thesis_series", import_optplay_results(db, opt_results, overwrite))
    from esbopfn.benchmarks.import_cluster import import_raw_dir

    for raw in raw_dirs:
        for tree in sorted(Path(raw).glob("*@g0.05@bcapped*")):
            if tree.is_dir():
                n = import_optplay_tree(db, tree, overwrite)
                add("optimization_timing" if tree.name.endswith("@timing") else "optimization", n)
        for preds in (Path(raw) / "preds", Path(raw) / "predplay" / "preds"):
            if preds.is_dir():
                add("prediction", import_predplay_preds(db, preds, home, overwrite))
        for kind, n in import_raw_dir(db, Path(raw), home, overwrite).items():
            add(kind, n)
    return counts
