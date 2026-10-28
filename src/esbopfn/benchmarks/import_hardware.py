"""Import of the thesis rig logs (old ``hwplay`` run trees) and of the old hardware prediction
benchmark results into the benchmark database.

``import_rig_logs(root, rig_name)`` converts a tree ``<root>/lqr_hardware/{manifest.json,
<cell>/seed<k>/{run.json, trials.jsonl}}`` into ``hardware_optimization`` units with
``provenance: "thesis"``. Cells map to methods by :data:`CELL_MAP` (``qc_pi_0.99_n10k`` and any
other unmapped cell are skipped). Conversion of one old trial line:

- ``t`` -> ``t_obs``, ``outcome`` completed / failed / early_stopped / budget_cut -> complete /
  crashed / early_stopped / budget_cut, ``crash_step = t`` for a failed trial,
- ``Y[:, 0:2]`` -> ``x`` (arm, pendulum), ``Y[:, 2:4]`` -> ``xdot``, ``Y[:, 4]`` (the commanded
  arm target, zero in every thesis trial) is dropped, the voltages were not logged (``u = None``),
- ``J`` is kept, except that the last entry of a failed trial carried the old crash tail
  penalty: it is replaced by the plain cumulative stage cost and the penalized value kept in
  ``extra.crash_tail_J``,
- ``successful`` (equal to ``held`` in every thesis trial) -> ``success``,
- timings: ``next_theta_seconds`` -> ``propose_s``, ``reset_seconds`` -> ``reset_s``,
  ``control_seconds`` -> ``control_s``, ``decision_seconds`` -> ``stop_rule_s``,
  ``observe_seconds`` -> ``end_s``, ``prepare_seconds`` -> ``prepare_s`` and ``total_s`` = the old
  operator wall-clock ``prepare + reset_pre + trial + next_theta + observe - overlap``,
- ``recovery`` -> ``reset`` (verbatim, plus ``phase_s`` from ``reset_phase_seconds`` and
  ``step_stats`` from ``reset_step_stats``), ``kick_sign`` from ``recovery.kick_sign``,
- the remaining old fields (``final_cost``, ``held``, ``n_decisions``, ...) under ``extra.thesis``.

``source`` is ``init`` for the first ``N_init = 8`` trials and ``strategy`` after (the old logs
do not say whether a later trial was a uniform fallback draw).

``import_hwpred_results(dir)`` converts ``hwpred_{ladder,ablation}_{reset_fixed,different_plant}``
(parquet + json) and ``hwpred_timing_*.json`` into ``hardware_prediction`` units with
``provenance: "thesis"``.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.benchmarks.db import DONE, Db
from esbopfn.config import pfn_home

log = logging.getLogger(__name__)

KIND_OPT = "hardware_optimization"
KIND_PRED = "hardware_prediction"
N_INIT = 8
T = 600

#: old hwplay cell -> method id
CELL_MAP: dict[str, str] = {
    "random_stenger": "esrs",
    "plain_gp": "bo",
    "stenger": "esbo-gp",
    "mtr0.99_k06binned": "decpfn",
    "htmr0.99_k06binned": "decpfn-h",
    "qc_pi_0.99": "xdecpfn-pi",
}
DROPPED_CELLS = ("qc_pi_0.99_n10k",)
METHOD_CELL = {m: c for c, m in CELL_MAP.items()}

#: tree directory name -> rig
TREE_RIG: dict[str, str] = {
    "runs_holdcost5_newbox_reset_fixed": "without_weight",
    "runs_holdcost5_newbox_reset_fixed_different_plant": "with_weight",
}
#: hwpred result tag -> rig
PRED_TAG_RIG: dict[str, str] = {"reset_fixed": "without_weight",
                                "different_plant": "with_weight"}

_DECPFN_CKPT = "rigreset_deadtime-strat_kick0.6_binned.pt"
_XDEC_CKPT = "c1r2_furuta_lnJ/f_s20260931_seg6_final.pt"

#: the parameters the old cells ran with (hwplay/field.py and the manifest), verbatim
THESIS_CELL_PARAMS: dict[str, dict[str, Any]] = {
    "random_stenger": {"cell": "random_stenger",
                       "optimizer": "RandomUniformStengerStopOptimizer",
                       "gp_cold_refit_growth": 0.05},
    "plain_gp": {"cell": "plain_gp", "optimizer": "PlainGPOptimizer",
                 "gp_cold_refit_growth": 0.05},
    "stenger": {"cell": "stenger", "optimizer": "StengerBOOptimizer",
                "gp_cold_refit_growth": 0.05},
    "mtr0.99_k06binned": {"cell": "mtr0.99_k06binned",
                          "optimizer": "MatchedTruncatedLivePredictionOptimizer",
                          "matched_threshold": 0.99, "fail_threshold": 0.9,
                          "fail_threshold_min_t": 0, "forecast": "every step (native resolution)",
                          "checkpoint": _DECPFN_CKPT, "gp_cold_refit_growth": 0.05},
    "htmr0.99_k06binned": {"cell": "htmr0.99_k06binned",
                           "optimizer": "HybridMatchedTruncatedLivePredictionOptimizer",
                           "matched_threshold": 0.99, "fail_threshold": 0.9,
                           "fail_threshold_min_t": 0, "monte_carlo": "gp",
                           "checkpoint": _DECPFN_CKPT, "gp_cold_refit_growth": 0.05},
    "qc_pi_0.99": {"cell": "qc_pi_0.99", "optimizer": "PureModelLoopOptimizer", "acq": "PI",
                   "threshold": 0.99, "checkpoint": _XDEC_CKPT},
}

#: old hwpred model id -> (predictor, checkpoint label)
PRED_MODEL_MAP: dict[str, tuple[str, str]] = {
    "stenger": ("segment-gp", "segment-gp (gp_cold_refit_growth 0.0)"),
    "decpfn": ("decpfn", "rigreset_deadtime-strat_kick0.6_binned"),
    "c1r2": ("xdecpfn", _XDEC_CKPT),
    "abl_dt_states": ("decpfn", "rigreset_deadtime-unif_kick0.6_binned"),
    "abl_dt_nostates": ("decpfn-nostates", "rigreset_deadtime-unif_kick0.6_nostates"),
    "abl_nd_states": ("decpfn-nodeadtime", "rigreset_nodeadtime_kick0.6_binned"),
    "abl_nd_nostates": ("decpfn-nodeadtime-nostates", "rigreset_nodeadtime_kick0.6_nostates"),
}

THESIS_METHODS = ("esrs", "bo", "esbo-gp", "decpfn", "decpfn-h", "xdecpfn-pi")


def _db(home: Path | str | None) -> Db:
    root = Path(home) if home is not None else pfn_home()
    return Db(root / "benchmarks" / "db.sqlite")


def thesis_key(rig: str, method: str, seed: int, n_max: int = 180,
               t_budget: int = 36000) -> dict[str, Any]:
    """Key of an imported thesis rig run."""
    from esbopfn.hardware.runner import unit_key

    return unit_key(rig, method, seed, THESIS_CELL_PARAMS[METHOD_CELL[method]], n_max, t_budget,
                    provenance="thesis")


# --------------------------------------------------------------------------- rig logs
def _stage_cost(pend: float) -> float:
    from esbopfn.hardware.system import stage_cost

    return float(stage_cost(pend))


def convert_trial(old: dict[str, Any], T: int = T) -> dict[str, Any]:
    """One old ``trials.jsonl`` line as a native trial dict (without incumbent fields)."""
    from esbopfn.hardware import records

    t = int(old["t"])
    Y = np.asarray(old["Y"], dtype=np.float64).reshape(t, -1)
    J = [float(v) for v in old["J"]]
    outcome = {"completed": records.COMPLETE, "failed": records.CRASHED,
               "early_stopped": records.EARLY_STOPPED,
               "budget_cut": records.BUDGET_CUT}[old["outcome"]]
    extra: dict[str, Any] = {}
    crash_step = None
    if outcome == records.CRASHED:
        crash_step = t
        prev = J[-2] if t >= 2 else 0.0
        extra["crash_tail_J"] = J[-1]
        J[-1] = float(np.float32(prev + _stage_cost(Y[-1, 1])))
    total = (old["prepare_seconds"] + old.get("reset_pre_seconds", 0.0) + old["trial_seconds"]
             + old["next_theta_seconds"] + old["observe_seconds"]
             - old.get("overlap_seconds", 0.0))
    timings = {"propose_s": old["next_theta_seconds"], "reset_s": old.get("reset_seconds"),
               "control_s": old.get("control_seconds"), "stop_rule_s": old["decision_seconds"],
               "end_s": old["observe_seconds"], "prepare_s": old["prepare_seconds"],
               "total_s": total}
    recovery = dict(old.get("recovery") or {})
    reset = {**recovery, "phase_s": dict(old.get("reset_phase_seconds") or {}),
             "step_stats": dict(old.get("reset_step_stats") or {})}
    thesis_fields = {k: old[k] for k in ("outcome", "failed", "successful", "held", "final_cost",
                                        "mean_alpha_final_deg", "max_arm_deg", "n_decisions",
                                        "decision_max_ms", "trial_seconds", "reset_pre_seconds",
                                        "overlap_seconds", "refit_overlapped",
                                        "refit_wait_seconds") if k in old}
    extra["thesis"] = thesis_fields
    index = int(old["index"])
    return {
        "index": index, "theta": [float(v) for v in old["theta"]],
        "source": "init" if index < N_INIT else "strategy", "T": int(old.get("T", T)),
        "t_obs": t, "outcome": outcome, "crash_step": crash_step,
        "success": bool(old["successful"]) and outcome == records.COMPLETE,
        "J_obs": math.inf if outcome == records.CRASHED else J[-1], "J": J,
        "x": Y[:, 0:2].tolist(), "xdot": Y[:, 2:4].tolist(), "u": None,
        "kick_sign": float(recovery.get("kick_sign", 0.0)),
        "incumbent_index": -1, "incumbent_J": math.inf, "timings": timings, "reset": reset,
        "max_arm_deg": float(old.get("max_arm_deg", math.nan)),
        "mean_pend_final_deg": float(old.get("mean_alpha_final_deg", math.nan)),
        "started_at": float(old.get("started_at", math.nan)), "extra": extra}


def convert_run(lines: Sequence[dict[str, Any]]) -> list[Any]:
    """Old trial lines -> :class:`HwTrial` records with the incumbent after each trial."""
    from esbopfn.hardware.records import HwTrial

    out = []
    best, best_i = math.inf, -1
    for old in sorted(lines, key=lambda d: int(d["index"])):
        rec = HwTrial.from_dict(convert_trial(old))
        if rec.success and rec.J_obs < best:
            best, best_i = rec.J_obs, rec.index
        rec.incumbent_index, rec.incumbent_J = best_i, best
        out.append(rec)
    return out


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def import_rig_logs(root: Path | str, rig_name: str | None = None,
                    home: Path | str | None = None, *, cells: Sequence[str] | None = None,
                    seeds: Sequence[int] | None = None, overwrite: bool = False
                    ) -> list[dict[str, Any]]:
    """Import one old hwplay tree as ``hardware_optimization`` units of ``rig_name``.

    ``root`` is the tree (``runs_holdcost5_...``) or its ``lqr_hardware`` directory.
    ``rig_name`` defaults to :data:`TREE_RIG` of the tree's name. ``cells``/``seeds`` restrict the
    import. Returns the keys of the imported (or already present) units.
    """
    from esbopfn.hardware import records
    from esbopfn.hardware.rigs import get_rig

    root = Path(root)
    tree = root / "lqr_hardware" if (root / "lqr_hardware").is_dir() else root
    if rig_name is None:
        name = tree.parent.name if tree.name == "lqr_hardware" else tree.name
        if name not in TREE_RIG:
            raise ValueError(f"cannot tell the rig of {root}, pass rig_name")
        rig_name = TREE_RIG[name]
    get_rig(rig_name)
    manifest_path = tree / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    n_max = int(manifest.get("trials_budget", 180))
    t_budget = int(manifest.get("t_budget", 36000))
    db = _db(home)
    keys = []
    for cell_dir in sorted(p for p in tree.iterdir() if p.is_dir()):
        cell = cell_dir.name
        if cell not in CELL_MAP:
            log.info("skipping cell %s (%s)", cell,
                     "not in the thesis" if cell in DROPPED_CELLS else "unmapped")
            continue
        if cells is not None and cell not in cells:
            continue
        method = CELL_MAP[cell]
        for seed_dir in sorted(cell_dir.glob("seed*"), key=lambda p: int(p.name[4:])):
            seed = int(seed_dir.name[4:])
            if seeds is not None and seed not in seeds:
                continue
            trials_path = seed_dir / "trials.jsonl"
            if not trials_path.exists():
                continue
            key = thesis_key(rig_name, method, seed, n_max, t_budget)
            keys.append(key)
            if db.has(KIND_OPT, key) and not overwrite:
                continue
            if overwrite:
                shutil.rmtree(db.blob_dir(KIND_OPT, key), ignore_errors=True)
                db.mark(KIND_OPT, key, "pending")
            old_run = json.loads((seed_dir / "run.json").read_text()) \
                if (seed_dir / "run.json").exists() else {}

            def write(key: dict[str, Any], blob: Path, trials_path: Path = trials_path,
                      old_run: dict[str, Any] = old_run, cell: str = cell) -> None:
                recs = convert_run(_read_lines(trials_path))
                records.write_trials(blob / records.TRIALS_FILE, recs)
                inc = recs[-1].incumbent_index if recs else -1
                records.write_run(blob, {
                    "kind": KIND_OPT, "key": key, "status": DONE, "provenance": "thesis",
                    "stop_reason": old_run.get("stop_reason", ""), "n_trials": len(recs),
                    "steps_used": int(sum(r.t_obs for r in recs)), "incumbent_index": inc,
                    "incumbent_J": recs[-1].incumbent_J if recs else math.inf,
                    "T": T, "n_init": N_INIT,
                    "source": {"tree": str(tree), "cell": cell, "run_json": old_run,
                               "manifest": manifest},
                    "record_version": records.RECORD_VERSION})

            db.run(KIND_OPT, key, write)
            log.info("imported %s %s seed %d", rig_name, method, seed)
    return keys


# --------------------------------------------------------------------------- hwpred results
def _pred_rows(df: Any, rig: str, predictor: str, ablation: bool) -> Any:
    import pandas as pd

    from esbopfn.hardware.prediction_bench import FULL, SCORE_COLUMNS

    run_id = df["run_id"].astype(str)
    cell = run_id.str.split("/").str[0]
    out = pd.DataFrame({
        "rig": rig, "method": cell.map(CELL_MAP),
        "seed": run_id.str.split("/").str[1].str.replace("seed", "").astype(int),
        "query_index": df["query_index"].astype(int),
        "n_ctx": FULL if ablation else df["n_ctx"].astype(int),
        "n_ctx_used": df["n_ctx_used"].astype(int),
        "cutoff_frac": df["cutoff_frac"].astype(float),
        "cutoff_step": df["cutoff_step"].astype(int), "step": df["cutoff_step"].astype(int),
        "predictor": predictor, "ll": df["ll"].astype(float),
        "p_crash": df["p_fail"].astype(float), "crashed": df["crashed"].astype(bool),
        "target_J": np.where(df["crashed"], np.inf, df["target_J"].astype(float)),
        "query_t": df["query_t"].astype(int), "infer_s": np.nan})
    dropped = out["method"].isna()
    if dropped.any():
        log.info("dropping %d rows of cells outside the thesis", int(dropped.sum()))
    return out[~dropped][list(SCORE_COLUMNS)].reset_index(drop=True)


def import_hwpred_results(results_dir: Path | str, home: Path | str | None = None, *,
                          overwrite: bool = False) -> list[dict[str, Any]]:
    """Import the old hardware prediction ladder, ablation and timing results."""
    import pandas as pd

    from esbopfn.benchmarks.kinds.hardware_prediction import unit_key
    from esbopfn.hardware.prediction_bench import summarize, write_blob

    results_dir = Path(results_dir)
    db = _db(home)
    keys = []
    for tag, rig in PRED_TAG_RIG.items():
        timing_path = results_dir / f"hwpred_timing_{tag}.json"
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        for bench in ("ladder", "ablation"):
            meta_path = results_dir / f"hwpred_{bench}_{tag}.json"
            pq = results_dir / f"hwpred_{bench}_{tag}.parquet"
            if not (meta_path.exists() and pq.exists()):
                continue
            meta = json.loads(meta_path.read_text())
            df = pd.read_parquet(pq)
            ablation = bench == "ablation"
            ladder = ["full"] if ablation else sorted(int(n) for n in meta["ladder"])
            eligible = ({"full": int(meta["n_queries"])} if ablation else
                        {str(k): int(v) for k, v in meta.get("eligible_per_rung", {}).items()})
            for old_model in meta["models"]:
                predictor, model = PRED_MODEL_MAP[old_model]
                key = unit_key(rig, bench, predictor, model, ladder=ladder,
                               cutoff_fracs=[float(f) for f in meta["cutoff_fracs"]],
                               methods=list(THESIS_METHODS), logs="thesis",
                               provenance="thesis")
                keys.append(key)
                if db.has(KIND_PRED, key) and not overwrite:
                    continue
                if overwrite:
                    shutil.rmtree(db.blob_dir(KIND_PRED, key), ignore_errors=True)
                    db.mark(KIND_PRED, key, "pending")
                rows = _pred_rows(df[df["model"] == old_model], rig, predictor, ablation)
                summary = {"rig": rig, "bench": bench, "predictor": predictor, "model": model,
                           "ladder": ladder, "cutoff_fracs": key["cutoff_fracs"],
                           "provenance": "thesis", "source": {"file": str(pq), "meta": meta},
                           "failures": {}, **summarize(rows)}
                # eligible queries per context size as the old run counted them (on the rig
                # without weight they include the 14 queries of the dropped qc_pi_0.99_n10k cell)
                summary["n_queries_thesis"] = eligible
                secs = (timing.get("seconds_per_prediction") or {}).get(old_model)
                if secs:
                    summary["seconds_per_prediction"] = {str(k): float(v)
                                                         for k, v in secs.items()}
                    summary["timing_source"] = {k: v for k, v in timing.items()
                                                if k != "seconds_per_prediction"}

                def write(key: dict[str, Any], blob: Path, rows: Any = rows,
                          summary: dict[str, Any] = summary) -> None:
                    write_blob(blob, rows, summary)

                db.run(KIND_PRED, key, write)
    return keys


def import_all(impl_root: Path | str, home: Path | str | None = None) -> dict[str, int]:
    """Import both thesis rig trees and the hwpred results from an old ``impl`` checkout."""
    impl_root = Path(impl_root)
    counts: dict[str, int] = {}
    for tree, rig in TREE_RIG.items():
        path = impl_root / "prediction" / "hwplay" / tree
        if path.exists():
            counts[rig] = len(import_rig_logs(path, rig, home))
    res = impl_root / "dynformer" / "experiments" / "hwpred" / "results"
    if res.exists():
        counts["hardware_prediction"] = len(import_hwpred_results(res, home))
    return counts
