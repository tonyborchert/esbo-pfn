"""Kind ``optimization``: one optimization run (thesis setup, closed-loop optimization benchmark).

A strategy runs once per screened instance under the thesis caps ``N_max = 3 * 15 * min(d, 4)``
trials and ``T_budget = 15 * T * min(d, 4)`` steps, whichever binds first (budget mode
``capped``). The run seed depends on (problem, instance, seed) only, so all strategies share the
initial design on an instance.

Key ``{problem, instance, strategy, params, seed, budget, instances_seed}`` (optional
``n_max``, ``t_budget`` override the caps, set from the same keys of the spec section, used
only for small test runs). ``params`` are the registry defaults of the strategy merged with the
group's overrides, plus ``checkpoint`` (sha256, 16 hex digits) when the strategy uses a
``model``. Imported thesis runs carry ``provenance: "thesis"`` and ``cell`` (the old cell name).
Blob ``run.json``: :meth:`RunRecord.to_json` (``problem, strategy, params, seed, instance_seed, T,
n_init, n_max, t_budget, trials[{theta, t_obs, J_obs, crashed, success, complete, J_true, source,
propose_s, stop_rule_s, end_s, J}], incumbent_index[], incumbent_J[], timing{gp_acq, segment_gp,
tabpfn_forecast, model_forecast, model_acq}, wall_s, version}``, curves off.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from esbopfn.benchmarks.instances import BENCH_SEED, N_OPT_INSTANCES, load_instances
from esbopfn.benchmarks.kinds import BenchmarkKind, Key
from esbopfn.util.seeds import derive_seed


def strategy_params(strategy: str, problem: str, overrides: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Registry defaults + overrides, model templates resolved, checkpoint hash added."""
    from esbopfn.benchmarks.kinds.prediction import checkpoint_hash, resolve_model
    from esbopfn.strategies import default_params

    params = default_params(strategy, problem)
    params.update(overrides or {})
    if isinstance(params.get("model"), str):
        params["model"] = resolve_model(params["model"], problem)
        params["checkpoint"] = checkpoint_hash(params["model"])
    elif strategy == "esbo-t-tabpfn":
        params["checkpoint"] = checkpoint_hash(f"tabpfn/{params.get('L', 50)}")
    return params


def strategy_label(strategy: str, overrides: dict[str, Any] | None = None) -> str:
    """``decpfn`` or ``decpfn@crash_threshold=0.9,tau=0.99`` for non-default overrides."""
    if not overrides:
        return strategy
    return strategy + "@" + ",".join(f"{k}={overrides[k]}" for k in sorted(overrides))


def _instance_list(sec: dict[str, Any], default: int) -> list[int]:
    spec = sec.get("instances", default)
    if isinstance(spec, int):
        return list(range(spec))
    if isinstance(spec, dict):
        return list(range(int(spec.get("start", 0)), int(spec["stop"])))
    return [int(i) for i in spec]


def optimization_keys(cfg: dict[str, Any], kind: str, budget: str, default_instances: int
                      ) -> list[Key]:
    from esbopfn.benchmarks.groups import instances_seed, problems, section

    sec = section(cfg, kind)
    caps = {k: int(sec[k]) for k in ("n_max", "t_budget") if sec.get(k) is not None}
    keys: list[Key] = []
    for p in problems(cfg, sec):
        for entry in sec.get("strategies", []):
            entry = {"id": entry} if isinstance(entry, str) else entry
            if entry.get("problems") and p not in entry["problems"]:
                continue
            params = strategy_params(entry["id"], p, entry.get("params"))
            for i in _instance_list(sec, default_instances):
                for s in sec.get("seeds", [0]):
                    keys.append({"problem": p, "instance": int(i), "strategy": entry["id"],
                                 "params": params, "seed": int(s), "budget": budget,
                                 "instances_seed": instances_seed(cfg), **caps})
    return keys


def run_seed(key: Key) -> int:
    return derive_seed(BENCH_SEED, "optimization", key["problem"], int(key["instance"]),
                       int(key["seed"]))


def run_unit(key: Key, blob_dir: Path) -> dict[str, Any]:
    from esbopfn.benchmarks.kinds.prediction import checkpoint_hash
    from esbopfn.problems import get_problem
    from esbopfn.strategies import make_strategy, run_optimization, strategy_rng
    from esbopfn.training.registry import register_predictor_families

    if key.get("provenance") == "thesis":
        raise RuntimeError("thesis runs are imported with `pfn import thesis-results`")
    register_predictor_families()
    problem = get_problem(key["problem"])
    instance = load_instances(key["problem"], key.get("instances_seed", BENCH_SEED))[
        int(key["instance"])]
    params = dict(key["params"])
    expected = params.pop("checkpoint", None)
    if expected is not None:
        model = params.get("model") or f"tabpfn/{params.get('L', 50)}"
        if checkpoint_hash(model) != expected:
            raise RuntimeError(f"{model}: checkpoint hash {checkpoint_hash(model)} differs from "
                               f"the unit key {expected}")
    seed = run_seed(key)
    strategy = make_strategy(key["strategy"], problem.spec, strategy_rng(seed), **params)
    record = run_optimization(problem, instance, strategy, seed, n_max=key.get("n_max"),
                              t_budget=key.get("t_budget"), record_curves=False)
    (Path(blob_dir) / "run.json").write_text(record.to_json())
    return json.loads(record.to_json())


def enumerate_units(cfg: dict[str, Any]) -> list[Key]:
    return optimization_keys(cfg, "optimization", "capped", N_OPT_INSTANCES)


def run(key: Key, blob_dir: Path) -> None:
    run_unit(key, blob_dir)


def read_run(blob_dir: Path):
    from esbopfn.strategies import RunRecord

    return RunRecord.from_json((Path(blob_dir) / "run.json").read_text())


KIND = BenchmarkKind(name="optimization", enumerate=enumerate_units, run=run)
