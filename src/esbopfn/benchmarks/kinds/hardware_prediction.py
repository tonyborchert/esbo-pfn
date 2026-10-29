"""Kind ``hardware_prediction``: the hardware prediction benchmark of one predictor on one rig.

Key: ``{"rig", "bench" ("ladder" | "ablation"), "predictor", "model", "ladder", "cutoff_fracs",
"methods", "logs", "provenance"}``. ``logs`` selects the rig runs the queries come from
(``native`` or ``thesis``), ``ladder`` is ``["full"]`` for the ablation. Blob: ``scores.parquet``
(one row per query, context size and cutoff, columns ``prediction_bench.SCORE_COLUMNS``) and
``summary.json`` (medians per context size, see ``prediction_bench.summarize``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from esbopfn.benchmarks.kinds import BenchmarkKind

NAME = "hardware_prediction"
DEFAULT_METHODS = ("esrs", "bo", "esbo-gp", "decpfn", "decpfn-h", "xdecpfn-pi")


def unit_key(rig: str, bench: str, predictor: str, model: str, *,
             ladder: list[Any] | None = None, cutoff_fracs: list[float] | None = None,
             methods: list[str] | None = None, logs: str = "thesis",
             provenance: str = "native") -> dict[str, Any]:
    from esbopfn.hardware.prediction_bench import CUTOFF_FRACS, LADDER

    if ladder is None:
        ladder = ["full"] if bench == "ablation" else list(LADDER)
    return {"rig": rig, "bench": bench, "predictor": predictor, "model": model,
            "ladder": list(ladder), "cutoff_fracs": list(cutoff_fracs or CUTOFF_FRACS),
            "methods": list(methods or DEFAULT_METHODS), "logs": logs, "provenance": provenance}


def resolve_models(models: dict[str, str] | None, home: Any = None) -> dict[str, str]:
    """Each model id replaced by the first trained of the native id and its imported thesis ids
    (:func:`esbopfn.training.registry.first_trained`), so the unit key names the weights used."""
    from esbopfn.training.registry import first_trained

    return {name: first_trained(model, home) for name, model in (models or {}).items()}


def enumerate_keys(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Native units for ``spec`` (``configs/benchmarks/hardware.yaml`` layout)."""
    pred = dict(spec.get("prediction") or {})
    rigs = spec.get("rigs") or ["without_weight", "with_weight"]
    methods = (spec.get("optimization") or {}).get("methods") or list(DEFAULT_METHODS)
    logs = pred.get("logs", "thesis")
    keys = []
    pred["predictors"] = resolve_models(pred.get("predictors"))
    pred["ablation"] = resolve_models(pred.get("ablation"))
    for rig in rigs:
        for name, model in (pred.get("predictors") or {}).items():
            keys.append(unit_key(rig, "ladder", name, model, ladder=pred.get("ladder"),
                                 cutoff_fracs=pred.get("cutoff_fracs"), methods=methods,
                                 logs=logs))
        for name, model in (pred.get("ablation") or {}).items():
            keys.append(unit_key(rig, "ablation", name, model,
                                 cutoff_fracs=pred.get("cutoff_fracs"), methods=methods,
                                 logs=logs))
    return keys


def run(key: dict[str, Any], blob_dir: Path, predictor: Any = None) -> None:
    """Score ``key["model"]`` (or the given ``predictor``) on the rig logs in the database."""
    from esbopfn.hardware.prediction_bench import run_bench
    from esbopfn.hardware.store import load_runs
    from esbopfn.predictors.base import get_predictor
    from esbopfn.problems import get_problem

    if key.get("provenance") == "thesis":
        raise RuntimeError("thesis hardware_prediction units are imported, not computed")
    spec = get_problem("furuta").spec
    runs = load_runs(key["rig"], provenance=key["logs"], methods=key["methods"])
    if not runs:
        raise RuntimeError(f"no {key['logs']} rig runs of {key['rig']!r} in the database "
                           "(import them with `pfn hw import-logs` or run `pfn hw run`)")
    predictor = predictor if predictor is not None else get_predictor(key["model"], spec)
    ladder = [n for n in key["ladder"] if n != "full"]
    run_bench(runs, predictor, spec, rig=key["rig"], predictor_name=key["predictor"],
              model=key["model"], bench=key["bench"], ladder=ladder or [0],
              cutoff_fracs=key["cutoff_fracs"], blob_dir=blob_dir)


KIND = BenchmarkKind(name=NAME, enumerate=enumerate_keys, run=run)
