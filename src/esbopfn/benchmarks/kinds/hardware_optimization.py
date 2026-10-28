"""Kind ``hardware_optimization``: one optimization run on the rig.

Key: ``{"rig", "method", "params", "seed", "n_max", "t_budget", "provenance"}`` where ``params``
are the strategy parameters (native runs) or the old cell's parameters verbatim (``provenance:
"thesis"``, imported by ``benchmarks.import_hardware``). Blob: ``run.json`` and ``trials.jsonl``
(``esbopfn.hardware.records``).

These units need the rig, so ``pfn bench run`` does not compute them: they are run with
``pfn hw run`` on the workstation (resumable, written in place) or imported from the thesis logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from esbopfn.benchmarks.kinds import BenchmarkKind

NAME = "hardware_optimization"


def enumerate_keys(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Keys of the rig field for ``spec`` (``configs/benchmarks/hardware.yaml`` layout:
    ``rigs``, ``optimization: {methods, seeds, provenance}``)."""
    from esbopfn.hardware.rigs import field_defaults
    from esbopfn.hardware.runner import method_params, unit_key

    opt = dict(spec.get("optimization") or {})
    fd = field_defaults()
    rigs = spec.get("rigs") or ["without_weight", "with_weight"]
    methods = opt.get("methods") or list(fd.methods)
    seeds = opt.get("seeds") or list(fd.seeds)
    provenance = opt.get("provenance", "native")
    keys = []
    for rig in rigs:
        for m in methods:
            for s in seeds:
                if provenance == "thesis":
                    from esbopfn.benchmarks.import_hardware import thesis_key

                    keys.append(thesis_key(rig, m, int(s)))
                else:
                    keys.append(unit_key(rig, m, int(s), method_params(m), fd.n_max,
                                         fd.t_budget))
    return keys


def run(key: dict[str, Any], blob_dir: Path) -> None:
    raise RuntimeError(
        f"hardware_optimization units run on the rig: `pfn hw run --rig {key.get('rig')} "
        f"--methods {key.get('method')} --seeds {key.get('seed')}` on the workstation, or import "
        "the thesis logs with `pfn hw import-logs`")


KIND = BenchmarkKind(name=NAME, enumerate=enumerate_keys, run=run)
