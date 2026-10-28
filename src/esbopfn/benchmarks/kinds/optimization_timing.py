"""Kind ``optimization_timing``: optimization runs for the decision-time split (thesis
fig:sim:optplay-timing).

Same unit as ``optimization`` on instances 0-2 with ``budget: "timing"``. The unit runs in one
process with one thread (``torch.set_num_threads(1)``) and should be the only job on its core,
so the strategy's timing categories are uncontended CPU times.

Key ``{problem, instance, strategy, params, seed, budget: "timing", instances_seed}``.
Blob ``run.json`` (as ``optimization``) and ``environment.json`` ``{host, cpu, torch_threads,
torch, device}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from esbopfn.benchmarks.kinds import BenchmarkKind, Key
from esbopfn.benchmarks.kinds.optimization import optimization_keys, run_unit

TIMING_INSTANCES = 3


def enumerate_units(cfg: dict[str, Any]) -> list[Key]:
    return optimization_keys(cfg, "optimization_timing", "timing", TIMING_INSTANCES)


def run(key: Key, blob_dir: Path) -> None:
    import torch

    from esbopfn.benchmarks.kinds.prediction import environment

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run_unit(key, blob_dir)
        (Path(blob_dir) / "environment.json").write_text(json.dumps(environment(), indent=1))
    finally:
        torch.set_num_threads(previous)


KIND = BenchmarkKind(name="optimization_timing", enumerate=enumerate_units, run=run)
