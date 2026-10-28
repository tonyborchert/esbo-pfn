"""Kind ``instances``: the screened instance list of one problem.

Key ``{problem, seed, count}`` (imported thesis lists: ``{problem, provenance: "thesis"}``).
Blob ``instances.json``: ``{problem, root_seed, count, screen{n_theta, repeats, min_pass},
instances: [{index, attempt, attempts, phi{}, phi_hat{}, noise_std{}, extras{}, seed,
screen{pass_count, theta_screened, trials_used}, rejected_trials, seconds}]}``.

Group config: ``problems`` and ``instances: {count, seed}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from esbopfn.benchmarks import instances as inst
from esbopfn.benchmarks.kinds import BenchmarkKind, Key


def enumerate_units(cfg: dict[str, Any]) -> list[Key]:
    from esbopfn.benchmarks.groups import problems, section

    sec = section(cfg, "instances")
    count = int(sec.get("count", inst.N_INSTANCES))
    seed = sec.get("seed", inst.BENCH_SEED)
    return [inst.instances_key(p, count, seed) for p in problems(cfg)]


def run(key: Key, blob_dir: Path) -> None:
    from esbopfn.benchmarks.db import Db

    if key.get("provenance") == "thesis":
        raise RuntimeError("thesis instance lists are imported with `pfn import thesis-results`")
    problem, seed, count = key["problem"], int(key["seed"]), int(key["count"])
    reuse: list[dict[str, Any]] = []
    for unit in Db().query("instances", problem=problem, seed=seed):
        if "provenance" not in unit.key and int(unit.key["count"]) < count:
            records = inst.read_blob(unit.blob_dir)["instances"]
            if len(records) > len(reuse):
                reuse = records
    records = inst.build_instances(problem, count, seed, reuse=reuse)
    inst.write_blob(blob_dir, problem, seed, records)


KIND = BenchmarkKind(name="instances", enumerate=enumerate_units, run=run)
