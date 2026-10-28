"""Kind ``generation_speed``: sampling cost of the priors on one CPU core (thesis
tab:app:prior-cost), wrapping :func:`esbopfn.generators.speed.measure` (the C++ binaries'
``speed`` subcommand).

Key ``{problem, count, sets, budget_s, seed}``.
Blob ``speed.json``: the :class:`SpeedResult` fields ``{problem, traj_s_per_10k, traj_acceptance,
system_acceptance, ftpfn_s_per_10k, xdecpfn_s_per_10k, single_trajectories, xdecpfn_sets,
ftpfn_sets, system_candidates}`` plus ``{host, cpu}``.
"""

from __future__ import annotations

import json
import platform
import socket
from pathlib import Path
from typing import Any

from esbopfn.benchmarks.kinds import BenchmarkKind, Key


def enumerate_units(cfg: dict[str, Any]) -> list[Key]:
    from esbopfn.benchmarks.groups import problems, section

    sec = section(cfg, "generation_speed")
    return [{"problem": p, "count": int(sec.get("count", 2000)), "sets": int(sec.get("sets", 10)),
             "budget_s": float(sec.get("budget_s", 120.0)), "seed": int(sec.get("seed", 0))}
            for p in problems(cfg, sec)]


def run(key: Key, blob_dir: Path) -> None:
    from esbopfn.generators.speed import measure

    res = measure(key["problem"], count=int(key["count"]), sets=int(key["sets"]),
                  budget_s=float(key["budget_s"]), seed=int(key["seed"]))
    out = {**res.to_dict(), "host": socket.gethostname(),
           "cpu": platform.processor() or platform.machine()}
    (Path(blob_dir) / "speed.json").write_text(json.dumps(out, indent=1))


KIND = BenchmarkKind(name="generation_speed", enumerate=enumerate_units, run=run)
