"""Screened system instances of the simulation benchmarks (thesis setup, prediction tasks and
optimization metrics).

Instance ``i`` of a problem is drawn from the universal prior with
``substream(root_seed, "instances", problem, i, attempt)`` for ``attempt = 0, 1, ...`` and kept at
the first attempt that passes the degeneracy screen (100 controllers, 10 repeats, at least 25
passing). Instance ``i`` therefore depends only on ``(root_seed, problem, i)`` and the list is
append-only: a longer list starts with the shorter one. Optimization uses instances 0-19, the
prediction fields use all 50.

The whole list is stored once per problem as the blob ``instances.json`` of the ``instances``
benchmark kind (see :mod:`esbopfn.benchmarks.kinds.instances` for the schema).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.core import Instance, Trajectory
from esbopfn.problems import Problem, get_problem
from esbopfn.problems.prior import sample_instance, screen
from esbopfn.util.seeds import substream

log = logging.getLogger(__name__)

#: Root seed of every simulation benchmark (instances, fields, optimization runs, landscapes).
BENCH_SEED = 20261029
#: Instances stored per problem.
N_INSTANCES = 50
#: Instances the optimization benchmark runs on.
N_OPT_INSTANCES = 20
#: The degeneracy screen (method, prior section): controllers, repeats, minimum passing.
SCREEN = {"n_theta": 100, "repeats": 10, "min_pass": 25}
#: Attempts per instance before giving up (Furuta accepts about 1.5 % of the draws).
MAX_ATTEMPTS = 5000
#: File name of the blob.
BLOB_FILE = "instances.json"


def instance_to_dict(inst: Instance) -> dict[str, Any]:
    return {"phi": dict(inst.phi), "phi_hat": dict(inst.phi_hat),
            "noise_std": dict(inst.noise_std), "extras": dict(inst.extras),
            "seed": int(inst.seed)}


def instance_from_dict(data: dict[str, Any]) -> Instance:
    return Instance(phi={k: float(v) for k, v in data["phi"].items()},
                    phi_hat={k: float(v) for k, v in data.get("phi_hat", {}).items()},
                    noise_std={k: float(v) for k, v in data["noise_std"].items()},
                    extras={k: float(v) for k, v in data.get("extras", {}).items()},
                    seed=int(data.get("seed", 0)))


class _CountingProblem:
    """A :class:`Problem` stand-in that records the success flag of every simulation."""

    def __init__(self, problem: Problem) -> None:
        self._problem = problem
        self.spec = problem.spec
        self.prior = problem.prior
        self.successes: list[bool] = []

    def simulate(self, theta: np.ndarray, instance: Instance, rng: np.random.Generator,
                 t_max: int | None = None) -> Trajectory:
        traj = self._problem.simulate(theta, instance, rng, t_max)
        self.successes.append(bool(traj.success))
        return traj


def screen_stats(successes: list[bool], repeats: int) -> dict[str, int]:
    """Passing and screened controllers from the success flags of one screen, in call order.

    The screen drops a controller at its first unsuccessful repeat, so the sequence splits into
    runs that end either with a failure or after ``repeats`` successes.
    """
    passed = screened = run = 0
    for ok in successes:
        if ok:
            run += 1
            if run == repeats:
                passed += 1
                screened += 1
                run = 0
        else:
            screened += 1
            run = 0
    return {"pass_count": passed, "theta_screened": screened, "trials_used": len(successes)}


@dataclass
class ScreenedInstance:
    index: int
    attempt: int
    instance: Instance
    stats: dict[str, int]
    rejected_trials: int

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "attempt": self.attempt, "attempts": self.attempt + 1,
                **instance_to_dict(self.instance), "screen": dict(self.stats),
                "rejected_trials": self.rejected_trials}


def screened_instance(problem: str | Problem, index: int, root_seed: int = BENCH_SEED,
                      max_attempts: int = MAX_ATTEMPTS, **screen_kw: int) -> ScreenedInstance:
    """Instance ``index`` of ``problem``: the first attempt that passes the screen."""
    prob = get_problem(problem) if isinstance(problem, str) else problem
    cfg = {**SCREEN, **screen_kw}
    rejected = 0
    for attempt in range(max_attempts):
        rng = substream(root_seed, "instances", prob.name, index, attempt)
        inst = sample_instance(prob, rng)
        counting = _CountingProblem(prob)
        ok = screen(counting, inst, rng, n_theta=cfg["n_theta"], repeats=cfg["repeats"],
                    min_pass=cfg["min_pass"])
        if ok:
            return ScreenedInstance(index, attempt, inst,
                                    screen_stats(counting.successes, cfg["repeats"]), rejected)
        rejected += len(counting.successes)
    raise RuntimeError(f"{prob.name}: instance {index} not accepted in {max_attempts} attempts")


def build_instances(problem: str, count: int, root_seed: int = BENCH_SEED,
                    reuse: list[dict[str, Any]] | None = None, **screen_kw: int
                    ) -> list[dict[str, Any]]:
    """Instance records ``0 .. count-1``. ``reuse`` holds already screened records of the same
    root seed (the list is append-only, so their prefix is valid)."""
    records = list(reuse or [])[:count]
    prob = get_problem(problem)
    for index in range(len(records), count):
        t0 = time.perf_counter()
        rec = screened_instance(prob, index, root_seed, **screen_kw).to_dict()
        rec["seconds"] = time.perf_counter() - t0
        log.info("%s instance %d accepted at attempt %d (%.1f s)", problem, index,
                 rec["attempt"], rec["seconds"])
        records.append(rec)
    return records


def write_blob(blob_dir: Path, problem: str, root_seed: int | str, records: list[dict[str, Any]],
               **extra: Any) -> Path:
    path = Path(blob_dir) / BLOB_FILE
    payload = {"problem": problem, "root_seed": root_seed, "count": len(records),
               "screen": dict(SCREEN), **extra, "instances": records}
    path.write_text(json.dumps(payload, indent=1, default=float))
    return path


def read_blob(blob_dir: Path) -> dict[str, Any]:
    return json.loads((Path(blob_dir) / BLOB_FILE).read_text())


def instances_key(problem: str, count: int = N_INSTANCES,
                  root_seed: int | str = BENCH_SEED) -> dict[str, Any]:
    """Unit key of an instance list. ``root_seed="thesis"`` is the imported thesis list."""
    if root_seed == "thesis":
        return {"problem": problem, "provenance": "thesis"}
    return {"problem": problem, "seed": int(root_seed), "count": int(count)}


def load_instances(problem: str, root_seed: int | str = BENCH_SEED, db: Any = None
                   ) -> list[Instance]:
    """The stored instance list of ``problem`` (the longest finished unit of that seed)."""
    from esbopfn.benchmarks.db import Db

    db = db or Db()
    if root_seed == "thesis":
        units = db.query("instances", problem=problem, provenance="thesis")
    else:
        units = [u for u in db.query("instances", problem=problem, seed=int(root_seed))
                 if "provenance" not in u.key]
    if not units:
        raise LookupError(f"no instances of {problem} (seed {root_seed}) in the benchmark database."
                          f" Run `pfn bench run <group> --kind instances` first.")
    best = max(units, key=lambda u: int(u.key.get("count", 0)))
    return [instance_from_dict(r) for r in read_blob(best.blob_dir)["instances"]]
