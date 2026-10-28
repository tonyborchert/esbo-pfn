"""Kind ``fields``: ESBO-Oracle runs whose trials become the prediction tasks (thesis setup,
prediction tasks).

Fields:

- ``oracle65``: 65 trials per run on the 50 screened instances with a per-problem number of seeds
  (boiler 4, threetank_pi 4, pt2_deadtime 7, cartpole 8, furuta 10).
- ``oracle181``: ``N_max + 1`` trials per run (181 on Cartpole and Furuta, 136 on PT2), 50 instances
  times 2 seeds, for the context ladder.

The run uses ``make_strategy("esbo-oracle")`` and :func:`run_optimization` with the budgets
overridden to ``n_max = n_trials`` and ``t_budget = n_trials * T``, so the run always reaches
``n_trials`` trials and the time budget never cuts it. Every trial is simulated in full by the loop,
and the full uncut trajectory of every trial is captured here (the oracle needs it anyway), so the
blob holds the hidden continuation the prediction tasks score against.

Key ``{problem, field, instance, seed, n_trials, instances_seed}``.
Blob:

- ``trials.npz``: ``theta (n, d) float64``, ``t_obs (n,) int32`` (the oracle's observed length),
  ``crash_step (n,) int32`` (1-based, 0 = no crash when run to the end), ``success (n,) bool``,
  ``J (n, T) float32`` (uncut cumulative cost, +inf from the crash step), ``x (n, T, C) float32``,
  ``xdot (n, T, C_rate) float32``.
- ``field.json``: ``{problem, field, instance, seed, run_seed, n_trials, T, d, channels,
  rate_channels, instance{phi, phi_hat, noise_std, extras, seed}}``.
- ``run.json``: the oracle's :class:`RunRecord` (no curves, incumbent trace included).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.benchmarks.instances import BENCH_SEED, N_INSTANCES, instance_to_dict, load_instances
from esbopfn.benchmarks.kinds import BenchmarkKind, Key
from esbopfn.core import Instance, ProblemSpec, Trajectory, Trial
from esbopfn.util.seeds import derive_seed

#: Seeds per problem of the 65-trial field (planning record of the thesis field).
ORACLE65_SEEDS = {"boiler": 4, "threetank_pi": 4, "pt2_deadtime": 7, "cartpole": 8, "furuta": 10}


@dataclass(frozen=True)
class FieldDef:
    name: str
    seeds: dict[str, int] | int
    instances: int = N_INSTANCES
    trials: int | None = None   # None = N_max + 1

    def n_trials(self, spec: ProblemSpec) -> int:
        return int(self.trials) if self.trials is not None else spec.n_max() + 1

    def n_seeds(self, problem: str) -> int:
        return self.seeds[problem] if isinstance(self.seeds, dict) else int(self.seeds)


FIELDS: dict[str, FieldDef] = {
    "oracle65": FieldDef("oracle65", ORACLE65_SEEDS, trials=65),
    "oracle181": FieldDef("oracle181", 2, trials=None),
}


def field_key(problem: str, name: str, instance: int, seed: int,
              instances_seed: int | str = BENCH_SEED, n_trials: int | None = None) -> Key:
    from esbopfn.problems.base import cached_spec

    fdef = FIELDS[name]
    n = n_trials if n_trials is not None else fdef.n_trials(cached_spec(problem))
    return {"problem": problem, "field": name, "instance": int(instance), "seed": int(seed),
            "n_trials": int(n), "instances_seed": instances_seed}


def field_keys(problem: str, name: str, instances_seed: int | str = BENCH_SEED,
               n_instances: int | None = None, n_seeds: int | None = None) -> list[Key]:
    """Every run of field ``name`` on ``problem``, seed-major."""
    fdef = FIELDS[name]
    n_inst = fdef.instances if n_instances is None else int(n_instances)
    n_seed = fdef.n_seeds(problem) if n_seeds is None else int(n_seeds)
    return [field_key(problem, name, i, s, instances_seed)
            for s in range(n_seed) for i in range(n_inst)]


def field_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``fields`` section as a list of ``{name, problems?, seeds?, instances?}``.

    Accepts a list of such entries, a mapping with ``name`` or ``names``, or nothing
    (``oracle65``)."""
    raw = cfg.get("fields")
    if isinstance(raw, list):
        return [{"name": e} if isinstance(e, str) else dict(e) for e in raw]
    raw = dict(raw or {})
    names = raw.pop("names", None) or [raw.pop("name", "oracle65")]
    return [{**raw, "name": n} for n in names]


def enumerate_units(cfg: dict[str, Any]) -> list[Key]:
    from esbopfn.benchmarks.groups import instances_seed, problems

    keys: list[Key] = []
    for entry in field_entries(cfg):
        seeds = entry.get("seeds")
        for p in problems(cfg, entry):
            n_seeds = seeds.get(p) if isinstance(seeds, dict) else seeds
            keys += field_keys(p, entry["name"], instances_seed(cfg), entry.get("instances"),
                               n_seeds)
    return keys


class _Recorder:
    """Problem proxy that keeps every simulated (uncut) trajectory in call order."""

    def __init__(self, problem: Any) -> None:
        self.problem = problem
        self.spec = problem.spec
        self.trajectories: list[Trajectory] = []

    def simulate(self, theta: np.ndarray, instance: Instance, rng: np.random.Generator,
                 t_max: int | None = None) -> Trajectory:
        traj = self.problem.simulate(theta, instance, rng, t_max)
        self.trajectories.append(traj)
        return traj


def run_seed(key: Key) -> int:
    return derive_seed(BENCH_SEED, "fields", key["field"], key["problem"], int(key["instance"]),
                       int(key["seed"]))


def simulate_field(problem_obj: Any, instance: Instance, n_trials: int, seed: int,
                   strategy_params: dict[str, Any] | None = None):
    """Run ESBO-Oracle for ``n_trials`` trials. Returns ``(RunRecord, trajectories)``."""
    from esbopfn.strategies import make_strategy, run_optimization, strategy_rng

    rec = _Recorder(problem_obj)
    strategy = make_strategy("esbo-oracle", problem_obj.spec, strategy_rng(seed),
                             **(strategy_params or {}))
    record = run_optimization(rec, instance, strategy, seed, n_max=n_trials,
                              t_budget=n_trials * problem_obj.spec.T, record_curves=False)
    if len(rec.trajectories) != len(record.trials):
        raise RuntimeError("recorded trajectories do not match the run's trials")
    return record, rec.trajectories


def write_field(blob_dir: Path, meta: dict[str, Any], spec: ProblemSpec,
                trajectories: list[Trajectory], t_obs: list[int]) -> None:
    n, T = len(trajectories), spec.T
    C = trajectories[0].x.shape[1] if trajectories and trajectories[0].x.ndim == 2 else 0
    xdot = [t.xdot for t in trajectories]
    Cr = xdot[0].shape[1] if xdot and xdot[0] is not None and xdot[0].ndim == 2 else 0
    np.savez_compressed(
        Path(blob_dir) / "trials.npz",
        theta=np.stack([np.asarray(t.theta, np.float64) for t in trajectories]),
        t_obs=np.asarray(t_obs, np.int32),
        crash_step=np.asarray([t.crash_step or 0 for t in trajectories], np.int32),
        success=np.asarray([bool(t.success) for t in trajectories]),
        J=np.stack([np.asarray(t.J, np.float32) for t in trajectories]),
        x=np.stack([np.asarray(t.x, np.float32) for t in trajectories]).reshape(n, T, C),
        xdot=(np.stack([np.asarray(v, np.float32) for v in xdot]).reshape(n, T, Cr) if Cr
              else np.zeros((n, T, 0), np.float32)),
    )
    meta = {**meta, "n_trials": n, "T": T, "d": spec.d, "channels": list(spec.channels),
            "rate_channels": list(spec.rate_channels)}
    (Path(blob_dir) / "field.json").write_text(json.dumps(meta, indent=1))


def run(key: Key, blob_dir: Path) -> None:
    from esbopfn.problems import get_problem

    problem = get_problem(key["problem"])
    instances = load_instances(key["problem"], key.get("instances_seed", BENCH_SEED))
    instance = instances[int(key["instance"])]
    seed = run_seed(key)
    record, trajs = simulate_field(problem, instance, int(key["n_trials"]), seed)
    meta = {k: key[k] for k in ("problem", "field", "instance", "seed")}
    meta.update(run_seed=seed, instance_params=instance_to_dict(instance))
    write_field(blob_dir, meta, problem.spec, trajs, [t.t_obs for t in record.trials])
    (Path(blob_dir) / "run.json").write_text(record.to_json())


KIND = BenchmarkKind(name="fields", enumerate=enumerate_units, run=run)


# ------------------------------------------------------------------------------ loading
@dataclass
class FieldRun:
    """One stored field run with lazy access to its trials."""

    key: Key
    theta: np.ndarray
    t_obs: np.ndarray
    crash_step: np.ndarray
    success: np.ndarray
    J: np.ndarray
    x: np.ndarray
    xdot: np.ndarray
    _trajs: dict[int, Trajectory] = field(default_factory=dict, repr=False)

    @property
    def n_trials(self) -> int:
        return int(self.t_obs.shape[0])

    @property
    def run_id(self) -> str:
        return f"i{int(self.key['instance']):04d}_s{int(self.key['seed']):02d}"

    def trajectory(self, i: int) -> Trajectory:
        if i not in self._trajs:
            cs = int(self.crash_step[i])
            self._trajs[i] = Trajectory(
                theta=self.theta[i], J=self.J[i], x=self.x[i],
                xdot=self.xdot[i] if self.xdot.shape[-1] else None,
                crash_step=cs if cs > 0 else None, success=bool(self.success[i]))
        return self._trajs[i]

    def trial(self, i: int, t: int) -> Trial:
        """Trial ``i`` observed up to step ``t``."""
        return self.trajectory(i).cut(int(t))

    def context(self, n: int) -> list[Trial]:
        """The first ``n`` trials as the oracle observed them (cut at their ``t_obs``)."""
        return [self.trial(i, int(self.t_obs[i])) for i in range(int(n))]

    def target(self, i: int) -> tuple[bool, float]:
        """``(crashed, J_T)`` of trial ``i`` run to the end (``J_T`` NaN if it crashes)."""
        if int(self.crash_step[i]) > 0:
            return True, float("nan")
        return False, float(self.J[i, -1])


def read_field(blob_dir: Path, key: Key | None = None) -> FieldRun:
    blob_dir = Path(blob_dir)
    with np.load(blob_dir / "trials.npz") as z:
        arrays = {k: z[k] for k in z.files}
    if key is None:
        key = json.loads((blob_dir / "field.json").read_text())
    return FieldRun(key=dict(key), **arrays)


def field_index(problem: str, name: str, instances_seed: int | str = BENCH_SEED, db: Any = None
                ) -> dict[tuple[int, int], tuple[Key, Path]]:
    """``(instance, seed) -> (key, blob_dir)`` of the finished runs of a field."""
    from esbopfn.benchmarks.db import Db

    db = db or Db()
    out = {}
    for u in db.query("fields", problem=problem, field=name, instances_seed=instances_seed):
        out[(int(u.key["instance"]), int(u.key["seed"]))] = (u.key, Path(u.blob_dir))
    return out


def load_field_runs(problem: str, name: str, instances_seed: int | str = BENCH_SEED,
                    db: Any = None, keys: list[Key] | None = None) -> list[FieldRun] | None:
    """All runs of a field (seed-major) or None if any of them is not done."""
    from esbopfn.benchmarks.db import Db

    db = db or Db()
    keys = keys if keys is not None else field_keys(problem, name, instances_seed)
    runs = []
    for k in keys:
        if not db.has("fields", k):
            return None
        runs.append(read_field(db.blob_dir("fields", k), k))
    return runs
