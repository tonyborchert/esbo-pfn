"""The outer optimization loop of Algorithm alg:ct:esbo (thesis background) for every strategy.

- Initial design: ``N_init = max(d, 2)`` iid uniform trials, run without early stopping.
- Then, while ``n < N_max`` and ``sum_i T_i <= T_budget - T_max`` (a full trial must still fit),
  the strategy proposes ``theta_{n+1}`` and the trial runs. Until a successful complete trial
  exists (no incumbent, the thesis assumes one exists after the initial design) the loop keeps
  drawing uniform controllers without early stopping.
- A trial is simulated in full once with the per-trial generator ``substream(seed, n)`` (which
  also draws the instance's per-trial randomness such as the Furuta kick sign). The stop rules are
  then replayed over the logged steps: at step ``t < T`` a crash ends the trial, otherwise the
  ``exceed`` rule ``J_t >= J^-_n`` (for every strategy with ``uses_exceed``) and then the
  strategy's own rule (at its ``decision_steps``, on the trial cut at ``t``) may stop it. The
  replay is equivalent to stepping the plant, because a stop decision at ``t`` only reads the
  prefix up to ``t``.
- ``T_i`` is the observed length (the crash step for crashed trials) and counts toward the budget.

Seeds: initial (and fallback uniform) design from ``substream(seed, "init")``, trial ``n`` from
``substream(seed, n)``. Strategies get their own generator, conventionally
:func:`strategy_rng` ``(seed)``. Equal seeds give equal initial designs across strategies.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from esbopfn.core import Instance, Trial
from esbopfn.strategies.base import Strategy, incumbent_index, sample_uniform
from esbopfn.util.seeds import substream

log = logging.getLogger(__name__)

RECORD_VERSION = 1


def strategy_rng(seed: int) -> np.random.Generator:
    """The generator a strategy of run ``seed`` should use."""
    return substream(seed, "strategy")


@dataclass
class TrialRecord:
    """One trial of a run. ``J_obs`` is ``inf`` for a crashed trial, ``J_true`` the total cost of
    the uncut trajectory (``inf`` if it crashes at any step)."""

    theta: list[float]
    t_obs: int
    J_obs: float
    crashed: bool
    success: bool
    complete: bool
    J_true: float
    source: str                 # "init" (uniform design) or "strategy"
    propose_s: float
    stop_rule_s: float
    end_s: float
    J: list[float] | None = None  # observed J_1 .. J_{t_obs}, float32 values, optional

    @property
    def early_stopped(self) -> bool:
        return not self.crashed and not self.complete


@dataclass
class RunRecord:
    """Everything a benchmark stores about one optimization run."""

    problem: str
    strategy: str
    params: dict[str, Any]
    seed: int
    instance_seed: int
    T: int
    n_init: int
    n_max: int
    t_budget: int
    trials: list[TrialRecord] = field(default_factory=list)
    incumbent_index: list[int] = field(default_factory=list)   # after each trial, -1 = none
    incumbent_J: list[float] = field(default_factory=list)     # after each trial, inf = none
    timing: dict[str, float] = field(default_factory=dict)     # strategy categories, seconds
    wall_s: float = 0.0
    version: int = RECORD_VERSION

    @property
    def steps_used(self) -> int:
        return int(sum(t.t_obs for t in self.trials))

    @property
    def best(self) -> tuple[int, float]:
        """Final incumbent ``(index, J^-)``."""
        if not self.incumbent_index:
            return -1, math.inf
        return self.incumbent_index[-1], self.incumbent_J[-1]

    def to_json(self) -> str:
        """Compact JSON. Infinite costs are written as ``Infinity`` (Python's json dialect)."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> RunRecord:
        data = json.loads(text)
        data["trials"] = [TrialRecord(**t) for t in data["trials"]]
        return cls(**data)


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = repr(v)
    return out


def _replay_stop(strategy: Strategy, traj: Any, history: list[Trial], armed: bool,
                 J_minus: float, T: int) -> tuple[int, float]:
    """Observed length of the trial and the seconds spent in the strategy's stop rule."""
    end = traj.crash_step if traj.crash_step is not None else T
    limit = end  # stops are possible at t < end (and t < T)
    exceed_t: int | None = None
    if armed and strategy.uses_exceed and math.isfinite(J_minus):
        J = np.asarray(traj.J[: min(end, T) - 1], dtype=np.float64)
        hit = np.flatnonzero(J >= J_minus)
        if hit.size:
            exceed_t = int(hit[0]) + 1
            limit = exceed_t
    seconds = 0.0
    if armed or strategy.records_unarmed:
        steps = strategy.decision_steps()
        for t in steps[(steps < limit) & (steps < T)]:
            t0 = time.perf_counter()
            stop = strategy.should_stop(traj.cut(int(t)), history)
            seconds += time.perf_counter() - t0
            if stop and armed:
                return int(t), seconds
    return (exceed_t if exceed_t is not None else end), seconds


def run_optimization(problem: Any, instance: Instance, strategy: Strategy, seed: int,
                     n_max: int | None = None, t_budget: int | None = None,
                     record_curves: bool = False) -> RunRecord:
    """Run ``strategy`` on ``instance`` of ``problem`` under the thesis budgets.

    ``problem`` needs ``.spec`` and ``.simulate(theta, instance, rng)``. ``n_max`` and
    ``t_budget`` default to the spec's thesis budgets. ``record_curves`` stores each trial's
    observed cumulative cost curve.
    """
    spec = problem.spec
    T = spec.T
    n_max = spec.n_max() if n_max is None else int(n_max)
    t_budget = spec.t_budget() if t_budget is None else int(t_budget)
    n_init = spec.n_init()
    init_rng = substream(seed, "init")
    rec = RunRecord(problem=spec.name, strategy=strategy.name,
                    params=_jsonable(strategy.strategy_spec.params), seed=int(seed),
                    instance_seed=int(instance.seed), T=T, n_init=n_init, n_max=n_max,
                    t_budget=t_budget)
    history: list[Trial] = []
    used = 0
    start = time.perf_counter()
    while len(history) < n_max:
        n = len(history)
        if n >= n_init and used > t_budget - T:
            break
        inc = incumbent_index(history)
        armed = n >= n_init and inc is not None
        J_minus = history[inc].J_obs if inc is not None else math.inf
        if armed:
            t0 = time.perf_counter()
            theta = np.asarray(strategy.propose(history), dtype=np.float64)
            propose_s, source = time.perf_counter() - t0, "strategy"
        else:
            theta, propose_s, source = sample_uniform(spec, init_rng), 0.0, "init"
        strategy.on_trial_start(history, armed)
        traj = problem.simulate(theta, instance, substream(seed, n))
        t_obs, stop_s = _replay_stop(strategy, traj, history, armed, J_minus, T)
        trial = traj.cut(t_obs)
        t0 = time.perf_counter()
        strategy.on_trial_end(trial, n)
        end_s = time.perf_counter() - t0
        history.append(trial)
        used += trial.t_obs
        rec.trials.append(TrialRecord(
            theta=[float(v) for v in theta], t_obs=int(trial.t_obs), J_obs=float(trial.J_obs),
            crashed=trial.crashed, success=trial.successful, complete=trial.complete,
            J_true=float(traj.J_total), source=source, propose_s=propose_s, stop_rule_s=stop_s,
            end_s=end_s,
            J=[float(v) for v in np.asarray(traj.J[:trial.t_obs], dtype=np.float32)]
            if record_curves else None))
        new_inc = incumbent_index(history)
        rec.incumbent_index.append(-1 if new_inc is None else new_inc)
        rec.incumbent_J.append(math.inf if new_inc is None else float(history[new_inc].J_obs))
        log.debug("%s trial %d: t_obs=%d J=%.4g crashed=%s", strategy.name, n, trial.t_obs,
                  trial.J_obs, trial.crashed)
    rec.wall_s = time.perf_counter() - start
    rec.timing = {k: float(v) for k, v in strategy.timing.items()}
    return rec
