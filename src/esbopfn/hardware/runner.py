"""The rig field: every (method, seed) of a rig run on the rig under the thesis protocol.

Protocol (thesis setup, hardware experiment): seeds are shared across methods, so every method
gets the same initial design for a seed. ``N_max = 180`` trials and ``T_budget = 36000`` steps.
The budget, initial design, incumbent and ``exceed`` logic are those of
``esbopfn.strategies.loop``, re-implemented here around an online trial because a rig trial
cannot be simulated in full and replayed: the stop rule must act while the pendulum moves. The
shared pieces are imported, not copied (``incumbent_index``, ``sample_uniform``,
``strategy_rng``, the seed streams ``substream(seed, "init")`` and ``substream(seed, n)``).

Per trial: propose (after the previous trial's reset is no longer running, no overlap), reset
(alg:app:hw-reset, the kick sign from ``substream(seed, n)``, a connection reset when ``n`` is a
positive multiple of 10), then the controller runs with the stop rules applied after every logged
step: ``exceed`` (``J_t >= J^-``) for strategies that use it, then the strategy's own
``should_stop`` at its decision steps.

Resume: each finished trial is appended to ``trials.jsonl`` in the unit's blob directory. A
rerun rebuilds the strategy from the seed and replays the stored trials into it off the rig: it
proposes again (and checks that the stored controller comes out), streams the stored prefixes
through the stop rule at the steps the live run asked, and ends each trial, then continues on the
rig. On the simulated plant the continued run is identical to an uninterrupted one.
"""

from __future__ import annotations

import logging
import math
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.benchmarks.db import DONE, PENDING, RUNNING, Db, code_version
from esbopfn.core import Trial
from esbopfn.hardware import records
from esbopfn.hardware.client import RigClient, RigError
from esbopfn.hardware.records import HwTrial
from esbopfn.hardware.reset import config_dict
from esbopfn.hardware.rigs import field_defaults, get_rig
from esbopfn.hardware.system import RigSystem, RigTrialResult
from esbopfn.strategies.base import Strategy, default_params, incumbent_index, make_strategy, \
    sample_uniform
from esbopfn.strategies.loop import strategy_rng
from esbopfn.util.seeds import derive_seed, substream

log = logging.getLogger(__name__)

KIND = "hardware_optimization"
PROVENANCE_NATIVE = "native"


class ReplayDiverged(RuntimeError):
    """A rebuilt strategy proposes a controller other than the stored one."""


def pause_flag(db: Db) -> Path:
    """File whose existence makes a running field stop after the current trial."""
    return db.path.parent / "hardware.pause"


def unit_key(rig: str, method: str, seed: int, params: dict[str, Any] | None = None,
             n_max: int = 180, t_budget: int = 36000,
             provenance: str = PROVENANCE_NATIVE) -> dict[str, Any]:
    """Key of one rig run (kind ``hardware_optimization``)."""
    return {"rig": rig, "method": method, "params": dict(params or {}), "seed": int(seed),
            "n_max": int(n_max), "t_budget": int(t_budget), "provenance": provenance}


def method_params(method: str, overrides: dict[str, Any] | None = None,
                  problem: str = "furuta") -> dict[str, Any]:
    """Thesis defaults of ``method`` on ``problem`` with the rig field's overrides
    (``configs/rigs.yaml``), JSON values only."""
    params = default_params(method, problem)
    params.update(field_defaults().method_params.get(method, {}))
    params.update(overrides or {})
    return {k: v for k, v in params.items() if v is None or isinstance(v, (bool, int, float, str))}


# --------------------------------------------------------------------------- stop rules
class StopDecider:
    """The stop rules of one trial, identical for the live run and for a replay."""

    def __init__(self, strategy: Strategy, history: list[Trial], armed: bool,
                 J_minus: float, T: int) -> None:
        self.strategy = strategy
        self.history = history
        self.armed = armed
        self.J_minus = J_minus
        self.T = T
        self.use_exceed = armed and strategy.uses_exceed and math.isfinite(J_minus)
        asks = armed or strategy.records_unarmed
        self.steps = set(int(s) for s in strategy.decision_steps()) if asks else set()
        self.stopped_by: str | None = None

    def __call__(self, k: int, running: Trial) -> bool:
        if k >= self.T:
            return False
        if self.use_exceed and float(running.traj.J[k - 1]) >= self.J_minus:
            self.stopped_by = "exceed"
            return True
        if k in self.steps:
            stop = self.strategy.should_stop(running, self.history)
            if stop and self.armed:
                self.stopped_by = "strategy"
                return True
        return False


# --------------------------------------------------------------------------- one run
@dataclass
class CellResult:
    key: dict[str, Any]
    blob_dir: Path
    n_trials: int
    new_trials: int
    finished: bool
    stop_reason: str


@dataclass
class _State:
    strategy: Strategy
    history: list[Trial] = field(default_factory=list)
    init_rng: np.random.Generator | None = None
    used: int = 0


def _to_record(n: int, T: int, res: RigTrialResult, source: str, propose_s: float,
               started_at: float, stopped_by: str | None) -> HwTrial:
    """The record of a finished rig trial (incumbent and ``end_s`` are filled in later)."""
    if res.crash_step is not None:
        outcome = records.CRASHED
    elif res.stopped:
        outcome = records.EARLY_STOPPED
    else:
        outcome = records.COMPLETE
    timings = {"propose_s": propose_s, "reset_s": res.reset.seconds, "control_s": res.control_s,
               "stop_rule_s": res.stop_rule_s, "end_s": 0.0}
    return HwTrial(
        index=n, theta=[float(v) for v in res.theta], source=source, T=T,
        t_obs=int(res.t_obs), outcome=outcome, crash_step=res.crash_step,
        success=bool(res.success),
        J_obs=math.inf if res.crash_step is not None else float(res.J[-1]),
        J=[float(v) for v in res.J], x=res.x.astype(float).tolist(),
        xdot=res.xdot.astype(float).tolist(), u=[float(v) for v in res.u],
        kick_sign=float(res.reset.kick_sign), incumbent_index=-1, incumbent_J=math.inf,
        timings=timings, reset=dict(res.reset.diagnostics), max_arm_deg=res.max_arm_deg,
        mean_pend_final_deg=res.mean_pend_final_deg, started_at=started_at,
        extra={"stopped_by": stopped_by})


class CellRunner:
    """Runs (or finishes) one (rig, method, seed) unit on an open :class:`RigSystem`."""

    def __init__(self, system: RigSystem, db: Db, key: dict[str, Any],
                 strategy_overrides: dict[str, Any] | None = None) -> None:
        from esbopfn.problems import get_problem

        self.system = system
        self.db = db
        self.key = key
        self.problem = get_problem("furuta")
        self.spec = self.problem.spec
        self.seed = int(key["seed"])
        self.n_max = int(key["n_max"])
        self.t_budget = int(key["t_budget"])
        self.T = self.spec.T
        self.n_init = self.spec.n_init()
        self.overrides = dict(strategy_overrides or {})
        self.blob_dir = db.blob_dir(KIND, key)
        self.trials_path = self.blob_dir / records.TRIALS_FILE

    def _new_state(self) -> _State:
        params = {**self.key["params"], **self.overrides}
        strategy = make_strategy(self.key["method"], self.problem, strategy_rng(self.seed),
                                 **params)
        return _State(strategy=strategy, init_rng=substream(self.seed, "init"))

    def _next_theta(self, st: _State) -> tuple[np.ndarray, str, float, bool, float]:
        n = len(st.history)
        inc = incumbent_index(st.history)
        armed = n >= self.n_init and inc is not None
        J_minus = st.history[inc].J_obs if inc is not None else math.inf
        if armed:
            t0 = time.perf_counter()
            theta = np.asarray(st.strategy.propose(st.history), dtype=np.float64)
            return theta, "strategy", time.perf_counter() - t0, armed, J_minus
        return sample_uniform(self.spec, st.init_rng), "init", 0.0, armed, J_minus

    def replay(self, stored: list[HwTrial]) -> _State:
        """Rebuild the strategy state from stored trials (no rig time)."""
        st = self._new_state()
        for rec in stored:
            n = len(st.history)
            if rec.index != n:
                raise ReplayDiverged(f"stored trial {rec.index} out of order (expected {n})")
            theta, _, _, armed, J_minus = self._next_theta(st)
            if not np.allclose(theta, rec.theta, rtol=1e-6, atol=1e-9):
                raise ReplayDiverged(
                    f"{self.key['method']} seed {self.seed}: trial {n} proposes "
                    f"{np.array2string(theta, precision=4)} but the stored run evaluated "
                    f"{np.array2string(np.asarray(rec.theta), precision=4)}. The configuration "
                    "changed or the method is not reproducible from its seed.")
            st.strategy.on_trial_start(st.history, armed)
            decider = StopDecider(st.strategy, st.history, armed, J_minus, self.T)
            live = rec.trajectory(live=True)
            for k in range(1, min(rec.t_obs, self.T - 1) + 1):
                if decider(k, Trial(traj=live, t_obs=k)):
                    break
            trial = rec.trial()
            st.strategy.on_trial_end(trial, n)
            st.history.append(trial)
            st.used += trial.t_obs
        return st

    def _budget_left(self, st: _State) -> bool:
        n = len(st.history)
        if n >= self.n_max:
            return False
        return not (n >= self.n_init and st.used > self.t_budget - self.T)

    def _write_run(self, status: str, st: _State, stop_reason: str = "",
                   extra: dict[str, Any] | None = None) -> None:
        run = records.read_run(self.blob_dir)
        inc = incumbent_index(st.history)
        run.update({
            "kind": KIND, "key": self.key, "status": status, "stop_reason": stop_reason,
            "n_trials": len(st.history), "steps_used": st.used,
            "incumbent_index": -1 if inc is None else inc,
            "incumbent_J": math.inf if inc is None else float(st.history[inc].J_obs),
            "strategy": st.strategy.name, "strategy_params": st.strategy.strategy_spec.params,
            "n_init": self.n_init, "T": self.T, "reset_config": config_dict(
                self.system.reset_config), "phi_hat": self.system.phi_hat,
            "server": dict(self.system.client.info), "url": self.system.client.url,
            "host": socket.gethostname(), "code_version": code_version(),
            "updated_at": time.time(), "record_version": records.RECORD_VERSION})
        run.setdefault("started_at", time.time())
        run.update(extra or {})
        records.write_run(self.blob_dir, run)

    def run(self, max_new_trials: int | None = None,
            should_pause: Callable[[], bool] | None = None) -> CellResult:
        """Run trials until the budget is exhausted, ``max_new_trials`` ran, or a pause."""
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        stored = records.read_trials(self.trials_path)
        records.write_trials(self.trials_path, stored)       # drops a torn last line
        st = self.replay(stored)
        self.db.mark(KIND, self.key, RUNNING, blob_dir=self.blob_dir, started=time.time(),
                     host=socket.gethostname(), code_version=code_version())
        self._write_run(RUNNING, st)
        new = 0
        status, reason = PENDING, "interrupted"
        try:
            while True:
                if not self._budget_left(st):
                    status = DONE
                    reason = ("trials budget" if len(st.history) >= self.n_max
                              else "step budget")
                    break
                if max_new_trials is not None and new >= max_new_trials:
                    reason = "trial limit"
                    break
                if should_pause is not None and should_pause():
                    reason = "paused"
                    break
                self._one_trial(st)
                new += 1
        finally:
            self._write_run(status, st, reason)
            if status == DONE:
                self.db.mark(KIND, self.key, DONE, blob_dir=self.blob_dir, finished=time.time())
            else:
                self.db.mark(KIND, self.key, PENDING, blob_dir=self.blob_dir)
        log.info("%s %s seed %d: %d trials (%d new), %s", self.key["rig"], self.key["method"],
                 self.seed, len(st.history), new, reason)
        return CellResult(self.key, self.blob_dir, len(st.history), new, status == DONE, reason)

    def _one_trial(self, st: _State) -> None:
        n = len(st.history)
        theta, source, propose_s, armed, J_minus = self._next_theta(st)
        st.strategy.on_trial_start(st.history, armed)
        decider = StopDecider(st.strategy, st.history, armed, J_minus, self.T)
        started = time.time()
        reopen = n > 0 and n % self.system.reset_config.reopen_every == 0
        res = self.system.run_trial(theta, substream(self.seed, n), stop=decider, reopen=reopen,
                                    plant_seed=derive_seed(self.seed, n, "plant"))
        rec = _to_record(n, self.T, res, source, propose_s, started, decider.stopped_by)
        trial = rec.trial()
        t0 = time.perf_counter()
        st.strategy.on_trial_end(trial, n)
        end_s = time.perf_counter() - t0
        st.history.append(trial)
        st.used += trial.t_obs
        inc = incumbent_index(st.history)
        rec.incumbent_index = -1 if inc is None else int(inc)
        rec.incumbent_J = math.inf if inc is None else float(st.history[inc].J_obs)
        rec.timings["end_s"] = end_s
        rec.timings["total_s"] = propose_s + res.reset.seconds + res.control_s + end_s
        records.append_trial(self.trials_path, rec)
        log.info("trial %d %s: t=%d %s J=%.4g reset %.1fs", n, source, rec.t_obs, rec.outcome,
                 rec.J_obs, rec.timings["reset_s"])


# --------------------------------------------------------------------------- the field
@dataclass
class FieldPlan:
    rig: str
    methods: list[str]
    seeds: list[int]
    n_max: int = 180
    t_budget: int = 36000
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def keys(self) -> list[dict[str, Any]]:
        return [unit_key(self.rig, m, s, method_params(m, self.params.get(m)), self.n_max,
                         self.t_budget) for m in self.methods for s in self.seeds]


def run_field(plan: FieldPlan, client: RigClient, db: Db | None = None, *,
              system: RigSystem | None = None, max_new_trials: int | None = None,
              strategy_overrides: dict[str, dict[str, Any]] | None = None,
              max_attempts: int = 3) -> list[CellResult]:
    """Run every unit of ``plan`` (method-major, then seed) on the rig behind ``client``.

    A rig error (timeout, dropped connection) rebuilds the unit from its stored trials and
    retries, up to ``max_attempts`` times per unit without progress.
    """
    get_rig(plan.rig)
    db = db or Db()
    system = system or RigSystem(client)
    flag = pause_flag(db)
    results: list[CellResult] = []
    system.open_session()
    for key in plan.keys():
        if db.has(KIND, key):
            log.info("%s %s seed %d already done", key["rig"], key["method"], key["seed"])
            continue
        if flag.exists():
            log.info("pause flag %s present, stopping the field", flag)
            break
        attempts = 0
        while True:
            runner = CellRunner(system, db, key, (strategy_overrides or {}).get(key["method"]))
            try:
                res = runner.run(max_new_trials, should_pause=flag.exists)
                break
            except RigError as exc:
                attempts += 1
                log.warning("rig error in %s seed %d: %s (attempt %d)", key["method"],
                            key["seed"], exc, attempts)
                if attempts >= max_attempts:
                    raise
                system.resetter.opened = False
                system.open_session()
        results.append(res)
        if not res.finished:
            break
    return results


def parse_seeds(text: str) -> list[int]:
    """``0-4`` or ``0,2,3`` or ``1``."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
