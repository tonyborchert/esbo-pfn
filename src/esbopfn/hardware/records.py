"""Per-trial and per-run records of the hardware field (the ``hardware_optimization`` blobs).

A run blob holds ``run.json`` (key, budgets, status, server info, resolved strategy parameters)
and ``trials.jsonl`` with one :class:`HwTrial` per line. Infinite costs are written as
``Infinity`` (Python's JSON dialect, as in ``strategies.RunRecord``).

Cost and channel conventions: ``J`` holds the cumulative stage cost of the ``t_obs`` logged
measurements (all finite, also for a crashed trial, whose last logged measurement precedes the
step that left the arm bounds), ``x`` the level channels ``[phi_arm, phi_pend]`` (trial frame,
pendulum from upright), ``xdot`` the measured rates. ``J_obs`` is ``inf`` for a crashed trial.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.core import Trajectory, Trial

TRIALS_FILE = "trials.jsonl"
RUN_FILE = "run.json"
RECORD_VERSION = 1

#: outcome values
COMPLETE, EARLY_STOPPED, CRASHED, BUDGET_CUT = "complete", "early_stopped", "crashed", "budget_cut"
OUTCOMES = (COMPLETE, EARLY_STOPPED, CRASHED, BUDGET_CUT)

#: keys of ``HwTrial.timings`` (seconds)
TIMING_KEYS = ("propose_s", "reset_s", "control_s", "stop_rule_s", "end_s", "total_s")


@dataclass
class HwTrial:
    """One rig trial."""

    index: int
    theta: list[float]
    source: str                         # "init" (uniform design) or "strategy"
    T: int
    t_obs: int
    outcome: str                        # complete | early_stopped | crashed | budget_cut
    crash_step: int | None
    success: bool                       # complete, not crashed, mean |phi_pend| last 2 s < 5 deg
    J_obs: float                        # J[t_obs - 1], inf if crashed
    J: list[float]                      # t_obs cumulative costs (finite)
    x: list[list[float]]                # t_obs rows [phi_arm, phi_pend]
    xdot: list[list[float]]             # t_obs rows [phidot_arm, phidot_pend]
    u: list[float] | None               # t_obs commanded voltages (None if not recorded)
    kick_sign: float
    incumbent_index: int                # after this trial, -1 = none
    incumbent_J: float                  # after this trial, inf = none
    timings: dict[str, float] = field(default_factory=dict)
    reset: dict[str, Any] = field(default_factory=dict)
    max_arm_deg: float = math.nan
    mean_pend_final_deg: float = math.nan
    started_at: float = math.nan
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def crashed(self) -> bool:
        return self.outcome == CRASHED

    @property
    def complete(self) -> bool:
        return self.outcome == COMPLETE

    @property
    def early_stopped(self) -> bool:
        return self.outcome in (EARLY_STOPPED, BUDGET_CUT)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HwTrial:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> HwTrial:
        return cls.from_dict(json.loads(text))

    def deterministic_view(self) -> dict[str, Any]:
        """Everything except wall-clock quantities (what a resumed simulated run reproduces)."""
        d = asdict(self)
        for k in ("timings", "started_at"):
            d.pop(k)
        d["reset"] = {k: v for k, v in self.reset.items()
                      if k not in ("phase_s", "seconds")}
        return d

    # -- conversion to the core data model --------------------------------------------------
    def trajectory(self, live: bool = False) -> Trajectory:
        """The trial as a full-length :class:`Trajectory` (values after ``t_obs`` repeat the last
        one, ``J`` is ``inf`` from the crash step on). ``live=True`` gives the view the strategy
        had while the trial ran (no crash marked)."""
        T, n = int(self.T), int(self.t_obs)
        J = np.empty(T, dtype=np.float32)
        x = np.empty((T, 2), dtype=np.float32)
        xd = np.empty((T, 2), dtype=np.float32)
        if n:
            J[:n] = np.asarray(self.J, dtype=np.float32)
            J[n:] = J[n - 1]
            x[:n] = np.asarray(self.x, dtype=np.float32).reshape(n, 2)
            x[n:] = x[n - 1]
            xd[:n] = np.asarray(self.xdot, dtype=np.float32).reshape(n, 2)
            xd[n:] = xd[n - 1]
        else:
            J[:] = 0.0
            x[:] = 0.0
            xd[:] = 0.0
        crash = None if live else self.crash_step
        if crash is not None:
            J[crash - 1:] = np.inf
        return Trajectory(theta=np.asarray(self.theta, dtype=np.float64), J=J, x=x, xdot=xd,
                          crash_step=crash, success=bool(self.success and crash is None))

    def trial(self) -> Trial:
        """The observed trial (cut at ``t_obs``)."""
        return Trial(traj=self.trajectory(), t_obs=int(self.t_obs))


def read_trials(path: Path) -> list[HwTrial]:
    """All complete lines of a ``trials.jsonl`` (a torn last line is ignored)."""
    path = Path(path)
    if path.is_dir():
        path = path / TRIALS_FILE
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(HwTrial.from_json(line))
        except (json.JSONDecodeError, TypeError):
            break
    return out


def write_trials(path: Path, trials: list[HwTrial]) -> None:
    """Rewrite a ``trials.jsonl`` with exactly ``trials``."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(t.to_json() + "\n" for t in trials))
    tmp.replace(path)


def append_trial(path: Path, trial: HwTrial) -> None:
    """Append one line and flush it to disk."""
    with open(path, "a") as f:
        f.write(trial.to_json() + "\n")
        f.flush()


def read_run(blob_dir: Path) -> dict[str, Any]:
    p = Path(blob_dir) / RUN_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def write_run(blob_dir: Path, run: dict[str, Any]) -> None:
    p = Path(blob_dir) / RUN_FILE
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=1, default=str))
    tmp.replace(p)


@dataclass
class HwRun:
    """One (rig, method, seed) run with its trials, as the metrics consume it."""

    rig: str
    method: str
    seed: int
    trials: list[HwTrial]
    key: dict[str, Any] = field(default_factory=dict)
