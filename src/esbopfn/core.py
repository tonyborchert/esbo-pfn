"""Core data model: problem specification, sampled plant instance, trajectory and observed trial.

All step indices are 1-based as in the thesis. ``J[t - 1]`` is the cumulative cost after step
``t``, and a crash at step ``k`` sets ``J[k - 1:]`` to ``+inf``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Budget constants of the thesis setup: ``N_max = 3 * 15 * min(d, 4)`` and
#: ``T_budget = 15 * T * min(d, 4)``.
BUDGET_D_CAP = 4
TRIALS_PER_DIM = 15
TRIAL_CAP_FACTOR = 3


@dataclass(frozen=True, eq=False)
class ProblemSpec:
    """Static description of one control problem (search box, horizon, observed channels)."""

    name: str
    d: int
    theta_lo: np.ndarray
    theta_hi: np.ndarray
    T: int
    channels: tuple[str, ...]
    rate_channels: tuple[str, ...]
    can_crash: bool
    has_success: bool
    theta_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        lo = np.asarray(self.theta_lo, dtype=np.float64)
        hi = np.asarray(self.theta_hi, dtype=np.float64)
        if lo.shape != (self.d,) or hi.shape != (self.d,):
            raise ValueError(f"{self.name}: theta box must have shape ({self.d},)")
        if np.any(hi <= lo):
            raise ValueError(f"{self.name}: theta_hi must exceed theta_lo")
        if self.theta_names and len(self.theta_names) != self.d:
            raise ValueError(f"{self.name}: theta_names must have {self.d} entries")
        lo.flags.writeable = False
        hi.flags.writeable = False
        object.__setattr__(self, "theta_lo", lo)
        object.__setattr__(self, "theta_hi", hi)
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "rate_channels", tuple(self.rate_channels))
        object.__setattr__(self, "theta_names", tuple(self.theta_names))

    def budget_d(self) -> int:
        """Dimension used for the budgets, ``min(d, 4)``."""
        return min(self.d, BUDGET_D_CAP)

    def n_max(self) -> int:
        """Trial cap ``N_max = 3 * 15 * min(d, 4)``."""
        return TRIAL_CAP_FACTOR * TRIALS_PER_DIM * self.budget_d()

    def t_budget(self) -> int:
        """Time budget in logged steps, ``15 * T * min(d, 4)``."""
        return TRIALS_PER_DIM * self.T * self.budget_d()

    def n_init(self) -> int:
        """Size of the initial design, ``max(d, 2)``."""
        return max(self.d, 2)

    @property
    def extent(self) -> np.ndarray:
        """Width of the search box per dimension."""
        return self.theta_hi - self.theta_lo

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-serializable form."""
        return {
            "name": self.name,
            "d": self.d,
            "theta_lo": self.theta_lo.tolist(),
            "theta_hi": self.theta_hi.tolist(),
            "T": self.T,
            "channels": list(self.channels),
            "rate_channels": list(self.rate_channels),
            "can_crash": self.can_crash,
            "has_success": self.has_success,
            "theta_names": list(self.theta_names),
        }

    def spec_hash(self) -> str:
        """SHA-256 of the canonical JSON form, used to tag pools and models."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProblemSpec):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(self.spec_hash())


@dataclass
class Instance:
    """One sampled plant: true parameters, the controller's estimate, sensor noise and extras."""

    phi: dict[str, float]
    phi_hat: dict[str, float]
    noise_std: dict[str, float]
    extras: dict[str, float] = field(default_factory=dict)
    seed: int = 0


@dataclass
class Trajectory:
    """Full simulation output of one controller on one instance, not cut.

    ``success`` is false for a crashed trajectory. For problems without a success criterion it
    equals ``not crashed``.
    """

    theta: np.ndarray
    J: np.ndarray
    x: np.ndarray
    xdot: np.ndarray | None
    crash_step: int | None
    success: bool

    @property
    def T(self) -> int:
        """Number of logged steps."""
        return int(self.J.shape[0])

    @property
    def crashed(self) -> bool:
        return self.crash_step is not None

    @property
    def J_total(self) -> float:
        """Total cost ``J_T`` (``inf`` if the trajectory crashed)."""
        return float(self.J[-1])

    def cut(self, t_obs: int) -> Trial:
        """Return the trial that observed this trajectory up to step ``t_obs``.

        ``t_obs = 0`` is an unrun controller. A crashed trajectory is never observed past its
        crash step, so ``t_obs`` is clamped to ``crash_step``.
        """
        if not 0 <= t_obs <= self.T:
            raise ValueError(f"t_obs must lie in [0, {self.T}], got {t_obs}")
        if self.crash_step is not None:
            t_obs = min(t_obs, self.crash_step)
        return Trial(traj=self, t_obs=int(t_obs))


@dataclass
class Trial:
    """A trajectory as seen by an optimizer or predictor, observed up to step ``t_obs``."""

    traj: Trajectory
    t_obs: int

    @property
    def theta(self) -> np.ndarray:
        return self.traj.theta

    @property
    def T(self) -> int:
        return self.traj.T

    @property
    def unrun(self) -> bool:
        """True if no step was observed (a query controller)."""
        return self.t_obs == 0

    @property
    def crashed(self) -> bool:
        """True if the crash happened within the observed steps."""
        cs = self.traj.crash_step
        return cs is not None and cs <= self.t_obs

    @property
    def complete(self) -> bool:
        """True if the full horizon was observed without a crash."""
        return self.t_obs == self.traj.T and not self.crashed

    @property
    def early_stopped(self) -> bool:
        """True if the trial was cut before the horizon without a crash."""
        return 0 < self.t_obs < self.traj.T and not self.crashed

    @property
    def successful(self) -> bool:
        """True if the trial is complete and the trajectory meets the success criterion."""
        return self.complete and self.traj.success

    @property
    def J_obs(self) -> float:
        """Observed cumulative cost ``J[t_obs - 1]`` (``inf`` if crashed, ``0`` if unrun)."""
        if self.t_obs == 0:
            return 0.0
        if self.crashed:
            return math.inf
        return float(self.traj.J[self.t_obs - 1])
