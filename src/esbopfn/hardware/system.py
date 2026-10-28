"""One trial of the Furuta controller on the rig.

The controller is ``esbopfn.problems.furuta.sim.Controller``, unchanged: energy swing-up with the
arm barrier and the discrete LQR designed from the rig's nominal estimate ``p_hat`` (the furuta
prior's nominal parameters, the same for every rig configuration). One trial is ``T = 600``
control periods at 120 Hz:

- step ``k``: the measured state is logged (``x``, ``xdot``) and its stage cost
  ``|phi_pend| (1 + 20 (1 - min(|phi_pend|, 5 deg) / 5 deg))`` added to ``J_k``,
- for ``k < T`` the early-stop callback sees the trial cut at ``k`` and may end it here (before
  the next command is sent, so no extra step runs after a stop),
- otherwise the command ``u_k`` is sent. The arm angle of the returned state is unwrapped, and if
  it leaves ``+-90 deg`` the trial crashed at step ``k`` (thesis crash criterion, the same rule as
  the simulator, which checks the state after the command of step ``k``).

Success: complete, not crashed and mean ``|phi_pend|`` over the final 2 s below 5 deg. An
undesignable LQR is a crash at the first step (as in the simulator). After the trial 0 V is
written, the next reset starts with the brake.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from esbopfn.core import Instance, Trajectory, Trial
from esbopfn.hardware.client import RigClient
from esbopfn.hardware.reset import ResetConfig, ResetResult, RigReset, wrap
from esbopfn.problems.furuta import sim

T_STEPS = 600
CRASH_ARM = sim.THETA_SAFE           # 90 deg
HOLD_STEPS = int(round(sim.HOLD_SECONDS * sim.FREQUENCY))

#: ``stop(k, running) -> bool`` called after step ``k < T`` is logged
StopFn = Callable[[int, Trial], bool]


def stage_cost(pend: np.ndarray | float) -> np.ndarray:
    """Hold cost ``|phi_pend| (1 + 20 (1 - min(|phi_pend|, 5 deg) / 5 deg))``."""
    a = np.abs(np.asarray(pend, dtype=np.float64))
    return a * (1.0 + sim.HOLD_WEIGHT * (1.0 - np.minimum(a, sim.UPRIGHT) / sim.UPRIGHT))


def held(pend: np.ndarray, n: int) -> tuple[bool, float]:
    """Success test over the last 2 s of ``n`` logged steps: ``(mean < 5 deg, mean in deg)``."""
    if n < HOLD_STEPS:
        return False, math.nan
    m = float(np.mean(np.abs(np.asarray(pend[n - HOLD_STEPS:n], dtype=np.float64))))
    return m < sim.UPRIGHT, math.degrees(m)


@dataclass
class RigTrialResult:
    """Raw outcome of one rig trial."""

    theta: np.ndarray
    t_obs: int
    crash_step: int | None
    stopped: bool
    success: bool
    J: np.ndarray                 # (t_obs,)
    x: np.ndarray                 # (t_obs, 2)
    xdot: np.ndarray              # (t_obs, 2)
    u: np.ndarray                 # (t_obs,) commands sent (0 where none was sent)
    reset: ResetResult
    control_s: float
    stop_rule_s: float
    max_arm_deg: float
    mean_pend_final_deg: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RigSystem:
    """Runs trials of the Furuta controller on a rig client."""

    def __init__(self, client: RigClient, phi_hat: dict[str, float] | None = None,
                 reset_config: ResetConfig | None = None, T: int = T_STEPS) -> None:
        if phi_hat is None:
            from esbopfn.hardware.sim_plant import nominal_rig_instance

            phi_hat = dict(nominal_rig_instance().phi_hat)
        self.client = client
        self.phi_hat = dict(phi_hat)
        self.T = int(T)
        self.resetter = RigReset(client, reset_config)
        self.trials_run = 0

    def open_session(self) -> dict[str, Any]:
        """Connect, ask the server what it drives and open the board (defines the arm frame)."""
        info = self.client.hello()
        self.resetter.open()
        return info

    @property
    def reset_config(self) -> ResetConfig:
        return self.resetter.cfg

    def controller(self, theta: np.ndarray) -> sim.Controller | None:
        try:
            return sim.Controller(self.phi_hat, np.asarray(theta, dtype=np.float64))
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            return None

    def run_trial(self, theta: np.ndarray, rng: np.random.Generator,
                  stop: StopFn | None = None, reopen: bool = False,
                  plant_seed: int | None = None) -> RigTrialResult:
        """Reset the rig, then run ``theta`` for up to ``T`` steps.

        ``rng`` is the trial's generator (kick sign). ``plant_seed`` is sent as ``begin_trial``
        (seeds the simulated plant). ``reopen`` resets the connection during the reset.
        """
        theta = np.asarray(theta, dtype=np.float64)
        T = self.T
        if plant_seed is not None:
            self.client.begin_trial(int(plant_seed))
            if self.client.info.get("reseed_state"):
                # the simulated plant restarts at its board zero, so the session frame does too
                self.resetter.batch_zero_raw = 0.0
        reset = self.resetter.run(rng, reopen=reopen)
        ctrl = self.controller(theta)

        J = np.zeros(T, dtype=np.float32)
        x = np.zeros((T, 2), dtype=np.float32)
        xd = np.zeros((T, 2), dtype=np.float32)
        u_log = np.zeros(T, dtype=np.float32)
        running = Trajectory(theta=theta, J=J, x=x, xdot=xd, crash_step=None, success=False)

        state = reset.state
        zero = reset.zero_raw
        raw_prev = float(reset.raw_state[0])
        cont = float(state[0])
        total = 0.0
        crash_step: int | None = None
        stopped = False
        stop_s = 0.0
        n = 0
        max_arm = abs(cont)
        t0 = time.perf_counter()
        try:
            for k in range(1, T + 1):
                arm, pend = wrap(float(state[0])), float(state[1])
                x[k - 1] = (arm, pend)
                xd[k - 1] = (float(state[2]), float(state[3]))
                total += float(stage_cost(pend))
                J[k - 1] = total
                n = k
                if ctrl is None:
                    crash_step = k
                    break
                if stop is not None and k < T:
                    s0 = time.perf_counter()
                    decision = bool(stop(k, Trial(traj=running, t_obs=k)))
                    stop_s += time.perf_counter() - s0
                    if decision:
                        stopped = True
                        break
                u = ctrl.action(arm, pend, float(state[2]), float(state[3]))
                u_log[k - 1] = u
                state = np.asarray(self.client.step(u), dtype=np.float64).copy()
                raw = float(state[0])
                cont += wrap(raw - raw_prev)
                raw_prev = raw
                state[0] = wrap(raw - zero)
                max_arm = max(max_arm, abs(cont))
                if not all(math.isfinite(v) for v in state) or abs(cont) > CRASH_ARM:
                    crash_step = k
                    break
        finally:
            control_s = time.perf_counter() - t0
            try:
                self.client.step(0.0)        # de-energize, the next reset brakes
            except Exception:  # noqa: BLE001 - never mask the trial's own error
                pass
        self.trials_run += 1
        success, mean_deg = (False, math.nan)
        if crash_step is None and not stopped and n == T:
            success, mean_deg = held(x[:n, 1], n)
        elif n >= HOLD_STEPS:
            mean_deg = held(x[:n, 1], n)[1]
        return RigTrialResult(theta=theta, t_obs=n, crash_step=crash_step, stopped=stopped,
                              success=bool(success), J=J[:n].copy(), x=x[:n].copy(),
                              xdot=xd[:n].copy(), u=u_log[:n].copy(), reset=reset,
                              control_s=control_s, stop_rule_s=stop_s,
                              max_arm_deg=math.degrees(max_arm), mean_pend_final_deg=mean_deg)


def rig_instance(phi_hat: dict[str, float]) -> Instance:
    """An :class:`Instance` describing a rig run (only ``phi_hat`` acts, the plant is physical)."""
    return Instance(phi={}, phi_hat=dict(phi_hat), noise_std={}, extras={}, seed=0)
