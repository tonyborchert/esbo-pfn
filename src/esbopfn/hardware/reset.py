"""The between-trial reset of the rig, thesis Algorithm alg:app:hw-reset step by step.

1. Brake the pendulum until ``|phidot_pend|`` and its angle from hanging stay small for 0.4 s.
2. Coast at 0 V for 1.5 s. If every channel stayed bit-constant (a frozen board), reopen it.
3. Repeat a PD drive of the arm to its center (``K_P = 3.0`` V/rad, ``K_D = 0.2`` V s/rad, at
   least 0.35 V while the arm is stuck) until ``|phi_arm| < 2`` deg, 3 s elapsed or the arm moves
   away from the center. Coast 0.3 s.
4. Repeat {recenter if ``|phi_arm| > 5`` deg, brake the pendulum, hold 0 V} until for 2 s
   continuously ``||phi_pend| - pi| < 2`` deg, ``|phidot_pend| < 0.05`` and
   ``|phidot_arm| < 0.05`` rad/s (bounded by a 25 s cap, after which the trial starts anyway and
   the diagnostics say so).
5. Set the zero of ``phi_arm`` to the current arm position for the coming trial.
6. Draw ``s`` in ``{-1, +1}`` from the trial's generator and kick with ``s * 0.6`` V for 0.03 s.

The connection is reset every tenth trial (:meth:`RigReset.run` with ``reopen=True``): the arm is
centered and settled first, its position is read and carried across the board reopen (which
re-zeroes the encoders wherever the arm is), so the arm frame of the session survives it.

All durations are counted in control periods (``round(seconds * 120)`` steps), so the procedure
is deterministic on the simulated plant. The numerics (brake law, guards, tolerances) are those
the rig ran, documented in ``docs/HARDWARE.md``.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from esbopfn.hardware.client import RigClient

TWO_PI = 2.0 * math.pi


def wrap(a: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return ((a + math.pi) % TWO_PI) - math.pi


def _sign(v: float) -> float:
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


@dataclass(frozen=True)
class ResetConfig:
    """Thesis values (alg:app:hw-reset) plus the rig's brake law and guards."""

    frequency: float = 120.0
    # 1. brake until still and hanging over a trailing window
    brake_window_s: float = 0.4
    brake_vel_tol: float = 0.2              # rad/s, max |phidot_pend| over the window
    brake_hang_deg: float = 3.0             # max ||phi_pend| - pi| over the window
    brake_max_s: float = 8.0
    brake_max_voltage: float = 2.0
    brake_k: float = 0.3                    # V per rad/s of pendulum speed
    brake_speed_cap: float = 6.0            # rad/s
    brake_deadband: float = 0.05            # rad/s
    brake_arm_kp: float = 0.8               # V per rad of arm displacement beyond the free zone
    brake_arm_free_deg: float = 20.0
    brake_arm_limit_deg: float = 80.0       # never brake toward an end stop
    # 2. free coast with the liveness check
    coast_s: float = 1.5
    liveness_samples: int = 24
    # 3. PD centering
    center_kp: float = 3.0
    center_kd: float = 0.2
    center_floor: float = 0.35              # V, minimum |u| while the arm is stuck
    center_stall_vel: float = 0.05          # rad/s, "stuck" below this arm speed
    center_tol_deg: float = 2.0
    center_settle_vel: float = 0.30         # rad/s, the arm must also be slow at the center
    center_dwell_s: float = 0.10
    center_max_s: float = 3.0
    center_max_voltage: float = 1.5
    center_runaway_deg: float = 8.0         # abort if the arm moves this far away from 0
    center_coast_s: float = 0.3
    # 4. settle
    settle_hang_deg: float = 2.0
    settle_vel_tol: float = 0.05
    settle_arm_deg: float = 5.0
    settle_dwell_s: float = 2.0
    settle_calm_s: float = 0.5              # brake-to-rest exit dwell
    settle_brake_max_s: float = 8.0
    settle_max_s: float = 25.0
    # 6. kick
    kick_voltage: float = 0.6
    kick_s: float = 0.03
    # connection reset cadence
    reopen_every: int = 10

    def n(self, seconds: float) -> int:
        """Number of control periods in ``seconds``."""
        return int(round(seconds * self.frequency))


@dataclass
class ResetResult:
    """State at the start of the trial plus the diagnostics recorded per trial."""

    state: np.ndarray                     # trial-frame state after the kick
    raw_state: np.ndarray                 # board-frame state after the kick
    zero_raw: float                       # board-frame arm angle defined as zero for the trial
    kick_sign: float
    seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


class _Phases:
    """Wall-clock and step count per named phase."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.steps: dict[str, int] = defaultdict(int)
        self.current = "other"

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        prev, self.current = self.current, name
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - t0
            self.current = prev


class RigReset:
    """The reset procedure over a :class:`RigClient`.

    Keeps the session's arm frame: ``batch_zero_raw`` is the board-frame reading of the physical
    center (0 when the board was first opened, carried across a reopen).
    """

    def __init__(self, client: RigClient, config: ResetConfig | None = None) -> None:
        self.client = client
        self.cfg = config or ResetConfig()
        self.batch_zero_raw = 0.0
        self.opened = False
        self._phases = _Phases()
        self._frozen_revivals = 0

    # -- primitives -----------------------------------------------------------------------
    def arm(self, state: np.ndarray) -> float:
        """Arm angle in the session frame (rad)."""
        return wrap(float(state[0]) - self.batch_zero_raw)

    def _step(self, u: float) -> np.ndarray:
        self._phases.steps[self._phases.current] += 1
        return self.client.step(float(u))

    def open(self) -> np.ndarray:
        """First board open of a session (defines the arm frame)."""
        state = self.client.reset_down()
        self.batch_zero_raw = 0.0
        self.opened = True
        return state

    def brake_voltage(self, alpha: float, alpha_dot: float, arm_cont: float) -> float:
        """Energy-removal brake: oppose the pendulum swing, hold the arm near the center."""
        c = self.cfg
        u = 0.0
        if abs(alpha_dot) > c.brake_deadband:
            u = -c.brake_k * min(abs(alpha_dot), c.brake_speed_cap) * _sign(
                -math.cos(alpha) * alpha_dot)
        free = math.radians(c.brake_arm_free_deg)
        if abs(arm_cont) > free:
            u += c.brake_arm_kp * (arm_cont - _sign(arm_cont) * free)
        return float(np.clip(u, -c.brake_max_voltage, c.brake_max_voltage))

    def _brake_loop(self, max_s: float, exit_fn: Any) -> tuple[np.ndarray, bool]:
        """Brake while ``exit_fn(state)`` is false. Returns the state and whether it exited."""
        c = self.cfg
        state = self._step(0.0)
        raw_prev, cont = float(state[0]), self.arm(state)
        limit = math.radians(c.brake_arm_limit_deg)
        for _ in range(c.n(max_s)):
            u = self.brake_voltage(float(state[1]), float(state[3]), cont)
            if abs(cont) > limit:
                u = 0.0
            state = self._step(u)
            cont += wrap(float(state[0]) - raw_prev)
            raw_prev = float(state[0])
            if exit_fn(state):
                return state, True
        return state, False

    # -- step 1 -------------------------------------------------------------------------------
    def brake_briefly(self) -> tuple[np.ndarray, bool]:
        """Brake until speed and hanging error stayed small over a 0.4 s trailing window."""
        c = self.cfg
        window = max(2, c.n(c.brake_window_s))
        speeds: list[float] = []
        hangs: list[float] = []
        hang_tol = math.radians(c.brake_hang_deg)

        def done(state: np.ndarray) -> bool:
            speeds.append(abs(float(state[3])))
            hangs.append(abs(abs(float(state[1])) - math.pi))
            if len(speeds) > window:
                speeds.pop(0)
                hangs.pop(0)
            return (len(speeds) == window and max(speeds) < c.brake_vel_tol
                    and max(hangs) <= hang_tol)

        return self._brake_loop(c.brake_max_s, done)

    # -- step 2 -------------------------------------------------------------------------------
    def coast(self, seconds: float) -> tuple[np.ndarray, bool]:
        """0 V for ``seconds``. Returns the state and whether the readings were frozen."""
        samples = []
        state = self._step(0.0)
        for _ in range(self.cfg.n(seconds)):
            state = self._step(0.0)
            samples.append(np.asarray(state, dtype=np.float64))
        return state, self._frozen(samples)

    def _frozen(self, samples: list[np.ndarray]) -> bool:
        if len(samples) < self.cfg.liveness_samples:
            return False
        arr = np.asarray(samples)
        return bool(all(np.ptp(arr[:, k]) == 0.0 for k in range(4)))

    def revive(self) -> np.ndarray:
        """Reopen a frozen board and check that it reads again (the arm frame is lost)."""
        self.client.connection_reset()
        self.batch_zero_raw = 0.0
        self._frozen_revivals += 1
        probe = [np.asarray(self._step(0.0)) for _ in range(self.cfg.liveness_samples + 6)]
        if self._frozen(probe):
            raise RuntimeError("rig board is wedged: every channel stayed bit-constant across a "
                               "reopen. Power-cycle or replug the QUBE.")
        return probe[-1]

    # -- step 3 -------------------------------------------------------------------------------
    def center_arm(self) -> tuple[np.ndarray, dict[str, Any]]:
        """PD drive of the arm to the session center with a stiction floor, then coast 0.3 s."""
        c = self.cfg
        tol = math.radians(c.center_tol_deg)
        runaway = math.radians(c.center_runaway_deg)
        limit = math.radians(c.brake_arm_limit_deg)
        need = max(1, c.n(c.center_dwell_s))
        with self._phases.phase("center_drive"):
            state = self._step(0.0)
            before = self.arm(state)
            converged = aborted = False
            floored = dwell = 0
            try:
                for _ in range(c.n(c.center_max_s)):
                    arm, arm_dot = self.arm(state), float(state[2])
                    if abs(arm) <= tol and abs(arm_dot) <= c.center_settle_vel:
                        dwell += 1
                        if dwell >= need:
                            converged = True
                            break
                    else:
                        dwell = 0
                    if ((_sign(arm) == _sign(before) and abs(arm) > abs(before) + runaway)
                            or abs(arm) > limit):
                        aborted = True       # moving away from the center
                        break
                    u = c.center_kp * arm + c.center_kd * arm_dot
                    if abs(arm_dot) < c.center_stall_vel and abs(u) < c.center_floor:
                        u = _sign(arm) * c.center_floor
                        floored += 1
                    state = self._step(float(np.clip(u, -c.center_max_voltage,
                                                     c.center_max_voltage)))
            finally:
                state = self._step(0.0)
        with self._phases.phase("center_coast"):
            for _ in range(c.n(c.center_coast_s)):
                state = self._step(0.0)
        after = self.arm(state)
        return state, {"center_before_deg": math.degrees(before),
                       "center_after_deg": math.degrees(after),
                       "center_converged": bool(converged and abs(after) <= tol),
                       "center_reached": bool(converged), "center_aborted": bool(aborted),
                       "center_floor_steps": int(floored)}

    # -- step 4 -------------------------------------------------------------------------------
    def _still(self, state: np.ndarray) -> bool:
        c = self.cfg
        return (abs(abs(float(state[1])) - math.pi) < math.radians(c.settle_hang_deg)
                and abs(float(state[3])) < c.settle_vel_tol
                and abs(float(state[2])) < c.settle_vel_tol)

    def settle(self, state: np.ndarray, prefix: str = "settle") -> tuple[np.ndarray, dict]:
        """Recenter if needed, brake, hold 0 V, until the rig was still for 2 s (25 s cap)."""
        c = self.cfg
        budget = c.n(c.settle_max_s)
        need = c.n(c.settle_dwell_s)
        calm_need = c.n(c.settle_calm_s)
        used_before = self._steps_total()
        attempts = recenters = restarts = longest = 0
        settled = False
        while not settled and self._steps_total() - used_before < budget:
            attempts += 1
            if abs(self.arm(state)) > math.radians(c.settle_arm_deg):
                state, _ = self.center_arm()
                recenters += 1
            rested = self._dwell_test(calm_need)
            left = budget - (self._steps_total() - used_before)
            with self._phases.phase(f"{prefix}_brake"):
                state, _ = self._brake_loop(min(c.settle_brake_max_s, max(0, left) / c.frequency),
                                            rested)
                for _ in range(c.n(c.center_coast_s)):
                    state = self._step(0.0)
            with self._phases.phase(f"{prefix}_wait"):
                run = 0
                while self._steps_total() - used_before < budget:
                    state = self._step(0.0)
                    if self._still(state):
                        run += 1
                        longest = max(longest, run)
                        if run >= need:
                            settled = True
                            break
                    else:
                        if run:
                            restarts += 1
                        run = 0
                        if abs(self.arm(state)) > math.radians(c.settle_arm_deg) or \
                                abs(float(state[3])) > 10 * c.settle_vel_tol:
                            break       # moving again: brake (and recenter) once more
        return state, {f"{prefix}d": bool(settled), f"{prefix}_attempts": attempts,
                       f"{prefix}_recenters": recenters, f"{prefix}_dwell_restarts": restarts,
                       f"{prefix}_longest_dwell_s": longest / c.frequency,
                       f"{prefix}_arm_deg": math.degrees(self.arm(state)),
                       f"{prefix}_hang_deg": math.degrees(abs(abs(float(state[1])) - math.pi)),
                       f"{prefix}_arm_dot": float(state[2]),
                       f"{prefix}_pend_dot": float(state[3])}

    def _dwell_test(self, need: int) -> Any:
        """``test(state)``: true once ``need`` consecutive states were still."""
        calm = [0]

        def test(state: np.ndarray) -> bool:
            calm[0] = calm[0] + 1 if self._still(state) else 0
            return calm[0] >= need

        return test

    def _steps_total(self) -> int:
        return sum(self._phases.steps.values())

    # -- connection reset ---------------------------------------------------------------------
    def reopen(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Center, settle, read the arm, reset the connection, carry the arm frame across."""
        state = self._step(0.0)
        before = math.degrees(self.arm(state))
        centered = abs(before) > self.cfg.center_tol_deg
        if centered:
            state, _ = self.center_arm()
        state, info = self.settle(state, prefix="reopen_settle")
        offset = self.arm(state)
        with self._phases.phase("connection_reset"):
            state = self.client.connection_reset()
        self.batch_zero_raw = -offset
        return state, {"reopen_center_before_deg": abs(before), "reopen_centered": centered,
                       "reopen_zero_shift_deg": math.degrees(offset), **info}

    # -- the whole procedure ------------------------------------------------------------------
    def run(self, rng: np.random.Generator, reopen: bool = False) -> ResetResult:
        """The reset before one trial. ``rng`` is the trial's generator (kick sign)."""
        c = self.cfg
        self._phases = _Phases()
        t0 = time.perf_counter()
        diag: dict[str, Any] = {"kind": "reset"}
        if not self.opened:
            with self._phases.phase("initial_reset_down"):
                self.open()
            diag["kind"] = "initial"
        with self._phases.phase("brake"):
            state, braked = self.brake_briefly()
        diag["brake_stopped"] = bool(braked)
        if reopen and diag["kind"] != "initial":
            state, info = self.reopen()
            diag.update(info)
            diag["kind"] = "connection_reset"
        with self._phases.phase("coast"):
            state, frozen = self.coast(c.coast_s)
        diag["frozen"] = bool(frozen)
        if frozen:
            with self._phases.phase("revive"):
                state = self.revive()
        state, center_info = self.center_arm()
        diag.update(center_info)
        state, settle_info = self.settle(state)
        diag.update(settle_info)
        zero_raw = float(state[0])
        sign = float(rng.choice([-1.0, 1.0]))
        with self._phases.phase("kick"):
            try:
                for _ in range(max(1, c.n(c.kick_s))):
                    state = self._step(sign * c.kick_voltage)
            finally:
                state = self._step(0.0)
        raw = np.asarray(state, dtype=np.float64).copy()
        trial_state = raw.copy()
        trial_state[0] = wrap(float(raw[0]) - zero_raw)
        diag.update({"kick_sign": sign, "kick_voltage": c.kick_voltage,
                     "zero_deg": math.degrees(wrap(zero_raw - self.batch_zero_raw)),
                     "arm_after_deg": math.degrees(self.arm(raw)),
                     "pend_dot": float(raw[3]),
                     "phase_s": dict(self._phases.seconds),
                     "phase_steps": dict(self._phases.steps),
                     "steps": self._steps_total()})
        seconds = time.perf_counter() - t0
        diag["seconds"] = seconds
        return ResetResult(state=trial_state, raw_state=raw, zero_raw=zero_raw, kick_sign=sign,
                           seconds=seconds, diagnostics=diag)


def config_dict(cfg: ResetConfig) -> dict[str, Any]:
    """The reset constants as a plain dict (written to ``run.json``)."""
    return asdict(cfg)
