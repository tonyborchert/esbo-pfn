"""Furuta closed loop: QUBE-Servo 2 model, swing-up with arm barrier and LQR hold, rig-style reset.

Plant: the rotary-pendulum equations of the Quanser workbook (port of ``gym_brt``'s forward model
with the torque constant ``kt`` in the motor law), parameters ``[Rm, kt, km, mr, Lr, Dr, mp, Lp,
Dp]``, rod inertias ``m L^2 / 12``, motor torque ``-kt (V - km theta_dot) / Rm``, applied voltage
clipped to 18 V, ten semi-implicit Euler sub-steps per 1/120 s control step, both angles wrapped to
``(-pi, pi]`` with the pendulum angle measured from upright.

Controller (designed from the estimate ``p_hat``): outside 20 deg of upright an energy pump
``mu (E - E_bar m_p g L_p) sign(-cos(alpha) alpha_dot)`` (clipped to 6 V, scaled by
``Rm m_r L_r / kt``) plus the barrier ``K_P^barrier sign(theta_c) theta_c^2`` on the controller's
continuous arm angle, inside 20 deg the discrete LQR with ``Q = diag(q1..q4)`` and ``R = r`` on the
zero-order-hold discretization of the upright linearization (a central-difference Jacobian of the
same dynamics) solved with scipy's DARE. The command is clipped to 6 V. An undesignable LQR is a
crash at the first step.

Actuation: the command passes a fractional dead-time line (``dead_time`` per instance, linear
interpolation between control periods, starting at rest), then Gaussian voltage noise is added.

Reset (the rig's brake-then-kick): rest hanging down plus a 0.01 Gaussian residual on all four
states, then a ``kick`` volt pulse for 30 ms (four control steps) through the dead-time line, its
sign drawn per trial. A pulse that drives the state past ``1e6`` or to a non-finite value is
treated as not having taken and the braked state is kept.

Sensors: Gaussian noise on both angles (shared ``angle`` level) and both rates (shared
``velocity`` level). Crash: the true continuous arm angle leaves ``+-90 deg`` or the state becomes
non-finite. Success: no crash and mean measured ``|phi_pend|`` over the final 2 s below 5 deg.
Stage cost ``|phi_pend| (1 + 20 (1 - min(|phi_pend|, 5 deg) / 5 deg))``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy import linalg

from esbopfn.core import Instance, Trajectory
from esbopfn.problems.base import cached_spec, check_theta, finish, horizon

NAME = "furuta"

G = 9.81
FREQUENCY = 120.0
INTEGRATION_STEPS = 10
PLANT_MAX_VOLTAGE = 18.0
CTRL_MAX_VOLTAGE = 6.0
BALANCE_THRESHOLD = 20.0 * np.pi / 180.0
BARRIER_POWER = 2.0  # fixed for control rates of 120 Hz and above
THETA_SAFE = np.pi / 2.0
RESIDUAL_STD = 0.01
KICK_SECONDS = 0.03
RESET_MAX_STATE = 1e6
HOLD_SECONDS = 2.0
UPRIGHT = np.deg2rad(5.0)
HOLD_WEIGHT = 20.0
PHI_NAMES = ("Rm", "kt", "km", "mr", "Lr", "Dr", "mp", "Lp", "Dp")

_PI = np.pi
_TWO_PI = 2 * np.pi


def _wrap(a: float) -> float:
    return ((a + _PI) % _TWO_PI) - _PI


def _sign(v: float) -> float:
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


class Dynamics:
    """The rotary-pendulum accelerations for one parameter set, with its constants precomputed
    (prefix products of the reference expressions, so results are bit-identical to them)."""

    def __init__(self, phi: Mapping[str, float]) -> None:
        Rm, kt, km, mr, Lr, Dr, mp, Lp, Dp = (float(phi[n]) for n in PHI_NAMES)
        self.Rm, self.kt, self.km, self.mr, self.Lr, self.mp, self.Lp = Rm, kt, km, mr, Lr, mp, Lp
        Lp2, Lr2 = Lp * Lp, Lr * Lr
        Jr = mr * Lr2 / 12.0
        Jp = mp * Lp2 / 12.0
        self.Jp = Jp
        self.Lp2 = Lp2
        self.c_den1 = 4.0 * Lp2 * Lr2 * (mp * mp)
        self.c_jp = 4.0 * Jp + Lp2 * mp
        self.c_4jr = 4.0 * Jr
        self.c_lp2mp = Lp2 * mp
        self.c_4lr2mp = 4.0 * Lr2 * mp
        self.c_m8dp = -8.0 * Dp
        self.c_4lpgmp = 4.0 * Lp * G * mp
        self.c_4dr = 4.0 * Dr
        self.c_2lplr = 2.0 * Lp * Lr
        self.c_mlplrmp = -Lp * Lr * mp
        self.c_2lplrmp = 2.0 * Lp * Lr * mp

    def accel(self, alpha: float, theta_dot: float, alpha_dot: float,
              tau: float) -> tuple[float, float]:
        sin_a, cos_a = math.sin(alpha), math.cos(alpha)
        sin_2a = math.sin(2.0 * alpha)
        inner = self.c_4jr + self.c_lp2mp * (sin_a * sin_a) + self.c_4lr2mp
        denom = self.c_den1 * (cos_a * cos_a) - self.c_jp * inner
        term_p = (self.c_m8dp * alpha_dot + self.c_lp2mp * (theta_dot * theta_dot) * sin_2a
                  + self.c_4lpgmp * sin_a)
        term_r = (self.c_4dr * theta_dot
                  + self.Lp2 * alpha_dot * self.mp * theta_dot * sin_2a
                  + self.c_2lplr * (alpha_dot * alpha_dot) * self.mp * sin_a - 4.0 * tau)
        theta_dd = (self.c_mlplrmp * term_p * cos_a + self.c_jp * term_r) / denom
        alpha_dd = (self.c_2lplrmp * term_r * cos_a - 0.5 * inner * term_p) / denom
        return theta_dd, alpha_dd

    def torque(self, volts: float, theta_dot: float) -> float:
        return -(self.kt * (volts - self.km * theta_dot)) / self.Rm


def step(dyn: Dynamics, state: list[float], volts: float) -> list[float]:
    """Advance the plant by one control period at constant ``volts`` (clipped to 18 V)."""
    vm = min(max(volts, -PLANT_MAX_VOLTAGE), PLANT_MAX_VOLTAGE)
    theta, alpha, theta_dot, alpha_dot = state
    dt = (1.0 / FREQUENCY) / INTEGRATION_STEPS
    kt, km, rm = dyn.kt, dyn.km, dyn.Rm
    accel = dyn.accel
    try:
        for _ in range(INTEGRATION_STEPS):
            tau = -(kt * (vm - km * theta_dot)) / rm
            theta_dd, alpha_dd = accel(alpha, theta_dot, alpha_dot, tau)
            theta_dot += theta_dd * dt
            alpha_dot += alpha_dd * dt
            theta = _wrap(theta + theta_dot * dt)
            alpha = _wrap(alpha + alpha_dot * dt)
    except (OverflowError, ValueError, ZeroDivisionError):
        return [math.nan] * 4
    return [theta, alpha, theta_dot, alpha_dot]


def linearize_upright(dyn: Dynamics) -> tuple[np.ndarray, np.ndarray]:
    """Continuous ``(A, B)`` about upright from central differences (step 1e-6) of the dynamics.
    ``B`` uses the workbook sign convention (the motor law carries a leading minus)."""

    def fdot(x: np.ndarray, volts: float) -> np.ndarray:
        _, alpha, theta_dot, alpha_dot = (float(v) for v in x)
        tau = dyn.torque(volts, theta_dot)
        theta_dd, alpha_dd = dyn.accel(alpha, theta_dot, alpha_dot, tau)
        return np.array([theta_dot, alpha_dot, theta_dd, alpha_dd])

    eps = 1e-6
    x0 = np.zeros(4)
    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    for j in range(4):
        d = np.zeros(4)
        d[j] = eps
        A[:, j] = (fdot(x0 + d, 0.0) - fdot(x0 - d, 0.0)) / (2 * eps)
    B[:, 0] = -(fdot(x0, eps) - fdot(x0, -eps)) / (2 * eps)
    return A, B


def lqr_gains(phi_hat: Mapping[str, float], q: np.ndarray, r: float) -> np.ndarray:
    """Discrete LQR gains ``(k_theta, k_alpha, k_theta_dot, k_alpha_dot)`` for the estimate."""
    A, B = linearize_upright(Dynamics(phi_hat))
    em = np.zeros((5, 5))
    em[:4, :4] = A
    em[:4, 4:] = B
    ms = linalg.expm(em / FREQUENCY)
    Ad, Bd = ms[:4, :4], ms[:4, 4:]
    R = np.array([[float(r)]])
    P = np.array(linalg.solve_discrete_are(Ad, Bd, np.diag(np.asarray(q, dtype=np.float64)), R))
    K = np.array(linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad))
    return K[0]


class DeadTime:
    """Fractional transport delay of ``samples`` control periods with linear interpolation,
    starting at rest. ``samples = 0`` is the identity."""

    def __init__(self, samples: float) -> None:
        if samples < 0.0:
            raise ValueError("dead time must be non-negative")
        self.n = int(math.floor(samples))
        self.frac = float(samples) - self.n
        self.buf = [0.0] * (self.n + 2)

    def __call__(self, v: float) -> float:
        buf = self.buf
        buf.append(float(v))
        del buf[0]
        return (1.0 - self.frac) * buf[-1 - self.n] + self.frac * buf[-2 - self.n]


class Controller:
    """Energy swing-up with polynomial arm barrier, LQR hold within 20 deg of upright."""

    def __init__(self, phi_hat: Mapping[str, float], theta: np.ndarray) -> None:
        dyn = Dynamics(phi_hat)
        self.mu = float(theta[0])
        self.ref_energy = float(theta[1]) * dyn.mp * G * dyn.Lp
        self.k = [float(v) for v in lqr_gains(phi_hat, theta[2:6], float(theta[6]))]
        self.barrier_kp = float(theta[7])
        self.half_mpglp = 0.5 * dyn.mp * G * dyn.Lp
        self.half_jp = dyn.Jp / 2.0
        self.mrlr = dyn.mr * dyn.Lr
        self.rm_over_kt = dyn.Rm / dyn.kt
        self.theta_cont: float | None = None
        self.prev_theta = 0.0

    def action(self, theta: float, alpha: float, theta_dot: float, alpha_dot: float) -> float:
        if self.theta_cont is None:
            self.theta_cont = self.prev_theta = theta
        else:
            self.theta_cont += _wrap(theta - self.prev_theta)
            self.prev_theta = theta
        if abs(alpha) <= BALANCE_THRESHOLD:
            k = self.k
            a = theta * k[0] + alpha * k[1] + theta_dot * k[2] + alpha_dot * k[3]
        else:
            energy = self.half_mpglp * (1.0 + math.cos(alpha)) + self.half_jp * alpha_dot * alpha_dot
            u = self.mu * (energy - self.ref_energy) * _sign(-math.cos(alpha) * alpha_dot)
            u = min(max(u, -CTRL_MAX_VOLTAGE), CTRL_MAX_VOLTAGE)
            pump = -(self.rm_over_kt * (self.mrlr * u))
            tc = self.theta_cont
            a = pump + self.barrier_kp * _sign(tc) * (abs(tc) * abs(tc))
        return min(max(a, -CTRL_MAX_VOLTAGE), CTRL_MAX_VOLTAGE)


def reset(dyn: Dynamics, delay: DeadTime, kick: float, rng: np.random.Generator) -> list[float]:
    """Brake to rest hanging down (plus residual), then the signed kick pulse through the delay."""
    braked = [0.0, np.pi, 0.0, 0.0] + rng.standard_normal(4) * RESIDUAL_STD
    braked = [_wrap(float(braked[0])), _wrap(float(braked[1])), float(braked[2]), float(braked[3])]
    sign = float(rng.choice([-1.0, 1.0]))
    state = braked
    for _ in range(max(1, int(round(KICK_SECONDS * FREQUENCY)))):
        state = step(dyn, state, delay(sign * kick))
    if not all(math.isfinite(s) and abs(s) < RESET_MAX_STATE for s in state):
        state = list(braked)
    return state


def simulate(theta: np.ndarray, instance: Instance, rng: np.random.Generator,
             t_max: int | None = None) -> Trajectory:
    """One Furuta trial. ``theta = [mu, E_bar, q1, q2, q3, q4, r, K_P_barrier]``."""
    spec = cached_spec(NAME)
    theta = check_theta(spec, theta)
    T = horizon(spec, t_max)
    dyn = Dynamics(instance.phi)
    delay = DeadTime(float(instance.extras.get("dead_time", 0.0)) * FREQUENCY)
    kick = float(instance.extras.get("kick", 0.0))
    try:
        ctrl: Controller | None = Controller(instance.phi_hat or instance.phi, theta)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        ctrl = None

    state = reset(dyn, delay, kick, rng)
    sa = float(instance.noise_std.get("angle", 0.0))
    sv = float(instance.noise_std.get("velocity", 0.0))
    su = float(instance.noise_std.get("voltage", 0.0))
    noise = np.zeros((T, 5))
    if sa > 0:
        noise[:, :2] = rng.standard_normal((T, 2)) * sa
    if sv > 0:
        noise[:, 2:4] = rng.standard_normal((T, 2)) * sv
    if su > 0:
        noise[:, 4] = rng.standard_normal(T) * su
    noise_l = noise.tolist()

    obs_log = np.empty((T, 4), dtype=np.float64)
    theta_cont = prev = state[0]
    crash_step = None
    n = T
    for i in range(T):
        nz = noise_l[i]
        o0, o1 = state[0] + nz[0], state[1] + nz[1]
        o2, o3 = state[2] + nz[2], state[3] + nz[3]
        obs_log[i] = (o0, o1, o2, o3)
        if ctrl is None:
            crash_step = n = i + 1
            break
        volts = delay(ctrl.action(o0, o1, o2, o3)) + nz[4]
        state = step(dyn, state, volts)
        if not all(math.isfinite(s) for s in state):
            crash_step = n = i + 1
            break
        theta_cont += _wrap(state[0] - prev)
        prev = state[0]
        if abs(theta_cont) > THETA_SAFE:
            crash_step = n = i + 1
            break

    # a diverging plant can reach finite rates beyond float32 before it crashes, so the logged
    # channels are clipped to RESET_MAX_STATE (the angles are wrapped and never reach it)
    obs = np.clip(obs_log[:n], -RESET_MAX_STATE, RESET_MAX_STATE).astype(np.float32)
    alpha = np.abs(obs[:, 1].astype(np.float64))
    j = alpha * (1.0 + HOLD_WEIGHT * (1.0 - np.minimum(alpha, UPRIGHT) / UPRIGHT))
    hold_n = min(n, int(round(HOLD_SECONDS * FREQUENCY)))
    success = crash_step is None and float(alpha[n - hold_n:n].mean()) < UPRIGHT
    return finish(theta, j, obs[:, :2], obs[:, 2:], T, crash_step, success)
