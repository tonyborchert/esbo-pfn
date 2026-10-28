"""PT2 dead-time closed loop (port of the ESBO benchmark ``getPT2DeatTimePID3Var``).

Plant: ``G(s) = K w0^2 / (s^2 + 2 D w0 s + w0^2)`` with a 0.1 s input dead time, explicit Euler at
0.01 s. Controller: parallel PID with first-order derivative filter ``N = 100``, forward Euler,
gains ``10**theta``, on the measured error to the unit step ``w = 1``. Logging every 0.1 s over
50 s gives 501 steps. Sensor noise is added to the output at every 0.01 s step, the controller
reads it and the logged value is the measured output. Crash: ``y_meas - w > 2`` at a logged step
(a non-finite output, never observed on the prior, also counts as a crash).
Stage cost (discrete ITAE) ``tau_t |w - y_meas|`` with ``tau_t = (t - 1) 0.1 s``.
"""

from __future__ import annotations

import math

import numpy as np

from esbopfn.core import Instance, Trajectory
from esbopfn.problems.base import cached_spec, check_theta, finish, horizon

NAME = "pt2_deadtime"

DT = 0.01
LOG_EVERY = 10  # 0.1 s / DT
T_OBJ = 0.1
DEAD_TIME_STEPS = 10  # 0.1 s / DT
PID_N = 100.0
REFERENCE = 1.0
OVERSHOOT_LIMIT = 2.0


def simulate(theta: np.ndarray, instance: Instance, rng: np.random.Generator,
             t_max: int | None = None) -> Trajectory:
    """One PT2 trial. ``theta = [log10 K_P, log10 K_I, log10 K_D]``."""
    spec = cached_spec(NAME)
    return rollout(check_theta(spec, theta), instance, rng, horizon(spec, t_max), DEAD_TIME_STEPS)


def rollout(theta: np.ndarray, instance: Instance, rng: np.random.Generator, T: int,
            dead_time_steps: int) -> Trajectory:
    """The closed loop with an input delay of ``dead_time_steps`` base steps of 0.01 s."""
    kp, ki, kd = (float(g) for g in 10.0 ** theta)
    K, w0, D = (float(instance.phi[n]) for n in ("K", "w0", "D"))
    std = float(instance.noise_std.get("y", 0.0))
    last = LOG_EVERY * (T - 1)
    noise = (rng.standard_normal(last + 1) * std).tolist() if std > 0 else [0.0] * (last + 1)

    a_u, a_d, a_y = K * w0 * w0, 2.0 * D * w0, w0 * w0
    w = REFERENCE
    y = ydot = 0.0
    integ = dstate = 0.0
    line = [0.0] * dead_time_steps
    head = 0
    ys = np.empty(T, dtype=np.float64)
    crash_step = None
    for k in range(last + 1):
        y_meas = y + noise[k]
        if k % LOG_EVERY == 0:
            i = k // LOG_EVERY
            ys[i] = y_meas
            if y_meas - w > OVERSHOOT_LIMIT or not math.isfinite(y_meas):
                crash_step = i + 1
                break
        if k == last:
            break
        e = w - y_meas
        u = kp * e + ki * integ + kd * PID_N * (e - dstate)
        integ += DT * e
        dstate += DT * PID_N * (e - dstate)
        # dead-time line: the command issued dead_time_steps steps ago leaves as the new one enters
        u_delayed = line[head]
        line[head] = u
        head = (head + 1) % dead_time_steps
        ydot += DT * (a_u * u_delayed - a_d * ydot - a_y * y)
        y += DT * ydot

    n = crash_step if crash_step is not None else T
    j = np.arange(n) * T_OBJ * np.abs(REFERENCE - ys[:n])
    return finish(theta, j, ys[:n], None, T, crash_step)
