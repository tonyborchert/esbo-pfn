"""Three-tank closed loop (port of the ESBO benchmark ``getThreeTankPI2Var`` / ``Dreitank.slx``).

Plant: tanks B102, B103, B104 of cross-section 0.029 m^2, connected by signed Torricelli throttles
(pipe 600e-6 m^2, V101 opening mapped through its identified valve table, B103 to B104 fully open).
The pump P101 feeds B102 through its identified map ``Q(U, B101 volume)``, a first-order lag of
1.2341 s and a transport delay of 1.1 s. B104 drains through valve V106 with its identified outflow
map. Volumes saturate at ``[0, 10e-3] m^3``, explicit Euler at 0.1 s.

Controller: the tuned outer PI ``theta = (K_P, K_I)`` acts on the measured level error of B104 and
feeds a fixed inner PI (``30e3, 0.6e3``) on the pump flow, offset by 50 %. The set point steps from
0 to ``3e-3 m^3`` at 5 s, where logging starts, and the run ends at 100 s: 476 logged steps every
0.2 s. Sensor noise is added to the B104 level that the controller reads and that is logged.
Stage cost ``(V_ref - V4_meas)^2``.
"""

from __future__ import annotations

import math

import numpy as np

from esbopfn.core import Instance, Trajectory
from esbopfn.problems.base import cached_spec, check_theta, finish, horizon

NAME = "threetank_pi"

DT = 0.1
REF_STEP = 50  # 5 s / DT
LOG_EVERY = 2  # 0.2 s / DT
A_TANK = 0.029
A_PIPE = 600e-6
V2_INIT, V3_INIT, V4_INIT = 1.48e-3, 1.42e-3, 1.255e-3
V_MAX = 10e-3
G = 9.81
VALVE_B103_B104 = 100.0
PUMP_LAG_TAU = 1.2341
PUMP_DELAY_STEPS = 11  # 1.1 s / DT
INNER_P, INNER_I = 30e3, 0.6e3
PUMP_OFFSET = 50.0
V_REF = 3e-3

PUMP_MAP_U = np.arange(50.0, 101.0, 5.0)
PUMP_MAP_VOL = np.array([40e-3, 130e-3])
PUMP_MAP_Q = np.array([
    [0, 0.0606, 0.2450, 0.3378, 0.4184, 0.5002, 0.5801, 0.6551, 0.7138, 0.7746, 0.8206],
    [0, 0.2799, 0.3497, 0.4278, 0.5053, 0.5810, 0.6481, 0.7168, 0.7695, 0.8350, 0.8786],
]) * 1e-3
VALVE_V101_IN = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100.0])
VALVE_V101_OUT = np.array([100, 96, 90, 88, 84, 76, 68, 60, 49, 41, 34, 27, 22, 19, 16, 12, 0.0])
VALVE_V106_U = np.arange(15.0, 86.0, 5.0)
VALVE_V106_Q = np.array([0, 0.032, 0.097, 0.179, 0.262, 0.361, 0.475, 0.604, 0.902, 1.089, 1.318,
                         1.536, 1.685, 1.853, 2.029]) * 1e-3
#: V106 outflow is zero at a B104 level of 2e-3 m^3 and reaches the table value at 2.8e-3 m^3.
V106_V4_LO, V106_V4_HI = 2e-3, 2.8e-3


def _throttle(valve_pct: float, v_up: float, v_down: float) -> float:
    gh = 2.0 * G * (v_up - v_down) / A_TANK
    return (valve_pct / 100.0) * A_PIPE * math.copysign(math.sqrt(abs(gh)), gh)


def simulate(theta: np.ndarray, instance: Instance, rng: np.random.Generator,
             t_max: int | None = None) -> Trajectory:
    """One three-tank trial. ``theta = [K_P, K_I]`` of the outer level PI."""
    spec = cached_spec(NAME)
    theta = check_theta(spec, theta)
    T = horizon(spec, t_max)
    kp, ki = float(theta[0]), float(theta[1])
    u_v101 = float(instance.phi["V101_opening"])
    u_v106 = float(instance.phi["V106_opening"])
    vol = float(instance.phi["B101_volume"])
    std = float(instance.noise_std.get("V4", 0.0))

    valve_23 = float(np.interp(u_v101, VALVE_V101_IN, VALVE_V101_OUT))
    w = (min(max(vol, PUMP_MAP_VOL[0]), PUMP_MAP_VOL[1]) - PUMP_MAP_VOL[0]) / (
        PUMP_MAP_VOL[1] - PUMP_MAP_VOL[0])
    pump_q = (PUMP_MAP_Q[0] + w * (PUMP_MAP_Q[1] - PUMP_MAP_Q[0])).tolist()
    q106 = float(np.interp(min(max(u_v106, VALVE_V106_U[0]), VALVE_V106_U[-1]), VALVE_V106_U,
                           VALVE_V106_Q))

    last = REF_STEP + LOG_EVERY * (T - 1)
    noise = (rng.standard_normal(last + 1) * std).tolist() if std > 0 else [0.0] * (last + 1)
    logs = np.empty((T, 3), dtype=np.float64)
    v2, v3, v4 = V2_INIT, V3_INIT, V4_INIT
    line = [0.0] * PUMP_DELAY_STEPS
    head = 0
    lag = q_pump = 0.0
    outer_int = inner_int = 0.0
    for k in range(last + 1):
        ref = V_REF if k >= REF_STEP else 0.0
        v4_meas = v4 + noise[k]
        if k >= REF_STEP and (k - REF_STEP) % LOG_EVERY == 0:
            logs[(k - REF_STEP) // LOG_EVERY] = (v2, v3, v4_meas)
        if k == last:
            break
        e = ref - v4_meas
        o = kp * e + ki * outer_int
        outer_int += DT * e
        e_in = o - q_pump
        u_pump = PUMP_OFFSET + INNER_P * e_in + INNER_I * inner_int
        inner_int += DT * e_in
        # pump map (clipped to its 50..100 % table), first-order lag, transport delay
        u = min(max(u_pump, 50.0), 100.0)
        idx = min(int((u - 50.0) / 5.0), 9)
        q_map = pump_q[idx] + (pump_q[idx + 1] - pump_q[idx]) * (u - (50.0 + 5.0 * idx)) / 5.0
        lag += DT * (q_map - lag) / PUMP_LAG_TAU
        q_pump = line[head]
        line[head] = lag
        head = (head + 1) % PUMP_DELAY_STEPS
        q23 = _throttle(valve_23, v2, v3)
        q34 = _throttle(VALVE_B103_B104, v3, v4)
        v4c = min(max(v4, V106_V4_LO), V_MAX)
        q_out = q106 * min(1.0, (v4c - V106_V4_LO) / (V106_V4_HI - V106_V4_LO))
        v2 = min(max(v2 + DT * (q_pump - q23), 0.0), V_MAX)
        v3 = min(max(v3 + DT * (q23 - q34), 0.0), V_MAX)
        v4 = min(max(v4 + DT * (q34 - q_out), 0.0), V_MAX)

    j = (V_REF - logs[:, 2]) ** 2
    return finish(theta, j, logs, None, T, None)
