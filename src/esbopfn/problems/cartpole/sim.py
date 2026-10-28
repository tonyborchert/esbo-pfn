"""Cartpole closed loop (Gymnasium cart-pole dynamics, explicit Euler at 0.02 s).

Plant: the classic Barto-Sutton-Anderson cart-pole with cart mass, pole mass, pole half-length and
maximum force from the instance, gravity 9.8, start state ``U(-0.05, 0.05)^4``. Controller: two
parallel PD loops on the measured cart position error ``x_ref - x`` and pole angle error
``-phi``, derivatives as per-step error differences, and a bang-bang force: ``+max_force`` when the
summed output is negative and ``-max_force`` otherwise (the sign convention of the ported code).
The controller acts on the measurement of the previous step. The position goal steps between
``+-2 m`` (0 for 100 steps, then +2, -2, +2, -2 for 200, 300, 200, 200 steps). Sensor noise is
added to all four state channels, which are read as float32 as in the Gymnasium environment.
Crash: cart outside ``+-3.5 m`` or pole outside ``+-30 deg``. Stage cost
``10 tanh^2(x_ref - x) + xdot^2 + phi^2 + phidot^2`` on the measured channels.
"""

from __future__ import annotations

import math

import numpy as np

from esbopfn.core import Instance, Trajectory
from esbopfn.problems.base import cached_spec, check_theta, finish, horizon

NAME = "cartpole"

TAU = 0.02
GRAVITY = 9.8
X_LIMIT = 3.5
PHI_LIMIT = 30.0 * 2.0 * math.pi / 360.0
GOALS = np.array([0.0] * 100 + [2.0] * 200 + [-2.0] * 300 + [2.0] * 200 + [-2.0] * 200,
                 dtype=np.float32)
NOISE_KEYS = ("x_cart", "xdot_cart", "phi_pole", "phidot_pole")


def simulate(theta: np.ndarray, instance: Instance, rng: np.random.Generator,
             t_max: int | None = None) -> Trajectory:
    """One cartpole trial. ``theta = [K_P_cart, K_D_cart, K_P_pole, K_D_pole]``."""
    spec = cached_spec(NAME)
    theta = check_theta(spec, theta)
    T = horizon(spec, t_max)
    kp_c, kd_c, kp_p, kd_p = (float(v) for v in theta)
    mc = float(instance.phi["cart_mass"])
    mp = float(instance.phi["pole_mass"])
    length = float(instance.phi["pole_half_length"])
    force_mag = float(instance.phi["max_force"])
    total_mass = mp + mc
    pml = mp * length
    std = np.array([instance.noise_std.get(k, 0.0) for k in NOISE_KEYS], dtype=np.float64)

    x, xd, th, thd = (float(v) for v in rng.uniform(-0.05, 0.05, size=4))
    if np.any(std > 0):
        noise = (rng.standard_normal((T, 4)) * std).astype(np.float32)
    else:
        noise = np.zeros((T, 4), dtype=np.float32)
    goals = GOALS[:T]
    # noise[:, 0] - goal, to which the float32 observation is added: the position error reading
    pos_base = noise[:, 0] - goals

    meas = np.empty((T, 4), dtype=np.float32)  # x, xdot, phi, phidot (measured)
    pos_err = np.empty(T, dtype=np.float32)    # measured x - goal
    obs = np.empty(4, dtype=np.float32)
    last_ec = last_ep = np.float32(0.0)
    crash_step = None
    n = T
    for i in range(T):
        if i:
            ec = -pos_err[i - 1]
            ep = -meas[i - 1, 2]
            u_cart = kp_c * ec + kd_c * (ec - last_ec)
            u_pole = kp_p * ep + kd_p * (ep - last_ep)
            last_ec, last_ep = ec, ep
            force = force_mag if u_pole + u_cart < 0 else -force_mag
            cos_t, sin_t = math.cos(th), math.sin(th)
            temp = (force + pml * (thd * thd) * sin_t) / total_mass
            thacc = (GRAVITY * sin_t - cos_t * temp) / (
                length * (4.0 / 3.0 - mp * (cos_t * cos_t) / total_mass))
            xacc = temp - pml * thacc * cos_t / total_mass
            x = x + TAU * xd
            xd = xd + TAU * xacc
            th = th + TAU * thd
            thd = thd + TAU * thacc
        obs[0], obs[1], obs[2], obs[3] = x, xd, th, thd
        pos_err[i] = pos_base[i] + obs[0]
        meas[i] = noise[i] + obs
        if i and (x < -X_LIMIT or x > X_LIMIT or th < -PHI_LIMIT or th > PHI_LIMIT):
            crash_step = i + 1
            n = i + 1
            break

    m = meas[:n].astype(np.float64)
    j = (10.0 * np.tanh(pos_err[:n].astype(np.float64)) ** 2 + m[:, 1] ** 2 + m[:, 2] ** 2
         + m[:, 3] ** 2)
    return finish(theta, j, meas[:n, [0, 2]], meas[:n, [1, 3]], T, crash_step)
