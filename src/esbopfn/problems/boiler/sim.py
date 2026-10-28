"""Boiler closed loop (port of the ESBO benchmark ``getBoilerBangBang1Var`` / ``sldemo_boiler``).

Plant: ``dT/dt = rate / 25`` with ``rate = +heating_rate`` while the burner is on and
``-cooling_rate`` while it is off, ``T(0) = 15 degC``, explicit Euler at 1 s. The thermometer adds
Gaussian sensor noise to the temperature and quantizes the sum with an 8-bit ADC (step
0.390625 degC, offset -15 degC). The bang-bang chart, clocked at 1 s, switches on once it has been
off for at least the turn-on delay ``theta`` and the reading is at or below 20 degC, and switches
off once the reading exceeds 20 degC or after 100 s on. Logging every 2 s over 1400 s gives 701
steps. Stage cost ``0.5 min(|20 - T_meas|, 2)^2 + u + 8 [u_t > u_{t-1}]`` on the logged samples.
"""

from __future__ import annotations

import math

import numpy as np

from esbopfn.core import Instance, Trajectory
from esbopfn.problems.base import cached_spec, check_theta, finish, horizon

NAME = "boiler"

T_INIT = 15.0
PLANT_GAIN = 1.0 / 25.0
ADC_STEP = 0.390625
ADC_BIAS = -15.0
SET_POINT = 20.0
SAFETY_OFF_TIME = 100.0
LOG_EVERY = 2  # 1 s plant and controller steps, logged every 2 s
COMFORT_CLIP = 2.0
SWITCH_PENALTY = 8.0


def adc(temp: float) -> float:
    """8-bit thermometer quantization (floor to the 0.390625 degC grid offset by -15 degC)."""
    return ADC_BIAS + ADC_STEP * math.floor((temp - ADC_BIAS) / ADC_STEP)


def simulate(theta: np.ndarray, instance: Instance, rng: np.random.Generator,
             t_max: int | None = None) -> Trajectory:
    """One boiler trial. ``theta = [turn_on_delay]`` in seconds."""
    spec = cached_spec(NAME)
    theta = check_theta(spec, theta)
    T = horizon(spec, t_max)
    delay = float(theta[0])
    heat = float(instance.phi["heating_rate"])
    cool = float(instance.phi["cooling_rate"])
    std = float(instance.noise_std.get("temperature", 0.0))
    n_steps = LOG_EVERY * (T - 1)
    noise = (rng.standard_normal(n_steps + 1) * std).tolist() if std > 0 else [0.0] * (n_steps + 1)

    meas_log = np.empty(T, dtype=np.float64)
    u_log = np.empty(T, dtype=np.float64)
    on = False
    time_in_state = 0.0
    temp = T_INIT
    for k in range(n_steps + 1):
        if k:
            temp += PLANT_GAIN * (heat if on else -cool)
        meas = adc(temp + noise[k])
        cold = meas <= SET_POINT
        if not on:
            if time_in_state >= delay and cold:
                on, time_in_state = True, 0.0
        elif (not cold) or time_in_state >= SAFETY_OFF_TIME:
            on, time_in_state = False, 0.0
        time_in_state += 1.0
        if k % LOG_EVERY == 0:
            i = k // LOG_EVERY
            meas_log[i] = meas
            u_log[i] = 1.0 if on else 0.0

    comfort = 0.5 * np.minimum(np.abs(SET_POINT - meas_log), COMFORT_CLIP) ** 2
    switched = np.zeros(T)
    switched[1:] = u_log[1:] > u_log[:-1]
    j = comfort + u_log + SWITCH_PENALTY * switched
    return finish(theta, j, meas_log, None, T, None)
