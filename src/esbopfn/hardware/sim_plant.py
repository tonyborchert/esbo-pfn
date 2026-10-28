"""A stepwise Furuta plant that stands in for the rig.

``esbopfn.problems.furuta.sim`` has no stepwise entry point (``simulate`` runs a whole closed-loop
trial), so this module is a thin stepper on its internals, used unchanged: ``Dynamics`` (the
QUBE-Servo 2 equations), ``step`` (one 1/120 s control period, ten semi-implicit Euler
sub-steps, voltage clipped to 18 V), ``DeadTime`` (the fractional actuation delay) and the
constants. What the rig adds around the plant is modeled here:

- encoders: the arm is read relative to an encoder offset that is set when the board opens
  (``reset_down``), both angles are quantized to 2048 counts per revolution,
- sensor and actuation noise from the instance's ``noise_std`` (zero by default, so that the
  settle tests of the reset can pass exactly as on the quiet rig),
- ``reset_down`` brings the pendulum to hanging rest without moving the arm and re-zeroes the
  arm encoder there, like opening the real board,
- ``begin_trial(seed)`` reseeds the noise and, with ``reseed_state``, restarts the plant from a
  seeded rest state (hanging plus a 0.01 residual, arm at the board zero), which makes a
  simulated field reproducible trial by trial and so resumable bit for bit.
"""

from __future__ import annotations

import math

import numpy as np

from esbopfn.core import Instance
from esbopfn.problems.furuta import sim
from esbopfn.util.seeds import substream

ENCODER_COUNTS = 2048
_COUNT = 2.0 * math.pi / ENCODER_COUNTS


def nominal_rig_instance() -> Instance:
    """The furuta prior's nominal plant without noise (default plant of the ``sim`` server)."""
    from esbopfn.problems import get_problem

    return get_problem("furuta").nominal_instance(noise=False)


def _quantize(a: float) -> float:
    return round(a / _COUNT) * _COUNT


class SimPlant:
    """One simulated rig. ``state()`` is what the server reports (see ``hardware.protocol``)."""

    frequency = sim.FREQUENCY

    def __init__(self, instance: Instance | None = None, seed: int = 0, *,
                 quantize: bool = True, reseed_state: bool = True) -> None:
        self.instance = instance if instance is not None else nominal_rig_instance()
        self.dyn = sim.Dynamics(self.instance.phi)
        self.dead_time_s = float(self.instance.extras.get("dead_time", 0.0))
        self.quantize = bool(quantize)
        self.reseed_state = bool(reseed_state)
        self.rng = substream(seed, "sim-plant")
        self.x = [0.0, math.pi, 0.0, 0.0]    # true [theta, alpha, theta_dot, alpha_dot]
        self.encoder_offset = 0.0
        self.delay = sim.DeadTime(self.dead_time_s * sim.FREQUENCY)
        self.steps = 0

    # -- noise ------------------------------------------------------------------------------
    def _std(self, name: str) -> float:
        return float(self.instance.noise_std.get(name, 0.0))

    def state(self) -> np.ndarray:
        """Measured ``[phi_arm, phi_pend, phidot_arm, phidot_pend]``."""
        theta, alpha, theta_dot, alpha_dot = self.x
        arm = sim._wrap(theta - self.encoder_offset)
        sa, sv = self._std("angle"), self._std("velocity")
        if sa > 0:
            arm += float(self.rng.normal(0.0, sa))
            alpha += float(self.rng.normal(0.0, sa))
        if sv > 0:
            theta_dot += float(self.rng.normal(0.0, sv))
            alpha_dot += float(self.rng.normal(0.0, sv))
        if self.quantize:
            arm, alpha = _quantize(arm), _quantize(alpha)
        return np.array([sim._wrap(arm), sim._wrap(alpha), theta_dot, alpha_dot])

    # -- commands ---------------------------------------------------------------------------
    def reset_down(self) -> np.ndarray:
        """Open the board: pendulum at hanging rest, arm where it is, encoders zeroed there."""
        self.x = [self.x[0], math.pi, 0.0, 0.0]
        if not all(math.isfinite(v) for v in self.x):
            self.x = [0.0, math.pi, 0.0, 0.0]
        self.encoder_offset = self.x[0]
        self.delay = sim.DeadTime(self.dead_time_s * sim.FREQUENCY)
        return self.state()

    def begin_trial(self, seed: int) -> None:
        """Reseed the noise and, with ``reseed_state``, restart from a seeded rest state."""
        self.rng = substream(int(seed), "sim-plant")
        if self.reseed_state:
            r = self.rng.standard_normal(4) * sim.RESIDUAL_STD
            self.x = [sim._wrap(self.encoder_offset + float(r[0])),
                      sim._wrap(math.pi + float(r[1])), float(r[2]), float(r[3])]
            self.delay = sim.DeadTime(self.dead_time_s * sim.FREQUENCY)

    def step(self, voltage: float) -> np.ndarray:
        """Apply ``voltage`` for one control period (through the dead time, plus noise)."""
        v = self.delay(float(voltage))
        su = self._std("voltage")
        if su > 0:
            v += float(self.rng.normal(0.0, su))
        nxt = sim.step(self.dyn, self.x, v)
        if not all(math.isfinite(s) for s in nxt):
            nxt = [self.x[0], math.pi, 0.0, 0.0]   # a diverged plant is caught and rests
        self.x = nxt
        self.steps += 1
        return self.state()
