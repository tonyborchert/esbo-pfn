"""Plant backends of the rig server: the simulated rig and the QUBE through ``gym_brt``.

A backend offers ``open()``, ``reset_down() -> state``, ``step(voltage) -> state``,
``begin_trial(seed)``, ``release()`` and the attributes ``name`` and ``frequency``. Voltages are
clipped to ``max_voltage`` by the server before they reach a backend.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from esbopfn.core import Instance

log = logging.getLogger(__name__)

GYM_BRT_HINT = (
    "the qube plant needs BlueRiverTech's gym_brt driver (quanser-openai-driver) built against "
    "the Quanser HIL SDK, which only exists on the rig workstation. See docs/HARDWARE.md."
)


class Backend(Protocol):
    name: str
    frequency: float

    def open(self) -> None: ...
    def reset_down(self) -> np.ndarray: ...
    def step(self, voltage: float) -> np.ndarray: ...
    def begin_trial(self, seed: int) -> None: ...
    def release(self) -> None: ...


class SimBackend:
    """The :class:`~esbopfn.hardware.sim_plant.SimPlant`, optionally paced at real time.

    The plant persists across connections, like the physical rig.
    """

    name = "sim"

    def __init__(self, instance: Instance | None = None, *, seed: int = 0,
                 realtime: bool = False, reseed_state: bool = True) -> None:
        from esbopfn.hardware.sim_plant import SimPlant

        self.plant = SimPlant(instance, seed, reseed_state=reseed_state)
        self.frequency = self.plant.frequency
        self.realtime = bool(realtime)
        self._next = 0.0
        self.opened = False

    def _pace(self) -> None:
        if not self.realtime:
            return
        now = time.perf_counter()
        if self._next > now:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + 1.0 / self.frequency

    def open(self) -> None:
        self.opened = True

    def reset_down(self) -> np.ndarray:
        self.opened = True
        return self.plant.reset_down()

    def step(self, voltage: float) -> np.ndarray:
        self._pace()
        return self.plant.step(voltage)

    def begin_trial(self, seed: int) -> None:
        self.plant.begin_trial(seed)

    def release(self) -> None:
        if self.opened:
            self.plant.step(0.0)
        self.opened = False


class QubeBackend:
    """The QUBE-Servo 2 through ``gym_brt.quanser.QubeHardware`` (documented driver API:
    a context manager that opens the board and zeroes the encoders, ``reset_down()`` and
    ``step(action)`` returning ``[theta, alpha, theta_dot, alpha_dot]`` at the given frequency,
    with ``alpha`` from upright). Written against that API, not tested without the rig."""

    name = "qube"

    def __init__(self, frequency: float = 120.0, max_voltage: float = 18.0) -> None:
        try:
            from gym_brt.quanser import QubeHardware  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(GYM_BRT_HINT) from exc
        self._cls = QubeHardware
        self.frequency = float(frequency)
        self.max_voltage = float(max_voltage)
        self._qube: Any = None

    def open(self) -> None:
        if self._qube is None:
            qube = self._cls(frequency=self.frequency, max_voltage=self.max_voltage)
            qube.__enter__()
            self._qube = qube

    def reset_down(self) -> np.ndarray:
        self.release()          # reopening the board zeroes the encoders where the arm is
        self.open()
        return np.asarray(self._qube.reset_down(), dtype=np.float64).reshape(-1)[:4]

    def step(self, voltage: float) -> np.ndarray:
        if self._qube is None:
            raise RuntimeError("board is not open (send reset_down first)")
        state = self._qube.step(np.array([float(voltage)], dtype=np.float64))
        return np.asarray(state, dtype=np.float64).reshape(-1)[:4]

    def begin_trial(self, seed: int) -> None:
        """The physical rig has nothing to seed."""

    def release(self) -> None:
        qube, self._qube = self._qube, None
        if qube is None:
            return
        try:
            qube.step(np.array([0.0]))
        finally:
            qube.__exit__(None, None, None)


def load_instance(path: str | Path) -> Instance:
    """An :class:`Instance` from JSON (``phi``, ``phi_hat``, ``noise_std``, ``extras``, ``seed``)."""
    data = json.loads(Path(path).read_text())
    return Instance(phi={k: float(v) for k, v in data["phi"].items()},
                    phi_hat={k: float(v) for k, v in data.get("phi_hat", {}).items()},
                    noise_std={k: float(v) for k, v in data.get("noise_std", {}).items()},
                    extras={k: float(v) for k, v in data.get("extras", {}).items()},
                    seed=int(data.get("seed", 0)))
