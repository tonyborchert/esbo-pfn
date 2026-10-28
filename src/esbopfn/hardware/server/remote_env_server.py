"""Websocket server that exposes one plant (the QUBE rig or its simulation) to a rig client.

    python -m esbopfn.hardware.server --plant {qube,sim} --port 8765 [--instance-json FILE]

One plant, one client at a time (the board admits one process). The server holds no controller
logic: it executes ``reset_down``, ``step``, ``center``, ``begin_trial`` and ``close``
(``esbopfn.hardware.protocol``). Two safety nets: commanded voltages are clipped to
``--max-voltage``, and more than ``--max-motor-steps`` consecutive steps with a nonzero voltage
are refused (0 V is written and an error returned). A dropped connection releases the plant.
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from esbopfn.hardware import protocol
from esbopfn.hardware.server.backends import Backend, QubeBackend, SimBackend, load_instance

log = logging.getLogger(__name__)

DEFAULT_PORT = 8765
DEFAULT_MAX_VOLTAGE = 6.0
DEFAULT_MAX_MOTOR_STEPS = 3600


class Session:
    """Executes requests on a backend and enforces the safety limits."""

    def __init__(self, backend: Backend, max_voltage: float = DEFAULT_MAX_VOLTAGE,
                 max_motor_steps: int = DEFAULT_MAX_MOTOR_STEPS) -> None:
        self.backend = backend
        self.max_voltage = float(max_voltage)
        self.max_motor_steps = int(max_motor_steps)
        self.motor_steps = 0
        self.opened = False

    def _step(self, voltage: float) -> np.ndarray:
        if not self.opened:
            raise RuntimeError("board is not open (send reset_down first)")
        v = min(max(float(voltage), -self.max_voltage), self.max_voltage)
        if v != 0.0:
            self.motor_steps += 1
            if self.motor_steps > self.max_motor_steps:
                self.backend.step(0.0)
                self.motor_steps = 0
                raise RuntimeError(f"more than {self.max_motor_steps} consecutive motor-on steps, "
                                   "0 V written")
        else:
            self.motor_steps = 0
        return self.backend.step(v)

    def _center(self, u_max: float, max_seconds: float, tol: float) -> np.ndarray:
        """Bounded PD drive of the arm to the board zero, ending at 0 V."""
        state = self._step(0.0)
        n = int(max_seconds * self.backend.frequency)
        try:
            for _ in range(n):
                arm, arm_dot = float(state[0]), float(state[2])
                if abs(arm) <= tol and abs(arm_dot) < 0.05:
                    break
                u = float(np.clip(3.0 * arm + 0.2 * arm_dot, -u_max, u_max))
                state = self._step(u)
            else:
                raise RuntimeError(f"center did not converge within {max_seconds:g} s")
        finally:
            state = self._step(0.0)
        return state

    def handle(self, text: str | bytes) -> str:
        """One request in, one reply out (errors become ``error`` replies)."""
        try:
            msg = protocol.decode_request(text)
            cmd = msg["cmd"]
            if cmd == "hello":
                plant = getattr(self.backend, "plant", None)
                return protocol.reply("hello", plant=self.backend.name,
                                      frequency=float(self.backend.frequency),
                                      protocol=protocol.PROTOCOL_VERSION,
                                      reseed_state=bool(getattr(plant, "reseed_state", False)))
            if cmd == "reset_down":
                state = self.backend.reset_down()
                self.opened = True
                self.motor_steps = 0
                return protocol.state_reply(state)
            if cmd == "step":
                return protocol.state_reply(self._step(msg["voltage"]))
            if cmd == "center":
                return protocol.state_reply(
                    self._center(msg["u_max"], msg["max_seconds"], msg["tol"]))
            if cmd == "begin_trial":
                self.backend.begin_trial(int(msg["seed"]))
                return protocol.reply("ok")
            if cmd == "close":
                self.release()
                return protocol.reply("closed")
            raise protocol.ProtocolError(f"unhandled command {cmd!r}")
        except Exception as exc:  # noqa: BLE001 - every failure goes back to the client
            log.warning("request failed: %s", exc)
            return protocol.reply("error", message=f"{type(exc).__name__}: {exc}")

    def release(self) -> None:
        try:
            self.backend.release()
        finally:
            self.opened = False
            self.motor_steps = 0


class RigServer:
    """A running server. ``url`` is where clients connect, ``shutdown()`` stops it."""

    def __init__(self, backend: Backend, host: str = "localhost", port: int = DEFAULT_PORT,
                 max_voltage: float = DEFAULT_MAX_VOLTAGE,
                 max_motor_steps: int = DEFAULT_MAX_MOTOR_STEPS) -> None:
        from websockets.sync.server import serve as ws_serve

        self.session = Session(backend, max_voltage, max_motor_steps)
        self._lock = threading.Lock()
        self._server = ws_serve(self._handler, host, port, ping_interval=None, max_size=None)
        self.host = host
        self.port = int(self._server.socket.getsockname()[1])
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def _handler(self, ws: Any) -> None:
        if not self._lock.acquire(blocking=False):
            ws.send(protocol.reply("error", message="another client holds the rig"))
            ws.close()
            return
        try:
            for text in ws:
                ws.send(self.session.handle(text))
        except Exception as exc:  # noqa: BLE001 - a dropped client must release the plant
            log.info("client connection ended: %s", exc)
        finally:
            self.session.release()
            self._lock.release()

    def serve_forever(self) -> None:
        log.info("rig server (%s plant) listening on %s", self.session.backend.name, self.url)
        self._server.serve_forever()

    def start(self) -> RigServer:
        """Serve in a daemon thread (tests, in-process use)."""
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def shutdown(self) -> None:
        self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5)


def make_backend(plant: str, *, instance_json: str | None = None, seed: int = 0,
                 realtime: bool = False, reseed_state: bool = True,
                 frequency: float = 120.0) -> Backend:
    """The backend for ``plant`` (``sim`` or ``qube``)."""
    if plant == "sim":
        instance = load_instance(instance_json) if instance_json else None
        return SimBackend(instance, seed=seed, realtime=realtime, reseed_state=reseed_state)
    if plant == "qube":
        if instance_json:
            raise ValueError("--instance-json only applies to the sim plant")
        return QubeBackend(frequency=frequency)
    raise ValueError(f"unknown plant {plant!r} (sim or qube)")


def serve(plant: str = "sim", host: str = "localhost", port: int = DEFAULT_PORT,
          **kwargs: Any) -> RigServer:
    """Build a server for ``plant`` (not started). ``port=0`` picks a free port."""
    limits = {k: kwargs.pop(k) for k in ("max_voltage", "max_motor_steps") if k in kwargs}
    return RigServer(make_backend(plant, **kwargs), host, port, **limits)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m esbopfn.hardware.server",
                                description="websocket server for the QUBE rig or its simulation")
    add_arguments(p)
    return p


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--plant", choices=("qube", "sim"), default="sim")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--instance-json", default=None,
                   help="sim plant parameters (phi, phi_hat, noise_std, extras) as JSON")
    p.add_argument("--seed", type=int, default=0, help="sim plant noise seed")
    p.add_argument("--realtime", action="store_true", help="pace the sim plant at 120 Hz")
    p.add_argument("--keep-state", action="store_true",
                   help="sim plant: do not restart from a seeded rest state at begin_trial")
    p.add_argument("--max-voltage", type=float, default=DEFAULT_MAX_VOLTAGE)
    p.add_argument("--max-motor-steps", type=int, default=DEFAULT_MAX_MOTOR_STEPS)


def run_from_args(args: argparse.Namespace) -> int:
    if not math.isfinite(args.max_voltage) or args.max_voltage <= 0:
        raise SystemExit("--max-voltage must be positive")
    server = serve(args.plant, args.host, args.port, instance_json=args.instance_json,
                   seed=args.seed, realtime=args.realtime, reseed_state=not args.keep_state,
                   max_voltage=args.max_voltage, max_motor_steps=args.max_motor_steps)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.session.release()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return run_from_args(build_parser().parse_args(argv))
