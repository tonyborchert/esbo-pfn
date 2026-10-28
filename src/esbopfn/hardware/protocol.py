"""The JSON messages between :class:`~esbopfn.hardware.client.RigClient` and the rig server.

Every request is one JSON object with a ``cmd`` field, every reply one JSON object with a
``type`` field. The state is ``[phi_arm, phi_pend, phidot_arm, phidot_pend]`` in rad and rad/s,
with ``phi_arm`` in the board's encoder frame (zeroed when the board opens) and ``phi_pend``
measured from upright, wrapped to ``(-pi, pi]`` (hanging is ``+-pi``).

Requests:

- ``{"cmd": "hello"}`` -> ``{"type": "hello", "plant": "sim"|"qube", "frequency": 120.0,
  "protocol": 1, "reseed_state": bool}`` (``reseed_state``: the sim plant restarts every
  trial from a seeded rest state)
- ``{"cmd": "reset_down"}`` opens the board (zeroing the encoders) and waits for the pendulum to
  hang, -> ``state``
- ``{"cmd": "step", "voltage": v}`` applies ``v`` volts for one control period, -> ``state``
- ``{"cmd": "center", "u_max": 1.0, "max_seconds": 15.0, "tol": 0.02}`` drives the arm to the
  board zero with a bounded PD loop on the server, -> ``state``
- ``{"cmd": "begin_trial", "seed": s}`` announces a trial. The ``sim`` plant reseeds its noise
  (and, by default, its rest state) from ``s``, the ``qube`` plant ignores it. -> ``ok``
- ``{"cmd": "close"}`` writes 0 V and releases the board, -> ``closed``

Replies: ``{"type": "state", "state": [...]}``, ``{"type": "ok"}``, ``{"type": "closed"}``,
``{"type": "error", "message": "..."}``.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
COMMANDS = ("hello", "reset_down", "step", "center", "begin_trial", "close")
REPLY_TYPES = ("hello", "state", "ok", "closed", "error")


class ProtocolError(ValueError):
    """A message that does not follow the protocol."""


def _check_number(name: str, value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ProtocolError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(v):
        raise ProtocolError(f"{name} must be finite, got {value!r}")
    return v


def encode_request(cmd: str, **fields: Any) -> str:
    """JSON text of a request."""
    if cmd not in COMMANDS:
        raise ProtocolError(f"unknown command {cmd!r}, known: {', '.join(COMMANDS)}")
    msg = {"cmd": cmd, **fields}
    validate_request(msg)
    return json.dumps(msg, separators=(",", ":"))


def validate_request(msg: Any) -> dict[str, Any]:
    """Check a decoded request and return it with numeric fields as floats."""
    if not isinstance(msg, dict) or "cmd" not in msg:
        raise ProtocolError("a request is a JSON object with a 'cmd' field")
    cmd = msg["cmd"]
    if cmd not in COMMANDS:
        raise ProtocolError(f"unknown command {cmd!r}, known: {', '.join(COMMANDS)}")
    out = dict(msg)
    if cmd == "step":
        if "voltage" not in msg:
            raise ProtocolError("step needs a 'voltage'")
        out["voltage"] = _check_number("voltage", msg["voltage"])
    elif cmd == "center":
        for name, default in (("u_max", 1.0), ("max_seconds", 15.0), ("tol", 0.02)):
            out[name] = _check_number(name, msg.get(name, default))
    elif cmd == "begin_trial":
        if not isinstance(msg.get("seed"), int) or isinstance(msg.get("seed"), bool):
            raise ProtocolError("begin_trial needs an integer 'seed'")
    return out


def decode_request(text: str | bytes) -> dict[str, Any]:
    """Parse and validate a request."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"not JSON: {exc}") from None
    return validate_request(msg)


def state_reply(state: Any) -> str:
    """JSON text of a ``state`` reply."""
    arr = np.asarray(state, dtype=np.float64).reshape(-1)
    if arr.shape != (4,):
        raise ProtocolError(f"a state has 4 entries, got {arr.shape}")
    return json.dumps({"type": "state", "state": [float(v) for v in arr]})


def reply(kind: str, **fields: Any) -> str:
    """JSON text of a reply of type ``kind``."""
    if kind not in REPLY_TYPES:
        raise ProtocolError(f"unknown reply type {kind!r}")
    return json.dumps({"type": kind, **fields})


def decode_reply(text: str | bytes) -> dict[str, Any]:
    """Parse a reply. A ``state`` reply gets its state as a float64 array."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"reply is not JSON: {exc}") from None
    if not isinstance(msg, dict) or msg.get("type") not in REPLY_TYPES:
        raise ProtocolError(f"malformed reply {msg!r}")
    if msg["type"] == "state":
        state = np.asarray(msg.get("state"), dtype=np.float64).reshape(-1)
        if state.shape != (4,):
            raise ProtocolError(f"a state has 4 entries, got {state.shape}")
        msg["state"] = state
    return msg
