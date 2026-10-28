"""Websocket client of the rig server (``esbopfn.hardware.server``).

Every command is request/response with a per-command reply timeout (``step`` 15 s,
``reset_down`` 90 s, ``center`` 60 s, ``close`` 30 s). These are watchdogs, not pacing: a server
wedged inside a blocking driver call becomes a :class:`RigTimeout` instead of a hang. On a timeout
the socket is discarded (the late reply must never be read as the answer to the next request)
and the next command reconnects. Keepalive pings are disabled (``ping_interval=None``) because
the server cannot answer a ping while it blocks in a driver call during a reset.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from esbopfn.hardware import protocol

log = logging.getLogger(__name__)

DEFAULT_URL = "ws://localhost:8765"
REPLY_TIMEOUTS: dict[str, float] = {"step": 15.0, "reset_down": 90.0, "center": 60.0,
                                    "close": 30.0}
DEFAULT_REPLY_TIMEOUT = 30.0


class RigError(RuntimeError):
    """The server reported an error, or the connection failed."""


class RigTimeout(RigError, TimeoutError):
    """The server did not answer within the command's reply timeout."""


class RigClient:
    """``reset_down()``, ``step(voltage) -> state``, ``center()``, ``close()`` on a rig server."""

    def __init__(self, url: str = DEFAULT_URL, open_timeout: float = 5.0,
                 reply_timeouts: dict[str, float] | None = None) -> None:
        self.url = url
        self.open_timeout = float(open_timeout)
        self.reply_timeouts = {**REPLY_TIMEOUTS, **(reply_timeouts or {})}
        self._ws: Any = None
        self.reconnects = 0
        self.connections = 0
        self.info: dict[str, Any] = {}

    # -- connection ---------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._ws is not None

    def connect(self) -> RigClient:
        if self._ws is None:
            from websockets.sync.client import connect

            try:
                ws = connect(self.url, open_timeout=self.open_timeout, ping_interval=None,
                                 max_size=None)
                # the connection is closed explicitly by close() and _discard()
                self._ws = ws.__enter__()
            except Exception as exc:
                raise RigError(f"cannot connect to the rig server at {self.url}: {exc}") from exc
            self.connections += 1
            if self.connections > 1:
                self.reconnects += 1
        return self

    def _discard(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close_socket()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> RigClient:
        return self.connect()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _send(self, cmd: str, **fields: Any) -> dict[str, Any]:
        self.connect()
        timeout = self.reply_timeouts.get(cmd, DEFAULT_REPLY_TIMEOUT)
        try:
            self._ws.send(protocol.encode_request(cmd, **fields))
            raw = self._ws.recv(timeout=timeout)
        except TimeoutError:
            self._discard()
            raise RigTimeout(f"rig server did not answer {cmd!r} within {timeout:g} s (check that "
                             "the board is not held by another process)") from None
        except protocol.ProtocolError:
            raise
        except Exception as exc:
            self._discard()
            raise RigError(f"connection to {self.url} failed during {cmd!r}: {exc}") from exc
        msg = protocol.decode_reply(raw)
        if msg["type"] == "error":
            raise RigError(f"rig server error on {cmd!r}: {msg.get('message')}")
        return msg

    # -- commands ------------------------------------------------------------------------------
    def hello(self) -> dict[str, Any]:
        """Plant name, frequency and protocol version of the server."""
        msg = self._send("hello")
        self.info = {k: v for k, v in msg.items() if k != "type"}
        return self.info

    def reset_down(self) -> np.ndarray:
        """Open (or reopen) the board, zeroing the encoders, and wait for a hanging pendulum."""
        return self._send("reset_down")["state"]

    def step(self, voltage: float) -> np.ndarray:
        """Apply ``voltage`` for one control period and return the new state."""
        return self._send("step", voltage=float(np.asarray(voltage).reshape(-1)[0]))["state"]

    def center(self, u_max: float = 1.0, max_seconds: float = 15.0,
               tol: float = 0.02) -> np.ndarray:
        """Server-side bounded PD drive of the arm to the board zero."""
        return self._send("center", u_max=u_max, max_seconds=max_seconds, tol=tol)["state"]

    def begin_trial(self, seed: int) -> None:
        """Announce a trial (seeds the simulated plant, ignored by the rig)."""
        self._send("begin_trial", seed=int(seed))

    def close(self) -> None:
        """Release the board (0 V) and close the connection. Safe to call twice."""
        if self._ws is None:
            return
        try:
            self._send("close")
        except RigError:
            pass
        finally:
            ws, self._ws = self._ws, None
            if ws is not None:
                try:
                    ws.close()
                except Exception:  # noqa: BLE001
                    pass

    def connection_reset(self) -> np.ndarray:
        """Close the connection (releasing the board), reconnect and reopen the board.

        Done every tenth trial (thesis appendix, hardware reset). Returns the state after
        ``reset_down``.
        """
        self.close()
        self.connect()
        return self.reset_down()
