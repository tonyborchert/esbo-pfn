"""SQLite index of benchmark units plus a blob directory per unit.

A unit is identified by its kind and a JSON key. The key hash is the SHA-256 of the canonical JSON
of the key (sorted keys, no whitespace, NaN rejected). Results live under
``<PFN_HOME>/benchmarks/blobs/<kind>/<key_hash>/``. A unit function writes into a temporary
directory that is renamed into place only on success, so a finished blob directory is always
complete.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import sqlite3
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import esbopfn
from esbopfn.config import paths

log = logging.getLogger(__name__)

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    kind TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    key_json TEXT NOT NULL,
    status TEXT NOT NULL,
    blob_dir TEXT,
    started REAL,
    finished REAL,
    duration_s REAL,
    code_version TEXT,
    host TEXT,
    PRIMARY KEY (kind, key_hash)
);
CREATE INDEX IF NOT EXISTS units_kind_status ON units (kind, status);
"""


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"key value of type {type(obj).__name__} is not JSON serializable")


def canonical_json(key: dict[str, Any]) -> str:
    """Canonical JSON form of a key."""
    return json.dumps(
        key, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        default=_json_default,
    )


def key_hash(key: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical JSON of ``key``."""
    return hashlib.sha256(canonical_json(key).encode("utf-8")).hexdigest()


def code_version() -> str:
    """``git describe`` of the source checkout if available, else the package version."""
    src = Path(esbopfn.__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=src, capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or esbopfn.__version__
    except (OSError, subprocess.SubprocessError):
        return esbopfn.__version__


@dataclass(frozen=True)
class Unit:
    """One row of the ``units`` table."""

    kind: str
    key_hash: str
    key: dict[str, Any]
    status: str
    blob_dir: Path | None
    started: float | None
    finished: float | None
    duration_s: float | None
    code_version: str | None
    host: str | None


UnitFn = Callable[[dict[str, Any], Path], None]


class Db:
    """The benchmark database. Connections are short lived so long units hold no lock."""

    def __init__(self, path: Path | str | None = None, blob_root: Path | str | None = None):
        tree = paths()
        self.path = Path(path) if path is not None else tree.root / "benchmarks" / "db.sqlite"
        self.blob_root = (
            Path(blob_root) if blob_root is not None else self.path.parent / "blobs"
        )
        self._code_version: str | None = None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=120.0)
        try:
            con.executescript(_SCHEMA)
            with con:
                yield con
        finally:
            con.close()

    def blob_dir(self, kind: str, key: dict[str, Any]) -> Path:
        """Blob directory of a unit (not created)."""
        return self.blob_root / kind / key_hash(key)

    def _row(self, kind: str, h: str) -> tuple | None:
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM units WHERE kind = ? AND key_hash = ?", (kind, h)
            ).fetchone()

    @staticmethod
    def _unit(row: tuple) -> Unit:
        return Unit(
            kind=row[0], key_hash=row[1], key=json.loads(row[2]), status=row[3],
            blob_dir=Path(row[4]) if row[4] else None, started=row[5], finished=row[6],
            duration_s=row[7], code_version=row[8], host=row[9],
        )

    def get(self, kind: str, key: dict[str, Any]) -> Unit | None:
        """The unit row, or None."""
        row = self._row(kind, key_hash(key))
        return self._unit(row) if row else None

    def has(self, kind: str, key: dict[str, Any]) -> bool:
        """True if the unit finished successfully and its blob directory exists."""
        unit = self.get(kind, key)
        return bool(unit and unit.status == DONE and unit.blob_dir and unit.blob_dir.is_dir())

    def mark(self, kind: str, key: dict[str, Any], status: str, **fields: Any) -> None:
        """Insert or update a unit's status and any of the timing columns."""
        if status not in (PENDING, RUNNING, DONE, FAILED):
            raise ValueError(f"unknown status {status!r}")
        allowed = {"blob_dir", "started", "finished", "duration_s", "code_version", "host"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown unit fields {sorted(unknown)}")
        fields = {k: (str(v) if isinstance(v, Path) else v) for k, v in fields.items()}
        h = key_hash(key)
        with self._connect() as con:
            con.execute(
                "INSERT INTO units (kind, key_hash, key_json, status) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(kind, key_hash) DO UPDATE SET status = excluded.status",
                (kind, h, canonical_json(key), status),
            )
            for col, value in fields.items():
                con.execute(
                    f"UPDATE units SET {col} = ? WHERE kind = ? AND key_hash = ?", (value, kind, h)
                )

    def run(self, kind: str, key: dict[str, Any], fn: UnitFn) -> Path:
        """Run ``fn(key, tmp_dir)`` unless the unit is done. Returns the blob directory.

        A unit left in ``running`` by a killed job is run again. On an exception the unit is
        marked ``failed`` and the exception propagates.
        """
        final = self.blob_dir(kind, key)
        if self.has(kind, key):
            log.debug("skip %s %s", kind, final.name)
            return final
        if self._code_version is None:
            self._code_version = code_version()
        tmp = final.with_name(f"{final.name}.tmp-{socket.gethostname()}-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        start = time.time()
        self.mark(kind, key, RUNNING, started=start, host=socket.gethostname(),
                  code_version=self._code_version, finished=None, duration_s=None)
        try:
            fn(key, tmp)
        except BaseException:
            self.mark(kind, key, FAILED, finished=time.time())
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
        end = time.time()
        self.mark(kind, key, DONE, blob_dir=final, finished=end, duration_s=end - start)
        return final

    def query(self, kind: str, status: str | None = DONE, **filters: Any) -> list[Unit]:
        """Units of a kind whose key matches every ``filters`` item (top-level key fields).

        ``status=None`` returns units of any status.
        """
        sql = "SELECT * FROM units WHERE kind = ?"
        params: list[Any] = [kind]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        with self._connect() as con:
            rows = con.execute(sql + " ORDER BY key_hash", params).fetchall()
        units = [self._unit(r) for r in rows]
        if filters:
            want = {k: json.loads(canonical_json({"v": v}))["v"] for k, v in filters.items()}
            units = [u for u in units if all(u.key.get(k) == v for k, v in want.items())]
        return units

    def counts(self) -> dict[tuple[str, str], int]:
        """Number of units per ``(kind, status)``."""
        if not self.path.exists():
            return {}
        with self._connect() as con:
            rows = con.execute(
                "SELECT kind, status, COUNT(*) FROM units GROUP BY kind, status ORDER BY kind"
            ).fetchall()
        return {(k, s): n for k, s, n in rows}
