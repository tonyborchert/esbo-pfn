"""On-disk formats of the generated data: row pools, set pools and FT-PFN context shards.

A pool is a directory holding

- ``rows.f32``: raw little-endian float32 rows of ``row_floats`` values each, read as a
  ``(N, R)`` memmap,
- ``spec.json``: the column layout (name, offset, length, shape, entry names) plus the stored
  steps, channel names and the convention of the ``xdot`` block,
- ``meta.json``: dataset id, seed, count, chunk size, generator version, problem spec hash,
- ``chunks.log``: one line ``<chunk> <units>`` per finished chunk (rows ``chunk * chunk_size`` on),
- ``pool.finished``: present once every chunk is done.

Kinds: ``traj`` (single-trajectory prior, every row carries its own instance), ``set`` (one
screened instance per unit, ``n_max`` rows per instance tagged with the instance index, instance
parameters in ``instances.json``, unrun label trials in the sub-pool ``labels/``) and ``labels``.
A pool at ``resolution: full`` stores every step ``1..T``. At ``resolution: grid`` it stores the
decoder token steps plus the horizon, with ``xdot`` holding the rates summed over each interval.

FT-PFN contexts are stored in shards, one directory per chunk (see :class:`ContextShard`).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.core import Instance, ProblemSpec, Trajectory

POOL_FORMAT = "esbopfn-pool"
CONTEXT_FORMAT = "esbopfn-contexts"
FORMAT_VERSION = 1

ROWS_FILE = "rows.f32"
SPEC_FILE = "spec.json"
META_FILE = "meta.json"
CHUNK_LOG = "chunks.log"
FINISHED_FILE = "pool.finished"
INSTANCES_FILE = "instances.json"
INSTANCES_DIR = "instances.d"
LABELS_DIR = "labels"
SHARDS_DIR = "shards"

#: Status codes of a trial in a context shard.
STATUS_COMPLETE, STATUS_TRUNCATED, STATUS_CRASHED = 0, 1, 2


# --------------------------------------------------------------------------- column layout
@dataclass(frozen=True)
class Column:
    """One block of a row: ``length`` floats from ``offset`` on, viewed with ``shape``."""

    name: str
    offset: int
    length: int
    shape: tuple[int, ...]
    names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "offset": self.offset, "length": self.length,
                               "shape": list(self.shape)}
        if self.names:
            out["names"] = list(self.names)
        return out


@dataclass(frozen=True)
class PoolSpec:
    """The parsed ``spec.json`` of a pool."""

    kind: str
    resolution: str
    problem: str
    row_floats: int
    T: int
    d: int
    steps: np.ndarray
    channels: tuple[str, ...]
    rate_channels: tuple[str, ...]
    xdot_mode: str
    columns: tuple[Column, ...]
    extra: dict[str, Any] = field(default_factory=dict)

    def column(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"pool has no column {name!r} (columns: {[c.name for c in self.columns]})")

    def has(self, name: str) -> bool:
        return any(c.name == name for c in self.columns)

    @property
    def full_resolution(self) -> bool:
        return self.resolution == "full"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "format": POOL_FORMAT, "format_version": FORMAT_VERSION, "kind": self.kind,
            "resolution": self.resolution, "problem": self.problem, "dtype": "float32",
            "row_floats": self.row_floats, "T": self.T, "d": self.d,
            "steps": "all" if self.full_resolution else [int(s) for s in self.steps],
            "channels": list(self.channels), "rate_channels": list(self.rate_channels),
            "xdot_mode": self.xdot_mode,
        }
        out.update(self.extra)
        out["columns"] = [c.to_dict() for c in self.columns]
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PoolSpec:
        if raw.get("format") != POOL_FORMAT:
            raise ValueError(f"not a pool spec (format {raw.get('format')!r})")
        if int(raw.get("format_version", 0)) != FORMAT_VERSION:
            raise ValueError(f"unsupported pool format version {raw.get('format_version')}")
        T = int(raw["T"])
        steps_raw = raw["steps"]
        steps = (np.arange(1, T + 1, dtype=np.int64) if steps_raw == "all"
                 else np.asarray(steps_raw, dtype=np.int64))
        cols = tuple(Column(name=c["name"], offset=int(c["offset"]), length=int(c["length"]),
                            shape=tuple(int(s) for s in c["shape"]),
                            names=tuple(c.get("names", ()))) for c in raw["columns"])
        known = {"format", "format_version", "kind", "resolution", "problem", "dtype", "row_floats",
                 "T", "d", "steps", "channels", "rate_channels", "xdot_mode", "columns"}
        spec = cls(kind=str(raw["kind"]), resolution=str(raw["resolution"]),
                   problem=str(raw["problem"]), row_floats=int(raw["row_floats"]), T=T,
                   d=int(raw["d"]), steps=steps, channels=tuple(raw["channels"]),
                   rate_channels=tuple(raw["rate_channels"]), xdot_mode=str(raw["xdot_mode"]),
                   columns=cols, extra={k: v for k, v in raw.items() if k not in known})
        spec.validate()
        return spec

    def validate(self) -> None:
        end = 0
        for col in self.columns:
            if col.offset != end or col.length != math.prod(col.shape):
                raise ValueError(f"column {col.name!r} does not tile the row")
            end += col.length
        if end != self.row_floats:
            raise ValueError(f"columns cover {end} floats, row_floats is {self.row_floats}")


def make_spec(*, kind: str, problem: str, d: int, T: int, channels: Sequence[str],
              rate_channels: Sequence[str] = (), resolution: str = "full",
              steps: Sequence[int] | None = None, theta_names: Sequence[str] = (),
              instance_fields: dict[str, Sequence[str]] | None = None,
              rows_per_unit: int = 1) -> PoolSpec:
    """Build a spec with the column order the generators write.

    ``instance_fields`` maps ``phi``, ``phi_hat``, ``noise_std`` and ``extras`` (in that order, each
    optional) to entry names. They are stored per row in ``traj`` pools only.
    """
    if kind not in ("traj", "set", "labels"):
        raise ValueError(f"unknown pool kind {kind!r}")
    if kind == "labels":
        resolution, step_arr = "horizon", np.array([T], dtype=np.int64)
    elif resolution == "full":
        step_arr = np.arange(1, T + 1, dtype=np.int64)
    elif resolution == "grid":
        if steps is None:
            raise ValueError("a grid pool needs its steps")
        step_arr = np.asarray(steps, dtype=np.int64)
    else:
        raise ValueError(f"unknown resolution {resolution!r}")
    S, C, CR = len(step_arr), len(channels), len(rate_channels)
    cols: list[Column] = []
    off = 0

    def add(name: str, shape: tuple[int, ...], names: Sequence[str] = ()) -> None:
        nonlocal off
        length = math.prod(shape)
        cols.append(Column(name, off, length, shape, tuple(names)))
        off += length

    if kind != "traj":
        add("instance", (1,))
    add("theta", (d,), theta_names)
    if kind == "traj":
        for group in ("phi", "phi_hat", "noise_std", "extras"):
            names = (instance_fields or {}).get(group)
            if names:
                add(group, (len(names),), names)
    add("crash_step", (1,))
    add("success", (1,))
    add("J", (S,))
    xdot_mode = "none"
    if kind != "labels":
        add("x", (S, C), channels)
        if resolution == "grid":
            xdot_mode = "interval_sum"
            add("xdot", (S, CR or C), rate_channels or channels)
        elif CR:
            xdot_mode = "per_step"
            add("xdot", (S, CR), rate_channels)
    spec = PoolSpec(kind=kind, resolution=resolution, problem=problem, row_floats=off, T=T, d=d,
                    steps=step_arr, channels=tuple(channels), rate_channels=tuple(rate_channels),
                    xdot_mode=xdot_mode, columns=tuple(cols),
                    extra={"rows_per_unit": int(rows_per_unit)})
    spec.validate()
    return spec


# --------------------------------------------------------------------------- chunk log
def read_chunk_log(path: Path) -> dict[int, int]:
    """Finished chunks and their unit counts from ``chunks.log`` (later lines win)."""
    done: dict[int, int] = {}
    if not path.is_file():
        return done
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            done[int(parts[0])] = int(parts[1])
    return done


def append_chunk_log(path: Path, chunk: int, units: int) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{chunk} {units}\n")
        fh.flush()
        os.fsync(fh.fileno())


def chunk_units(count: int, chunk_size: int, chunk: int) -> int:
    """Units in chunk ``chunk`` of a dataset of ``count`` units."""
    return max(0, min(chunk_size, count - chunk * chunk_size))


def n_chunks(count: int, chunk_size: int) -> int:
    return -(-int(count) // int(chunk_size))


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- trajectories on a grid
@dataclass
class GridTrajectory:
    """A trajectory stored at a subset of steps (a ``grid`` pool row).

    ``xdot_sum[l]`` is the sum of the rates over ``(steps[l-1], steps[l]]``, so the DecPFN
    covariates are ``[J[l] - J[l-1], x[l], xdot_sum[l]]``.
    """

    theta: np.ndarray
    steps: np.ndarray
    J: np.ndarray
    x: np.ndarray
    xdot_sum: np.ndarray | None
    crash_step: int | None
    success: bool
    instance: int | None = None

    @property
    def crashed(self) -> bool:
        return self.crash_step is not None

    @property
    def J_total(self) -> float:
        return float(self.J[-1])


# --------------------------------------------------------------------------- pools
class Pool:
    """A row pool directory. Rows are read through a read-only memmap."""

    def __init__(self, path: Path, spec: PoolSpec, limit: int | None = None):
        self.path = Path(path)
        self.spec = spec
        self.limit = limit
        self._mm: np.memmap | None = None
        self._mm_rows = -1
        self._instances: list[dict[str, Any]] | None = None

    # ---- construction
    @classmethod
    def open(cls, path: Path | str, limit: int | None = None) -> Pool:
        """Open an existing pool. ``limit`` caps the visible rows (prefix datasets)."""
        path = Path(path)
        spec_path = path / SPEC_FILE
        if not spec_path.is_file():
            raise FileNotFoundError(f"no pool at {path} ({SPEC_FILE} missing)")
        spec = PoolSpec.from_dict(json.loads(spec_path.read_text()))
        return cls(path, spec, limit)

    @classmethod
    def create(cls, path: Path | str, spec: PoolSpec, meta: dict[str, Any] | None = None) -> Pool:
        """Create an empty pool (or reopen one with the identical layout)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        spec_path = path / SPEC_FILE
        if spec_path.is_file():
            have = PoolSpec.from_dict(json.loads(spec_path.read_text()))
            if have.to_dict() != spec.to_dict():
                raise ValueError(f"{path} already holds a pool with a different layout")
        else:
            write_json_atomic(spec_path, spec.to_dict())
        if meta is not None:
            write_json_atomic(path / META_FILE, meta)
        (path / ROWS_FILE).touch()
        return cls(path, spec)

    # ---- metadata
    @property
    def meta(self) -> dict[str, Any]:
        p = self.path / META_FILE
        return json.loads(p.read_text()) if p.is_file() else {}

    @property
    def finished(self) -> bool:
        return (self.path / FINISHED_FILE).is_file()

    def mark_finished(self, info: dict[str, Any] | None = None) -> None:
        write_json_atomic(self.path / FINISHED_FILE, info or {"rows": self.rows_on_disk()})

    def done_chunks(self) -> dict[int, int]:
        return read_chunk_log(self.path / CHUNK_LOG)

    @property
    def row_bytes(self) -> int:
        return 4 * self.spec.row_floats

    @property
    def rows_per_unit(self) -> int:
        """Rows per chunk unit: ``n_max`` (set pools) or the label count (labels), else 1."""
        return int(self.spec.extra.get("rows_per_unit", 1))

    def rows_on_disk(self) -> int:
        p = self.path / ROWS_FILE
        return p.stat().st_size // self.row_bytes if p.is_file() else 0

    def __len__(self) -> int:
        n = self.rows_on_disk()
        return n if self.limit is None else min(n, self.limit)

    # ---- raw access
    @property
    def rows(self) -> np.ndarray:
        """All visible rows as a read-only ``(N, R)`` float32 memmap."""
        n = len(self)
        if self._mm is None or self._mm_rows != n:
            if n == 0:
                return np.zeros((0, self.spec.row_floats), dtype=np.float32)
            self._mm = np.memmap(self.path / ROWS_FILE, dtype="<f4", mode="r",
                                 shape=(n, self.spec.row_floats))
            self._mm_rows = n
        return self._mm

    def row(self, i: int) -> np.ndarray:
        n = len(self)
        if not 0 <= i < n:
            raise IndexError(f"row {i} out of range [0, {n})")
        return np.asarray(self.rows[i])

    def slice(self, a: int, b: int) -> np.ndarray:
        """Rows ``a`` to ``b`` (exclusive) as a ``(b - a, R)`` view."""
        return self.rows[a:b]

    def column(self, name: str, rows: slice | np.ndarray | None = None) -> np.ndarray:
        """Column ``name`` of the selected rows, shaped ``(n, *shape)``."""
        col = self.spec.column(name)
        block = self.rows[rows if rows is not None else slice(None)]
        block = np.asarray(block)
        if block.ndim == 1:
            block = block[None, :]
        return block[:, col.offset:col.offset + col.length].reshape((block.shape[0], *col.shape))

    def _field(self, r: np.ndarray, name: str) -> np.ndarray:
        col = self.spec.column(name)
        return r[col.offset:col.offset + col.length].reshape(col.shape)

    # ---- typed access
    def read(self, i: int) -> Trajectory:
        """Row ``i`` as a :class:`~esbopfn.core.Trajectory` (full-resolution pools only)."""
        if not self.spec.full_resolution:
            raise ValueError("this pool stores a step grid, use read_grid()")
        r = self.row(i)
        cs = int(self._field(r, "crash_step")[0])
        xdot = (np.array(self._field(r, "xdot"), dtype=np.float32)
                if self.spec.xdot_mode == "per_step" else None)
        return Trajectory(theta=np.array(self._field(r, "theta"), dtype=np.float64),
                          J=np.array(self._field(r, "J"), dtype=np.float32),
                          x=np.array(self._field(r, "x"), dtype=np.float32), xdot=xdot,
                          crash_step=cs if cs > 0 else None,
                          success=bool(self._field(r, "success")[0] > 0.5))

    def read_grid(self, i: int) -> GridTrajectory:
        """Row ``i`` at its stored steps (any pool kind)."""
        r = self.row(i)
        cs = int(self._field(r, "crash_step")[0])
        has_x = self.spec.has("x")
        xd = self.spec.has("xdot")
        return GridTrajectory(
            theta=np.array(self._field(r, "theta"), dtype=np.float64),
            steps=self.spec.steps.copy(), J=np.array(self._field(r, "J"), dtype=np.float32),
            x=(np.array(self._field(r, "x"), dtype=np.float32) if has_x
               else np.zeros((len(self.spec.steps), 0), np.float32)),
            xdot_sum=np.array(self._field(r, "xdot"), dtype=np.float32) if xd else None,
            crash_step=cs if cs > 0 else None,
            success=bool(self._field(r, "success")[0] > 0.5),
            instance=int(self._field(r, "instance")[0]) if self.spec.has("instance") else None)

    def instance(self, i: int) -> Instance:
        """The instance of row ``i`` (``traj`` pools) or instance ``i`` (``set`` pools)."""
        if self.spec.kind == "traj":
            r = self.row(i)
            groups = {}
            for g in ("phi", "phi_hat", "noise_std", "extras"):
                if self.spec.has(g):
                    col = self.spec.column(g)
                    groups[g] = {n: float(v) for n, v in zip(col.names, self._field(r, g))}
                else:
                    groups[g] = {}
            return Instance(phi=groups["phi"], phi_hat=groups["phi_hat"],
                            noise_std=groups["noise_std"], extras=groups["extras"], seed=int(i))
        rec = self.instances()[i]["instance"]
        return Instance(phi=dict(rec["phi"]), phi_hat=dict(rec["phi_hat"]),
                        noise_std=dict(rec["noise_std"]), extras=dict(rec["extras"]),
                        seed=int(rec["seed"]))

    # ---- set pools
    def instances(self) -> list[dict[str, Any]]:
        """Instance records of a set pool (``instances.json`` or the per-chunk files)."""
        if self._instances is None:
            p = self.path / INSTANCES_FILE
            if p.is_file():
                recs = json.loads(p.read_text())
            else:
                recs = []
                for f in sorted((self.path / INSTANCES_DIR).glob("chunk_*.jsonl")):
                    recs += [json.loads(line) for line in f.read_text().splitlines() if line]
                recs.sort(key=lambda r: r["index"])
            self._instances = recs
        return self._instances

    def instance_rows(self, i: int) -> np.ndarray:
        """Rows of instance ``i`` in a set pool."""
        k = self.rows_per_unit
        return self.slice(i * k, (i + 1) * k)

    def labels(self) -> Pool:
        """The unrun label trials of a set pool."""
        return Pool.open(self.path / LABELS_DIR)

    # ---- writing (Python-side generators and tests)
    def append_chunk(self, chunk: int, rows: np.ndarray, chunk_size: int | None = None) -> None:
        """Write ``rows`` as chunk ``chunk`` (rows ``chunk * chunk_size`` on) and log it.

        ``chunk_size`` defaults to ``meta.json``'s chunk size, else to ``len(rows)``.
        """
        rows = np.ascontiguousarray(rows, dtype="<f4")
        if rows.ndim != 2 or rows.shape[1] != self.spec.row_floats:
            raise ValueError(f"rows must have shape (n, {self.spec.row_floats}), got {rows.shape}")
        rpu = self.rows_per_unit
        if len(rows) % rpu:
            raise ValueError(f"a chunk holds whole units of {rpu} rows")
        units = len(rows) // rpu
        size = int(chunk_size or self.meta.get("chunk") or units)
        if units > size:
            raise ValueError(f"chunk of {units} units exceeds the chunk size {size}")
        offset = chunk * size * rpu * self.row_bytes
        fd = os.open(self.path / ROWS_FILE, os.O_RDWR | os.O_CREAT, 0o664)
        try:
            os.pwrite(fd, rows.tobytes(), offset)
            os.fsync(fd)
        finally:
            os.close(fd)
        self._mm = None
        append_chunk_log(self.path / CHUNK_LOG, chunk, units)


def pool_from_spec(spec: ProblemSpec, kind: str, instance_fields: dict[str, Sequence[str]],
                   resolution: str = "full", steps: Sequence[int] | None = None) -> PoolSpec:
    """:func:`make_spec` filled in from a :class:`~esbopfn.core.ProblemSpec`."""
    return make_spec(kind=kind, problem=spec.name, d=spec.d, T=spec.T, channels=spec.channels,
                     rate_channels=spec.rate_channels, resolution=resolution, steps=steps,
                     theta_names=spec.theta_names, instance_fields=instance_fields)


def pack_trajectory(spec: PoolSpec, traj: Trajectory, instance: Instance | None = None,
                    instance_index: int | None = None) -> np.ndarray:
    """One row of a full-resolution pool from a trajectory (the inverse of :meth:`Pool.read`)."""
    if not spec.full_resolution:
        raise ValueError("pack_trajectory writes full-resolution rows only")
    row = np.zeros(spec.row_floats, dtype=np.float32)

    def put(name: str, value: np.ndarray | float) -> None:
        col = spec.column(name)
        row[col.offset:col.offset + col.length] = np.asarray(value, dtype=np.float32).reshape(-1)

    if spec.has("instance"):
        put("instance", float(instance_index or 0))
    put("theta", traj.theta)
    if spec.kind == "traj" and instance is not None:
        for g in ("phi", "phi_hat", "noise_std", "extras"):
            if spec.has(g):
                col = spec.column(g)
                vals = getattr(instance, g)
                put(g, [vals[n] for n in col.names])
    put("crash_step", float(traj.crash_step or 0))
    put("success", 1.0 if traj.success else 0.0)
    put("J", traj.J)
    put("x", traj.x)
    if spec.xdot_mode == "per_step":
        if traj.xdot is None:
            raise ValueError("the pool stores measured rates, the trajectory has none")
        put("xdot", traj.xdot)
    return row


# --------------------------------------------------------------------------- FT-PFN contexts
#: Arrays of a context shard: file name -> (dtype, trailing shape key).
SHARD_ARRAYS: dict[str, tuple[str, str]] = {
    "example_ptr.i64": ("<i8", "examples+1"),
    "example_n.i32": ("<i4", "examples"),
    "example_meta.f32": ("<f4", "examples,3"),
    "example_seed.i64": ("<i8", "examples"),
    "trial_theta.f32": ("<f4", "trials,d"),
    "trial_len.i32": ("<i4", "trials"),
    "trial_status.i8": ("i1", "trials"),
    "trial_target.f32": ("<f4", "trials"),
    "trial_tok_t.i16": ("<i2", "trials,n_tokens"),
    "trial_tok_J.f32": ("<f4", "trials,n_tokens"),
    "unrun_theta.f32": ("<f4", "examples,n_unrun,d"),
    "unrun_target.f32": ("<f4", "examples,n_unrun"),
}


@dataclass
class ContextExample:
    """One FT-PFN training example.

    Trial ``i`` was observed for ``length[i]`` steps with ``status[i]`` (complete, truncated or
    crashed). Its tokens are ``(theta[i], tok_t[i, k], tok_J[i, k])`` for ``tok_t[i, k] > 0``,
    where ``tok_J = inf`` marks a crash. ``target[i]`` is its total cost (``inf`` for a trajectory
    that crashes), the query target of truncated trials. ``unrun_theta`` and ``unrun_target`` are
    the unrun queries.
    """

    theta: np.ndarray
    length: np.ndarray
    status: np.ndarray
    target: np.ndarray
    tok_t: np.ndarray
    tok_J: np.ndarray
    unrun_theta: np.ndarray
    unrun_target: np.ndarray
    p_cut: float = 0.0

    @property
    def n(self) -> int:
        return int(self.theta.shape[0])

    def tokens(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Steps and cumulative costs of the tokens of trial ``i``."""
        keep = self.tok_t[i] > 0
        return self.tok_t[i][keep].astype(np.int64), self.tok_J[i][keep]


class ContextShard:
    """One chunk of FT-PFN examples, the arrays of :data:`SHARD_ARRAYS` as memmaps."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.meta: dict[str, Any] = json.loads((self.path / "shard.json").read_text())
        if self.meta.get("format") != CONTEXT_FORMAT:
            raise ValueError(f"{self.path} is not a context shard")
        dims = {"examples": int(self.meta["examples"]), "trials": int(self.meta["trials"]),
                "d": int(self.meta["d"]), "n_tokens": int(self.meta["n_tokens"]),
                "n_unrun": int(self.meta["n_unrun"])}
        dims["examples+1"] = dims["examples"] + 1
        self.arrays: dict[str, np.ndarray] = {}
        for fname, (dtype, shape_key) in SHARD_ARRAYS.items():
            shape = tuple(int(k) if k.isdigit() else dims[k] for k in shape_key.split(","))
            key = fname.split(".")[0]
            if math.prod(shape) == 0:
                self.arrays[key] = np.zeros(shape, dtype=dtype)
            else:
                self.arrays[key] = np.memmap(self.path / fname, dtype=dtype, mode="r", shape=shape)

    def __len__(self) -> int:
        return int(self.meta["examples"])

    def example(self, i: int) -> ContextExample:
        a = self.arrays
        lo, hi = int(a["example_ptr"][i]), int(a["example_ptr"][i + 1])
        return ContextExample(theta=np.asarray(a["trial_theta"][lo:hi]),
                              length=np.asarray(a["trial_len"][lo:hi]),
                              status=np.asarray(a["trial_status"][lo:hi]),
                              target=np.asarray(a["trial_target"][lo:hi]),
                              tok_t=np.asarray(a["trial_tok_t"][lo:hi]),
                              tok_J=np.asarray(a["trial_tok_J"][lo:hi]),
                              unrun_theta=np.asarray(a["unrun_theta"][i]),
                              unrun_target=np.asarray(a["unrun_target"][i]),
                              p_cut=float(a["example_meta"][i, 0]))


class ContextStore:
    """All finished shards of a ``contexts`` dataset, indexed by global example number."""

    def __init__(self, path: Path | str, limit: int | None = None):
        self.path = Path(path)
        self.shards = [ContextShard(p) for p in sorted((self.path / SHARDS_DIR).glob("chunk_*"))
                       if (p / "shard.json").is_file()]
        self._starts = [int(s.meta["first_example"]) for s in self.shards]
        self.limit = limit

    def __len__(self) -> int:
        n = sum(len(s) for s in self.shards)
        return n if self.limit is None else min(n, self.limit)

    def __iter__(self) -> Iterator[ContextExample]:
        for s in self.shards:
            for i in range(len(s)):
                yield s.example(i)

    def example(self, index: int) -> ContextExample:
        for start, shard in zip(self._starts, self.shards):
            if start <= index < start + len(shard):
                return shard.example(index - start)
        raise IndexError(f"example {index} is not in a finished shard")


def write_context_shard(path: Path | str, examples: Sequence[ContextExample], *, problem: str,
                        T: int, d: int, chunk: int = 0, first_example: int = 0,
                        grid: Sequence[int] = ()) -> Path:
    """Write examples in the shard layout (the generators write the same files from C++)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    n_tok = examples[0].tok_t.shape[1] if examples else 9
    n_un = examples[0].unrun_theta.shape[0] if examples else 0
    ptr = np.cumsum([0] + [e.n for e in examples]).astype("<i8")

    def cat(name: str, dtype: str, width: tuple[int, ...]) -> np.ndarray:
        parts = [np.asarray(getattr(e, name)).reshape((-1, *width)) for e in examples]
        return (np.concatenate(parts) if parts else np.zeros((0, *width))).astype(dtype)

    arrays = {
        "example_ptr.i64": ptr,
        "example_n.i32": np.array([e.n for e in examples], dtype="<i4"),
        "example_meta.f32": np.array([[e.p_cut, 0.0, 0.0] for e in examples], dtype="<f4"),
        "example_seed.i64": np.zeros(len(examples), dtype="<i8"),
        "trial_theta.f32": cat("theta", "<f4", (d,)),
        "trial_len.i32": cat("length", "<i4", ()),
        "trial_status.i8": cat("status", "i1", ()),
        "trial_target.f32": cat("target", "<f4", ()),
        "trial_tok_t.i16": cat("tok_t", "<i2", (n_tok,)),
        "trial_tok_J.f32": cat("tok_J", "<f4", (n_tok,)),
        "unrun_theta.f32": np.stack([e.unrun_theta for e in examples]).astype("<f4")
        if examples else np.zeros((0, n_un, d), "<f4"),
        "unrun_target.f32": np.stack([e.unrun_target for e in examples]).astype("<f4")
        if examples else np.zeros((0, n_un), "<f4"),
    }
    for fname, arr in arrays.items():
        (path / fname).write_bytes(np.ascontiguousarray(arr).tobytes())
    meta = {"format": CONTEXT_FORMAT, "format_version": FORMAT_VERSION, "problem": problem,
            "chunk": chunk, "first_example": first_example, "examples": len(examples),
            "trials": int(ptr[-1]), "d": d, "T": T, "n_tokens": n_tok, "n_unrun": n_un,
            "grid": [int(g) for g in grid],
            "status_codes": {"complete": 0, "truncated": 1, "crashed": 2}}
    write_json_atomic(path / "shard.json", meta)
    return path
