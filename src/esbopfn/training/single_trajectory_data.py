"""Batches of the single-trajectory prior for DecPFN and LC-PFN.

The data are the full-resolution row pools ``traj/<p>/<size>`` (see ``docs/DATA.md``). A training
run makes exactly one pass over ``N_examples`` rows in a seeded random order, ``floor(N / batch)``
optimizer steps, without recycling rows. Every batch is a pure function of its step index, so a
resumed run continues at the same place and workers can build batches in any order.

DecPFN tokens sit on :func:`esbopfn.grids.decoder_grid`. Each token holds the cost ``J_{t_l}``,
the covariates ``[ΔJ, ξ, Σ ξ̇]`` of :func:`esbopfn.grids.tokens` and the time ``t_l``
(variant ``nostates``: ``J, ΔJ, t``). The target is ``J_T`` (``+inf`` on a crash) and a position is
valid strictly before the crash.

LC-PFN tokens are ``(t_l, J_{t_l})`` on :func:`esbopfn.grids.curve_grid` with ``N = N_grid``. One
cutoff ``κ ~ U{1..N-1}`` is drawn per batch, the first ``κ`` points are the context and the
remaining ``N - κ`` points the queries (variant ``last``: only the final point). Rows that crash
at or before ``t_κ`` cannot supply the full context and are removed from the batch (no requeue).
Query targets at or after a crash are the crash atom (``+inf``).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from esbopfn import grids
from esbopfn.core import ProblemSpec
from esbopfn.generators.pools import Pool
from esbopfn.training.normalization import EDGE_ROWS, STAT_ROWS, Calib, make_calib
from esbopfn.util.seeds import substream

#: Number of training examples per size label (exact, no recycling).
SIZES: dict[str, int] = {"100k": 100_000, "1M": 1_000_000}


# ----------------------------------------------------------------------------- raw rows
@dataclass
class RowBlock:
    """Raw arrays of some pool rows at full resolution."""

    J: np.ndarray            # (B, T) float64, +inf from the crash step on
    x: np.ndarray            # (B, T, C)
    rates: np.ndarray        # (B, T, C_rate) measured rates or level increments
    crash_step: np.ndarray   # (B,) 1-based, 0 = no crash


class PoolReader:
    """Pickle-friendly lazy access to a full-resolution ``traj`` pool (one memmap per process)."""

    def __init__(self, path: Path | str, limit: int | None = None):
        self.path = Path(path)
        self.limit = limit
        self._pool: Pool | None = None
        pool = self.pool
        if pool.spec.kind != "traj" or not pool.spec.full_resolution:
            raise ValueError(f"{self.path} is not a full-resolution traj pool")

    @classmethod
    def of(cls, pool: Pool | Path | str, limit: int | None = None) -> PoolReader:
        if isinstance(pool, Pool):
            return cls(pool.path, limit if limit is not None else pool.limit)
        return cls(pool, limit)

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "limit": self.limit, "_pool": None}

    @property
    def pool(self) -> Pool:
        if self._pool is None:
            self._pool = Pool.open(self.path, limit=self.limit)
        return self._pool

    def __len__(self) -> int:
        return len(self.pool)

    @property
    def T(self) -> int:
        return self.pool.spec.T

    @property
    def problem(self) -> str:
        return self.pool.spec.problem

    def read(self, idx: np.ndarray) -> RowBlock:
        idx = np.asarray(idx, dtype=np.int64)
        order = np.argsort(idx, kind="stable")
        rows = np.empty((len(idx), self.pool.spec.row_floats), dtype=np.float32)
        rows[order] = self.pool.rows[idx[order]]
        spec = self.pool.spec

        def block(name: str) -> np.ndarray:
            col = spec.column(name)
            return rows[:, col.offset:col.offset + col.length].reshape((len(idx), *col.shape))

        J = block("J").astype(np.float64)
        x = block("x").astype(np.float64)
        if spec.xdot_mode == "per_step":
            rates = block("xdot").astype(np.float64)
        else:
            rates = np.zeros_like(x)
            rates[:, 1:] = x[:, 1:] - x[:, :-1]
        crash = block("crash_step")[:, 0].astype(np.int64)
        return RowBlock(J=J, x=x, rates=rates, crash_step=crash)

    def horizon_costs(self, idx: np.ndarray) -> np.ndarray:
        """``J_T`` of the given rows, reading a single float per row."""
        col = self.pool.spec.column("J")
        idx = np.sort(np.asarray(idx, dtype=np.int64))
        return np.asarray(self.pool.rows[idx, col.offset + self.T - 1], dtype=np.float64)


# ----------------------------------------------------------------------------- features
def decpfn_feature_names(spec: ProblemSpec, states: bool = True) -> list[str]:
    names = ["J", "dJ"]
    if states:
        rates = spec.rate_channels or tuple(f"d_{c}" for c in spec.channels)
        names += list(spec.channels) + [f"sum_{r}" for r in rates]
    return names + ["t"]


def decpfn_features(block: RowBlock, grid: np.ndarray, states: bool = True
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Raw DecPFN inputs without the time, ``(B, L, F - 1)``, and validity ``(B, L)``.

    Matches :func:`esbopfn.grids.tokens` on every valid position. Invalid positions (at or after
    the crash) are zero.
    """
    t = np.asarray(grid, dtype=np.int64)
    B = block.J.shape[0]
    valid = (block.crash_step[:, None] == 0) | (t[None, :] < block.crash_step[:, None])
    J_t = block.J[:, t - 1]
    J_t = np.where(valid, J_t, 0.0)
    prev = np.concatenate([np.zeros((B, 1)), J_t[:, :-1]], axis=1)
    dJ = J_t - prev
    parts = [J_t[..., None], dJ[..., None]]
    if states:
        csum = np.concatenate([np.zeros((B, 1, block.rates.shape[2])),
                               np.cumsum(block.rates, axis=1)], axis=1)
        t_prev = np.concatenate([[0], t[:-1]])
        parts += [block.x[:, t - 1, :], csum[:, t, :] - csum[:, t_prev, :]]
    feats = np.concatenate(parts, axis=2)
    feats = np.where(valid[..., None], feats, 0.0)
    return feats, valid


def lcpfn_grid(spec_or_T: ProblemSpec | int) -> np.ndarray:
    return grids.curve_grid(spec_or_T, grids.n_grid(spec_or_T))


# ----------------------------------------------------------------------------- calibration
def _calib_rows(n: int, rows: int, seed: int, label: str) -> np.ndarray:
    k = min(n, rows)
    return np.sort(substream(seed, "calib", label).choice(n, size=k, replace=False))


def calibrate_decpfn(reader: PoolReader, spec: ProblemSpec, model: str, head: str, seed: int,
                     states: bool = True, n_bin: int = 1000, stat_rows: int = STAT_ROWS,
                     edge_rows: int = EDGE_ROWS) -> Calib:
    """Statistics of the DecPFN inputs on a seeded subset of the training pool. Time range
    ``{1..T-1}``, bar edges on finite horizon samples ``J_T``."""
    grid = grids.decoder_grid(spec)
    idx = _calib_rows(len(reader), stat_rows, seed, "stats")
    feats, valid = decpfn_features(reader.read(idx), grid, states)
    edge_samples = reader.horizon_costs(_calib_rows(len(reader), edge_rows, seed, "edges"))
    return make_calib(model=model, problem=spec.name, head=head,
                      features=decpfn_feature_names(spec, states), feature_values=feats,
                      feature_valid=valid, time_range=(1, spec.T - 1), J_index=0,
                      edge_samples=edge_samples, grid=grid, T=spec.T, n_bin=n_bin,
                      samples={"rows": int(len(idx)), "edge_rows": int(min(len(reader), edge_rows)),
                               "seed": int(seed), "pool": str(reader.path)})


def calibrate_lcpfn(reader: PoolReader, spec: ProblemSpec, model: str, head: str, seed: int,
                    n_bin: int = 1000, stat_rows: int = STAT_ROWS) -> Calib:
    """Statistics of ``J`` on the curve grid. Time range ``{1..T}``, bar edges on samples of the
    full trajectory value range (every grid point)."""
    grid = lcpfn_grid(spec)
    idx = _calib_rows(len(reader), stat_rows, seed, "stats")
    J = reader.read(idx).J[:, grid - 1]
    return make_calib(model=model, problem=spec.name, head=head, features=["J", "t"],
                      feature_values=J[..., None], feature_valid=None, time_range=(1, spec.T),
                      J_index=0, edge_samples=J.reshape(-1), grid=grid, T=spec.T, n_bin=n_bin,
                      samples={"rows": int(len(idx)), "seed": int(seed),
                               "pool": str(reader.path)})


# ----------------------------------------------------------------------------- batches
@dataclass
class DecPFNBatch:
    x: torch.Tensor       # (B, L, F) normalized inputs, time last
    valid: torch.Tensor   # (B, L) bool
    y: torch.Tensor       # (B,) raw J_T, +inf on a crash


@dataclass
class LCPFNBatch:
    t: torch.Tensor       # (B, κ + Q) normalized time of the context and query points
    y_ctx: torch.Tensor   # (B, κ) normalized cost of the context points
    y: torch.Tensor       # (B, Q) raw query targets, +inf where crashed
    kappa: int


class _Data:
    """Shared step bookkeeping: seeded permutation, one batch per step, optional workers.

    By default a run is one pass over ``n_examples`` rows (``floor(n / batch)`` steps, no
    recycling). ``total_steps`` beyond one pass continues with a fresh seeded permutation per
    pass, which only debugging and tests use (``pfn train --steps``).
    """

    def __init__(self, reader: PoolReader, calib: Calib, batch_size: int, seed: int,
                 n_examples: int | None = None, shuffle: bool = True,
                 total_steps: int | None = None):
        self.reader = reader
        self.calib = calib
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        n = len(reader) if n_examples is None else int(n_examples)
        if n > len(reader):
            raise ValueError(f"{reader.path} holds {len(reader)} rows, {n} requested")
        self.n_examples = n
        self.shuffle = shuffle
        self.steps_per_pass = n // self.batch_size
        self.total_steps = self.steps_per_pass if total_steps is None else int(total_steps)
        self._perm: tuple[int, np.ndarray] | None = None

    @property
    def steps(self) -> int:
        return self.total_steps

    def _order(self, p: int) -> np.ndarray:
        if self._perm is None or self._perm[0] != p:
            order = (substream(self.seed, "shuffle", p).permutation(self.n_examples)
                     if self.shuffle else np.arange(self.n_examples, dtype=np.int64))
            self._perm = (p, order)
        return self._perm[1]

    def indices(self, step: int) -> np.ndarray:
        if not 0 <= step < self.steps or self.steps_per_pass == 0:
            raise IndexError(f"step {step} out of range [0, {self.steps})")
        p, i = divmod(step, self.steps_per_pass)
        return self._order(p)[i * self.batch_size:(i + 1) * self.batch_size]

    def batch(self, step: int) -> Any:
        raise NotImplementedError

    def iterate(self, start_step: int = 0, workers: int = 0) -> Iterator[Any]:
        """Batches from ``start_step`` to the end of the pass."""
        if workers <= 0:
            return (self.batch(s) for s in range(start_step, self.steps))
        loader = torch.utils.data.DataLoader(
            _Steps(self, start_step), batch_size=None, shuffle=False, num_workers=workers,
            persistent_workers=False, prefetch_factor=4)
        return iter(loader)


class _Steps(torch.utils.data.Dataset):
    def __init__(self, data: _Data, start: int):
        self.data, self.start = data, start

    def __len__(self) -> int:
        return self.data.steps - self.start

    def __getitem__(self, i: int) -> Any:
        return self.data.batch(self.start + i)


class DecPFNData(_Data):
    def __init__(self, reader: PoolReader, calib: Calib, batch_size: int = 128, seed: int = 0,
                 n_examples: int | None = None, states: bool = True, shuffle: bool = True,
                 total_steps: int | None = None):
        super().__init__(reader, calib, batch_size, seed, n_examples, shuffle, total_steps)
        self.states = states
        self.grid = np.asarray(calib.grid, dtype=np.int64)

    def make(self, idx: np.ndarray) -> DecPFNBatch:
        block = self.reader.read(idx)
        feats, valid = decpfn_features(block, self.grid, self.states)
        t = np.broadcast_to(self.grid.astype(np.float64), valid.shape)[..., None]
        x = self.calib.normalize_inputs(np.concatenate([feats, t], axis=2))
        x = np.where(valid[..., None], x, 0.0).astype(np.float32)
        y = block.J[:, -1].astype(np.float32)
        return DecPFNBatch(torch.from_numpy(x), torch.from_numpy(valid), torch.from_numpy(y))

    def batch(self, step: int) -> DecPFNBatch:
        return self.make(self.indices(step))


class LCPFNData(_Data):
    def __init__(self, reader: PoolReader, calib: Calib, batch_size: int = 128, seed: int = 0,
                 n_examples: int | None = None, last_only: bool = False, shuffle: bool = True,
                 total_steps: int | None = None):
        super().__init__(reader, calib, batch_size, seed, n_examples, shuffle, total_steps)
        self.last_only = last_only
        self.grid = np.asarray(calib.grid, dtype=np.int64)
        n = len(self.grid)
        self.kappas = substream(seed, "cutoff").integers(1, n, size=max(self.steps, 1))

    def make(self, idx: np.ndarray, kappa: int) -> LCPFNBatch:
        block = self.reader.read(idx)
        g = self.grid
        cs = block.crash_step
        keep = (cs == 0) | (cs > g[kappa - 1])
        J = block.J[keep][:, g - 1]
        q = np.array([len(g) - 1]) if self.last_only else np.arange(kappa, len(g))
        pos = np.concatenate([np.arange(kappa), q])
        t = np.broadcast_to(self.calib.normalize_time(g[pos].astype(np.float64)),
                            (J.shape[0], len(pos)))
        y_ctx = self.calib.normalize_J(J[:, :kappa])
        y = J[:, q].astype(np.float32)
        return LCPFNBatch(torch.from_numpy(np.ascontiguousarray(t, dtype=np.float32)),
                          torch.from_numpy(y_ctx), torch.from_numpy(y), int(kappa))

    def batch(self, step: int) -> LCPFNBatch:
        return self.make(self.indices(step), int(self.kappas[step]))


# ----------------------------------------------------------------------------- validation
def cutoff_range(n_cutoffs: int, lo: float = 0.20, hi: float = 0.99) -> np.ndarray:
    """1-based cutoff indices ``κ`` between ``lo`` and ``hi`` of the ``n_cutoffs`` possible ones
    (``N_grid - 1`` for both models): positions ``round(lo L) .. round(hi L) - 1``, plus one."""
    start = int(round(lo * n_cutoffs))
    end = max(start + 1, min(int(round(hi * n_cutoffs)), n_cutoffs))
    return np.arange(start, end, dtype=np.int64) + 1


#: Validation ranges reported in ``metrics.json``. ``val_nll`` (20..99 %) is the HPO objective.
VAL_RANGES: dict[str, tuple[float, float]] = {"val_nll": (0.20, 0.99), "val_nll_0_99": (0.0, 0.99)}


# ----------------------------------------------------------------------------- datasets
class DataMissing(FileNotFoundError):
    """The training or validation pool of a model is not (fully) generated."""


@dataclass
class TrainingPools:
    train: PoolReader
    val: PoolReader
    n_examples: int
    train_id: str
    val_id: str


def training_pools(model: str, problem: str, size: str,
                   home: Path | str | None = None) -> TrainingPools:
    """Readers of the pools ``configs/datasets.yaml`` lists for ``model`` on ``problem``. The
    training reader is limited to the first ``SIZES[size]`` rows of the (1M) pool."""
    from esbopfn.generators.datasets import dataset_dir, load_catalog

    if size not in SIZES:
        raise ValueError(f"size must be one of {sorted(SIZES)}, got {size!r}")
    cat = load_catalog()
    train_id, val_id = cat.for_model(model, problem)
    n = SIZES[size]
    hint = f"Generate it with `pfn data gen --for train {model} --problem {problem}`."
    out = []
    for did, count in ((train_id, n), (val_id, cat.resolve(val_id).count)):
        path = dataset_dir(did, home, cat)
        if not (path / "spec.json").is_file():
            raise DataMissing(f"dataset {did} is missing at {path}. {hint}")
        reader = PoolReader(path, limit=count)
        if len(reader) < count:
            raise DataMissing(f"dataset {did} has {len(reader)} of {count} rows. {hint}")
        out.append(reader)
    return TrainingPools(out[0], out[1], n, train_id, val_id)


def simulate_pool(path: Path | str, problem: str, n: int, seed: int = 0,
                  dead_time: bool = True) -> Pool:
    """Write a small ``traj`` pool of ``n`` rows with the Python simulators (one fresh instance
    and a uniform ``θ`` per row, no rejection of unsuccessful trajectories). Meant for tests and
    smoke runs, the training pools come from ``pfn data gen``."""
    from esbopfn.generators.pools import pack_trajectory, pool_from_spec
    from esbopfn.problems import get_problem
    from esbopfn.problems.prior import sample_instance, sample_theta

    prob = get_problem(problem, dead_time=dead_time)
    rng = substream(seed, "simulate_pool", problem)
    rows, spec = [], None
    for _ in range(n):
        inst = sample_instance(prob, rng)
        traj = prob.simulate(sample_theta(prob.spec, rng), inst, rng)
        if spec is None:
            fields = {g: list(getattr(inst, g)) for g in ("phi", "phi_hat", "noise_std", "extras")
                      if getattr(inst, g)}
            spec = pool_from_spec(prob.spec, "traj", fields)
        rows.append(pack_trajectory(spec, traj, inst))
    pool = Pool.create(path, spec, meta={"seed": seed, "count": n, "chunk": n,
                                         "generator": "python"})
    pool.append_chunk(0, np.stack(rows), chunk_size=n)
    pool.mark_finished()
    return pool
