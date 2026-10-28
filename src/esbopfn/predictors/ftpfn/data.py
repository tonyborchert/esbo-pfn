"""FT-PFN data: calibration, tokenization and packed batches of context shards.

A training example is one context of ``contexts/<p>/ftpfn`` (a fresh trajectory set): the tokens
of every trial on the 8-point curve grid (grid steps ``t_l <= T_i``, then ``T_i`` itself if the
trial is truncated, a crash as ``J = inf``), one horizon query for every truncated trial and the
16 unrun queries. A batch of 32 consecutive contexts is packed into rows (first-fit decreasing
with the longest context as the row length) under the PFN block mask of
:func:`~esbopfn.predictors.ftpfn.model.block_mask`.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from esbopfn.core import ProblemSpec, Trial
from esbopfn.generators.pools import STATUS_TRUNCATED, ContextExample, ContextStore
from esbopfn.grids import FTPFN_N_GRID, curve_grid
from esbopfn.predictors.ftpfn.model import KIND_CRASH, KIND_PAD, KIND_QUERY, KIND_VALUE
from esbopfn.predictors.xdecpfn.data import _prefetch, theta_stats
from esbopfn.predictors.xdecpfn.model import bin_index, equal_mass_edges, time_stats

QUERY_HORIZON, QUERY_UNRUN = 0, 1


# --------------------------------------------------------------------------- calibration
@dataclass
class FCalib:
    """Normalization and bins of one FT-PFN model (``calib.json``).

    Raw cost inputs are clamped into ``[y_lo, y_hi]`` (the range of the prior samples, ``y_hi``
    the top bin edge) and z-scored with ``y_mean`` / ``y_std``. Time is z-scored over
    ``{1..T}``, the controller with the moments of the uniform prior on the box.
    """

    T: int
    grid: list[int]
    edges: np.ndarray
    y_mean: float
    y_std: float
    y_lo: float
    y_hi: float
    t_mean: float
    t_std: float
    theta_mean: np.ndarray
    theta_std: np.ndarray
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def log_widths(self) -> np.ndarray:
        return np.log(np.diff(self.edges))

    def to_dict(self) -> dict[str, Any]:
        return {"model": "ftpfn", "target": "raw J", "T": self.T, "grid": list(self.grid),
                "time_range": [1, self.T], "edges": self.edges.tolist(), "y_mean": self.y_mean,
                "y_std": self.y_std, "y_lo": self.y_lo, "y_hi": self.y_hi, "t_mean": self.t_mean,
                "t_std": self.t_std, "theta_mean": self.theta_mean.tolist(),
                "theta_std": self.theta_std.tolist(), "source": self.source}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FCalib:
        f = lambda k: float(d[k])
        return cls(T=int(d["T"]), grid=[int(g) for g in d["grid"]],
                   edges=np.asarray(d["edges"], dtype=np.float64), y_mean=f("y_mean"),
                   y_std=f("y_std"), y_lo=f("y_lo"), y_hi=f("y_hi"), t_mean=f("t_mean"),
                   t_std=f("t_std"), theta_mean=np.asarray(d["theta_mean"], dtype=np.float64),
                   theta_std=np.asarray(d["theta_std"], dtype=np.float64),
                   source=dict(d.get("source", {})))


def build_calib(store: ContextStore, spec: ProblemSpec, n_bins: int = 1000,
                n_examples: int = 20000) -> FCalib:
    """Bin edges from the total costs of the first ``n_examples`` contexts (all non-crashed
    trials and unrun queries), value statistics from their finite context tokens."""
    k = min(n_examples, len(store))
    if k == 0:
        raise ValueError(f"{store.path} holds no context")
    vals, tgts = [], []
    idx = ContextIndex(store)
    for i in range(k):
        ex = idx.example(i)
        keep = ex.tok_t > 0
        v = ex.tok_J[keep]
        vals.append(v[np.isfinite(v)])
        tgts += [ex.target[np.isfinite(ex.target)], ex.unrun_target[np.isfinite(ex.unrun_target)]]
    y = np.concatenate(tgts).astype(np.float64)
    edges = equal_mass_edges(y, n_bins)
    v = np.concatenate(vals).astype(np.float64)
    lo, hi = float(min(v.min(), edges[0])), float(edges[-1])
    vc = np.clip(v, lo, hi)
    t_mean, t_std = time_stats(spec.T)
    th_mean, th_std = theta_stats(spec)
    return FCalib(T=spec.T, grid=curve_grid(spec, FTPFN_N_GRID).tolist(), edges=edges,
                  y_mean=float(vc.mean()), y_std=float(max(vc.std(), 1e-12)), y_lo=lo, y_hi=hi,
                  t_mean=t_mean, t_std=t_std, theta_mean=th_mean, theta_std=th_std,
                  source={"store": str(store.path), "examples": k, "targets": int(y.size),
                          "values": int(v.size)})


# --------------------------------------------------------------------------- tokens
@dataclass
class ContextTokens:
    """One context in raw units: context tokens and queries (``inf`` = crash)."""

    theta: np.ndarray       # (n, d)
    t: np.ndarray           # (n,)
    J: np.ndarray           # (n,)
    q_theta: np.ndarray     # (Q, d)
    q_target: np.ndarray    # (Q,)  nan where unknown (inference)
    q_kind: np.ndarray      # (Q,)

    def __len__(self) -> int:
        return int(self.t.size + self.q_kind.size)


def example_tokens(ex: ContextExample) -> ContextTokens:
    """Tokens and queries of a stored context: horizon queries for the truncated trials plus the
    unrun queries."""
    keep = ex.tok_t > 0
    rows = np.nonzero(keep)[0]
    trunc = ex.status == STATUS_TRUNCATED
    q_theta = np.concatenate([ex.theta[trunc], ex.unrun_theta]).astype(np.float64)
    return ContextTokens(theta=ex.theta[rows].astype(np.float64),
                         t=ex.tok_t[keep].astype(np.float64), J=ex.tok_J[keep].astype(np.float64),
                         q_theta=q_theta,
                         q_target=np.concatenate([ex.target[trunc], ex.unrun_target]).astype(np.float64),
                         q_kind=np.concatenate([np.full(int(trunc.sum()), QUERY_HORIZON),
                                                np.full(ex.unrun_target.size, QUERY_UNRUN)]))


def trial_tokens(trial: Trial, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Steps and values of one observed trial (thesis eq. ftpfn:context).

    The trial's length ``T_i`` is its crash step if it crashed, else ``t_obs``. Its tokens are the
    grid steps ``t_l <= T_i`` and, if ``T_i < T``, the step ``T_i`` itself, with ``J = inf`` from
    the crash step on. An unrun trial has no token.
    """
    if trial.unrun:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    T = trial.T
    Ti = int(trial.traj.crash_step) if trial.crashed else int(trial.t_obs)
    steps = grid[grid <= Ti]
    if Ti < T:
        steps = np.concatenate([steps, [Ti]])
    J = np.asarray(trial.traj.J, dtype=np.float64)[steps - 1]
    return steps.astype(np.int64), J


# --------------------------------------------------------------------------- packing
@dataclass
class FBatch:
    """Packed rows of several contexts and the bookkeeping of their queries."""

    theta: torch.Tensor     # (R, L, d) z-scored
    t: torch.Tensor         # (R, L)
    y: torch.Tensor         # (R, L)
    kind: torch.Tensor      # (R, L)
    ctx: torch.Tensor       # (R, L) context id, -1 = padding
    q_row: torch.Tensor     # (Q,)
    q_pos: torch.Tensor     # (Q,)
    q_bin: torch.Tensor     # (Q,)
    q_crash: torch.Tensor   # (Q,)
    q_valid: torch.Tensor   # (Q,) target known
    q_example: torch.Tensor  # (Q,)
    q_kind: torch.Tensor    # (Q,)
    n_examples: int = 0


def _rows_first_fit(lengths: Sequence[int]) -> tuple[list[int], list[int], int]:
    """Row and offset of every context, first-fit decreasing into rows of the longest length."""
    cap = max(lengths)
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    free: list[int] = []
    row, off = [0] * len(lengths), [0] * len(lengths)
    for i in order:
        for r, f in enumerate(free):
            if f >= lengths[i]:
                row[i], off[i] = r, cap - f
                free[r] -= lengths[i]
                break
        else:
            row[i], off[i] = len(free), 0
            free.append(cap - lengths[i])
    return row, off, cap


def pack(contexts: Sequence[ContextTokens], calib: FCalib) -> FBatch:
    """Normalize and pack contexts into rows (one context never spans two rows)."""
    lengths = [max(len(c), 1) for c in contexts]
    row, off, L = _rows_first_fit(lengths)
    R = max(row) + 1
    d = calib.theta_mean.size
    theta = np.zeros((R, L, d), dtype=np.float32)
    t = np.zeros((R, L), dtype=np.float32)
    y = np.zeros((R, L), dtype=np.float32)
    kind = np.full((R, L), KIND_PAD, dtype=np.int64)
    ctx = np.full((R, L), -1, dtype=np.int64)
    q_row, q_pos, q_tgt, q_ex, q_kind = [], [], [], [], []
    for e, c in enumerate(contexts):
        r, a = row[e], off[e]
        n, Q = c.t.size, c.q_kind.size
        tok = slice(a, a + n)
        qs = slice(a + n, a + n + Q)
        crash = ~np.isfinite(c.J)
        theta[r, tok] = (c.theta - calib.theta_mean) / calib.theta_std
        theta[r, qs] = (c.q_theta - calib.theta_mean) / calib.theta_std
        t[r, tok] = (c.t - calib.t_mean) / calib.t_std
        t[r, qs] = (calib.T - calib.t_mean) / calib.t_std
        yv = (np.clip(np.where(crash, calib.y_hi, c.J), calib.y_lo, calib.y_hi) - calib.y_mean) \
            / calib.y_std
        y[r, tok] = np.where(crash, 0.0, yv)
        kind[r, tok] = np.where(crash, KIND_CRASH, KIND_VALUE)
        kind[r, qs] = KIND_QUERY
        ctx[r, a:a + n + Q] = e
        q_row += [r] * Q
        q_pos += list(range(a + n, a + n + Q))
        q_tgt.append(c.q_target)
        q_ex += [e] * Q
        q_kind.append(c.q_kind)
    tgt = np.concatenate(q_tgt) if q_tgt else np.zeros(0)
    crashed = np.isposinf(tgt)
    valid = ~np.isnan(tgt)
    b = bin_index(np.where(crashed | ~valid, calib.edges[0], tgt), calib.edges)
    T = torch.from_numpy
    return FBatch(theta=T(theta), t=T(t), y=T(y), kind=T(kind), ctx=T(ctx),
                  q_row=torch.as_tensor(q_row, dtype=torch.long),
                  q_pos=torch.as_tensor(q_pos, dtype=torch.long), q_bin=T(np.asarray(b)),
                  q_crash=T(crashed), q_valid=T(valid),
                  q_example=torch.as_tensor(q_ex, dtype=torch.long),
                  q_kind=T(np.concatenate(q_kind).astype(np.int64)) if q_kind
                  else torch.zeros(0, dtype=torch.long),
                  n_examples=len(contexts))


# --------------------------------------------------------------------------- batch streams
class ContextIndex:
    """Random access to the examples of a :class:`ContextStore` by global index."""

    def __init__(self, store: ContextStore):
        self.store = store
        self.starts = [int(s.meta["first_example"]) for s in store.shards]
        self.n = len(store)

    def example(self, i: int) -> ContextExample:
        k = bisect.bisect_right(self.starts, i) - 1
        if k < 0:
            raise IndexError(i)
        shard = self.store.shards[k]
        j = i - self.starts[k]
        if not 0 <= j < len(shard):
            raise IndexError(f"example {i} is not in a finished shard")
        return shard.example(j)


class ContextBatches:
    """Batch ``b`` = contexts ``[b B, (b + 1) B)`` in store order (wrapping around only if the
    store is shorter than the run)."""

    def __init__(self, store: ContextStore, calib: FCalib, batch_size: int = 32):
        self.index = ContextIndex(store)
        self.calib = calib
        self.batch_size = batch_size
        if self.index.n == 0:
            raise ValueError(f"{store.path} holds no context")

    def __len__(self) -> int:
        return (self.index.n + self.batch_size - 1) // self.batch_size

    def batch(self, b: int) -> FBatch:
        B, n = self.batch_size, self.index.n
        return self.pack_ids([(b * B + k) % n for k in range(B)])

    def pack_ids(self, ids: Sequence[int]) -> FBatch:
        return pack([example_tokens(self.index.example(i)) for i in ids], self.calib)

    def partition(self) -> list[FBatch]:
        """Every context exactly once, in batches of at most ``batch_size`` (validation)."""
        n, B = self.index.n, self.batch_size
        return [self.pack_ids(range(a, min(a + B, n))) for a in range(0, n, B)]

    def stream_from(self, start: int, prefetch: int = 2) -> Iterator[FBatch]:
        def gen() -> Iterator[FBatch]:
            b = start
            while True:
                yield self.batch(b)
                b += 1
        return _prefetch(gen(), prefetch)
