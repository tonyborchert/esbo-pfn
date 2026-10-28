"""XDecPFN data: calibration, stage-1 features and the training batch of thesis M:1122-1132.

A training step takes ``N_sys`` consecutive instances of the set pool ``sets/<p>/xdecpfn``. Each
instance holds ``N_max`` full trajectories (run until they complete or crash) stored on the
decoder grid, plus 128 unrun label trials. Stage 1 runs once over the ``N_sys * N_max``
trajectories. Every instance is then reused for ``N_reuse`` examples, each of which draws

- the number of trajectories ``n ~ U{1..N_max}`` and which ``n`` of the pool (without
  replacement, so the numbers of crashed, unsuccessful and successful trajectories of the example
  follow from the pool's counts),
- one cut probability ``p ~ U(0, 1)``, every trajectory with at least one grid token being cut
  independently with probability ``p`` at a cutoff uniform over its grid tokens,
- ``N_unrun`` unrun queries from the instance's label trials (without replacement).

Cut trajectories and unrun queries are the predictions that enter the loss. The examples of a
batch are sorted by ``n`` so the cross stage can run in chunks padded to the chunk's longest
example.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from esbopfn.core import ProblemSpec
from esbopfn.generators.pools import Pool
from esbopfn.grids import decoder_grid
from esbopfn.predictors.xdecpfn.model import (
    MARK_COMPLETE,
    MARK_CRASH,
    MARK_CUT,
    bin_index,
    equal_mass_edges,
    time_stats,
)
from esbopfn.util.seeds import substream

#: Floor under ``J`` before the logarithm.
LN_FLOOR = 1e-12


# --------------------------------------------------------------------------- calibration
@dataclass
class XCalib:
    """Normalization statistics and bin edges of one XDecPFN model (``calib.json``).

    ``feat_mean`` / ``feat_std`` z-score the stage-1 channels ``[ln J, Delta J, xi..., sum
    xi_dot...]``, ``t_mean`` / ``t_std`` the time over ``{1..T-1}``, ``theta_mean`` /
    ``theta_std`` the controller (the moments of the uniform prior on the box). ``ln_edges`` are
    the ``N_bin + 1`` equal-mass edges of ``ln J_T``.
    """

    T: int
    grid: list[int]
    ln_edges: np.ndarray
    feat_mean: np.ndarray
    feat_std: np.ndarray
    t_mean: float
    t_std: float
    theta_mean: np.ndarray
    theta_std: np.ndarray
    ln_floor: float = LN_FLOOR
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def d_in(self) -> int:
        return int(self.feat_mean.size) + 1

    @property
    def edges(self) -> np.ndarray:
        """Bin edges in raw cost units, ``exp(ln_edges)``."""
        return np.exp(self.ln_edges)

    @property
    def log_widths(self) -> np.ndarray:
        """Log bin widths in ln-J units (the density term of the loss)."""
        return np.log(np.diff(self.ln_edges))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "xdecpfn", "target": "ln J", "T": self.T, "grid": list(self.grid),
            "time_range": [1, self.T - 1], "ln_edges": self.ln_edges.tolist(),
            "edges": self.edges.tolist(), "feat_mean": self.feat_mean.tolist(),
            "feat_std": self.feat_std.tolist(), "t_mean": self.t_mean, "t_std": self.t_std,
            "theta_mean": self.theta_mean.tolist(), "theta_std": self.theta_std.tolist(),
            "ln_floor": self.ln_floor, "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> XCalib:
        arr = lambda k: np.asarray(d[k], dtype=np.float64)
        return cls(T=int(d["T"]), grid=[int(g) for g in d["grid"]], ln_edges=arr("ln_edges"),
                   feat_mean=arr("feat_mean"), feat_std=arr("feat_std"),
                   t_mean=float(d["t_mean"]), t_std=float(d["t_std"]),
                   theta_mean=arr("theta_mean"), theta_std=arr("theta_std"),
                   ln_floor=float(d.get("ln_floor", LN_FLOOR)), source=dict(d.get("source", {})))


def theta_stats(spec: ProblemSpec) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of ``U(Theta)``, the z-score of the controller inputs."""
    lo, hi = spec.theta_lo.astype(np.float64), spec.theta_hi.astype(np.float64)
    return (lo + hi) / 2.0, (hi - lo) / np.sqrt(12.0)


def raw_features(J: np.ndarray, cov: np.ndarray, t: np.ndarray,
                 ln_floor: float = LN_FLOOR) -> np.ndarray:
    """Un-normalized stage-1 inputs ``[ln J, Delta J, xi..., sum xi_dot..., t]`` per token."""
    lnJ = np.log(np.maximum(np.asarray(J, dtype=np.float64), ln_floor))
    return np.concatenate([lnJ[:, None], np.asarray(cov, dtype=np.float64),
                           np.asarray(t, dtype=np.float64)[:, None]], axis=1)


def normalize(feats: np.ndarray, calib: XCalib) -> np.ndarray:
    """z-score raw features (last column = time) with the calibration."""
    mean = np.concatenate([calib.feat_mean, [calib.t_mean]])
    std = np.concatenate([calib.feat_std, [calib.t_std]])
    return ((feats - mean) / std).astype(np.float32)


# --------------------------------------------------------------------------- pool access
@dataclass
class InstanceData:
    """One set-pool instance in model units."""

    tokens: np.ndarray       # (N, L, d_in) float32 normalized, zero after n_tok
    n_tok: np.ndarray        # (N,) tokens before the crash
    theta: np.ndarray        # (N, d) z-scored
    target: np.ndarray       # (N,) raw J_T, inf on crash
    label_theta: np.ndarray  # (M, d) z-scored
    label_target: np.ndarray  # (M,) raw J_T, inf on crash


def check_grid_pool(pool: Pool, spec: ProblemSpec) -> np.ndarray:
    """The decoder grid of ``pool`` after checking it is a grid set pool of ``spec``."""
    grid = decoder_grid(spec)
    steps = pool.spec.steps
    if pool.spec.kind != "set" or pool.spec.resolution != "grid":
        raise ValueError(f"{pool.path} is not a set pool on the decoder grid")
    if steps.size != grid.size + 1 or not np.array_equal(steps[:-1], grid) or steps[-1] != spec.T:
        raise ValueError(f"{pool.path} is stored on a different grid than {spec.name}'s decoder grid")
    return grid


def _block(pool: Pool, rows: np.ndarray, name: str) -> np.ndarray:
    col = pool.spec.column(name)
    return np.asarray(rows[:, col.offset:col.offset + col.length]).reshape(
        (rows.shape[0], *col.shape))


def raw_instance(pool: Pool, i: int) -> dict[str, np.ndarray]:
    """Raw stage-1 features ``(N, L, d_in)``, token counts, theta and targets of instance ``i``."""
    rows = np.asarray(pool.instance_rows(i), dtype=np.float32)
    L = pool.spec.steps.size - 1
    steps = pool.spec.steps[:L].astype(np.float64)
    J = _block(pool, rows, "J").astype(np.float64)
    x = _block(pool, rows, "x").astype(np.float64)
    parts = [x[:, :L]]
    if pool.spec.has("xdot"):
        parts.append(_block(pool, rows, "xdot").astype(np.float64)[:, :L])
    crash = _block(pool, rows, "crash_step")[:, 0].astype(np.int64)
    n_tok = np.where(crash > 0, np.searchsorted(steps, crash, side="left"), L)
    Jt = J[:, :L]
    with np.errstate(invalid="ignore"):
        dJ = np.diff(np.concatenate([np.zeros((J.shape[0], 1)), Jt], axis=1), axis=1)
    lnJ = np.log(np.maximum(Jt, LN_FLOOR))
    t = np.broadcast_to(steps, Jt.shape)
    feats = np.concatenate([lnJ[..., None], dJ[..., None], *parts, t[..., None]], axis=2)
    valid = np.arange(L)[None, :] < n_tok[:, None]
    feats = np.where(valid[..., None], feats, 0.0)
    return {"feats": feats, "valid": valid, "n_tok": n_tok,
            "theta": _block(pool, rows, "theta").astype(np.float64),
            "target": np.where(crash > 0, np.inf, J[:, -1])}


def load_instance(pool: Pool, labels: Pool, i: int, calib: XCalib) -> InstanceData:
    raw = raw_instance(pool, i)
    tok = np.where(raw["valid"][..., None], normalize(raw["feats"], calib), 0.0).astype(np.float32)
    lab = np.asarray(labels.instance_rows(i), dtype=np.float32)
    lab_crash = _block(labels, lab, "crash_step")[:, 0] > 0
    lab_J = _block(labels, lab, "J")[:, -1].astype(np.float64)
    zt = lambda th: ((th - calib.theta_mean) / calib.theta_std).astype(np.float32)
    return InstanceData(tokens=tok, n_tok=raw["n_tok"], theta=zt(raw["theta"]),
                        target=raw["target"], label_theta=zt(_block(labels, lab, "theta")),
                        label_target=np.where(lab_crash, np.inf, lab_J))


def build_calib(pool: Pool, spec: ProblemSpec, n_bins: int = 1000,
                n_instances: int = 2000) -> XCalib:
    """Statistics and ``ln J_T`` bin edges from the first ``n_instances`` instances of ``pool``.

    The channel statistics are taken over all tokens before a crash, the edges over the horizon
    values of the non-crashed pool trajectories and label trials (prior samples of ``J_T``).
    """
    grid = check_grid_pool(pool, spec)
    n_inst = len(pool) // pool.rows_per_unit
    k = max(1, min(n_instances, n_inst))
    labels = pool.labels()
    feats, targets = [], []
    for i in range(k):
        raw = raw_instance(pool, i)
        feats.append(raw["feats"][raw["valid"]][:, :-1])
        targets.append(raw["target"])
        lab = np.asarray(labels.instance_rows(i), dtype=np.float32)
        lab_ok = _block(labels, lab, "crash_step")[:, 0] <= 0
        targets.append(_block(labels, lab, "J")[lab_ok, -1].astype(np.float64))
    f = np.concatenate(feats)
    f = f[np.all(np.isfinite(f), axis=1)]
    y = np.concatenate(targets)
    y = y[np.isfinite(y)]
    std = f.std(0)
    std = np.where(std > 1e-12, std, 1.0)
    t_mean, t_std = time_stats(spec.T - 1)
    th_mean, th_std = theta_stats(spec)
    return XCalib(T=spec.T, grid=grid.tolist(),
                  ln_edges=equal_mass_edges(np.log(np.maximum(y, LN_FLOOR)), n_bins),
                  feat_mean=f.mean(0), feat_std=std, t_mean=t_mean, t_std=t_std,
                  theta_mean=th_mean, theta_std=th_std,
                  source={"pool": str(pool.path), "instances": k, "tokens": int(f.shape[0]),
                          "targets": int(y.size)})


# --------------------------------------------------------------------------- examples
@dataclass
class Examples:
    """The examples drawn from one instance (indices into its pool and label trials)."""

    traj: list[np.ndarray]     # per example: pool rows
    pos: list[np.ndarray]      # per example: last observed token index (-1 = none)
    marker: list[np.ndarray]   # per example: complete / cut / crash
    labels: np.ndarray         # (E, Q) label trial indices


def draw_examples(n_tok: np.ndarray, crashed: np.ndarray, n_examples: int, n_unrun: int,
                  n_labels: int, rng: np.random.Generator) -> Examples:
    """Thesis M:1122-1132: per example ``n ~ U{1..N_max}`` trajectories without replacement, one
    cut probability ``p ~ U(0, 1)``, cutoffs uniform over a trajectory's grid tokens, and
    ``n_unrun`` label trials without replacement."""
    N = n_tok.size
    traj, pos, marker = [], [], []
    labels = np.empty((n_examples, n_unrun), dtype=np.int64)
    for e in range(n_examples):
        n = int(rng.integers(1, N + 1))
        idx = rng.choice(N, size=n, replace=False)
        nt = n_tok[idx]
        p = rng.random()
        cut = (rng.random(n) < p) & (nt > 0)
        cut_pos = np.floor(rng.random(n) * np.maximum(nt, 1)).astype(np.int64)
        traj.append(idx)
        pos.append(np.where(cut, cut_pos, nt - 1))
        marker.append(np.where(cut, MARK_CUT, np.where(crashed[idx], MARK_CRASH, MARK_COMPLETE)))
        labels[e] = rng.choice(n_labels, size=n_unrun, replace=False)
    return Examples(traj, pos, marker, labels)


@dataclass
class XBatch:
    """A training batch: stage-1 inputs of all pool trajectories and the stage-2 examples.

    Examples are sorted by their number of trajectories (descending), ``ex_n`` holds the counts.
    """

    tokens: torch.Tensor      # (S, L, d_in)
    theta: torch.Tensor       # (S, d)
    ex_traj: torch.Tensor     # (E, n) rows into S
    ex_pos: torch.Tensor      # (E, n) token index, -1 = none
    ex_marker: torch.Tensor   # (E, n)
    ex_valid: torch.Tensor    # (E, n)
    ex_scored: torch.Tensor   # (E, n) cut trajectories, predicted at the horizon
    ex_bin: torch.Tensor      # (E, n) target bin of ln J_T
    ex_crash: torch.Tensor    # (E, n) target is a crash
    q_theta: torch.Tensor     # (E, Q, d)
    q_bin: torch.Tensor       # (E, Q)
    q_crash: torch.Tensor     # (E, Q)
    ex_n: tuple[int, ...] = ()


def assemble(instances: list[InstanceData], draws: list[Examples], calib: XCalib) -> XBatch:
    """Stack the instances' trajectories and pad their examples into one :class:`XBatch`."""
    offsets = np.cumsum([0] + [inst.n_tok.size for inst in instances])
    rows: list[tuple[np.ndarray, ...]] = []
    for g, (inst, ex) in enumerate(zip(instances, draws)):
        tgt_bin = bin_index(np.log(np.maximum(inst.target, LN_FLOOR)), calib.ln_edges)
        lab_bin = bin_index(np.log(np.maximum(inst.label_target, LN_FLOOR)), calib.ln_edges)
        for e in range(len(ex.traj)):
            idx = ex.traj[e]
            q = ex.labels[e]
            rows.append((idx + offsets[g], ex.pos[e], ex.marker[e], ex.marker[e] == MARK_CUT,
                         tgt_bin[idx], ~np.isfinite(inst.target[idx]), inst.label_theta[q],
                         lab_bin[q], ~np.isfinite(inst.label_target[q])))
    rows.sort(key=lambda r: -r[0].size)
    E, n_max = len(rows), max(r[0].size for r in rows)
    Q, d = rows[0][6].shape
    ex = {k: np.zeros((E, n_max), dtype=np.int64) for k in ("traj", "marker", "bin")}
    ex["pos"] = np.full((E, n_max), -1, dtype=np.int64)
    flags = {k: np.zeros((E, n_max), dtype=bool) for k in ("valid", "scored", "crash")}
    q_theta = np.zeros((E, Q, d), dtype=np.float32)
    q_bin = np.zeros((E, Q), dtype=np.int64)
    q_crash = np.zeros((E, Q), dtype=bool)
    for e, r in enumerate(rows):
        n = r[0].size
        ex["traj"][e, :n], ex["pos"][e, :n], ex["marker"][e, :n] = r[0], r[1], r[2]
        flags["valid"][e, :n], flags["scored"][e, :n] = True, r[3]
        ex["bin"][e, :n], flags["crash"][e, :n] = r[4], r[5]
        q_theta[e], q_bin[e], q_crash[e] = r[6], r[7], r[8]
    t = torch.from_numpy
    return XBatch(tokens=t(np.concatenate([i.tokens for i in instances])),
                  theta=t(np.concatenate([i.theta for i in instances])),
                  ex_traj=t(ex["traj"]), ex_pos=t(ex["pos"]), ex_marker=t(ex["marker"]),
                  ex_valid=t(flags["valid"]), ex_scored=t(flags["scored"]),
                  ex_bin=t(ex["bin"]), ex_crash=t(flags["crash"]), q_theta=t(q_theta),
                  q_bin=t(q_bin), q_crash=t(q_crash), ex_n=tuple(int(r[0].size) for r in rows))


# --------------------------------------------------------------------------- batch streams
class SetPoolBatches:
    """Training or validation batches of ``n_sys`` instances from a set pool.

    Batch ``b`` uses instances ``b * n_sys ..`` in pool order (wrapping around only if the pool
    is shorter than the run) and draws its examples from ``substream(seed, stream, b)``, so a
    resumed run continues the identical stream.
    """

    def __init__(self, pool: Pool, spec: ProblemSpec, calib: XCalib, *, n_sys: int,
                 n_reuse: int, n_unrun: int, seed: int, stream: str = "train"):
        check_grid_pool(pool, spec)
        self.pool, self.labels, self.calib = pool, pool.labels(), calib
        self.n_sys, self.n_reuse, self.n_unrun = n_sys, n_reuse, n_unrun
        self.seed, self.stream = seed, stream
        self.n_instances = len(pool) // pool.rows_per_unit
        if self.n_instances == 0:
            raise ValueError(f"{pool.path} holds no complete instance")

    def __len__(self) -> int:
        return (self.n_instances + self.n_sys - 1) // self.n_sys

    def batch(self, b: int) -> XBatch:
        rng = substream(self.seed, "xdecpfn", self.stream, b)
        insts, draws = [], []
        for g in range(self.n_sys):
            i = (b * self.n_sys + g) % self.n_instances
            inst = load_instance(self.pool, self.labels, i, self.calib)
            insts.append(inst)
            draws.append(draw_examples(inst.n_tok, ~np.isfinite(inst.target), self.n_reuse,
                                       self.n_unrun, inst.label_target.size, rng))
        return assemble(insts, draws, self.calib)

    def stream_from(self, start: int, prefetch: int = 2) -> Iterator[XBatch]:
        """Endless batches ``start, start + 1, ...`` built in a background thread."""
        return _prefetch((self.batch(b) for b in _count(start)), prefetch)


def _count(start: int) -> Iterator[int]:
    b = start
    while True:
        yield b
        b += 1


def _prefetch(it: Iterator[Any], depth: int) -> Iterator[Any]:
    if depth <= 0:
        yield from it
        return
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def work() -> None:
        try:
            for item in it:
                if stop.is_set():
                    return
                q.put(item)
        except Exception as exc:  # noqa: BLE001  (re-raised in the consumer)
            q.put(exc)
        q.put(done)

    threading.Thread(target=work, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
