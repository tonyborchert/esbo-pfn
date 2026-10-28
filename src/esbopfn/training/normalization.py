"""Input normalization and the ``calib.json`` of a trained model.

Every continuous input channel is z-scored with a per-channel mean and standard deviation
estimated from samples of the prior (a seeded subset of the training pool). The time ``t`` is
z-scored over its known range instead, ``{1, ..., T}`` for LC-PFN and ``{1, ..., T - 1}`` for
DecPFN, which is the mean and standard deviation of the uniform distribution on that set. The
Gaussian head z-scores its target with the statistics of the cost channel ``J`` and the bar heads
place their edges in the same normalized space, so the change of variables back to raw cost
units adds ``log std_J`` to every finite-target NLL.

``calib.json`` layout (version 1)::

    {
      "version": 1,
      "model": "decpfn",                  # model family incl. variant, e.g. decpfn-nostates
      "problem": "cartpole",
      "head": "binned",                   # binned | linear | gaussian
      "features": ["J", "dJ", "x_cart", ..., "t"],
      "input": {"mean": [...], "std": [...]},   # per feature, the last entry is the time
      "time": {"lo": 1, "hi": 999, "mean": 500.0, "std": 288.4},
      "target": {"mean": m_J, "std": s_J},      # statistics of the J channel
      "edges": [...] | null,              # N_bin + 1 bar edges in normalized target space
      "n_bin": 1000,
      "grid": [15, 30, ...],              # token steps of the model
      "T": 1000,
      "samples": {"rows": 20000, "seed": 0, "pool": "<path>"}
    }
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CALIB_VERSION = 1

#: Rows drawn from the training pool to estimate the input statistics.
STAT_ROWS = 20_000
#: Rows drawn to place the bar edges on horizon samples (DecPFN, only ``J_T`` is read).
EDGE_ROWS = 100_000


def time_stats(lo: int, hi: int) -> tuple[float, float]:
    """Mean and standard deviation of the uniform distribution on ``{lo, ..., hi}``."""
    n = hi - lo + 1
    if n < 1:
        raise ValueError(f"empty time range [{lo}, {hi}]")
    mean = 0.5 * (lo + hi)
    std = math.sqrt((n * n - 1) / 12.0) if n > 1 else 1.0
    return mean, std


def channel_stats(values: np.ndarray, valid: np.ndarray | None = None,
                  min_std: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation over the last axis of ``values`` (``(..., F)``),
    counting only finite entries (and ``valid`` ones when a mask ``(...)`` is given).
    A channel with (near) zero spread gets standard deviation 1."""
    v = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    ok = np.isfinite(v)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool).reshape(-1)[:, None]
    n = ok.sum(axis=0)
    if np.any(n == 0):
        raise ValueError("a channel has no finite sample to estimate its statistics")
    vz = np.where(ok, v, 0.0)
    mean = vz.sum(axis=0) / n
    var = (np.where(ok, (v - mean) ** 2, 0.0)).sum(axis=0) / n
    std = np.sqrt(var)
    std = np.where(std > min_std * np.maximum(1.0, np.abs(mean)), std, 1.0)
    return mean, std


def strictly_increasing(edges: np.ndarray, rel: float = 1e-6) -> np.ndarray:
    """Push repeated edges apart by a tiny step so every bin has positive width."""
    e = np.asarray(edges, dtype=np.float64).copy()
    step = rel * max(float(e[-1] - e[0]), 1.0)
    for i in range(1, len(e)):
        if e[i] <= e[i - 1]:
            e[i] = e[i - 1] + step
    return e


def bar_edges(samples: np.ndarray, n_bin: int, kind: str) -> np.ndarray:
    """``n_bin + 1`` edges from finite samples: ``binned`` = equal-mass quantiles (the first and
    last edge are the sample extremes), ``linear`` = equal width over the sample range."""
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 2:
        raise ValueError("need at least two finite samples to place bar edges")
    if kind == "binned":
        edges = np.quantile(s, np.linspace(0.0, 1.0, n_bin + 1))
    elif kind == "linear":
        edges = np.linspace(s.min(), s.max(), n_bin + 1)
    else:
        raise ValueError(f"no bar edges for head {kind!r}")
    return strictly_increasing(edges)


@dataclass
class Calib:
    """Normalization statistics and head geometry of one model (the content of ``calib.json``)."""

    model: str
    problem: str
    head: str
    features: list[str]
    input_mean: np.ndarray
    input_std: np.ndarray
    time_lo: int
    time_hi: int
    target_mean: float
    target_std: float
    edges: np.ndarray | None
    grid: np.ndarray
    T: int
    n_bin: int = 1000
    samples: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Inputs without the time channel."""
        return len(self.features) - 1

    def normalize_inputs(self, values: np.ndarray) -> np.ndarray:
        """z-score ``(..., F)`` values whose last feature is the raw time."""
        return ((np.asarray(values, dtype=np.float64) - self.input_mean) / self.input_std
                ).astype(np.float32)

    def normalize_time(self, t: np.ndarray) -> np.ndarray:
        return ((np.asarray(t, dtype=np.float64) - self.input_mean[-1]) / self.input_std[-1]
                ).astype(np.float32)

    def normalize_J(self, J: np.ndarray) -> np.ndarray:
        return ((np.asarray(J, dtype=np.float64) - self.target_mean) / self.target_std
                ).astype(np.float32)

    def raw_edges(self) -> np.ndarray:
        if self.edges is None:
            raise ValueError("the gaussian head has no bar edges")
        return strictly_increasing(self.edges * self.target_std + self.target_mean, rel=1e-12)

    @property
    def log_jacobian(self) -> float:
        """``log std_J``: added to a finite-target NLL to express it in raw cost units."""
        return math.log(self.target_std)

    def to_dict(self) -> dict[str, Any]:
        mean_t, std_t = time_stats(self.time_lo, self.time_hi)
        return {
            "version": CALIB_VERSION, "model": self.model, "problem": self.problem,
            "head": self.head, "features": list(self.features),
            "input": {"mean": [float(v) for v in self.input_mean],
                      "std": [float(v) for v in self.input_std]},
            "time": {"lo": int(self.time_lo), "hi": int(self.time_hi), "mean": mean_t,
                     "std": std_t},
            "target": {"mean": float(self.target_mean), "std": float(self.target_std)},
            "edges": None if self.edges is None else [float(v) for v in self.edges],
            "n_bin": int(self.n_bin), "grid": [int(v) for v in self.grid], "T": int(self.T),
            "samples": dict(self.samples),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Calib:
        if int(raw.get("version", 0)) != CALIB_VERSION:
            raise ValueError(f"unsupported calib.json version {raw.get('version')}")
        edges = raw.get("edges")
        return cls(
            model=raw["model"], problem=raw["problem"], head=raw["head"],
            features=list(raw["features"]),
            input_mean=np.asarray(raw["input"]["mean"], dtype=np.float64),
            input_std=np.asarray(raw["input"]["std"], dtype=np.float64),
            time_lo=int(raw["time"]["lo"]), time_hi=int(raw["time"]["hi"]),
            target_mean=float(raw["target"]["mean"]), target_std=float(raw["target"]["std"]),
            edges=None if edges is None else np.asarray(edges, dtype=np.float64),
            grid=np.asarray(raw["grid"], dtype=np.int64), T=int(raw["T"]),
            n_bin=int(raw.get("n_bin", 1000)), samples=dict(raw.get("samples", {})),
        )


def make_calib(*, model: str, problem: str, head: str, features: list[str],
               feature_values: np.ndarray, feature_valid: np.ndarray | None,
               time_range: tuple[int, int], J_index: int, edge_samples: np.ndarray,
               grid: np.ndarray, T: int, n_bin: int = 1000,
               samples: dict[str, Any] | None = None) -> Calib:
    """Estimate a :class:`Calib`.

    ``feature_values`` is ``(..., F - 1)`` without the time, ``features`` names all ``F``
    inputs with ``"t"`` last, ``J_index`` locates the cost channel whose statistics normalize the
    target, ``edge_samples`` are raw cost samples for the bar edges (ignored by ``gaussian``).
    """
    mean, std = channel_stats(feature_values, feature_valid)
    t_mean, t_std = time_stats(*time_range)
    input_mean = np.concatenate([mean, [t_mean]])
    input_std = np.concatenate([std, [t_std]])
    target_mean, target_std = float(mean[J_index]), float(std[J_index])
    edges = None
    if head != "gaussian":
        z = (np.asarray(edge_samples, dtype=np.float64) - target_mean) / target_std
        edges = bar_edges(z, n_bin, head)
    return Calib(model=model, problem=problem, head=head, features=list(features),
                 input_mean=input_mean, input_std=input_std, time_lo=time_range[0],
                 time_hi=time_range[1], target_mean=target_mean, target_std=target_std,
                 edges=edges, grid=np.asarray(grid, dtype=np.int64), T=int(T), n_bin=n_bin,
                 samples=dict(samples or {}))
