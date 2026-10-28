"""Crash-aware output heads (thesis, crash-aware output heads).

Both heads score a raw cost target ``y`` (``+inf`` = crash) against the model outputs. They work in
the normalized target space ``z = (y - mean_J) / std_J`` of :class:`~.normalization.Calib`, so
their NLL is a density in ``z``. ``raw_units=True`` adds ``log std_J`` to finite targets, the
Jacobian that turns it into a density in raw cost units.

:class:`BarHead` has ``N_bin + 1`` outputs, the last one is the crash atom. With the softmax
masses ``a_1 .. a_N, a_inf`` the loss is ``-log a_inf`` on a crash, else
``-log a_k + log(b_k - b_{k-1}) = -log(1 - w_inf) - log w_k + log(b_k - b_{k-1})`` for the bin
``b_{k-1} <= z < b_k``. Targets outside ``[b_0, b_N)`` are clamped into the outermost bins.

:class:`GaussianHead` has three outputs ``(mu, log sigma^2, logit w_inf)``, the loss is
``-log w_inf`` on a crash, else
``-log(1 - w_inf) + 0.5 log(2 pi sigma^2) + (z - mu)^2 / (2 sigma^2)``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from esbopfn.predictors.base import BarPrediction, GaussianPrediction
from esbopfn.training.normalization import Calib

HEADS = ("binned", "linear", "gaussian")
LOG_2PI = math.log(2.0 * math.pi)


class BarHead:
    """Crash-aware bar distribution with fixed edges in normalized target space."""

    def __init__(self, edges: np.ndarray, target_mean: float = 0.0, target_std: float = 1.0):
        e = np.asarray(edges, dtype=np.float64)
        if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0):
            raise ValueError("edges must be a strictly increasing 1-d array")
        self.edges = e
        self.n_bin = e.size - 1
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self._t_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def n_outputs(self) -> int:
        return self.n_bin + 1

    def _tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in self._t_cache:
            e = torch.as_tensor(self.edges, dtype=torch.float32, device=device)
            logw = torch.as_tensor(np.log(np.diff(self.edges)), dtype=torch.float32, device=device)
            self._t_cache[device] = (e, logw)
        return self._t_cache[device]

    def bin_index(self, z: torch.Tensor) -> torch.Tensor:
        """Bin of every normalized finite target, clamped into ``0 .. N_bin - 1``."""
        edges, _ = self._tensors(z.device)
        idx = torch.bucketize(z.contiguous(), edges, right=True) - 1
        return idx.clamp(0, self.n_bin - 1)

    def nll(self, out: torch.Tensor, y: torch.Tensor, raw_units: bool = False) -> torch.Tensor:
        """Per-target NLL. ``out`` is ``(..., N_bin + 1)``, ``y`` raw costs ``(...)``."""
        logp = F.log_softmax(out.float(), dim=-1)
        crash = torch.isinf(y)
        z = torch.where(crash, torch.zeros_like(y), (y - self.target_mean) / self.target_std)
        idx = self.bin_index(z.float())
        _, logw = self._tensors(out.device)
        finite = -logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1) + logw[idx]
        if raw_units:
            finite = finite + math.log(self.target_std)
        return torch.where(crash, -logp[..., -1], finite)

    def predictions(self, out: torch.Tensor) -> list[BarPrediction]:
        """Bar predictions with raw-unit edges for a ``(n, N_bin + 1)`` output."""
        a = out.detach().double().cpu().numpy().reshape(-1, self.n_outputs)
        a = a - a.max(axis=1, keepdims=True)
        p = np.exp(a)
        p /= p.sum(axis=1, keepdims=True)
        edges = self.edges * self.target_std + self.target_mean
        return [BarPrediction(edges=edges, probs=row[:-1], p_crash=float(row[-1])) for row in p]


class GaussianHead:
    """Crash-aware Gaussian on the normalized target."""

    n_outputs = 3

    def __init__(self, target_mean: float = 0.0, target_std: float = 1.0):
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def nll(self, out: torch.Tensor, y: torch.Tensor, raw_units: bool = False) -> torch.Tensor:
        o = out.float()
        mu, log_var, logit = o[..., 0], o[..., 1], o[..., 2]
        crash = torch.isinf(y)
        z = torch.where(crash, torch.zeros_like(y), (y - self.target_mean) / self.target_std)
        finite = (-F.logsigmoid(-logit) + 0.5 * (LOG_2PI + log_var)
                  + 0.5 * (z.float() - mu) ** 2 * torch.exp(-log_var))
        if raw_units:
            finite = finite + math.log(self.target_std)
        return torch.where(crash, -F.logsigmoid(logit), finite)

    def predictions(self, out: torch.Tensor) -> list[GaussianPrediction]:
        o = out.detach().double().cpu().numpy().reshape(-1, 3)
        res = []
        for mu, log_var, logit in o:
            p_crash = float(1.0 / (1.0 + math.exp(-logit))) if logit > -700 else 0.0
            res.append(GaussianPrediction(
                mean=float(mu * self.target_std + self.target_mean),
                var=float(math.exp(min(log_var, 700.0)) * self.target_std ** 2),
                p_crash=min(max(p_crash, 0.0), 1.0)))
        return res


Head = BarHead | GaussianHead


def make_head(calib: Calib) -> Head:
    """The head described by a :class:`Calib`."""
    if calib.head == "gaussian":
        return GaussianHead(calib.target_mean, calib.target_std)
    if calib.head in ("binned", "linear"):
        if calib.edges is None:
            raise ValueError(f"calib of a {calib.head} head has no edges")
        return BarHead(calib.edges, calib.target_mean, calib.target_std)
    raise ValueError(f"unknown head {calib.head!r}, expected one of {HEADS}")


def n_outputs(head: str, n_bin: int = 1000) -> int:
    return 3 if head == "gaussian" else n_bin + 1
