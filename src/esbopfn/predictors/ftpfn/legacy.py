"""The FT-PFN checkpoints used for the thesis results, served as ``ftpfn/<p>/thesis/default``.

These weights were trained with an earlier implementation whose embeddings differ from
:class:`~esbopfn.predictors.ftpfn.model.FTPFN` (planning record of the old code base):

- time enters as ``MLP([t / T, ln(t / T)])``, the value as ``MLP(z(J))`` with ``J`` clamped into
  the calibrated range, the controller as ``Linear(z(theta))``,
- every trial has a token at its last observed step with a marker vector (cutoff or crash, the
  crash token sitting on the step before the crash), on the grid ``round(T j / 8)``,
- every context starts with a learned null token, the crash atom is the first logit.

:class:`LegacyFTPFN` reproduces that forward pass with the original parameter names (strict
state-dict loading), :class:`LegacyFTPFNPredictor` tokenizes trials the way the old optimization
loop did: a completed trial without marker, a crashed one with the crash marker, an early-stopped
one and the running trial with the cutoff marker.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from esbopfn.core import ProblemSpec, Trial
from esbopfn.predictors.base import BarPrediction, Prediction, PredictorBase
from esbopfn.predictors.xdecpfn.legacy import LegacyBlock, _mlp
from esbopfn.predictors.xdecpfn.predictor import known_outcome

KIND_CTX, KIND_NULL, KIND_QUERY = 0, 1, 2
MARK_NONE, MARK_CUTOFF, MARK_CRASH = 0, 1, 2


class LegacyFTPFN(nn.Module):
    """The old packed-sequence model, forward pass over one context at a time."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        d = int(cfg["d_model"])
        self.t_enc = _mlp(2, 128, d)
        self.th_enc = nn.Linear(int(cfg["d_theta"]), d)
        self.v_enc = _mlp(1, 64, d)
        self.cutoff_vec = nn.Parameter(torch.zeros(d))
        self.crash_vec = nn.Parameter(torch.zeros(d))
        self.null_ctx = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(LegacyBlock(d, int(cfg["n_heads"]), int(cfg["d_ffn"]))
                                    for _ in range(int(cfg["n_layers"])))
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, int(cfg["n_bins"]) + 1)

    def forward(self, kind: torch.Tensor, t: torch.Tensor, theta: torch.Tensor, yn: torch.Tensor,
                marker: torch.Tensor) -> torch.Tensor:
        """One context ``(L,)`` (null token first) -> ``[atom, bars]`` logits at its queries."""
        tt = t.clamp_min(1e-6)
        base = self.t_enc(torch.stack([tt, torch.log(tt)], -1)) + self.th_enc(theta)
        val = self.v_enc(yn[:, None])
        mark = (marker == MARK_CUTOFF)[:, None] * self.cutoff_vec \
            + (marker == MARK_CRASH)[:, None] * self.crash_vec
        k = kind[:, None]
        x = torch.where(k == KIND_CTX, base + val + mark,
                        torch.where(k == KIND_QUERY, base, self.null_ctx.expand_as(base)))
        L = kind.shape[0]
        is_key = (kind == KIND_CTX) | (kind == KIND_NULL)
        mask = (is_key[None, :] | torch.eye(L, dtype=torch.bool, device=kind.device))[None, None]
        x = x[None]
        for blk in self.blocks:
            x = blk(x, mask=mask)
        return self.head(self.nf(x[0, kind == KIND_QUERY]))


def legacy_grid(T: int) -> np.ndarray:
    """``round_half_up(T j / 8)`` for ``j = 1..8``."""
    return np.floor(np.arange(1, 9) * T / 8 + 0.5).astype(np.int64)


class LegacyFTPFNPredictor(PredictorBase):
    """Thesis FT-PFN checkpoint behind the :class:`Predictor` protocol (old token rules)."""

    name = "ftpfn-thesis"
    supports_unrun = True
    context_aware = True

    def __init__(self, model: LegacyFTPFN, calib: dict[str, Any], spec: ProblemSpec,
                 device: str = "cpu", name: str | None = None):
        if int(calib["T"]) != spec.T:
            raise ValueError(f"checkpoint has T={calib['T']}, {spec.name} has T={spec.T}")
        self.spec = spec
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.grid = legacy_grid(spec.T)
        self.edges = np.asarray(calib["edges"], dtype=np.float64)
        self.lo_in, self.hi_in = float(calib["lo_input"]), float(calib["hi_input"])
        self.y_mean, self.y_std = float(calib["y_mean"]), float(calib["y_std"])
        self.log1p = calib.get("value_space", "raw") == "log1p"
        self.th_mean = np.asarray(calib["theta_mean"], dtype=np.float32)
        self.th_std = np.asarray(calib["theta_std"], dtype=np.float32)
        self.name = name or self.name

    @classmethod
    def from_dir(cls, path: Path | str, spec: ProblemSpec, device: str = "cpu",
                 name: str | None = None) -> LegacyFTPFNPredictor:
        from esbopfn.training.registry import load_model

        files = load_model(path)
        model = LegacyFTPFN(files.model_cfg["cfg"])
        model.load_state_dict(files.state_dict, strict=True)
        return cls(model, files.calib, spec, device, name)

    def predictable_steps(self, spec: ProblemSpec) -> np.ndarray:
        g = legacy_grid(spec.T)
        return g[g < spec.T]

    def _tokens(self, trial: Trial, running: bool):
        if trial.unrun:
            return None
        if running or not (trial.crashed or trial.complete):
            L, mark = int(trial.t_obs), MARK_CUTOFF
        elif trial.crashed:
            L, mark = max(int(trial.traj.crash_step) - 1, 1), MARK_CRASH
        else:
            L, mark = int(trial.t_obs), MARK_NONE
        steps = np.concatenate([self.grid[self.grid < L], [L]]).astype(np.int64)
        marker = np.zeros(steps.size, dtype=np.int64)
        marker[-1] = mark
        J = np.asarray(trial.traj.J, dtype=np.float64)[steps - 1]
        return steps, J, marker, np.repeat(np.asarray(trial.theta, np.float32)[None], steps.size, 0)

    def _logits(self, parts: list, q_theta: np.ndarray) -> torch.Tensor:
        d = self.spec.d
        steps = np.concatenate([[0]] + [p[0] for p in parts]).astype(np.float64)
        J = np.concatenate([[np.nan]] + [p[1] for p in parts])
        marker = np.concatenate([[0]] + [p[2] for p in parts])
        theta = np.concatenate([np.zeros((1, d), np.float32)] + [p[3] for p in parts])
        M = q_theta.shape[0]
        kind = np.concatenate([[KIND_NULL], np.full(steps.size - 1, KIND_CTX), np.full(M, KIND_QUERY)])
        t = np.concatenate([np.zeros(1), steps[1:] / self.spec.T, np.ones(M)])
        theta = np.concatenate([theta, q_theta.astype(np.float32)])
        Jall = np.concatenate([J, np.full(M, np.nan)])
        marker = np.concatenate([marker, np.zeros(M, np.int64)])
        is_tok = kind == KIND_CTX
        Jc = np.where(np.isfinite(Jall), Jall, self.hi_in)
        Jc = np.clip(Jc, self.lo_in, self.hi_in)
        Jin = np.log1p(Jc) if self.log1p else Jc
        yn = np.where(is_tok, (Jin - self.y_mean) / self.y_std, 0.0)
        th_n = (theta - self.th_mean) / self.th_std
        f = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)
        with torch.no_grad():
            return self.model(f(kind, torch.long), f(t), f(th_n), f(yn), f(marker, torch.long))

    def _bars(self, logits: torch.Tensor) -> list[BarPrediction]:
        a = logits.double().cpu().numpy().reshape(-1, self.edges.size)
        a = a - a.max(axis=1, keepdims=True)
        p = np.exp(a)
        p /= p.sum(axis=1, keepdims=True)
        return [BarPrediction(self.edges, row[1:], float(row[0])) for row in p]

    def _context(self, context: Sequence[Trial], exclude: set[int]) -> list:
        parts = [self._tokens(tr, running=False) for tr in context if id(tr.traj) not in exclude]
        return [p for p in parts if p is not None]

    def predict(self, query: Trial, context: Sequence[Trial]) -> Prediction:
        return self.predict_many([query], context)[0]

    def predict_many(self, queries: Sequence[Trial], context: Sequence[Trial]) -> list[Prediction]:
        out: list[Prediction | None] = [known_outcome(q) for q in queries]
        parts = self._context(context, {id(q.traj) for q in queries})
        for i, q in enumerate(queries):
            if out[i] is None:
                own = self._tokens(q, running=True)
                logits = self._logits(parts + ([own] if own is not None else []),
                                      np.asarray(q.theta, dtype=np.float64)[None])
                out[i] = self._bars(logits)[0]
        return out  # type: ignore[return-value]

    def predict_unrun(self, thetas: np.ndarray, context: Sequence[Trial]) -> list[BarPrediction]:
        parts = self._context(context, set())
        q = np.asarray(thetas, dtype=np.float64).reshape(-1, self.spec.d)
        preds: list[BarPrediction] = []
        for a in range(0, q.shape[0], 256):
            preds += self._bars(self._logits(parts, q[a:a + 256]))
        return preds
