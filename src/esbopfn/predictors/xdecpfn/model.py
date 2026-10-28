"""XDecPFN: a causal DecPFN stage per trajectory followed by attention across trajectories.

Thesis, method chapter (XDecPFN) and appendix (training of the predictors):

- Stage 1 is a causal pre-LN decoder over the ``N_grid - 1`` decoder-grid tokens of one
  trajectory. A token is the linear embedding of the z-scored ``[ln J_{t_l}, c_l, t_l]`` with the
  DecPFN covariates ``c_l = [Delta J, xi, sum xi_dot]``. Its output at the trajectory's last
  observed token is the trajectory's representation, so stage 1 can be cached per trajectory.
- The stage-2 token of a trajectory is that representation plus ``e_theta(theta)`` (linear) plus
  one of the constant embeddings ``e_complete``, ``e_cut``, ``e_crash``. An unrun controller
  ``theta*`` enters as ``e_theta(theta*) + e_unrun``.
- Stage 2 is a stack of bidirectional pre-LN self-attention blocks over all trajectory and query
  tokens of one example. No token attends to a query token (a query attends to the trajectory
  tokens and to itself).
- A final LayerNorm and one linear layer give ``N_bin + 1`` logits per token: the crash-aware bar
  head over ``ln J_T`` (the last logit is the crash atom).

A trajectory that crashes before the first grid point has no stage-1 token. Its representation is
the learned vector ``empty`` (the thesis does not cover this case).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#: Stage-2 marker ids, the order of the constant embeddings.
MARK_COMPLETE, MARK_CUT, MARK_CRASH, MARK_UNRUN = 0, 1, 2, 3


# --------------------------------------------------------------------------- shared layers
class PreLNBlock(nn.Module):
    """Pre-LN transformer block: ``x + Attn(LN(x))`` then ``x + FFN(LN(x))`` with a GELU FFN.

    ``mask`` is a boolean ``(B, 1, L, L)`` (or broadcastable) tensor, True where a query position
    may attend to a key position. ``causal=True`` uses the causal mask instead.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                causal: bool = False) -> torch.Tensor:
        B, L, d = x.shape
        h = self.n_heads
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, h, d // h).permute(2, 0, 3, 1, 4)
        p = self.dropout if self.training else 0.0
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None if causal else mask,
                                           dropout_p=p, is_causal=causal)
        o = o.transpose(1, 2).reshape(B, L, d)
        x = x + F.dropout(self.proj(o), p, self.training)
        return x + F.dropout(self.ff(self.norm2(x)), p, self.training)


def crash_bar_nll(logits: torch.Tensor, target_bin: torch.Tensor, crashed: torch.Tensor,
                  log_widths: torch.Tensor) -> torch.Tensor:
    """Per-target NLL of the crash-aware bar head (thesis eq. crash-aware bar loss).

    ``logits[..., :N]`` are the bars, ``logits[..., N]`` the crash atom. With ``a = softmax``,
    ``w_inf = a_inf`` and ``w_k = a_k / (1 - w_inf)`` the loss is ``-log w_inf`` for a crash and
    ``-log(1 - w_inf) - log w_k + log(b_k - b_{k-1}) = -log a_k + log width_k`` otherwise.
    ``target_bin`` must already be clamped into ``[0, N)`` (out-of-support targets go to the
    outermost bins).
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    n = logp.shape[-1] - 1
    lp_crash = logp[..., n]
    idx = target_bin.clamp(0, n - 1).unsqueeze(-1)
    lp_bin = torch.gather(logp[..., :n], -1, idx).squeeze(-1)
    nll_value = -lp_bin + log_widths.to(logp.dtype)[target_bin.clamp(0, n - 1)]
    return torch.where(crashed, -lp_crash, nll_value)


def equal_mass_edges(samples: np.ndarray, n_bins: int) -> np.ndarray:
    """``n_bins + 1`` strictly increasing edges at equal-mass quantiles of finite ``samples``.

    The outer edges are the sample minimum and maximum. Tied quantiles are pushed apart by
    ``1e-9`` of the range so every bin has a positive width.
    """
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("need at least two finite samples to place bin edges")
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    lo, hi = float(x.min()), float(x.max())
    eps = max(hi - lo, abs(hi), 1.0) * 1e-9
    edges[0] = lo
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return edges


def bin_index(y: np.ndarray | torch.Tensor, edges: np.ndarray | torch.Tensor):
    """Bin of each target, ``edges[k] <= y < edges[k + 1]``, clamped into ``[0, N)``."""
    if isinstance(y, torch.Tensor):
        e = torch.as_tensor(edges, dtype=torch.float64, device=y.device)
        idx = torch.bucketize(y.double().contiguous(), e, right=True) - 1
        return idx.clamp(0, e.numel() - 2)
    e = np.asarray(edges, dtype=np.float64)
    idx = np.searchsorted(e, np.asarray(y, dtype=np.float64), side="right") - 1
    return np.clip(idx, 0, e.size - 2)


# --------------------------------------------------------------------------- the model
@dataclass
class XDecPFNConfig:
    """Architecture of XDecPFN (thesis table of the cross-trajectory configurations)."""

    d_in: int                      # stage-1 token width: ln J, Delta J, xi, sum xi_dot, t
    d_theta: int
    d_model: int = 384
    n_heads: int = 6
    layers_causal: int = 3
    layers_cross: int = 6
    d_ff: int = 768
    dropout: float = 0.0
    n_bins: int = 1000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> XDecPFNConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


class XDecPFN(nn.Module):
    """Two-stage cross-trajectory decoder with a crash-aware bar head over ``ln J_T``."""

    def __init__(self, cfg: XDecPFNConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.token_embed = nn.Linear(cfg.d_in, d)
        self.causal = nn.ModuleList(PreLNBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                    for _ in range(cfg.layers_causal))
        self.empty = nn.Parameter(torch.zeros(d))
        self.theta_embed = nn.Linear(cfg.d_theta, d)
        self.marker = nn.Embedding(4, d)           # complete, cut, crash, unrun
        self.cross = nn.ModuleList(PreLNBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                   for _ in range(cfg.layers_cross))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.n_bins + 1)
        nn.init.normal_(self.empty, std=0.02)

    # ---------------------------------------------------------------- stage 1
    def stage1(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(S, L, d_in)`` z-scored stage-1 inputs -> ``(S, L, d)`` causal representations.

        Position ``l`` depends on positions ``0..l`` only, so a trajectory's representation at
        any cutoff is the output at its last observed position, whatever follows it.
        """
        x = self.token_embed(tokens)
        for blk in self.causal:
            x = blk(x, causal=True)
        return x

    def gather(self, h: torch.Tensor, traj: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Stage-1 outputs ``h[traj, pos]`` with ``pos = -1`` meaning no token (``empty``)."""
        out = h[traj, pos.clamp_min(0)]
        return torch.where((pos < 0).unsqueeze(-1), self.empty.to(out.dtype), out)

    # ---------------------------------------------------------------- stage 2
    def stage2(self, rep: torch.Tensor, theta: torch.Tensor, marker: torch.Tensor,
               valid: torch.Tensor, q_theta: torch.Tensor | None = None,
               ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Cross-trajectory stage for a batch of examples.

        ``rep`` ``(E, n, d)`` stage-1 representations, ``theta`` ``(E, n, d_theta)`` z-scored,
        ``marker`` ``(E, n)``, ``valid`` ``(E, n)`` (False = padding), ``q_theta``
        ``(E, Q, d_theta)`` unrun queries (optional). Returns the logits of the
        trajectory tokens ``(E, n, N + 1)`` and of the queries ``(E, Q, N + 1)``.
        """
        E, n, _ = rep.shape
        x = rep + self.theta_embed(theta) + self.marker(marker)
        keys = valid
        Q = 0
        if q_theta is not None and q_theta.shape[1] > 0:
            Q = q_theta.shape[1]
            q = self.theta_embed(q_theta) + self.marker.weight[MARK_UNRUN]
            x = torch.cat([x, q.to(x.dtype)], dim=1)
            keys = torch.cat([valid, torch.zeros(E, Q, dtype=torch.bool, device=valid.device)], 1)
        mask = cross_mask(keys)
        for blk in self.cross:
            x = blk(x, mask=mask)
        logits = self.head(self.norm(x))
        return logits[:, :n], (logits[:, n:] if Q else None)


def cross_mask(keys: torch.Tensor) -> torch.Tensor:
    """Stage-2 attention mask ``(E, 1, L, L)``: position ``i`` may attend to ``j`` iff ``j`` is a
    valid trajectory token or ``j == i``. Queries and padding are never keys, so no token reads a
    query and a padding row only sees itself."""
    L = keys.shape[1]
    eye = torch.eye(L, dtype=torch.bool, device=keys.device)
    return (keys[:, None, :] | eye[None]).unsqueeze(1)


def time_stats(T_max: int) -> tuple[float, float]:
    """Mean and standard deviation of the uniform distribution on ``{1, ..., T_max}``."""
    return (T_max + 1) / 2.0, math.sqrt((T_max * T_max - 1) / 12.0) if T_max > 1 else 1.0
