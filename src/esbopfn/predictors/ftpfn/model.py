"""FT-PFN adapted to controller trajectories (thesis method chapter, adapting FT-PFN).

A context token is ``((theta_i, t_l), J_{t_l})``: the sum of linear embeddings of the z-scored
controller, the z-scored time and the z-scored raw cost. A crash (``J = inf``) replaces the value
embedding by the learned constant ``e_inf``. A query ``(theta, T)`` carries no value embedding.
Pre-LN blocks with a final LayerNorm and one linear layer give ``N_bin + 1`` logits (the last one
is the crash atom) over raw ``J_T``.

Several contexts share one packed row. The attention mask is the PFN block mask: a token attends
to the context tokens of its own context and to itself, so no token reads a query and contexts
never interact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import nn

from esbopfn.predictors.xdecpfn.model import PreLNBlock

#: Token kinds of a packed row.
KIND_VALUE, KIND_CRASH, KIND_QUERY, KIND_PAD = 0, 1, 2, 3


@dataclass
class FTPFNConfig:
    """Architecture of FT-PFN (thesis table of the cross-trajectory configurations)."""

    d_theta: int
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    n_bins: int = 1000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FTPFNConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def block_mask(kind: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """``(R, 1, L, L)`` boolean mask of packed rows: ``i`` may attend to ``j`` iff both belong to
    the same context and ``j`` is a context token, or ``j == i``."""
    key = (kind == KIND_VALUE) | (kind == KIND_CRASH)
    same = ctx[:, :, None] == ctx[:, None, :]
    eye = torch.eye(kind.shape[1], dtype=torch.bool, device=kind.device)
    return ((same & key[:, None, :]) | eye[None]).unsqueeze(1)


class FTPFN(nn.Module):
    """Encoder over packed contexts with a crash-aware bar head on raw ``J_T``."""

    def __init__(self, cfg: FTPFNConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.theta_embed = nn.Linear(cfg.d_theta, d)
        self.t_embed = nn.Linear(1, d)
        self.y_embed = nn.Linear(1, d)
        self.e_inf = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(PreLNBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                    for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.n_bins + 1)
        nn.init.normal_(self.e_inf, std=0.02)

    def embed(self, theta: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
              kind: torch.Tensor) -> torch.Tensor:
        """Token embeddings ``(R, L, d)`` from z-scored ``theta`` ``(R, L, d_theta)``, ``t`` and
        ``y`` ``(R, L)`` and the token kinds."""
        base = self.theta_embed(theta) + self.t_embed(t.unsqueeze(-1))
        value = self.y_embed(y.unsqueeze(-1))
        k = kind.unsqueeze(-1)
        add = torch.where(k == KIND_VALUE, value,
                          torch.where(k == KIND_CRASH, self.e_inf.to(value.dtype),
                                      torch.zeros_like(value)))
        return torch.where(k == KIND_PAD, torch.zeros_like(base), base + add)

    def forward(self, theta: torch.Tensor, t: torch.Tensor, y: torch.Tensor, kind: torch.Tensor,
                ctx: torch.Tensor, q_row: torch.Tensor | None = None,
                q_pos: torch.Tensor | None = None) -> torch.Tensor:
        """Logits ``(Q, N + 1)`` at the positions ``(q_row, q_pos)`` of the packed rows, or
        ``(R, L, N + 1)`` at every position when no positions are given."""
        x = self.embed(theta, t, y, kind)
        mask = block_mask(kind, ctx)
        for blk in self.blocks:
            x = blk(x, mask=mask)
        if q_row is not None and q_pos is not None:
            x = x[q_row, q_pos]
        return self.head(self.norm(x))
