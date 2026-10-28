"""LC-PFN network, the architecture of the released LC-PFN written anew.

Context tokens embed ``(t_l, J_{t_l})`` as ``x_enc(t) + y_enc(J)``, query tokens embed only
``x_enc(t)``. Post-LN encoder blocks (attention, then FFN, each followed by a residual add and a
layer norm) run under the PFN attention mask: context tokens attend to the context, a query token
attends to the context and to itself. The output head is the released two-layer FFN
(``Linear(d, 2d) - GELU - Linear(2d, n_outputs)``) widened to the crash-aware head. As in the
released code, the attention output projection and the second FFN layer of every block start at
zero. ``n_heads = d_model / 16``, FFN width ``2 d_model``, GELU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

HEAD_DIM = 16


@dataclass
class LCPFNConfig:
    n_outputs: int
    d_model: int = 512
    n_layers: int = 3
    dropout: float = 0.0

    @property
    def n_heads(self) -> int:
        if self.d_model % HEAD_DIM:
            raise ValueError(f"d_model must be a multiple of {HEAD_DIM}, got {self.d_model}")
        return self.d_model // HEAD_DIM

    def to_dict(self) -> dict:
        return asdict(self)


def pfn_mask(n_ctx: int, n_total: int, device: torch.device | None = None) -> torch.Tensor:
    """Boolean attention mask (True = blocked): every token sees the context, a query token also
    sees itself, nothing sees another query."""
    blocked = torch.zeros(n_total, n_total, dtype=torch.bool, device=device)
    blocked[:, n_ctx:] = True
    q = torch.arange(n_ctx, n_total, device=device)
    blocked[q, q] = False
    return blocked


class LCPFN(nn.Module):
    def __init__(self, cfg: LCPFNConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.x_encoder = nn.Linear(1, d)
        self.y_encoder = nn.Linear(1, d)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(d, cfg.n_heads, dim_feedforward=2 * d,
                                       dropout=cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=False)
            for _ in range(cfg.n_layers))
        self.decoder = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                     nn.Linear(2 * d, cfg.n_outputs))
        for block in self.blocks:
            nn.init.zeros_(block.linear2.weight)
            nn.init.zeros_(block.linear2.bias)
            nn.init.zeros_(block.self_attn.out_proj.weight)
            nn.init.zeros_(block.self_attn.out_proj.bias)

    def forward(self, t: torch.Tensor, y_ctx: torch.Tensor) -> torch.Tensor:
        """``t``: ``(B, κ + Q)`` normalized times of context then query points, ``y_ctx``:
        ``(B, κ)`` normalized context costs -> outputs ``(B, Q, n_outputs)`` at the queries."""
        k = y_ctx.shape[1]
        n = t.shape[1]
        h = self.x_encoder(t.unsqueeze(-1))
        h = torch.cat([h[:, :k] + self.y_encoder(y_ctx.unsqueeze(-1)), h[:, k:]], dim=1)
        mask = pfn_mask(k, n, device=t.device)
        for block in self.blocks:
            h = block(h, src_mask=mask)
        return self.decoder(h[:, k:])
