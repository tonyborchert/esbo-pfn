"""DecPFN network: a causal pre-LN Transformer decoder with a crash-aware head on every position.

The input ``[z-scored covariates, z-scored t]`` of each token is embedded by one linear layer,
passed through ``n_layers`` pre-LN blocks under a causal mask (each token attends to itself and
its predecessors), a final layer norm and a single linear output layer. Heads have dimension 16
(``n_heads = d_model / 16``), the FFN width is ``2 d_model`` with GELU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

HEAD_DIM = 16


@dataclass
class DecPFNConfig:
    n_features: int          # inputs per token without the time
    n_outputs: int           # N_bin + 1 (bar heads) or 3 (gaussian)
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


class DecPFN(nn.Module):
    def __init__(self, cfg: DecPFNConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Linear(cfg.n_features + 1, d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, dim_feedforward=2 * d, dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layers, norm=nn.LayerNorm(d),
                                            enable_nested_tensor=False)
        self.out = nn.Linear(d, cfg.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, L, n_features + 1)`` -> outputs ``(B, L, n_outputs)``, causal in L."""
        L = x.shape[1]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.blocks(self.embed(x), mask=mask, is_causal=True)
        return self.out(h)
