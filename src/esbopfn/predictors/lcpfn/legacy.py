"""The LC-PFN checkpoints of the earlier implementation, served as ``lcpfn/<p>/thesis-<head>[-linearnorm]/<size>``.

These weights were trained with the old code base, a vendored copy of the released LC-PFN
(planning record of the prediction code base, LC-PFN section and discrepancies D1, D2, D6, D9,
D15). Their forward pass differs from :class:`~esbopfn.predictors.lcpfn.model.LCPFN`:

- the released ``TransformerModel``: post-LN encoder layers (``nn.MultiheadAttention``, GELU FFN),
  no final norm, decoder ``Linear-GELU-Linear``, sequence-first tensors,
- ``x`` = grid index ``/(L - 1)`` standardized as a uniform variable, ``y`` = the cost clipped to
  ``[0, cap]`` and mapped to ``[0.02, 1]`` (crash marker 0), then the input normalization of the
  checkpoint: ``linear`` (equal-width CDF back onto ``[0.02, 1]``), ``binned`` (quantile CDF) or
  ``zscore``, finally standardized as a uniform variable,
- training tokens ``x + y`` for the observed grid points, query tokens ``x`` only, every query
  attends to the observed prefix and itself,
- grid ``linspace(0, T - 1, L).astype(int)`` (0-based, includes the first step),
- head: ``FailureBarDistribution`` with the crash atom as bucket 0 and value bars in normalized
  cost units (mapped back to raw units with ``cap``), or the Gaussian curve head
  ``(mean, log var, crash logit)`` on a target z-scored in normalized cost units.

:class:`LegacyLCPFN` reproduces that forward pass with the original parameter names (the
checkpoints' ``criterion.*`` buffers live in ``calib.json``), so the exported state dicts load
strictly. :class:`LegacyLCPFNPredictor` serves it through the :class:`Predictor` protocol.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from esbopfn.core import ProblemSpec, Trial
from esbopfn.predictors.base import BarPrediction, GaussianPrediction, Prediction, PredictorBase
from esbopfn.predictors.lcpfn.predictor import CurvePrediction

_UNIFORM_STD = float(np.sqrt(1 / 12))


class _Normalize(nn.Module):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean, self.std = mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class _NanLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.nan_to_num(x, nan=0.0))


class _GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class LegacyLCPFNLayer(nn.Module):
    """The released LC-PFN encoder layer with ``pre_norm=False`` (post-LN)."""

    def __init__(self, d: int, h: int, d_ff: int, eps: float = 1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, h, dropout=0.0)
        self.linear1 = nn.Linear(d, d_ff)
        self.linear2 = nn.Linear(d_ff, d)
        self.norm1 = nn.LayerNorm(d, eps=eps)
        self.norm2 = nn.LayerNorm(d, eps=eps)

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        src = self.norm1(src + self.self_attn(src, src, src, attn_mask=mask)[0])
        return self.norm2(src + self.linear2(F.gelu(self.linear1(src))))


class _Encoder(nn.Module):
    def __init__(self, layers: list[LegacyLCPFNLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, mask)
        return src


def dq_mask(seq_len: int, k: int, device: torch.device | None = None) -> torch.Tensor:
    """Additive mask: every position attends to the first ``k`` positions and to itself."""
    allowed = torch.ones(seq_len, seq_len, dtype=torch.bool)
    allowed[:, k:] = False
    allowed |= torch.eye(seq_len, dtype=torch.bool)
    return torch.zeros(seq_len, seq_len).masked_fill(~allowed, float("-inf")).to(device)


class LegacyLCPFN(nn.Module):
    """The old LC-PFN ``TransformerModel``, forward pass only (``arch`` = ``arch.json``)."""

    def __init__(self, arch: dict[str, Any]):
        super().__init__()
        d = int(arch["d_model"])
        eps = float(arch.get("layer_norm_eps", 1e-5))
        (xm0, xs0), (xm1, xs1) = arch["x_encoding"]
        ym, ys = arch["y_encoding"]
        self.transformer_encoder = _Encoder([
            LegacyLCPFNLayer(d, int(arch["n_heads"]), int(arch["d_ff"]), eps)
            for _ in range(int(arch["n_layers"]))])
        self.encoder = nn.Sequential(_Normalize(xm0, xs0), _Normalize(xm1, xs1), _NanLinear(1, d))
        self.y_encoder = nn.Sequential(_Normalize(ym, ys), _NanLinear(1, d))
        self.decoder = nn.Sequential(nn.Linear(d, int(arch["decoder_hidden"])), _GELU(),
                                     nn.Linear(int(arch["decoder_hidden"]), int(arch["n_out"])))
        self.seq_len = int(arch["seq_len"])

    def forward(self, y: torch.Tensor, k: int) -> torch.Tensor:
        """``y (L, B)`` input-normalized curve (only the first ``k`` rows are read) -> outputs
        ``(L - k, B, n_out)`` at the query positions ``k .. L - 1``."""
        L, B = y.shape
        x = torch.arange(0, L, dtype=torch.float32, device=y.device).view(L, 1, 1).expand(L, B, 1)
        x_src = self.encoder(x)
        y_src = self.y_encoder(y.unsqueeze(-1))
        src = torch.cat([x_src[:k] + y_src[:k], x_src[k:]], 0)
        out = self.transformer_encoder(src, dq_mask(L, k, y.device))
        return self.decoder(out)[k:]


# --------------------------------------------------------------------------- normalization
def normalize_cost(J: np.ndarray, spec: dict[str, float]) -> np.ndarray:
    """Old ``_normalize_np`` for a curve without crash: clip to ``[0, cap]``, map to ``[low, high]``."""
    J = np.asarray(J, dtype=np.float32)
    y = np.clip(J, 0.0, spec["cap"]) / spec["cap"] * (spec["high"] - spec["low"]) + spec["low"]
    return y.astype(np.float32)


def input_normalize(y: np.ndarray, edges: np.ndarray, spec: dict[str, float]) -> np.ndarray:
    """Old ``_input_normalize_np``: zscore for two edges, else the piecewise CDF onto
    ``[low, high]`` with the crash marker kept at 0."""
    y = np.asarray(y, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float32)
    if edges.size == 2:
        return (0.5 + _UNIFORM_STD * (y - edges[0]) / edges[1]).astype(np.float32)
    idx = np.clip(np.searchsorted(edges, y) - 1, 0, len(edges) - 2)
    u = np.clip(((y - edges[idx]) / (edges[idx + 1] - edges[idx]) + idx) / (len(edges) - 1), 0, 1)
    valid = u * (spec["high"] - spec["low"]) + spec["low"]
    return np.where(y > spec["fail_split"], valid, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- predictor
class LegacyLCPFNPredictor(PredictorBase):
    """An imported thesis LC-PFN behind the :class:`Predictor` protocol (old grid and scaling).

    Context-blind. ``predict`` returns the prediction at the final grid point (the horizon),
    :meth:`predict_curve` the predictions at every grid point after the observed prefix.
    """

    supports_unrun = False
    context_aware = False

    def __init__(self, model: LegacyLCPFN, arch: dict[str, Any], calib: dict[str, Any],
                 spec: ProblemSpec, device: str = "cpu", name: str | None = None,
                 batch_size: int = 256):
        if arch["problem"] != spec.name:
            raise ValueError(f"model trained on {arch['problem']}, asked for {spec.name}")
        self.spec = spec
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.seq_len = int(arch["seq_len"])
        self.grid = np.linspace(0, spec.T - 1, self.seq_len).astype(int)
        self.norm_spec = dict(calib["norm_spec"])
        self.in_edges = np.asarray(calib["input_norm_edges"], dtype=np.float32)
        self.gaussian = arch["head"] == "gaussian"
        if self.gaussian:
            self.target_mean = float(calib["target_mean"])
            self.target_std = float(calib["target_std"])
            self.scale = float(calib["to_cost_scale"])
        else:
            borders = np.asarray(calib["borders"], dtype=np.float64)
            centers = 0.5 * (borders[:-1] + borders[1:])
            self.n_fail = int((centers < float(calib["fail_threshold"])).sum())
            if self.n_fail != 1:
                raise ValueError(f"expected one crash bucket, found {self.n_fail}")
            self.edges = np.asarray(calib["edges_raw"], dtype=np.float64)
        self.name = name or "lcpfn-thesis"
        self.batch_size = batch_size

    @classmethod
    def from_dir(cls, path: Path | str, spec: ProblemSpec, device: str = "cpu",
                 name: str | None = None, **kw: Any) -> LegacyLCPFNPredictor:
        from esbopfn.training.registry import load_model

        files = load_model(path)
        model = LegacyLCPFN(files.model_cfg)
        model.load_state_dict(files.state_dict, strict=True)
        return cls(model, files.model_cfg, files.calib, spec, device, name, **kw)

    def predictable_steps(self, spec: ProblemSpec | None = None) -> np.ndarray:
        """1-based steps of the old grid points before the horizon."""
        return np.unique(self.grid[:-1]) + 1

    def kappa(self, query: Trial) -> int:
        """Observed grid points (old ``k``), clamped to ``[1, L - 1]``."""
        k = int(np.searchsorted(self.grid, int(query.t_obs)))
        return max(1, min(k, self.seq_len - 1))

    def inputs(self, query: Trial, k: int) -> np.ndarray:
        """Model input curve ``(L,)``: the normalized cost at the first ``k`` grid points, 0 after."""
        J = np.asarray(query.traj.J, dtype=np.float32)[:int(query.t_obs)]
        y = input_normalize(normalize_cost(J, self.norm_spec), self.in_edges, self.norm_spec)
        out = np.zeros(self.seq_len, dtype=np.float32)
        out[:k] = y[self.grid[:k]]
        return out

    @torch.no_grad()
    def _forward(self, queries: Sequence[Trial], k: int) -> np.ndarray:
        """Outputs ``(n, L - k, n_out)`` (float64) for queries sharing ``k``."""
        res = []
        for a in range(0, len(queries), self.batch_size):
            part = queries[a:a + self.batch_size]
            y = torch.as_tensor(np.stack([self.inputs(q, k) for q in part], axis=1),
                                device=self.device)
            res.append(self.model(y, k).double().cpu().numpy().transpose(1, 0, 2))
        return np.concatenate(res, axis=0)

    def prediction(self, o: np.ndarray) -> Prediction:
        if self.gaussian:
            mean, log_var, logit = (float(v) for v in o[:3])
            norm_mean = mean * self.target_std + self.target_mean
            cost_mean = (norm_mean - self.norm_spec["low"]) / (
                self.norm_spec["high"] - self.norm_spec["low"]) * self.norm_spec["cap"]
            var = math.exp(log_var) * self.target_std ** 2 * self.scale ** 2
            return GaussianPrediction(cost_mean, var, p_crash=float(1.0 / (1.0 + math.exp(-logit))))
        p = np.exp(o - o.max())
        p /= p.sum()
        return BarPrediction(self.edges, p[self.n_fail:], float(p[:self.n_fail].sum()))

    def predict(self, query: Trial, context: Sequence[Trial] = ()) -> Prediction:
        return self.predict_many([query], context)[0]

    def predict_many(self, queries: Sequence[Trial], context: Sequence[Trial] = ()
                     ) -> list[Prediction]:
        results: list[Prediction | None] = [None] * len(queries)
        groups: dict[int, list[int]] = {}
        for i, q in enumerate(queries):
            self._check_query(q)
            if q.crashed:
                results[i] = GaussianPrediction(0.0, 0.0, p_crash=1.0)
            elif q.complete:
                results[i] = GaussianPrediction(float(q.J_obs), 0.0)
            else:
                groups.setdefault(self.kappa(q), []).append(i)
        for k, idx in groups.items():
            out = self._forward([queries[i] for i in idx], k)[:, -1]
            for i, o in zip(idx, out):
                results[i] = self.prediction(o)
        return results  # type: ignore[return-value]

    def predict_curve(self, query: Trial) -> CurvePrediction:
        """Predictions at the old grid points after the observed prefix (1-based steps)."""
        self._check_query(query)
        if query.complete:
            return CurvePrediction(np.zeros(0, dtype=np.int64), [])
        if query.crashed:
            rest = self.grid[self.grid + 1 > query.t_obs] + 1
            return CurvePrediction(rest.astype(np.int64),
                                   [GaussianPrediction(0.0, 0.0, p_crash=1.0) for _ in rest])
        k = self.kappa(query)
        out = self._forward([query], k)[0]
        return CurvePrediction((self.grid[k:] + 1).astype(np.int64),
                               [self.prediction(o) for o in out])


def load(model_id: str, spec: ProblemSpec, home: Path | str | None = None, **kw: Any
         ) -> LegacyLCPFNPredictor:
    """The predictor of an imported thesis LC-PFN id, ``ModelNotTrained`` when it is absent."""
    from esbopfn.training.registry import ModelId, resolve

    path = resolve(model_id, home, hint="Import it with `pfn import thesis-models --from DIR`.")
    kw.setdefault("device", "cpu")
    return LegacyLCPFNPredictor.from_dir(path, spec, name=ModelId.parse(model_id).short, **kw)
