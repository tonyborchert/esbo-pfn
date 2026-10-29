"""The DecPFN checkpoints of the earlier implementation, served as ``decpfn/<p>/thesis-<head>/<size>``.

These weights were trained with the old code base (planning record of the prediction code base,
DecPFN section and discrepancies D5, D9, D10, D14, D15). Their forward pass differs from
:class:`~esbopfn.predictors.decpfn.model.DecPFN`:

- input ``Linear([normalized features, t])`` with ``t = linspace(-sqrt 3, sqrt 3, L - 1)`` at the
  token position, the features z-scored with the old per-feature statistics,
- features ``(y, Y, j, J)`` per token, ``Y`` the old level channels, ``y`` their per-step
  increments (Cartpole: the measured velocities and the action increment), ``j`` the stage cost.
  ``y`` and ``j`` are summed over the steps since the previous token. DecPFN-noStates reads
  ``(j, J)`` only,
- the old channels include extras the thesis channels do not have: the Boiler heater command, the
  Cartpole bang-bang action with the position taken relative to the goal and the constant Furuta
  target (0),
- tokens at the 0-based steps ``0, s, 2s, ... < T - 1`` (the first token covers one step), at most
  ``L - 1`` of them, the horizon ``T`` is the target,
- ``nn.TransformerEncoder`` (pre-LN, causal mask) followed by a final LayerNorm, one linear head
  with the crash logit first (bar heads) or ``(mean, log var, crash logit)`` on a target z-scored
  with the horizon statistics (Gaussian head).

:class:`LegacyDecPFN` reproduces that forward pass with the original parameter names, so the
exported state dicts load strictly. :class:`LegacyDecPFNPredictor` builds the old features from a
:class:`~esbopfn.core.Trial` of this package (:func:`legacy_states`).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from esbopfn.core import ProblemSpec, Trajectory, Trial
from esbopfn.predictors.base import BarPrediction, GaussianPrediction, Prediction, PredictorBase

_ACTIVATIONS = {"relu": F.relu, "leaky_relu": F.leaky_relu, "gelu": F.gelu, "silu": F.silu}


# --------------------------------------------------------------------------- old channels
def _cartpole_states(traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """Old ``Y = [x - x_ref, phi, action]`` and ``y = [xdot, phidot, action increment]``.

    The action (``+1`` / ``-1``, 0 at the first step) is recomputed with the bang-bang rule of
    the controller from ``theta`` and the measurements of the previous step."""
    from esbopfn.predictors.xdecpfn.legacy import _goals

    x = np.asarray(traj.x, dtype=np.float32).reshape(len(traj.J), -1)
    xd = np.asarray(traj.xdot, dtype=np.float32).reshape(len(traj.J), -1)
    pos = x[:, 0] - _goals(x.shape[0])
    kp_c, kd_c, kp_p, kd_p = (float(v) for v in np.asarray(traj.theta).reshape(-1))
    ec = -pos[:-1].astype(np.float64)
    ep = -x[:-1, 1].astype(np.float64)
    last_c = np.concatenate([[0.0], ec[:-1]])
    last_p = np.concatenate([[0.0], ep[:-1]])
    u = kp_c * ec + kd_c * (ec - last_c) + kp_p * ep + kd_p * (ep - last_p)
    act = np.concatenate([[0.0], np.where(u < 0, 1.0, -1.0)]).astype(np.float32)
    d_act = np.zeros_like(act)
    d_act[1:] = np.diff(act)
    Y = np.stack([pos, x[:, 1], act], axis=1).astype(np.float32)
    y = np.stack([xd[:, 0], xd[:, 1], d_act], axis=1).astype(np.float32)
    return Y, y


def legacy_states(problem: str, traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """The old ``(Y, y)`` blocks ``(T, C)`` of a trajectory (float32, as the old pools)."""
    from esbopfn.predictors.xdecpfn.legacy import _boiler_u

    n = len(traj.J)
    x = np.asarray(traj.x, dtype=np.float32).reshape(n, -1)
    if problem == "cartpole":
        return _cartpole_states(traj)
    if problem == "furuta":
        xd = np.asarray(traj.xdot, dtype=np.float32).reshape(n, -1)
        Y = np.concatenate([x[:, :2], xd[:, :2], np.zeros((n, 1), np.float32)], axis=1)
    elif problem == "boiler":
        Y = np.stack([x[:, 0], _boiler_u(traj)], axis=1).astype(np.float32)
    elif problem in ("threetank_pi", "pt2_deadtime"):
        Y = x
    else:
        raise ValueError(f"no legacy DecPFN channels for problem {problem!r}")
    y = np.zeros_like(Y)
    y[1:] = np.diff(Y, axis=0)
    return Y, y


def legacy_token_steps(T: int, resolution: int, seq_len: int) -> np.ndarray:
    """0-based steps of the old tokens, ``arange(0, T - 1, resolution)`` capped at ``L - 1``."""
    return np.arange(0, T - 1, resolution, dtype=np.int64)[:seq_len - 1]


def legacy_features(problem: str, trial: Trial, resolution: int, seq_len: int,
                    states: bool = True) -> np.ndarray:
    """Old DecPFN inputs ``(n_tokens, F)`` (``[y, Y, j, J]`` or ``[j, J]``, raw units) of the
    observed prefix: tokens at the old grid steps before ``t_obs``, ``y`` and ``j`` summed since
    the previous token."""
    t = int(trial.t_obs)
    grid = legacy_token_steps(trial.T, resolution, seq_len)
    grid = grid[grid < t]
    J = np.asarray(trial.traj.J, dtype=np.float64)[:t]
    if not np.all(np.isfinite(J)):
        raise ValueError("legacy DecPFN features need a prefix before the crash")
    j = np.diff(J, prepend=0.0)

    def summed(a: np.ndarray) -> np.ndarray:
        c = np.cumsum(a, axis=0)[grid]
        return np.diff(c, axis=0, prepend=np.zeros((1,) + c.shape[1:]))

    cols = [summed(j)[:, None], J[grid][:, None]]
    if states:
        Y, y = legacy_states(problem, trial.traj)
        cols = [summed(y[:t].astype(np.float64)), Y[:t].astype(np.float64)[grid]] + cols
    return np.concatenate(cols, axis=1)


# --------------------------------------------------------------------------- network
class _LinearNorm(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean, False)
        self.register_buffer("std", std, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class _RawBinnedNorm(nn.Module):
    """Per-feature piecewise-linear CDF through the quantile edges ``(F, n + 1)``."""

    def __init__(self, edges: torch.Tensor):
        super().__init__()
        self.register_buffer("bin_edges", edges.contiguous(), False)
        self.n_bins = edges.shape[-1] - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).T.contiguous()
        idx = (torch.searchsorted(self.bin_edges, flat) - 1).clamp(0, self.n_bins - 1)
        start = self.bin_edges.gather(1, idx)
        end = self.bin_edges.gather(1, idx + 1)
        return (((flat - start) / (end - start) + idx) / self.n_bins).T.reshape(shape)


class _BinnedNorm(_RawBinnedNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (super().forward(x) - 0.5) * 12 ** 0.5


class _GaussianNorm(_RawBinnedNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = super().forward(x).clamp(1e-6, 1 - 1e-6)
        return torch.erfinv(2 * u - 1) * 2 ** 0.5


class _Head(nn.Module):
    def __init__(self, d: int, n_out: int):
        super().__init__()
        self.linear = nn.Linear(d, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LegacyDecPFN(nn.Module):
    """The old ``UnifiedDecPFN`` plus its head, forward pass only.

    ``arch`` is the ``arch.json`` of the export, ``calib`` its ``calib.json`` (normalization)."""

    def __init__(self, arch: dict[str, Any], calib: dict[str, Any]):
        super().__init__()
        d = int(arch["d_model"])
        self.seq_len = int(arch["sequence_length"])
        f32 = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float32))
        kind = arch["input_norm"]
        if kind == "linear":
            self.y_input_norm = _LinearNorm(f32(calib["input_mean"]), f32(calib["input_std"]))
        elif kind == "binned":
            self.y_input_norm = _BinnedNorm(f32(calib["input_bin_edges"]))
        elif kind == "gaussian":
            self.y_input_norm = _GaussianNorm(f32(calib["input_bin_edges"]))
        else:
            raise ValueError(f"unknown input_norm {kind!r}")
        self.register_buffer("t", torch.linspace(-3.0 ** 0.5, 3.0 ** 0.5, self.seq_len - 1)
                             .unsqueeze(1), False)
        self.register_buffer("causal_mask",
                             nn.Transformer.generate_square_subsequent_mask(self.seq_len - 1), False)
        self.embedding = nn.Linear(int(arch["n_features"]) + 1, d)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=int(arch["n_heads"]), dim_feedforward=int(arch["d_ff"]),
                dropout=float(arch.get("dropout", 0.0)),
                activation=_ACTIVATIONS[arch["activation"]],
                layer_norm_eps=float(arch.get("layer_norm_eps", 1e-5)),
                norm_first=True, batch_first=True),
            num_layers=int(arch["n_layers"]), enable_nested_tensor=False)
        self.fn_norm = nn.LayerNorm(d)
        self.head = _Head(d, int(arch["n_outputs"]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """``feats (B, L, F)`` raw features of equally long prefixes -> head outputs at the last
        token ``(B, n_outputs)``."""
        B, L, _ = feats.shape
        if L > self.seq_len - 1:
            raise ValueError(f"at most {self.seq_len - 1} tokens, got {L}")
        x = torch.cat([self.y_input_norm(feats), self.t[:L].expand(B, L, 1)], dim=-1)
        x = self.embedding(x)
        pad = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        with warnings.catch_warnings():     # the old call mixes a float mask and a bool padding mask
            warnings.filterwarnings("ignore", message=".*mismatched src_key_padding_mask.*")
            x = self.encoder(src=x, mask=self.causal_mask[:L, :L], src_key_padding_mask=pad)
        return self.head(self.fn_norm(x[:, -1, :]))


# --------------------------------------------------------------------------- predictor
class LegacyDecPFNPredictor(PredictorBase):
    """An imported thesis DecPFN behind the :class:`Predictor` protocol (old token rules).

    Context-blind. ``predict`` feeds the old tokens before ``query.t_obs`` and returns the
    prediction of ``J_T`` at the last of them: a :class:`BarPrediction` with the old raw-unit
    edges or a :class:`GaussianPrediction`, ``p_crash`` the crash output.
    """

    supports_unrun = False
    context_aware = False

    def __init__(self, model: LegacyDecPFN, arch: dict[str, Any], calib: dict[str, Any],
                 spec: ProblemSpec, device: str = "cpu", name: str | None = None,
                 batch_size: int = 256):
        if arch["problem"] != spec.name:
            raise ValueError(f"model trained on {arch['problem']}, asked for {spec.name}")
        self.spec = spec
        self.arch = arch
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.resolution = int(arch["resolution"])
        self.seq_len = int(arch["sequence_length"])
        self.states = bool(arch["extra_inputs"])
        self.gaussian = bool(arch["is_gaussian_head"])
        self.edges = None if self.gaussian else np.asarray(calib["edges"], dtype=np.float64)
        self.target_mean = calib.get("target_mean")
        self.target_std = calib.get("target_std")
        self.name = name or f"{arch['family']}-thesis"
        self.batch_size = batch_size

    @classmethod
    def from_dir(cls, path: Path | str, spec: ProblemSpec, device: str = "cpu",
                 name: str | None = None, **kw: Any) -> LegacyDecPFNPredictor:
        from esbopfn.training.registry import load_model

        files = load_model(path)
        model = LegacyDecPFN(files.model_cfg, files.calib)
        model.load_state_dict(files.state_dict, strict=True)
        return cls(model, files.model_cfg, files.calib, spec, device, name, **kw)

    def predictable_steps(self, spec: ProblemSpec | None = None) -> np.ndarray:
        """1-based steps at which a new old token arrives (the first is step 1)."""
        return legacy_token_steps(self.spec.T, self.resolution, self.seq_len) + 1

    def features(self, query: Trial) -> np.ndarray:
        return legacy_features(self.spec.name, query, self.resolution, self.seq_len, self.states)

    @torch.no_grad()
    def outputs(self, feats: list[np.ndarray]) -> np.ndarray:
        """Head outputs ``(n, n_outputs)`` (float64), prefixes grouped by length."""
        out = np.zeros((len(feats), self.model.head.linear.out_features), dtype=np.float64)
        groups: dict[int, list[int]] = {}
        for i, f in enumerate(feats):
            groups.setdefault(len(f), []).append(i)
        for idx in groups.values():
            for a in range(0, len(idx), self.batch_size):
                part = idx[a:a + self.batch_size]
                x = torch.as_tensor(np.stack([feats[i] for i in part]), dtype=torch.float32,
                                    device=self.device)
                out[part] = self.model(x).double().cpu().numpy()
        return out

    def prediction(self, o: np.ndarray) -> Prediction:
        if self.gaussian:
            mean, log_var, logit = (float(v) for v in o[:3])
            return GaussianPrediction(mean * self.target_std + self.target_mean,
                                      float(np.exp(log_var)) * self.target_std ** 2,
                                      p_crash=float(1.0 / (1.0 + np.exp(-logit))))
        p = np.exp(o - o.max())
        p /= p.sum()
        return BarPrediction(self.edges, p[1:], float(p[0]))

    def predict(self, query: Trial, context: Sequence[Trial] = ()) -> Prediction:
        return self.predict_many([query], context)[0]

    def predict_many(self, queries: Sequence[Trial], context: Sequence[Trial] = ()
                     ) -> list[Prediction]:
        results: list[Prediction | None] = [None] * len(queries)
        open_idx, feats = [], []
        for i, q in enumerate(queries):
            self._check_query(q)
            if q.crashed:
                results[i] = GaussianPrediction(0.0, 0.0, p_crash=1.0)
            elif q.complete:
                results[i] = GaussianPrediction(float(q.J_obs), 0.0)
            else:
                open_idx.append(i)
                feats.append(self.features(q))
        if open_idx:
            for i, o in zip(open_idx, self.outputs(feats)):
                results[i] = self.prediction(o)
        return results  # type: ignore[return-value]


def load(model_id: str, spec: ProblemSpec, home: Path | str | None = None, **kw: Any
         ) -> LegacyDecPFNPredictor:
    """The predictor of an imported thesis DecPFN id, ``ModelNotTrained`` when it is absent."""
    from esbopfn.training.registry import ModelId, resolve

    path = resolve(model_id, home, hint="Import it with `pfn import thesis-models --from DIR`.")
    kw.setdefault("device", "cpu")
    return LegacyDecPFNPredictor.from_dir(path, spec, name=ModelId.parse(model_id).short, **kw)
