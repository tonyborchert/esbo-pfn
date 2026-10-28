"""The XDecPFN checkpoints used for the thesis results, served as ``xdecpfn/<p>/thesis/default``.

These weights were trained with an earlier implementation whose architecture differs from
:class:`~esbopfn.predictors.xdecpfn.model.XDecPFN` (planning record of the old code base):

- stage-1 tokens ``Linear([Z, Delta Z, y, Delta y, t / 1000, ln t])`` with the channels ``Z`` of the
  old generators, ``y`` the z-scored ``ln J`` and differences taken between consecutive tokens,
- a token grid ``arange(0, T - 1, k)[1:] + 1`` plus one token at a finished trajectory's last
  observed step (the step before a crash),
- ``e_theta`` = ``MLP(LN(Linear(theta)))`` and a readout skip ``head(LN(x + skip))``,
- the crash atom as the first logit.

:class:`LegacyXDecPFN` reproduces that forward pass with the original parameter names, so the
checkpoints load with strict state-dict matching. :class:`LegacyXDecPFNPredictor` maps the
trajectories of this package onto the old channels: Cartpole ``[x - x_ref, xdot, phi, phidot]``,
Furuta ``[phi_arm, phi_pend, phidot_arm, phidot_pend]``, PT2 ``[y]``, Three-Tank ``[V2, V3, V4]``
and Boiler ``[temperature, u]`` with the heater command ``u`` recovered from the stage costs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from esbopfn.core import ProblemSpec, Trajectory, Trial
from esbopfn.predictors.base import BarPrediction, Prediction, PredictorBase
from esbopfn.predictors.xdecpfn.predictor import known_outcome

#: Old marker rows.
OLD_CUT, OLD_CRASH, OLD_COMPLETE, OLD_QUERY = 0, 1, 2, 3
LN_FLOOR = 1e-30


def _mlp(din: int, dh: int, dout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(din, dh), nn.GELU(), nn.Linear(dh, dout))


class LegacyBlock(nn.Module):
    """The old pre-LN block (parameter names ``n1 n2 qkv proj ff``)."""

    def __init__(self, d: int, h: int, d_ff: int):
        super().__init__()
        self.h, self.dh = h, d // h
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = _mlp(d, d_ff, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                causal: bool = False) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        sp = lambda z: z.view(B, L, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(sp(q), sp(k), sp(v), attn_mask=None if causal else mask,
                                           is_causal=causal)
        x = x + self.proj(o.transpose(1, 2).reshape(B, L, d))
        return x + self.ff(self.n2(x))


class LegacyXDecPFN(nn.Module):
    """The old two-stage model (``noinj1q`` arm), forward pass only."""

    def __init__(self, mcfg: dict[str, Any]):
        super().__init__()
        d, h = int(mcfg["d_model"]), int(mcfg["n_heads"])
        d_ff = int(mcfg.get("ffn_mult", 2)) * d
        self.c_chan, self.d_theta = int(mcfg["c_chan"]), int(mcfg["d_theta"])
        self.skip = bool(mcfg.get("skip", True))
        self.tok = nn.Linear(2 * self.c_chan + 4, d)
        self.th_enc2 = nn.Linear(self.d_theta, d)
        self.th_mlp = _mlp(d, d, d)
        self.th_ln = nn.LayerNorm(d)
        self.marker = nn.Embedding(4, d)
        self.within = nn.ModuleList(LegacyBlock(d, h, d_ff) for _ in range(int(mcfg["layers_within"])))
        self.cross = nn.ModuleList(LegacyBlock(d, h, d_ff) for _ in range(int(mcfg["layers_cross"])))
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, int(mcfg["n_bins"]) + 1)

    def encode(self, Z: torch.Tensor, yn: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """``Z (S, L, C)``, ``yn (S, L)``, ``step (S, L)`` -> ``(S, L, d)`` causal outputs."""
        dZ = torch.zeros_like(Z)
        dZ[:, 1:] = Z[:, 1:] - Z[:, :-1]
        dy = torch.zeros_like(yn)
        dy[:, 1:] = yn[:, 1:] - yn[:, :-1]
        feats = torch.cat([Z, dZ, yn[..., None], dy[..., None], (step / 1000.0)[..., None],
                           torch.log(step.clamp_min(1.0))[..., None]], dim=-1)
        x = self.tok(feats)
        for blk in self.within:
            x = blk(x, causal=True)
        return x

    def theta_embed(self, theta: torch.Tensor) -> torch.Tensor:
        return self.th_mlp(self.th_ln(self.th_enc2(theta)))

    def cross_logits(self, h_gat: torch.Tensor, theta: torch.Tensor, marker: torch.Tensor,
                     q_theta: torch.Tensor | None = None):
        """One group of ``n`` trajectory tokens per row (``(E, n, ...)``) plus optional unrun
        queries ``(E, Q, d_theta)``. Returns ``[atom, bars]`` logits of both."""
        n = h_gat.shape[1]
        th = self.theta_embed(theta)
        x = h_gat + th + self.marker(marker)
        skip = h_gat
        Q = 0 if q_theta is None else q_theta.shape[1]
        if Q:
            qth = self.theta_embed(q_theta)
            x = torch.cat([x, qth + self.marker.weight[OLD_QUERY]], dim=1)
            skip = torch.cat([h_gat, qth], dim=1)
        L = n + Q
        key = torch.zeros(L, dtype=torch.bool, device=x.device)
        key[:n] = True
        mask = (key[None, :] | torch.eye(L, dtype=torch.bool, device=x.device))[None, None]
        for blk in self.cross:
            x = blk(x, mask=mask)
        if self.skip:
            x = x + skip
        out = self.head(self.nf(x))
        return out[:, :n], (out[:, n:] if Q else None)


# --------------------------------------------------------------------------- channels
def _goals(T: int) -> np.ndarray:
    from esbopfn.problems.cartpole.sim import GOALS

    return np.asarray(GOALS[:T], dtype=np.float32)


def _boiler_u(traj: Trajectory) -> np.ndarray:
    """Heater command per step, recovered from the stage cost
    ``0.5 min(|20 - T_meas|, 2)^2 + u + 8 [u_t > u_{t-1}]``."""
    from esbopfn.problems.boiler.sim import COMFORT_CLIP, SET_POINT

    J = np.asarray(traj.J, dtype=np.float64)
    j = np.diff(np.concatenate([[0.0], np.where(np.isfinite(J), J, 0.0)]))
    temp = np.asarray(traj.x, dtype=np.float64).reshape(len(J), -1)[:, 0]
    comfort = 0.5 * np.minimum(np.abs(SET_POINT - temp), COMFORT_CLIP) ** 2
    return ((j - comfort) > 0.5).astype(np.float32)


def legacy_channels(problem: str, traj: Trajectory) -> np.ndarray:
    """The ``(T, C)`` channel block the old encoders were trained on."""
    x = np.asarray(traj.x, dtype=np.float32)
    x = x.reshape(x.shape[0], -1)
    xd = None if traj.xdot is None else np.asarray(traj.xdot, dtype=np.float32).reshape(x.shape[0], -1)
    if problem == "cartpole":
        return np.stack([x[:, 0] - _goals(x.shape[0]), xd[:, 0], x[:, 1], xd[:, 1]], -1)
    if problem == "furuta":
        return np.stack([x[:, 0], x[:, 1], xd[:, 0], xd[:, 1]], -1)
    if problem == "boiler":
        return np.stack([x[:, 0], _boiler_u(traj)], -1)
    return x                                    # pt2_deadtime [y], threetank_pi [V2, V3, V4]


# --------------------------------------------------------------------------- predictor
class LegacyXDecPFNPredictor(PredictorBase):
    """Thesis XDecPFN checkpoint behind the :class:`Predictor` protocol (old token rules)."""

    name = "xdecpfn-thesis"
    supports_unrun = True
    context_aware = True

    def __init__(self, model: LegacyXDecPFN, calib: dict[str, Any], meta: dict[str, Any],
                 spec: ProblemSpec, device: str = "cpu", name: str | None = None):
        self.spec = spec
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.k = int(meta["k"])
        self.grid = np.arange(0, spec.T - 1, self.k, dtype=np.int64)[1:] + 1
        self.units = str(calib.get("units", "raw_J"))
        edges = np.asarray(calib["edges"], dtype=np.float64)
        self.edges = np.exp(edges) if self.units == "ln_J" else edges
        self.j_mean, self.j_std = float(calib["j_mean"]), float(calib["j_std"])
        self.th_mean = np.asarray(calib["theta_mean"], dtype=np.float64)
        self.th_std = np.asarray(calib["theta_std"], dtype=np.float64)
        self.z_mean = np.asarray(calib["z_mean"], dtype=np.float32)
        self.z_std = np.asarray(calib["z_std"], dtype=np.float32)
        if self.th_mean.size != spec.d:
            raise ValueError(f"checkpoint has d_theta={self.th_mean.size}, {spec.name} has {spec.d}")
        self.name = name or self.name
        self._cache: dict[tuple[int, int, int], tuple[Trajectory, torch.Tensor]] = {}

    @classmethod
    def from_dir(cls, path: Path | str, spec: ProblemSpec, device: str = "cpu",
                 name: str | None = None) -> LegacyXDecPFNPredictor:
        from esbopfn.training.registry import load_model

        files = load_model(path)
        model = LegacyXDecPFN(files.model_cfg["mcfg"])
        model.load_state_dict(files.state_dict, strict=True)
        return cls(model, files.calib, files.model_cfg, spec, device, name)

    def predictable_steps(self, spec: ProblemSpec) -> np.ndarray:
        return self.grid.copy()

    # ---------------------------------------------------------------- rows
    def _steps(self, trial: Trial, at_end: bool) -> tuple[np.ndarray, int]:
        if at_end:
            crashed = trial.crashed
            nat = int(trial.traj.crash_step) if crashed else int(trial.t_obs)
            last = max(nat - 1, 1) if crashed else nat
            g = self.grid[self.grid <= last]
            steps = g if g.size and g[-1] == last else np.concatenate([g, [last]])
            return steps.astype(np.int64), (OLD_CRASH if crashed else OLD_COMPLETE)
        return self.grid[self.grid <= trial.t_obs], OLD_CUT

    def _vector(self, trial: Trial, at_end: bool) -> tuple[torch.Tensor, int] | None:
        steps, marker = self._steps(trial, at_end)
        if steps.size == 0:
            return None
        key = (id(trial.traj), int(trial.t_obs), int(at_end))
        hit = self._cache.get(key)
        if hit is not None and hit[0] is trial.traj:
            return hit[1], marker
        chan = legacy_channels(self.spec.name, trial.traj)
        Z = (chan[steps - 1] - self.z_mean) / self.z_std
        v = np.asarray(trial.traj.J, dtype=np.float64)[steps - 1]
        if self.units == "ln_J":
            v = np.log(np.maximum(v, LN_FLOOR))
        yn = ((v - self.j_mean) / self.j_std).astype(np.float32)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)[None]
        with torch.no_grad():
            h = self.model.encode(t(Z), t(yn), t(steps.astype(np.float32)))[0, -1]
        if len(self._cache) > 20000:
            self._cache.clear()
        self._cache[key] = (trial.traj, h)
        return h, marker

    def _theta(self, th: np.ndarray) -> torch.Tensor:
        z = ((np.asarray(th, dtype=np.float64).reshape(-1, self.spec.d) - self.th_mean)
             / self.th_std).astype(np.float32)
        return torch.as_tensor(z, device=self.device)

    def _context(self, context: Sequence[Trial], exclude: set[int]):
        vecs, marks, ths = [], [], []
        for tr in context:
            if tr.unrun or id(tr.traj) in exclude:
                continue
            at_end = tr.crashed or tr.complete
            got = self._vector(tr, at_end)
            if got is None:
                continue
            vecs.append(got[0])
            marks.append(got[1])
            ths.append(tr.theta)
        return vecs, marks, ths

    def _bars(self, logits: torch.Tensor) -> list[BarPrediction]:
        a = logits.detach().double().cpu().numpy().reshape(-1, self.edges.size)
        a = a - a.max(axis=1, keepdims=True)
        p = np.exp(a)
        p /= p.sum(axis=1, keepdims=True)
        return [BarPrediction(self.edges, row[1:], float(row[0])) for row in p]

    # ---------------------------------------------------------------- predictions
    def predict(self, query: Trial, context: Sequence[Trial]) -> Prediction:
        return self.predict_many([query], context)[0]

    def predict_many(self, queries: Sequence[Trial], context: Sequence[Trial]) -> list[Prediction]:
        out: list[Prediction | None] = [known_outcome(q) for q in queries]
        vecs, marks, ths = self._context(context, {id(q.traj) for q in queries})
        unrun = []
        with torch.no_grad():
            for i, q in enumerate(queries):
                if out[i] is not None:
                    continue
                got = None if q.unrun else self._vector(q, at_end=False)
                if got is None:
                    unrun.append(i)
                    continue
                n = len(vecs) + 1
                h = torch.stack(vecs + [got[0]])[None]
                th = self._theta(np.stack(ths + [q.theta]))[None]
                mk = torch.as_tensor(marks + [OLD_CUT], device=self.device)[None]
                logits, _ = self.model.cross_logits(h, th, mk)
                out[i] = self._bars(logits[0, n - 1])[0]
        if unrun:
            preds = self._unrun(np.stack([queries[i].theta for i in unrun]), vecs, marks, ths)
            for i, p in zip(unrun, preds):
                out[i] = p
        return out  # type: ignore[return-value]

    def predict_unrun(self, thetas: np.ndarray, context: Sequence[Trial]) -> list[BarPrediction]:
        vecs, marks, ths = self._context(context, set())
        return self._unrun(np.asarray(thetas), vecs, marks, ths)

    def _unrun(self, thetas, vecs, marks, ths) -> list[BarPrediction]:
        d = self.model.head.in_features
        h = torch.stack(vecs)[None] if vecs else torch.zeros(1, 0, d, device=self.device)
        th = self._theta(np.stack(ths))[None] if ths else torch.zeros(1, 0, self.spec.d,
                                                                    device=self.device)
        mk = torch.as_tensor(marks, dtype=torch.long, device=self.device)[None]
        q = self._theta(thetas)
        preds: list[BarPrediction] = []
        with torch.no_grad():
            for a in range(0, q.shape[0], 128):
                _, lq = self.model.cross_logits(h, th, mk, q[None, a:a + 128])
                preds += self._bars(lq[0])
        return preds
