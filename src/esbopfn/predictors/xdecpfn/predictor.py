"""XDecPFN inference: :class:`~esbopfn.predictors.base.Predictor` over a trained model directory.

Every context trial enters as observed: a completed trial with all its grid tokens and
``e_complete``, a crashed one with its tokens before the crash and ``e_crash``, an early-stopped
one with the tokens up to its cutoff and ``e_cut``. The running trial is one more cut trajectory
whose token gives the prediction. Stage 1 is causal, so it runs once per trajectory (cached) and
any cutoff reads the output at its last observed token. An unrun controller, or a running trial
that has not reached the first grid point yet, is predicted through an unrun query token.

Predictions are :class:`~esbopfn.predictors.base.BarPrediction` over raw cost: the bins are the
``exp`` of the ln-J edges with the same masses, read as uniform in J inside a bin.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from esbopfn.core import ProblemSpec, Trajectory, Trial
from esbopfn.grids import decoder_grid, tokens
from esbopfn.predictors.base import (
    BarPrediction,
    GaussianPrediction,
    Prediction,
    PredictorBase,
)
from esbopfn.predictors.xdecpfn.data import XCalib, normalize, raw_features
from esbopfn.predictors.xdecpfn.model import (
    MARK_COMPLETE,
    MARK_CRASH,
    MARK_CUT,
    XDecPFN,
    XDecPFNConfig,
)

#: Rows per cross-stage call and unrun queries per call.
ROW_CHUNK = 64
QUERY_CHUNK = 512
_CACHE_MAX = 20000


def bar_predictions(logits: torch.Tensor, edges: np.ndarray) -> list[BarPrediction]:
    """``(M, N + 1)`` logits (last = crash atom) -> bar predictions on ``edges``."""
    a = logits.detach().double().cpu().numpy().reshape(-1, edges.size)
    a = a - a.max(axis=1, keepdims=True)
    p = np.exp(a)
    p /= p.sum(axis=1, keepdims=True)
    return [BarPrediction(edges, row[:-1], float(row[-1])) for row in p]


def known_outcome(query: Trial) -> Prediction | None:
    """The degenerate prediction of a crashed or completed query, else None."""
    if query.crashed:
        return GaussianPrediction(0.0, 0.0, p_crash=1.0)
    if query.complete:
        return GaussianPrediction(float(query.traj.J[-1]), 0.0)
    return None


class XDecPFNPredictor(PredictorBase):
    """A trained XDecPFN conditioned on the other trials of the run."""

    name = "xdecpfn"
    supports_unrun = True
    context_aware = True

    def __init__(self, model: XDecPFN, calib: XCalib, spec: ProblemSpec, device: str = "cpu",
                 name: str | None = None):
        if calib.T != spec.T:
            raise ValueError(f"model was trained for T={calib.T}, {spec.name} has T={spec.T}")
        self.spec = spec
        self.calib = calib
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.grid = decoder_grid(spec)
        if not np.array_equal(self.grid, np.asarray(calib.grid)):
            raise ValueError("the model's decoder grid differs from the problem's")
        self.edges = calib.edges
        self.name = name or self.name
        self._cache: dict[int, tuple[Trajectory, torch.Tensor]] = {}

    @classmethod
    def from_dir(cls, path: Path | str, spec: ProblemSpec, device: str = "cpu",
                 name: str | None = None) -> XDecPFNPredictor:
        from esbopfn.training.registry import load_model

        files = load_model(path)
        cfg = XDecPFNConfig.from_dict(files.model_cfg)
        model = XDecPFN(cfg)
        model.load_state_dict(files.state_dict)
        return cls(model, XCalib.from_dict(files.calib), spec, device, name)

    def predictable_steps(self, spec: ProblemSpec) -> np.ndarray:
        """The decoder grid steps before ``T``, where the trial gains a token."""
        return decoder_grid(spec)

    # ---------------------------------------------------------------- stage 1
    def _stage1(self, traj: Trajectory) -> torch.Tensor:
        """Causal outputs ``(L_i, d)`` at every grid token of ``traj`` before a crash (cached)."""
        hit = self._cache.get(id(traj))
        if hit is not None and hit[0] is traj:
            return hit[1]
        tk = tokens(traj, self.grid, traj.T)
        if len(tk) == 0:
            h = torch.zeros(0, self.model.cfg.d_model, device=self.device)
        else:
            x = normalize(raw_features(tk.J, tk.covariates, tk.t, self.calib.ln_floor), self.calib)
            with torch.no_grad():
                h = self.model.stage1(torch.from_numpy(x).to(self.device)[None])[0]
        if len(self._cache) >= _CACHE_MAX:
            self._cache.clear()
        self._cache[id(traj)] = (traj, h)
        return h

    def _n_obs(self, trial: Trial) -> int:
        cs = trial.traj.crash_step
        limit = trial.t_obs if cs is None else min(trial.t_obs, cs - 1)
        return int(np.searchsorted(self.grid, limit, side="right"))

    def _token(self, trial: Trial) -> tuple[torch.Tensor, int] | None:
        """Stage-1 representation and marker of an observed trial, None if it has no token."""
        n = self._n_obs(trial)
        if trial.unrun or (n == 0 and not trial.crashed):
            return None                    # no grid token observed yet
        h = self._stage1(trial.traj)
        rep = h[n - 1] if n > 0 else self.model.empty.detach()
        if trial.crashed:
            return rep, MARK_CRASH
        return rep, (MARK_COMPLETE if trial.complete else MARK_CUT)

    def _theta(self, thetas: np.ndarray) -> torch.Tensor:
        z = (np.asarray(thetas, dtype=np.float64).reshape(-1, self.spec.d) - self.calib.theta_mean) \
            / self.calib.theta_std
        return torch.as_tensor(z, dtype=torch.float32, device=self.device)

    def _context(self, context: Sequence[Trial], exclude: set[int]) -> tuple[torch.Tensor, ...]:
        reps, markers, thetas = [], [], []
        for tr in context:
            if id(tr.traj) in exclude:
                continue
            tok = self._token(tr)
            if tok is None:
                continue
            reps.append(tok[0])
            markers.append(tok[1])
            thetas.append(tr.theta)
        d = self.model.cfg.d_model
        if not reps:
            z = torch.zeros(0, d, device=self.device)
            return z, torch.zeros(0, dtype=torch.long, device=self.device), self._theta(
                np.zeros((0, self.spec.d)))
        return (torch.stack(reps), torch.as_tensor(markers, dtype=torch.long, device=self.device),
                self._theta(np.stack(thetas)))

    # ---------------------------------------------------------------- predictions
    def predict(self, query: Trial, context: Sequence[Trial]) -> Prediction:
        return self.predict_many([query], context)[0]

    def predict_many(self, queries: Sequence[Trial], context: Sequence[Trial]) -> list[Prediction]:
        """Predictions for several running trials (or cutoffs of one trial) against one context.

        Each query that has at least one grid token joins the context as a cut trajectory in its
        own cross-stage row. Queries without a token are scored as unrun queries in one pass.
        """
        out: list[Prediction | None] = [known_outcome(q) for q in queries]
        traj_q = [i for i, q in enumerate(queries) if out[i] is None and self._n_obs(q) > 0
                  and not q.unrun]
        unrun_q = [i for i, q in enumerate(queries) if out[i] is None and i not in set(traj_q)]
        exclude = {id(q.traj) for q in queries}
        rep, marker, theta = self._context(context, exclude)
        n = rep.shape[0]
        with torch.no_grad():
            for a in range(0, len(traj_q), ROW_CHUNK):
                idx = traj_q[a:a + ROW_CHUNK]
                E = len(idx)
                q_rep = torch.stack([self._token(queries[i])[0] for i in idx])
                q_th = self._theta(np.stack([queries[i].theta for i in idx]))
                reps = torch.cat([rep.expand(E, n, -1), q_rep[:, None]], dim=1)
                ths = torch.cat([theta.expand(E, n, -1), q_th[:, None]], dim=1)
                mks = torch.cat([marker.expand(E, n),
                                 torch.full((E, 1), MARK_CUT, device=self.device)], dim=1)
                valid = torch.ones(E, n + 1, dtype=torch.bool, device=self.device)
                lt, _ = self.model.stage2(reps, ths, mks, valid)
                for i, p in zip(idx, bar_predictions(lt[:, n], self.edges)):
                    out[i] = p
        if unrun_q:
            preds = self._unrun(np.stack([queries[i].theta for i in unrun_q]), rep, marker, theta)
            for i, p in zip(unrun_q, preds):
                out[i] = p
        return out  # type: ignore[return-value]

    def predict_unrun(self, thetas: np.ndarray, context: Sequence[Trial]) -> list[BarPrediction]:
        """Predictions of ``J_T`` for unrun controllers ``thetas`` ``(M, d)`` given ``context``."""
        rep, marker, theta = self._context(context, set())
        return self._unrun(np.asarray(thetas), rep, marker, theta)

    def _unrun(self, thetas: np.ndarray, rep, marker, theta) -> list[BarPrediction]:
        q = self._theta(thetas)
        n = rep.shape[0]
        valid = torch.ones(1, n, dtype=torch.bool, device=self.device)
        preds: list[BarPrediction] = []
        with torch.no_grad():
            for a in range(0, q.shape[0], QUERY_CHUNK):
                _, lq = self.model.stage2(rep[None], theta[None], marker[None], valid,
                                          q[None, a:a + QUERY_CHUNK])
                preds += bar_predictions(lq[0], self.edges)
        return preds

    def clear_cache(self) -> None:
        self._cache.clear()
