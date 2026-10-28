"""FT-PFN inference: :class:`~esbopfn.predictors.base.Predictor` over a trained model directory.

Every context trial is tokenized as observed (:func:`~esbopfn.predictors.ftpfn.data.trial_tokens`),
the running trial adds its observed prefix as one more truncated trial, and the query is
``(theta, T)`` without a value. Several queries are packed as separate contexts into the same rows.
Unrun controllers are queries against the context alone, many per context.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from esbopfn.core import ProblemSpec, Trial
from esbopfn.grids import FTPFN_N_GRID, curve_grid
from esbopfn.predictors.base import BarPrediction, Prediction, PredictorBase
from esbopfn.predictors.ftpfn.data import (
    QUERY_HORIZON,
    QUERY_UNRUN,
    ContextTokens,
    FCalib,
    pack,
    trial_tokens,
)
from esbopfn.predictors.ftpfn.model import FTPFN, FTPFNConfig
from esbopfn.predictors.xdecpfn.predictor import bar_predictions, known_outcome

#: Contexts per forward pass and unrun queries per context.
CONTEXT_CHUNK = 32
QUERY_CHUNK = 512


class FTPFNPredictor(PredictorBase):
    """A trained FT-PFN conditioned on the other trials of the run."""

    name = "ftpfn"
    supports_unrun = True
    context_aware = True

    def __init__(self, model: FTPFN, calib: FCalib, spec: ProblemSpec, device: str = "cpu",
                 name: str | None = None):
        if calib.T != spec.T:
            raise ValueError(f"model was trained for T={calib.T}, {spec.name} has T={spec.T}")
        self.spec = spec
        self.calib = calib
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.grid = curve_grid(spec, FTPFN_N_GRID)
        self.edges = calib.edges
        self.name = name or self.name

    @classmethod
    def from_dir(cls, path: Path | str, spec: ProblemSpec, device: str = "cpu",
                 name: str | None = None) -> FTPFNPredictor:
        from esbopfn.training.registry import load_model

        files = load_model(path)
        model = FTPFN(FTPFNConfig.from_dict(files.model_cfg))
        model.load_state_dict(files.state_dict)
        return cls(model, FCalib.from_dict(files.calib), spec, device, name)

    def predictable_steps(self, spec: ProblemSpec) -> np.ndarray:
        """The 8-point curve grid steps before ``T``."""
        g = curve_grid(spec, FTPFN_N_GRID)
        return g[g < spec.T]

    # ---------------------------------------------------------------- tokens
    def _context_arrays(self, trials: Sequence[Trial]) -> tuple[np.ndarray, ...]:
        th, ts, js = [], [], []
        for tr in trials:
            t, J = trial_tokens(tr, self.grid)
            if t.size:
                th.append(np.repeat(np.asarray(tr.theta, dtype=np.float64)[None], t.size, 0))
                ts.append(t.astype(np.float64))
                js.append(J)
        if not ts:
            return np.zeros((0, self.spec.d)), np.zeros(0), np.zeros(0)
        return np.concatenate(th), np.concatenate(ts), np.concatenate(js)

    def _run(self, contexts: list[ContextTokens]) -> list[BarPrediction]:
        preds: list[BarPrediction] = []
        with torch.no_grad():
            for a in range(0, len(contexts), CONTEXT_CHUNK):
                b = pack(contexts[a:a + CONTEXT_CHUNK], self.calib)
                d = self.device
                logits = self.model(b.theta.to(d), b.t.to(d), b.y.to(d), b.kind.to(d),
                                    b.ctx.to(d), b.q_row.to(d), b.q_pos.to(d))
                preds += bar_predictions(logits, self.edges)
        return preds

    # ---------------------------------------------------------------- predictions
    def predict(self, query: Trial, context: Sequence[Trial]) -> Prediction:
        return self.predict_many([query], context)[0]

    def predict_many(self, queries: Sequence[Trial], context: Sequence[Trial]) -> list[Prediction]:
        out: list[Prediction | None] = [known_outcome(q) for q in queries]
        exclude = {id(q.traj) for q in queries}
        base = [tr for tr in context if id(tr.traj) not in exclude]
        th, ts, js = self._context_arrays(base)
        todo = [i for i, q in enumerate(queries) if out[i] is None]
        contexts = []
        for i in todo:
            q = queries[i]
            qt, qj = trial_tokens(q, self.grid)
            theta = np.asarray(q.theta, dtype=np.float64)
            contexts.append(ContextTokens(
                theta=np.concatenate([th, np.repeat(theta[None], qt.size, 0)]),
                t=np.concatenate([ts, qt]), J=np.concatenate([js, qj]), q_theta=theta[None],
                q_target=np.array([np.nan]),
                q_kind=np.array([QUERY_UNRUN if q.unrun else QUERY_HORIZON])))
        for i, p in zip(todo, self._run(contexts)):
            out[i] = p
        return out  # type: ignore[return-value]

    def predict_unrun(self, thetas: np.ndarray, context: Sequence[Trial]) -> list[BarPrediction]:
        """Predictions of ``J_T`` for unrun controllers ``thetas`` ``(M, d)`` given ``context``."""
        th, ts, js = self._context_arrays(context)
        q = np.asarray(thetas, dtype=np.float64).reshape(-1, self.spec.d)
        chunks = [ContextTokens(theta=th, t=ts, J=js, q_theta=q[a:a + QUERY_CHUNK],
                                q_target=np.full(min(QUERY_CHUNK, q.shape[0] - a), np.nan),
                                q_kind=np.full(min(QUERY_CHUNK, q.shape[0] - a), QUERY_UNRUN))
                  for a in range(0, q.shape[0], QUERY_CHUNK)]
        return self._run(chunks)
