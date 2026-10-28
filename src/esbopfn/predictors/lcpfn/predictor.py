"""LC-PFN inference: extrapolation of the cost curve from the observed grid points."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from esbopfn.core import ProblemSpec, Trial
from esbopfn.predictors.base import GaussianPrediction, Prediction, PredictorBase
from esbopfn.predictors.lcpfn.model import LCPFN, LCPFNConfig
from esbopfn.training import registry
from esbopfn.training.common import resolve_device
from esbopfn.training.heads import make_head
from esbopfn.training.normalization import Calib


@dataclass
class CurvePrediction:
    """Predictions of ``J_t`` at the grid points after the observed prefix."""

    steps: np.ndarray                 # (Q,) 1-based grid steps t_{κ+1} .. t_N
    predictions: list[Prediction]


class LCPFNPredictor(PredictorBase):
    """A trained LC-PFN (``lcpfn`` or ``lcpfn-last``). Context-blind.

    The context is every curve grid point ``<= query.t_obs``. ``predict`` returns the prediction
    at the final grid point ``T``, :meth:`predict_curve` the predictions at every remaining grid
    point (for ``lcpfn-last`` models only the final one is trained, the others are returned
    anyway).
    """

    supports_unrun = False
    context_aware = False

    def __init__(self, path: Path | str, spec: ProblemSpec, device: str = "auto",
                 name: str | None = None, batch_size: int = 512):
        files = registry.load_model(path)
        cfg = dict(files.model_cfg)
        self.family = cfg.pop("family", "lcpfn")
        self.cfg = LCPFNConfig(**cfg)
        self.calib = Calib.from_dict(files.calib)
        if self.calib.problem != spec.name:
            raise ValueError(f"model trained on {self.calib.problem}, asked for {spec.name}")
        self.spec = spec
        self.head = make_head(self.calib)
        self.device = resolve_device(device)
        self.model = LCPFN(self.cfg)
        self.model.load_state_dict(files.state_dict)
        self.model.to(self.device).eval()
        self.grid = np.asarray(self.calib.grid, dtype=np.int64)
        self.name = name or str(path)
        self.batch_size = batch_size
        self.path = Path(path)

    def predictable_steps(self, spec: ProblemSpec | None = None) -> np.ndarray:
        return self.grid[:-1].copy()

    def kappa(self, query: Trial) -> int:
        k = int(np.searchsorted(self.grid, query.t_obs, side="right"))
        if k == 0:
            raise ValueError(f"{self.name}: no curve grid step observed (t_obs = {query.t_obs},"
                             f" first step {int(self.grid[0])})")
        return min(k, len(self.grid) - 1)

    @torch.no_grad()
    def _forward(self, queries: Sequence[Trial], kappa: int, q_pos: np.ndarray) -> torch.Tensor:
        """Outputs ``(n, len(q_pos), n_outputs)`` for queries sharing the cutoff ``kappa``."""
        pos = np.concatenate([np.arange(kappa), q_pos])
        t_norm = self.calib.normalize_time(self.grid[pos].astype(np.float64))
        outs = []
        for a in range(0, len(queries), self.batch_size):
            part = queries[a:a + self.batch_size]
            J = np.stack([np.asarray(q.traj.J, dtype=np.float64)[self.grid[:kappa] - 1]
                          for q in part])
            t = torch.from_numpy(np.broadcast_to(t_norm, (len(part), len(pos))).copy())
            y = torch.from_numpy(self.calib.normalize_J(J))
            outs.append(self.model(t.to(self.device), y.to(self.device)).float().cpu())
        return torch.cat(outs, dim=0)

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
                results[i] = GaussianPrediction(q.J_obs, 0.0)
            else:
                groups.setdefault(self.kappa(q), []).append(i)
        last = np.array([len(self.grid) - 1])
        for k, idx in groups.items():
            out = self._forward([queries[i] for i in idx], k, last)[:, 0]
            for i, p in zip(idx, self.head.predictions(out)):
                results[i] = p
        return results  # type: ignore[return-value]

    def predict_curve(self, query: Trial) -> CurvePrediction:
        """Predictions at every grid point after the observed prefix."""
        self._check_query(query)
        if query.complete:
            return CurvePrediction(np.zeros(0, dtype=np.int64), [])
        if query.crashed:   # every later point is the crash atom
            rest = self.grid[self.grid > query.t_obs]
            return CurvePrediction(rest.copy(),
                                   [GaussianPrediction(0.0, 0.0, p_crash=1.0) for _ in rest])
        k = self.kappa(query)
        q_pos = np.arange(k, len(self.grid))
        out = self._forward([query], k, q_pos)[0]
        return CurvePrediction(self.grid[q_pos].copy(), list(self.head.predictions(out)))


def load(model_id: str, spec: ProblemSpec, home: Path | str | None = None, **kw: Any
         ) -> LCPFNPredictor:
    """The predictor of a trained model id, raising ``ModelNotTrained`` when it is absent."""
    path = registry.resolve(model_id, home)
    return LCPFNPredictor(path, spec, name=registry.ModelId.parse(model_id).short, **kw)
