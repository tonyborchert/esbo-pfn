"""DecPFN inference: the horizon prediction after the tokens observed so far."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from esbopfn import grids
from esbopfn.core import ProblemSpec, Trial
from esbopfn.predictors.base import GaussianPrediction, Prediction, PredictorBase
from esbopfn.predictors.decpfn.model import DecPFN, DecPFNConfig
from esbopfn.training import registry
from esbopfn.training.common import resolve_device
from esbopfn.training.heads import make_head
from esbopfn.training.normalization import Calib


class DecPFNPredictor(PredictorBase):
    """A trained DecPFN (any variant). Context-blind, predicts at every decoder grid step.

    ``predict`` tokenizes the query on the decoder grid up to the largest grid step
    ``<= query.t_obs`` and returns the head's prediction at that position, a
    :class:`~esbopfn.predictors.base.BarPrediction` with raw-unit edges or a
    :class:`~esbopfn.predictors.base.GaussianPrediction`, with ``p_crash`` the crash atom.
    """

    supports_unrun = False
    context_aware = False

    def __init__(self, path: Path | str, spec: ProblemSpec, device: str = "auto",
                 name: str | None = None, batch_size: int = 256):
        files = registry.load_model(path)
        cfg = dict(files.model_cfg)
        self.family = cfg.pop("family", "decpfn")
        self.cfg = DecPFNConfig(**cfg)
        self.calib = Calib.from_dict(files.calib)
        if self.calib.problem != spec.name:
            raise ValueError(f"model trained on {self.calib.problem}, asked for {spec.name}")
        self.spec = spec
        self.states = not self.family.endswith("-nostates")
        self.head = make_head(self.calib)
        self.device = resolve_device(device)
        self.model = DecPFN(self.cfg)
        self.model.load_state_dict(files.state_dict)
        self.model.to(self.device).eval()
        self.grid = np.asarray(self.calib.grid, dtype=np.int64)
        self.name = name or str(path)
        self.batch_size = batch_size
        self.path = Path(path)

    def predictable_steps(self, spec: ProblemSpec | None = None) -> np.ndarray:
        return self.grid.copy()

    def features(self, query: Trial) -> np.ndarray:
        """Normalized inputs ``(n_tokens, F)`` of the observed prefix."""
        tok = grids.tokens(query.traj, self.grid, query.t_obs)
        if len(tok) == 0:
            raise ValueError(f"{self.name}: no decoder grid step observed (t_obs = {query.t_obs},"
                             f" first step {int(self.grid[0])})")
        cov = tok.covariates if self.states else tok.covariates[:, :1]
        raw = np.concatenate([tok.J[:, None], cov, tok.t[:, None].astype(np.float64)], axis=1)
        return self.calib.normalize_inputs(raw)

    @torch.no_grad()
    def outputs(self, feats: list[np.ndarray]) -> torch.Tensor:
        """Head outputs at the last token of every feature block, ``(n, n_outputs)``."""
        res = []
        for a in range(0, len(feats), self.batch_size):
            part = feats[a:a + self.batch_size]
            L = max(f.shape[0] for f in part)
            x = np.zeros((len(part), L, part[0].shape[1]), dtype=np.float32)
            for i, f in enumerate(part):
                x[i, :len(f)] = f
            out = self.model(torch.from_numpy(x).to(self.device))
            last = torch.as_tensor([len(f) - 1 for f in part], device=self.device)
            res.append(out[torch.arange(len(part), device=self.device), last].float().cpu())
        return torch.cat(res, dim=0)

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
                results[i] = GaussianPrediction(q.J_obs, 0.0)
            else:
                open_idx.append(i)
                feats.append(self.features(q))
        if open_idx:
            for i, p in zip(open_idx, self.head.predictions(self.outputs(feats))):
                results[i] = p
        return results  # type: ignore[return-value]


def load(model_id: str, spec: ProblemSpec, home: Path | str | None = None, **kw: Any
         ) -> DecPFNPredictor:
    """The predictor of a trained model id, raising ``ModelNotTrained`` when it is absent."""
    path = registry.resolve(model_id, home)
    return DecPFNPredictor(path, spec, name=registry.ModelId.parse(model_id).short, **kw)
