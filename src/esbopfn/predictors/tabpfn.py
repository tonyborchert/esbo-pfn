"""Trajectory-TabPFN (thesis method, off-the-shelf TabPFN-3 predictors).

The released TabPFN v3 regressor is used without fine-tuning. Every non-crashed context trial
``k`` contributes one row ``[t_l, theta_k]`` with target ``J_{t_l}(theta_k)`` (raw cumulative cost)
for each point of ``grids.curve_grid(spec, L)`` with ``t_l <= T_k``. A running query also
contributes its own observed prefix when it is not already in the context. The query row is
``[T, theta_i]``. TabPFN's output bar distribution (``output_type="full"``: softmax of the logits
over ``criterion.borders`` in raw units) becomes a :class:`BarPrediction` with ``p_crash = 0``.
TabPFN's half-normal tails beyond the outer borders are folded into the outer bars.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.core import ProblemSpec, Trial
from esbopfn.grids import curve_grid
from esbopfn.predictors.base import (
    BarPrediction,
    GaussianPrediction,
    Prediction,
    PredictorBase,
)
from esbopfn.predictors.checkpoints import tabpfn_checkpoint
from esbopfn.predictors.segment_gp import same_trial

#: TabPFN ensemble size, as in the thesis runs.
N_ESTIMATORS = 8
#: Grid sizes evaluated in the thesis (``L = 50`` drives ESBO-T-TabPFN).
GRID_SIZES = (25, 50, 100, 250)
#: Query rows per forward pass, keeps long query batches within GPU/MPS memory.
PREDICT_CHUNK = 64


def default_device() -> str:
    """``mps`` if available, else ``cuda``, else ``cpu``."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@functools.cache
def _regressor(path: str, n_estimators: int, device: str) -> Any:
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        model_path=path,
        device=device,
        n_estimators=n_estimators,
        inference_precision="auto",
        ignore_pretraining_limits=True,
    )


def load_regressor(
    n_estimators: int = N_ESTIMATORS,
    device: str | None = None,
    checkpoint: Path | str | None = None,
) -> Any:
    """The TabPFN v3 regressor loaded from ``$PFN_HOME/checkpoints/tabpfn`` (cached per process).

    Raises :class:`~esbopfn.predictors.checkpoints.CheckpointMissing` if the file is absent.
    """
    path = Path(checkpoint) if checkpoint is not None else tabpfn_checkpoint()
    return _regressor(str(path.resolve()), int(n_estimators), device or default_device())


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        import torch

        x = x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, dtype=np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.where(np.isnan(logits), -np.inf, logits)
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def fit_predict_bars(reg: Any, X: np.ndarray, y: np.ndarray, Xq: np.ndarray) -> list[BarPrediction]:
    """Fit TabPFN on ``(X, y)`` and return one bar distribution per query row."""
    import torch

    out: list[BarPrediction] = []
    with torch.no_grad():
        reg.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        for i in range(0, len(Xq), PREDICT_CHUNK):
            res = reg.predict(np.asarray(Xq[i:i + PREDICT_CHUNK], dtype=np.float64),
                              output_type="full")
            probs = _softmax(_to_numpy(res["logits"]))
            edges = _to_numpy(res["criterion"].borders)
            out += [BarPrediction(edges, p / p.sum(), 0.0) for p in probs]
    return out


def degenerate_table(y: np.ndarray) -> bool:
    """Fewer than two rows or a constant target, where TabPFN is not fit."""
    return len(y) < 2 or bool(np.allclose(y, y[0]))


@dataclass
class TrajectoryTabPFN(PredictorBase):
    """Trajectory-TabPFN with ``L`` grid points (id ``tabpfn/<L>``)."""

    spec: ProblemSpec
    L: int = 50
    n_estimators: int = N_ESTIMATORS
    device: str | None = None
    checkpoint: Path | str | None = None
    supports_unrun: bool = True
    context_aware: bool = True

    def __post_init__(self) -> None:
        self.L = int(self.L)
        self.grid = curve_grid(self.spec, self.L)
        self.name = f"tabpfn/{self.L}"
        # Fail early with the download instructions.
        if self.checkpoint is None:
            tabpfn_checkpoint()

    @property
    def regressor(self) -> Any:
        return load_regressor(self.n_estimators, self.device, self.checkpoint)

    def predictable_steps(self, spec: ProblemSpec) -> np.ndarray:
        """The grid points before the horizon."""
        return self.grid[self.grid < spec.T]

    def _rows(self, trials: Sequence[Trial]) -> tuple[list[list[float]], list[float]]:
        X: list[list[float]] = []
        y: list[float] = []
        for tr in trials:
            if tr.crashed or tr.unrun:
                continue
            theta = [float(v) for v in tr.theta]
            for t in self.grid[self.grid <= tr.t_obs]:
                X.append([float(t), *theta])
                y.append(float(tr.traj.J[t - 1]))
        return X, y

    def predict(self, query: Trial, context: Sequence[Trial]) -> Prediction:
        return self.predict_many([query], context)[0]

    def predict_many(self, queries: Sequence[Trial], context: Sequence[Trial]) -> list[Prediction]:
        """One TabPFN fit on the context (plus the prefixes of queries not in it) for all queries."""
        extra = [q for q in queries if not any(same_trial(q, c) for c in context)]
        X, y = self._rows([*context, *extra])
        results: list[Prediction | None] = [None] * len(queries)
        open_idx: list[int] = []
        for i, q in enumerate(queries):
            if q.crashed:
                results[i] = GaussianPrediction(0.0, 0.0, p_crash=1.0)
            elif q.complete:
                results[i] = GaussianPrediction(q.J_obs, 0.0)
            else:
                open_idx.append(i)
        if open_idx:
            yy = np.asarray(y)
            if degenerate_table(yy):
                for i in open_idx:  # nothing to fit: the observed cost, no spread
                    results[i] = GaussianPrediction(queries[i].J_obs, 0.0)
            else:
                Xq = np.array([[float(self.spec.T), *map(float, queries[i].theta)]
                               for i in open_idx])
                for i, bar in zip(open_idx, fit_predict_bars(self.regressor, np.array(X), yy, Xq)):
                    results[i] = bar
        return [r for r in results if r is not None]


def make_predictor(args: tuple[str, ...], spec: ProblemSpec, **kwargs: Any) -> TrajectoryTabPFN:
    """Registry factory for ``tabpfn/<L>``."""
    if len(args) != 1 or not args[0].isdigit():
        raise ValueError(f"expected tabpfn/<L> with an integer L, got tabpfn/{'/'.join(args)}")
    L = int(args[0])
    if not 1 <= L <= spec.T:
        raise ValueError(f"L must lie in [1, {spec.T}]")
    return TrajectoryTabPFN(spec, L=L, **kwargs)
