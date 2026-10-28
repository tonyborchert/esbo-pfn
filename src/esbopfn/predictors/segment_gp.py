"""Segment-GPs of ESBO-GP (thesis background, early stopping BO, and appendix, our port).

The segment boundaries ``T~_1 < ... < T~_n~ = T`` are the sorted unique termination times of the
non-crashed context trials together with the horizon ``T``. One GP per segment ``(a, b]`` models the
segment cost ``J_b - J_a`` over the controller parameters and is trained on every non-crashed
context trial that observed the segment completely. Crashed context trials contribute neither
boundaries nor data. A query cut at ``t_obs`` that is not a boundary gets ``t_obs`` as its leading
boundary, an unrun query starts at ``0``. The predicted total is Gaussian with

    mean = J_{t_obs} + sum_q max(mu_q, 0),    var = sum_q sigma_q^2,

where ``sigma_q^2`` includes the learned noise of segment GP ``q``. Segments that no non-crashed
context trial observed completely are skipped.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from esbopfn.core import ProblemSpec, Trial
from esbopfn.predictors.base import GaussianPrediction, PredictorBase
from esbopfn.predictors.gp import DEFAULT_ITERS, DEFAULT_LR, ExactGP

Segment = tuple[int, int]


def segment_cost(trial: Trial, a: int, b: int) -> float:
    """``J_b - J_a`` of a trial (1-based steps, ``J_0 = 0``)."""
    J = trial.traj.J
    return float(J[b - 1]) - (float(J[a - 1]) if a > 0 else 0.0)


@dataclass
class SegmentLayout:
    """Boundaries and segment training sets derived from one context."""

    T: int
    trials: list[Trial]
    edges: np.ndarray            # sorted unique boundaries, the horizon included
    lengths: np.ndarray          # observed length of each usable trial
    thetas: np.ndarray           # (n, d)

    @classmethod
    def from_context(cls, context: Sequence[Trial], T: int) -> SegmentLayout:
        usable = [c for c in context if not c.crashed and not c.unrun]
        lengths = np.array([c.t_obs for c in usable], dtype=np.int64)
        edges = np.unique(np.concatenate([lengths, [T]])).astype(np.int64)
        d = usable[0].theta.shape[0] if usable else 0
        thetas = (np.stack([np.asarray(c.theta, dtype=np.float64) for c in usable])
                  if usable else np.zeros((0, d)))
        return cls(T=T, trials=usable, edges=edges, lengths=lengths, thetas=thetas)

    def segments_after(self, t_obs: int) -> list[Segment]:
        """Segments covering ``(t_obs, T]``, with ``t_obs`` as the leading boundary."""
        above = self.edges[self.edges > t_obs]
        if above.size == 0:
            return []
        bounds = np.concatenate([[t_obs], above])
        return [(int(a), int(b)) for a, b in itertools.pairwise(bounds)]

    def data(self, seg: Segment) -> tuple[np.ndarray, np.ndarray]:
        """``(thetas, segment costs)`` of the usable trials that observed ``seg`` completely."""
        a, b = seg
        idx = np.flatnonzero(self.lengths >= b)
        y = np.array([segment_cost(self.trials[i], a, b) for i in idx], dtype=np.float64)
        return self.thetas[idx], y


class _ConstGP:
    """Segment model for fewer than two rows or a constant target: mean and spread of the data."""

    def __init__(self, y: np.ndarray) -> None:
        self.mu = float(np.mean(y))
        self.var = float(np.var(y)) if len(y) > 1 else 0.0

    def predict(self, X: np.ndarray, with_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        n = np.atleast_2d(X).shape[0]
        return np.full(n, self.mu), np.full(n, self.var)


def same_trial(a: Trial, b: Trial) -> bool:
    """True if both refer to the same trajectory observed to the same step."""
    return a is b or (a.traj is b.traj and a.t_obs == b.t_obs)


@dataclass
class SegmentGP(PredictorBase):
    """ESBO-GP Segment-GP predictor (id ``segment-gp``)."""

    spec: ProblemSpec
    iters: int = DEFAULT_ITERS
    lr: float = DEFAULT_LR
    cold_refit_growth: float = 0.0
    name: str = "segment-gp"
    supports_unrun: bool = True
    context_aware: bool = True
    _chains: dict[Segment, ExactGP] = field(default_factory=dict, repr=False)

    def reset(self) -> None:
        """Forget the hyperparameter chains (the next fit of every segment is cold)."""
        self._chains.clear()

    def predictable_steps(self, spec: ProblemSpec) -> np.ndarray:
        """Every cutoff before the horizon."""
        return np.arange(1, spec.T, dtype=np.int64)

    def _fit_segment(self, layout: SegmentLayout, seg: Segment) -> Any:
        X, y = layout.data(seg)
        if y.size == 0:
            return None
        if y.size < 2 or np.allclose(y, y[0]):
            return _ConstGP(y)
        gp = self._chains.get(seg)
        if gp is None:
            gp = ExactGP(self.spec.theta_lo, self.spec.theta_hi, iters=self.iters, lr=self.lr,
                         cold_refit_growth=self.cold_refit_growth)
            self._chains[seg] = gp
        return gp.fit(X, y)

    def predict(self, query: Trial, context: Sequence[Trial]) -> GaussianPrediction:
        return self.predict_many([query], context)[0]

    def predict_many(
        self, queries: Sequence[Trial], context: Sequence[Trial]
    ) -> list[GaussianPrediction]:
        """Predict every query from one set of segment fits (shared segments are fit once)."""
        layout = SegmentLayout.from_context(context, self.spec.T)
        fitted: dict[Segment, Any] = {}

        def model(seg: Segment) -> Any:
            if seg not in fitted:
                fitted[seg] = self._fit_segment(layout, seg)
            return fitted[seg]

        return [self._predict_one(q, layout, model) for q in queries]

    def predict_finals(
        self, queries: Sequence[Trial], context: Sequence[Trial]
    ) -> list[GaussianPrediction]:
        """Batch path of the ESBO loop, identical to :meth:`predict_many`."""
        return self.predict_many(queries, context)

    def _predict_one(
        self, query: Trial, layout: SegmentLayout, model: Callable[[Segment], Any]
    ) -> GaussianPrediction:
        if query.crashed:
            return GaussianPrediction(0.0, 0.0, p_crash=1.0)
        t_obs = int(query.t_obs)
        mean, var = float(query.J_obs), 0.0
        xq = np.asarray(query.theta, dtype=np.float64)[None, :]
        for seg in layout.segments_after(t_obs):
            m = model(seg)
            if m is None:
                continue
            mu, v = m.predict(xq, with_noise=True)
            mean += max(float(mu[0]), 0.0)
            var += float(v[0])
        return GaussianPrediction(mean, var, p_crash=0.0)


def make_predictor(args: tuple[str, ...], spec: ProblemSpec, **kwargs: Any) -> SegmentGP:
    """Registry factory for ``segment-gp`` (no further id components)."""
    if args:
        raise ValueError(f"segment-gp takes no id arguments, got {'/'.join(args)}")
    return SegmentGP(spec, **kwargs)
