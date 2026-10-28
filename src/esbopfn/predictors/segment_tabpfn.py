"""Segment-TabPFN (thesis method, Segment-TabPFN): TabPFN in place of the Segment-GPs.

The segment layout is that of :mod:`esbopfn.predictors.segment_gp`. Two modes:

- ``split``: one TabPFN per segment ``q`` with context ``D^q_n = {(theta_k, J_{T~_q, T~_q+1})}``,
  queried at ``theta_i``.
- ``rate``: one shared TabPFN over all segments with rows ``[theta_k, q]`` and target the segment
  cost divided by the segment length. ``q`` is the index of the segment's end in the boundary list,
  so a leading segment ``(t_obs, T~_q+1]`` shares the index of the segment it is a tail of. The
  predicted rate is scaled back by the segment length.

Each segment's bar distribution is reduced to its mean (clamped at ``>= 0``) and variance, and
the total is the Gaussian ``J_{t_obs} + sum mean_q`` with variance ``sum var_q`` (moment matching
over independent segments). A segment table with fewer than two rows or a constant target uses
the mean and spread of its data. ``p_crash`` is 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.core import ProblemSpec, Trial
from esbopfn.predictors.base import GaussianPrediction, PredictorBase
from esbopfn.predictors.checkpoints import tabpfn_checkpoint
from esbopfn.predictors.segment_gp import Segment, SegmentLayout
from esbopfn.predictors.tabpfn import (
    N_ESTIMATORS,
    degenerate_table,
    fit_predict_bars,
    load_regressor,
)

MODES = ("split", "rate")


def _moments_fit(reg: Any, X: np.ndarray, y: np.ndarray, Xq: np.ndarray) -> list[tuple[float, float]]:
    """Mean and variance per query row, from TabPFN or from the data when it is degenerate."""
    if len(Xq) == 0:
        return []
    if degenerate_table(y):
        mu = float(np.mean(y)) if y.size else 0.0
        var = float(np.var(y)) if y.size > 1 else 0.0
        return [(mu, var)] * len(Xq)
    return [bar.mean_var() for bar in fit_predict_bars(reg, X, y, Xq)]


@dataclass
class SegmentTabPFN(PredictorBase):
    """Segment-TabPFN (id ``segment-tabpfn/<split|rate>``)."""

    spec: ProblemSpec
    mode: str = "split"
    n_estimators: int = N_ESTIMATORS
    device: str | None = None
    checkpoint: Path | str | None = None
    supports_unrun: bool = True
    context_aware: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        self.name = f"segment-tabpfn/{self.mode}"
        if self.checkpoint is None:
            tabpfn_checkpoint()

    @property
    def regressor(self) -> Any:
        return load_regressor(self.n_estimators, self.device, self.checkpoint)

    def predictable_steps(self, spec: ProblemSpec) -> np.ndarray:
        return np.arange(1, spec.T, dtype=np.int64)

    def predict(self, query: Trial, context: Sequence[Trial]) -> GaussianPrediction:
        return self.predict_many([query], context)[0]

    def predict_many(
        self, queries: Sequence[Trial], context: Sequence[Trial]
    ) -> list[GaussianPrediction]:
        layout = SegmentLayout.from_context(context, self.spec.T)
        needs: list[list[Segment]] = []
        for q in queries:
            segs = [] if q.crashed else layout.segments_after(int(q.t_obs))
            needs.append([s for s in segs if layout.data(s)[1].size > 0])
        moments = self._split(layout, queries, needs) if self.mode == "split" else \
            self._rate(layout, queries, needs)
        out: list[GaussianPrediction] = []
        for i, q in enumerate(queries):
            if q.crashed:
                out.append(GaussianPrediction(0.0, 0.0, p_crash=1.0))
                continue
            mean, var = float(q.J_obs), 0.0
            for seg in needs[i]:
                mu, v = moments[(i, seg)]
                mean += max(mu, 0.0)
                var += v
            out.append(GaussianPrediction(mean, var, p_crash=0.0))
        return out

    def _split(self, layout: SegmentLayout, queries: Sequence[Trial],
               needs: list[list[Segment]]) -> dict[tuple[int, Segment], tuple[float, float]]:
        moments: dict[tuple[int, Segment], tuple[float, float]] = {}
        for seg in sorted({s for segs in needs for s in segs}):
            X, y = layout.data(seg)
            who = [i for i, segs in enumerate(needs) if seg in segs]
            Xq = np.array([np.asarray(queries[i].theta, dtype=np.float64) for i in who])
            for i, mv in zip(who, _moments_fit(self.regressor, X, y, Xq)):
                moments[(i, seg)] = mv
        return moments

    def _rate(self, layout: SegmentLayout, queries: Sequence[Trial],
              needs: list[list[Segment]]) -> dict[tuple[int, Segment], tuple[float, float]]:
        def index(seg: Segment) -> float:
            return float(np.searchsorted(layout.edges, seg[1]))

        rows_X: list[np.ndarray] = []
        rows_y: list[np.ndarray] = []
        for seg in sorted({s for segs in needs for s in segs}):
            X, y = layout.data(seg)
            rows_X.append(np.column_stack([X, np.full(len(y), index(seg))]))
            rows_y.append(y / (seg[1] - seg[0]))
        pairs = [(i, seg) for i, segs in enumerate(needs) for seg in segs]
        if not pairs:
            return {}
        Xq = np.array([[*map(float, queries[i].theta), index(seg)] for i, seg in pairs])
        fitted = _moments_fit(self.regressor, np.concatenate(rows_X), np.concatenate(rows_y), Xq)
        moments = {}
        for (i, seg), (mu, var) in zip(pairs, fitted):
            width = float(seg[1] - seg[0])
            moments[(i, seg)] = (mu * width, var * width * width)
        return moments


def make_predictor(args: tuple[str, ...], spec: ProblemSpec, **kwargs: Any) -> SegmentTabPFN:
    """Registry factory for ``segment-tabpfn/<split|rate>``."""
    if len(args) != 1 or args[0] not in MODES:
        raise ValueError(f"expected segment-tabpfn/<split|rate>, got {'/'.join(args)!r}")
    return SegmentTabPFN(spec, mode=args[0], **kwargs)
