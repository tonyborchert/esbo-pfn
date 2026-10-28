"""Early stopping rules as pure functions (thesis background and method, single-trajectory ESBO).

- ``exceed`` (Stenger et al.): stop as soon as ``J_t >= J^-_n``.
- ``naive``: stop when ``Pr[J~ >= J^-_n] > tau`` under the running trial's prediction. A crash is
  ``J~ = inf`` and counts as ``>=``.
- ``matched``: stop when ``Pr[J~(theta_{n+1}) >= J~(theta^-_n) | J^-_n <= J~(theta^-_n)] > tau``
  with both predictions made at the same cutoff from their own prefixes. The incumbent's prediction
  is conditioned on ``J >= J^-_n`` as a whole mixture (:meth:`truncate_below`, the crash atom lies
  in that event), exactly the conditioning event of the thesis equation.

``crash_threshold`` (``None`` by default) adds the appendix variant that also stops when the
running trial's predicted crash probability exceeds it.
"""

from __future__ import annotations

import math

import numpy as np

from esbopfn.predictors.base import BarPrediction, GaussianPrediction, Prediction

#: Bars used to discretize a Gaussian prediction for the matched comparison.
GAUSSIAN_BARS = 512
#: Standard deviations covered by that discretization.
GAUSSIAN_SPAN = 8.0


def exceed(J_t: float, J_minus: float) -> bool:
    """``J_t >= J^-_n`` (never fires without an incumbent, ``J^-_n = inf``)."""
    return math.isfinite(J_minus) and float(J_t) >= J_minus


def _crash_stop(pred: Prediction, crash_threshold: float | None) -> bool:
    return crash_threshold is not None and pred.p_crash > crash_threshold


def prob_at_least(pred: Prediction, j: float) -> float:
    """``Pr[J~ >= j]`` with the crash mass counted as ``>=``."""
    return float(pred.survival(float(j)))


def naive(pred: Prediction, J_minus: float, tau: float,
          crash_threshold: float | None = None) -> bool:
    """The ``naive`` rule ``Pr[J~ >= J^-_n] > tau`` (plus the optional crash threshold)."""
    if _crash_stop(pred, crash_threshold):
        return True
    if not math.isfinite(J_minus):
        return False
    return prob_at_least(pred, J_minus) > tau


def as_bars(pred: Prediction) -> BarPrediction:
    """A bar view of a prediction. Gaussians are discretized over ``mean +- 8 sd`` (truncation
    respected), the crash mass is kept."""
    if isinstance(pred, BarPrediction):
        return pred
    if not isinstance(pred, GaussianPrediction):
        raise TypeError(f"unsupported prediction type {type(pred).__name__}")
    mean, sd = pred.mean, pred.std
    lo = max(mean - GAUSSIAN_SPAN * sd, pred.lower) if math.isfinite(pred.lower) else (
        mean - GAUSSIAN_SPAN * sd)
    hi = max(mean, lo) + GAUSSIAN_SPAN * sd
    if sd <= 0 or hi <= lo:
        point = max(mean, pred.lower)
        width = max(abs(point) * 1e-9, 1e-12)
        return BarPrediction(np.array([point, point + width]), np.array([pred.value_mass]),
                             pred.p_crash)
    edges = np.linspace(lo, hi, GAUSSIAN_BARS + 1)
    cdf = np.asarray(pred.cdf(edges), dtype=np.float64)
    probs = np.clip(np.diff(cdf), 0.0, None)
    total = probs.sum()
    if total > 0:
        probs = probs * (pred.value_mass / total)
    else:
        probs = np.full(GAUSSIAN_BARS, pred.value_mass / GAUSSIAN_BARS)
    return BarPrediction(edges, probs, pred.p_crash)


def prob_geq(a: Prediction, b: Prediction) -> float:
    """``Pr[A >= B]`` for independent crash-aware predictions ``A`` and ``B``.

    ``A = inf`` always counts as ``>=`` (``inf >= inf`` included), ``B = inf`` with finite ``A``
    never does. The value parts are piecewise uniform, so the probability is exact on the union of
    both edge sets: inside one shared interval two independent uniforms tie with probability one
    half.
    """
    A, B = as_bars(a), as_bars(b)
    p = A.p_crash
    if A.value_mass <= 0 or B.value_mass <= 0:
        return min(1.0, p)
    grid = np.union1d(A.edges, B.edges)
    ma = np.diff(np.asarray(A.cdf(grid), dtype=np.float64))   # unconditional masses per interval
    mb = np.diff(np.asarray(B.cdf(grid), dtype=np.float64))
    below_b = np.concatenate([[0.0], np.cumsum(mb)[:-1]])      # B mass strictly below interval
    value = float(np.sum(ma * below_b) + 0.5 * np.sum(ma * mb))
    return float(np.clip(p + value, 0.0, 1.0))


def matched(pred_running: Prediction, pred_incumbent: Prediction, J_minus: float, tau: float,
            crash_threshold: float | None = None) -> bool:
    """The ``matched`` rule. ``pred_incumbent`` is the incumbent's prediction at the same cutoff,
    not yet conditioned. It is conditioned here on ``J >= J^-_n``."""
    if _crash_stop(pred_running, crash_threshold):
        return True
    if not math.isfinite(J_minus):
        return False
    reference = pred_incumbent.truncate_below(J_minus)
    return prob_geq(pred_running, reference) > tau
