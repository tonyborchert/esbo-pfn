"""Pure model loops XDecPFN-{PI,EI} and FT-PFN-{EI,PI} (method, cross-trajectory ESBO, and setup,
implementation details).

No GP. The cross-trajectory model is the only surrogate:

- proposal: closed-form EI or PI (eq:method:pure-loop:ei, eq:method:pure-loop:pi) on the
  crash-aware bar prediction of every unrun candidate given the history, maximized over 400 uniform
  candidates and 1600 axis-aligned Gaussian perturbations of the five best complete controllers
  (standard deviation 0.10 times the box extent per dimension, clipped to the box),
- stopping: ``exceed`` (applied by the loop) and ``naive`` with the prediction conditioned on the
  history and the running prefix, evaluated at the decoder grid token steps for both models.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.special import ndtr

from esbopfn.core import ProblemSpec, Trial
from esbopfn.grids import decoder_grid
from esbopfn.predictors.base import BarPrediction, GaussianPrediction, Prediction, get_predictor
from esbopfn.strategies import rules
from esbopfn.strategies.base import Strategy, sample_uniform

N_UNIFORM = 400
N_LOCAL = 1600
TOP_K = 5
LOCAL_SCALE = 0.10
ACQUISITIONS = ("ei", "pi")


# ------------------------------------------------------------------------------- acquisition
def _clamped(edges: np.ndarray, J_minus: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = edges[:-1], edges[1:]
    c = np.maximum(np.minimum(J_minus, hi), lo)
    return lo, hi, c


def bar_pi(pred: BarPrediction, J_minus: float) -> float:
    """``(1 - w_inf) sum_k w_k (c_k - b_{k-1}) / (b_k - b_{k-1})``. ``probs`` are the
    unconditional bar masses ``(1 - w_inf) w_k``."""
    lo, hi, c = _clamped(pred.edges, J_minus)
    return float(np.sum(pred.probs * (c - lo) / (hi - lo)))


def bar_ei(pred: BarPrediction, J_minus: float) -> float:
    """``(1 - w_inf) sum_k w_k (c_k - b_{k-1}) (J^- - (b_{k-1} + c_k) / 2) / (b_k - b_{k-1})``."""
    lo, hi, c = _clamped(pred.edges, J_minus)
    return float(np.sum(pred.probs * (c - lo) * (J_minus - 0.5 * (lo + c)) / (hi - lo)))


def gaussian_pi(pred: GaussianPrediction, J_minus: float) -> float:
    """PI of an untruncated crash-aware Gaussian, ``(1 - p_crash) Phi((J^- - mu) / sigma)``."""
    if pred.var <= 0:
        return pred.value_mass * float(pred.mean < J_minus)
    return pred.value_mass * float(ndtr((J_minus - pred.mean) / pred.std))


def gaussian_ei(pred: GaussianPrediction, J_minus: float) -> float:
    """EI of an untruncated crash-aware Gaussian."""
    if pred.var <= 0:
        return pred.value_mass * max(J_minus - pred.mean, 0.0)
    z = (J_minus - pred.mean) / pred.std
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return pred.value_mass * ((J_minus - pred.mean) * float(ndtr(z)) + pred.std * pdf)


def acquisition(preds: Sequence[Prediction], J_minus: float, acq: str) -> np.ndarray:
    """EI or PI of every prediction with respect to the incumbent cost ``J^-_n``."""
    if acq not in ACQUISITIONS:
        raise ValueError(f"acq must be one of {ACQUISITIONS}, got {acq!r}")
    out = np.empty(len(preds))
    for i, p in enumerate(preds):
        if isinstance(p, BarPrediction):
            out[i] = bar_ei(p, J_minus) if acq == "ei" else bar_pi(p, J_minus)
        else:
            out[i] = gaussian_ei(p, J_minus) if acq == "ei" else gaussian_pi(p, J_minus)
    return out


# -------------------------------------------------------------------------------- candidates
def candidate_set(spec: ProblemSpec, rng: np.random.Generator, best: np.ndarray, *,
                  n_uniform: int = N_UNIFORM, n_local: int = N_LOCAL,
                  scale: float = LOCAL_SCALE) -> np.ndarray:
    """``n_uniform`` uniform draws plus ``n_local`` Gaussian perturbations, each around one
    uniformly chosen row of ``best`` with standard deviation ``scale * extent`` per dimension,
    clipped to the box. Without ``best`` rows all candidates are uniform."""
    best = np.asarray(best, dtype=np.float64).reshape(-1, spec.d)
    if best.shape[0] == 0:
        return sample_uniform(spec, rng, n_uniform + n_local)
    uniform = sample_uniform(spec, rng, n_uniform)
    centers = best[rng.integers(0, best.shape[0], size=n_local)]
    local = centers + rng.standard_normal((n_local, spec.d)) * (scale * spec.extent)
    local = np.clip(local, spec.theta_lo, spec.theta_hi)
    return np.vstack([uniform, local])


def top_complete(history: Sequence[Trial], k: int = TOP_K) -> np.ndarray:
    """Controllers of the ``k`` complete trials with the lowest total cost, shape ``(<=k, d)``."""
    done = sorted((tr.J_obs, i) for i, tr in enumerate(history) if tr.complete)
    rows = [np.asarray(history[i].theta, dtype=np.float64) for _, i in done[:k]]
    d = history[0].theta.shape[0] if history else 0
    return np.array(rows) if rows else np.zeros((0, d))


# -------------------------------------------------------------------------------- strategy
class PureModelLoop(Strategy):
    """XDecPFN-PI/EI and FT-PFN-EI/PI. The predictor must offer ``predict(query, context)`` and
    ``predict_unrun(thetas, context) -> list[BarPrediction]``."""

    name = "pure-loop"

    def __init__(self, spec: ProblemSpec, rng: np.random.Generator, *, acq: str = "ei",
                 tau: float = 0.99, crash_threshold: float | None = None,
                 model: str | None = None, predictor: Any = None, n_uniform: int = N_UNIFORM,
                 n_local: int = N_LOCAL, top_k: int = TOP_K, local_scale: float = LOCAL_SCALE,
                 **params: Any) -> None:
        super().__init__(spec, rng, **params)
        acq = acq.lower()
        if acq not in ACQUISITIONS:
            raise ValueError(f"acq must be one of {ACQUISITIONS}, got {acq!r}")
        self.acq = acq
        self.tau = float(tau)
        self.crash_threshold = None if crash_threshold is None else float(crash_threshold)
        self.model = model
        self._predictor = predictor
        self.n_uniform, self.n_local = int(n_uniform), int(n_local)
        self.top_k, self.local_scale = int(top_k), float(local_scale)
        self._steps = decoder_grid(self.spec)
        self._steps = self._steps[self._steps < self.spec.T]
        self._set_params(acq=acq, tau=self.tau, crash_threshold=self.crash_threshold, model=model,
                         n_uniform=self.n_uniform, n_local=self.n_local, top_k=self.top_k,
                         local_scale=self.local_scale)

    @property
    def predictor(self) -> Any:
        if self._predictor is None:
            if self.model is None:
                raise ValueError("no model id and no predictor given")
            self._predictor = get_predictor(self.model, self.spec)
        return self._predictor

    def decision_steps(self) -> np.ndarray:
        return self._steps

    def propose(self, history: list[Trial]) -> np.ndarray:
        _, J_minus = self.incumbent(history)
        with self.timed("model_acq"):
            cands = candidate_set(self.spec, self.rng, top_complete(history, self.top_k),
                                  n_uniform=self.n_uniform, n_local=self.n_local,
                                  scale=self.local_scale)
            preds = self.predictor.predict_unrun(cands, history)
            scores = acquisition(preds, J_minus, self.acq)
        return cands[int(np.argmax(scores))]

    def should_stop(self, running: Trial, history: list[Trial]) -> bool:
        if not self.armed:
            return False
        with self.timed("model_forecast"):
            pred = self.predictor.predict(running, history)
        return rules.naive(pred, self.J_minus, self.tau, self.crash_threshold)
