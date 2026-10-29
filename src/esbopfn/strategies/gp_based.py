"""Strategies built on the objective GP of ESBO-GP, and ESRS.

Every GP strategy follows Algorithm alg:ct:esbo (thesis background) in :meth:`GPStrategy.propose`:

1. virtual observations ``E_n`` for the early-stopped trials, resampled every iteration from the
   strategy's predictor conditioned on ``J >= J_{T_i}`` (``inf`` draws go to the crash rule),
2. the objective GP fit on the complete trials ``O_n`` and the crash virtual objective
   ``J_virt = min{max{mu_O, J^-} + 3 sigma_O, J^+}`` for every crashed trial (set ``F_n``),
3. the objective GP fit on ``O_n u E_n u F_n`` and max-value entropy search over 2000 uniform
   random candidates.

The two objective GPs are separate :class:`ExactGP` chains (``cold_refit_growth`` acts on each).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from esbopfn.core import ProblemSpec, Trial
from esbopfn.predictors.base import GaussianPrediction, Prediction, get_predictor
from esbopfn.predictors.gp import ExactGP, mes_acquisition, sample_positive
from esbopfn.predictors.segment_gp import SegmentGP
from esbopfn.strategies import rules
from esbopfn.strategies.base import Strategy, incumbent_index, j_plus, sample_uniform

#: Size of the random candidate set of the GP acquisition (setup, ESRS/BO/Oracle paragraph).
N_CANDIDATES = 2000
#: Default ``cold_refit_growth`` of the optimization runs (docs/PREDICTORS.md).
COLD_REFIT_GROWTH = 0.05


def crash_virtual_objective(mu: np.ndarray, sigma: np.ndarray, J_minus: float,
                            J_plus: float) -> np.ndarray:
    """eq:esbo:crashed-virtual-objective, ``min{max{mu, J^-} + 3 sigma, J^+}``."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    return np.minimum(np.maximum(mu, J_minus) + 3.0 * sigma, J_plus)


def sample_virtual(pred: Prediction, j_obs: float, rng: np.random.Generator) -> float:
    """One draw of ``J~`` conditioned on ``J~ >= j_obs`` (crash mass kept, ``inf`` = crash)."""
    if isinstance(pred, GaussianPrediction) and pred.p_crash == 0.0:
        return float(sample_positive(pred.mean, math.sqrt(pred.var), max(pred.lower, j_obs), rng))
    return float(pred.conditioned_above(j_obs).sample(rng, 1)[0])


class ESRS(Strategy):
    """``ESRS``: theta ~ U(Theta) with the ``exceed`` rule (setup, baselines)."""

    name = "esrs"

    def propose(self, history: list[Trial]) -> np.ndarray:
        return sample_uniform(self.spec, self.rng)


class GPStrategy(Strategy):
    """Objective GP + MES over random candidates, virtual observations from
    :meth:`virtual_early`."""

    def __init__(self, spec: ProblemSpec, rng: np.random.Generator, *,
                 cold_refit_growth: float = COLD_REFIT_GROWTH, n_candidates: int = N_CANDIDATES,
                 gp_iters: int = 100, **params: Any) -> None:
        super().__init__(spec, rng, **params)
        self.cold_refit_growth = float(cold_refit_growth)
        self.n_candidates = int(n_candidates)
        self.gp_iters = int(gp_iters)
        self.gp_complete = self._new_gp()
        self.gp_virtual = self._new_gp()
        self._set_params(cold_refit_growth=self.cold_refit_growth, n_candidates=self.n_candidates,
                         gp_iters=self.gp_iters)
        self.last_virtual: dict[int, float] = {}
        self._segments: SegmentGP | None = None

    def _new_gp(self) -> ExactGP:
        return ExactGP(self.spec.theta_lo, self.spec.theta_hi, iters=self.gp_iters,
                       cold_refit_growth=self.cold_refit_growth)

    def _segment_gp(self) -> SegmentGP:
        if self._segments is None:
            self._segments = SegmentGP(self.spec, iters=self.gp_iters,
                                       cold_refit_growth=self.cold_refit_growth)
        return self._segments

    # -- virtual observations -----------------------------------------------------------------
    def virtual_early(self, early: list[tuple[int, Trial]], history: list[Trial]) -> np.ndarray:
        """Virtual totals of the early-stopped trials, ``inf`` where a crash was sampled."""
        if early:
            raise NotImplementedError(f"{self.name} has no virtual observations")
        return np.zeros(0)

    def _segment_gp_virtual(self, early: list[tuple[int, Trial]],
                            history: list[Trial]) -> np.ndarray:
        trials = [tr for _, tr in early]
        with self.timed("segment_gp"):
            preds = self._segment_gp().predict_many(trials, history)
        means = np.array([p.mean for p in preds])
        stds = np.sqrt(np.array([p.var for p in preds]))
        lower = np.array([tr.J_obs for tr in trials])
        return np.atleast_1d(sample_positive(means, stds, lower, self.rng))

    def virtual_dataset(self, history: list[Trial]) -> tuple[np.ndarray, np.ndarray]:
        """``D_virt = O u E u F`` as ``(X, y)``."""
        complete = [i for i, tr in enumerate(history) if tr.complete]
        early = [(i, tr) for i, tr in enumerate(history) if tr.early_stopped]
        crashed = [i for i, tr in enumerate(history) if tr.crashed]
        X_O = np.array([history[i].theta for i in complete], dtype=np.float64)
        y_O = np.array([history[i].J_obs for i in complete], dtype=np.float64)
        y_E = self.virtual_early(early, history) if early else np.zeros(0)
        idx = complete + [i for i, _ in early]
        y = np.concatenate([y_O, y_E])
        to_crash = crashed + [i for (i, _), v in zip(early, y_E) if not math.isfinite(v)]
        values = dict(zip(idx, y))
        if to_crash:
            _, J_minus = self.incumbent(history)
            with self.timed("gp_acq"):
                self.gp_complete.fit(X_O, y_O)
                X_c = np.array([history[i].theta for i in to_crash], dtype=np.float64)
                mu, var = self.gp_complete.predict(X_c, with_noise=True)
            jv = crash_virtual_objective(mu, np.sqrt(var), J_minus, j_plus(history))
            values.update(zip(to_crash, jv))
        order = sorted(values)
        self.last_virtual = {i: float(values[i]) for i in order if not history[i].complete}
        X = np.array([history[i].theta for i in order], dtype=np.float64)
        return X, np.array([values[i] for i in order], dtype=np.float64)

    def propose(self, history: list[Trial]) -> np.ndarray:
        X, y = self.virtual_dataset(history)
        _, J_minus = self.incumbent(history)
        with self.timed("gp_acq"):
            self.gp_virtual.fit(X, y)
            candidates = sample_uniform(self.spec, self.rng, self.n_candidates)
            scores = mes_acquisition(self.gp_virtual, candidates, self.rng, incumbent=J_minus)
        return candidates[int(np.argmax(scores))]


class BO(GPStrategy):
    """``BO``: the same BO procedure without early stopping (crashes get the crash rule)."""

    name = "bo"
    uses_exceed = False


class ESBOGP(GPStrategy):
    """``ESBO-GP``: Segment-GP virtual observations resampled every iteration, ``exceed`` rule."""

    name = "esbo-gp"

    def virtual_early(self, early: list[tuple[int, Trial]], history: list[Trial]) -> np.ndarray:
        return self._segment_gp_virtual(early, history)


class ESBOOracle(GPStrategy):
    """``ESBO-Oracle``: the virtual observation is the true total cost of the uncut trajectory
    (a point mass), ``inf`` when the uncut trajectory crashes (then the crash rule applies)."""

    name = "esbo-oracle"

    def __init__(self, spec: ProblemSpec, rng: np.random.Generator, **params: Any) -> None:
        super().__init__(spec, rng, **params)
        self.true_total: dict[int, float] = {}

    def on_trial_end(self, trial: Trial, index: int) -> None:
        self.true_total[index] = float(trial.traj.J_total)

    def virtual_early(self, early: list[tuple[int, Trial]], history: list[Trial]) -> np.ndarray:
        return np.array([self.true_total[i] for i, _ in early], dtype=np.float64)


class _PredictorMixin:
    """Lazy predictor resolution from ``model`` unless a ``predictor`` object was given."""

    spec: ProblemSpec
    model: str | None
    _predictor: Any

    @property
    def predictor(self) -> Any:
        if self._predictor is None:
            if self.model is None:
                raise ValueError("no model id and no predictor given")
            self._predictor = get_predictor(self.model, self.spec)
        return self._predictor


class ESBOTTabPFN(_PredictorMixin, GPStrategy):
    """``ESBO-T-TabPFN``: Trajectory-TabPFN (``tabpfn/<L>``) replaces the Segment-GPs."""

    name = "esbo-t-tabpfn"

    def __init__(self, spec: ProblemSpec, rng: np.random.Generator, *, L: int = 50,
                 model: str | None = None, predictor: Any = None, **params: Any) -> None:
        super().__init__(spec, rng, **params)
        self.L = int(L)
        self.model = model or f"tabpfn/{self.L}"
        self._predictor = predictor
        self._set_params(L=self.L, model=self.model)

    def virtual_early(self, early: list[tuple[int, Trial]], history: list[Trial]) -> np.ndarray:
        trials = [tr for _, tr in early]
        with self.timed("tabpfn_forecast"):
            preds = self.predictor.predict_many(trials, history)
        return np.array([sample_virtual(p, tr.J_obs, self.rng) for p, tr in zip(preds, trials)])


class DecPFN(_PredictorMixin, GPStrategy):
    """``DecPFN``: ``exceed`` + ``matched`` rule and virtual observations sampled from the static
    DecPFN horizon prediction of each early-stopped trial (method, single-trajectory ESBO).

    The stop rule is evaluated at every step of ``predictor.predictable_steps`` (the decoder grid).
    Predictions of the running trial are recorded at every such step, also during the initial
    design, and the curve of the current incumbent is kept for the matched comparison. The
    virtual observation of an early-stopped trial uses the prediction at its last grid token
    ``<= T_i``, conditioned on ``J >= J_{T_i}``. A trial stopped before the first grid token has no
    DecPFN prediction and gets its virtual observation from the Segment-GPs.
    """

    name = "decpfn"
    records_unarmed = True

    def __init__(self, spec: ProblemSpec, rng: np.random.Generator, *, tau: float = 0.9,
                 crash_threshold: float | None = None, model: str | None = None,
                 predictor: Any = None, **params: Any) -> None:
        super().__init__(spec, rng, **params)
        self.tau = float(tau)
        self.crash_threshold = None if crash_threshold is None else float(crash_threshold)
        self.model = model
        self._predictor = predictor
        self._set_params(tau=self.tau, crash_threshold=self.crash_threshold, model=self.model)
        self.curves: dict[int, dict[int, Prediction]] = {}
        self.final_pred: dict[int, Prediction | None] = {}
        self._incumbent: tuple[int, Trial] | None = None
        self._steps: np.ndarray | None = None

    def decision_steps(self) -> np.ndarray:
        if self._steps is None:
            steps = np.asarray(self.predictor.predictable_steps(self.spec), dtype=np.int64)
            self._steps = np.unique(steps[(steps >= 1) & (steps < self.spec.T)])
        return self._steps

    def _predict_prefix(self, trial: Trial) -> Prediction:
        with self.timed("model_forecast"):
            return self.predictor.predict(trial, [])

    def on_trial_start(self, history: list[Trial], armed: bool) -> None:
        super().on_trial_start(history, armed)
        i = incumbent_index(history)
        self._incumbent = None if i is None else (i, history[i])
        self.curves[self.running_index] = {}

    def incumbent_prediction(self, t: int) -> Prediction | None:
        """The incumbent's prediction at cutoff ``t`` (reused from its run, computed if absent)."""
        if self._incumbent is None:
            return None
        i, trial = self._incumbent
        curve = self.curves.setdefault(i, {})
        if t not in curve:
            curve[t] = self._predict_prefix(trial.traj.cut(t))
        return curve[t]

    def should_stop(self, running: Trial, history: list[Trial]) -> bool:
        t = int(running.t_obs)
        pred = self._predict_prefix(running)
        self.curves.setdefault(self.running_index, {})[t] = pred
        if not self.armed:
            return False
        ref = self.incumbent_prediction(t)
        if ref is None:
            return False
        return rules.matched(pred, ref, self.J_minus, self.tau, self.crash_threshold)

    def on_trial_end(self, trial: Trial, index: int) -> None:
        curve = self.curves.get(index, {})
        if trial.early_stopped:
            steps = self.decision_steps()
            below = steps[steps <= trial.t_obs]
            if below.size == 0:
                self.final_pred[index] = None
            else:
                g = int(below[-1])
                pred = curve.get(g)
                self.final_pred[index] = (pred if pred is not None
                                          else self._predict_prefix(trial.traj.cut(g)))
        # keep only the curves that can still serve as the matched reference
        keep = {index} if trial.successful and trial.J_obs < self.J_minus else set()
        if not keep and self._incumbent is not None:
            keep = {self._incumbent[0]}
        self.curves = {i: c for i, c in self.curves.items() if i in keep}

    def _decpfn_virtual(self, i: int, trial: Trial) -> float | None:
        pred = self.final_pred.get(i)
        if pred is None:
            return None
        return sample_virtual(pred, trial.J_obs, self.rng)

    def virtual_early(self, early: list[tuple[int, Trial]], history: list[Trial]) -> np.ndarray:
        out = np.empty(len(early))
        missing: list[int] = []
        for k, (i, tr) in enumerate(early):
            v = self._decpfn_virtual(i, tr)
            if v is None:
                missing.append(k)
            else:
                out[k] = v
        if missing:
            out[missing] = self._segment_gp_virtual([early[k] for k in missing], history)
        return out


class DecPFNH(DecPFN):
    """``DecPFN-H``: the ``hybrid`` virtual observations. Every iteration and for every
    early-stopped trial, the updated Segment-GP prediction and the static DecPFN prediction
    (conditioned on ``J != inf``) are compared, and the one with the lower variance is sampled.
    A Segment-GP prediction with zero variance (no segment GP could be fit) is not used."""

    name = "decpfn-h"

    def __init__(self, spec: ProblemSpec, rng: np.random.Generator, **params: Any) -> None:
        super().__init__(spec, rng, **params)
        self.last_sources: dict[int, str] = {}

    @staticmethod
    def choose(seg: GaussianPrediction, dec: Prediction | None) -> str:
        """``"segment-gp"`` or ``"decpfn"``, whichever has the lower variance."""
        seg_var = seg.var
        if dec is None:
            return "segment-gp"
        _, dec_var = dec.mean_var()  # value part, i.e. conditioned on J != inf
        if not (math.isfinite(seg_var) and seg_var > 0.0):
            return "decpfn"
        if not math.isfinite(dec_var):
            return "segment-gp"
        return "segment-gp" if seg_var < dec_var else "decpfn"

    def virtual_early(self, early: list[tuple[int, Trial]], history: list[Trial]) -> np.ndarray:
        trials = [tr for _, tr in early]
        with self.timed("segment_gp"):
            segs = self._segment_gp().predict_many(trials, history)
        out = np.empty(len(early))
        for k, ((i, tr), seg) in enumerate(zip(early, segs)):
            dec = self.final_pred.get(i)
            source = self.choose(seg, dec)
            self.last_sources[i] = source
            pred = seg if source == "segment-gp" else dec
            out[k] = sample_virtual(pred, tr.J_obs, self.rng)
        return out

