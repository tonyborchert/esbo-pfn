"""Prediction types, the conditioned log-likelihood and the predictor registry."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from esbopfn.core import ProblemSpec
from esbopfn.predictors import base
from esbopfn.predictors.base import (
    BarPrediction,
    GaussianPrediction,
    PredictorBase,
    conditioned_ll_details,
    conditioned_log_likelihood,
    get_predictor,
    predictor_families,
    register_predictor_family,
)


def _bar(p_crash: float = 0.2) -> BarPrediction:
    edges = np.array([0.0, 1.0, 3.0, 4.0, 8.0])
    probs = np.array([0.1, 0.4, 0.3, 0.2]) * (1 - p_crash)
    return BarPrediction(edges, probs, p_crash)


def _integrate(pred, lo, hi, n=400_001):
    y = np.linspace(lo, hi, n)
    return np.trapezoid(np.exp(pred.log_density(y)), y)


SPEC = ProblemSpec("toy", 2, np.zeros(2), np.ones(2), 100, ("x",), (), True, False)


# --------------------------------------------------------------------------- bars
def test_bar_density_integrates_to_value_mass():
    pred = _bar(0.2)
    assert _integrate(pred, -1, 9) == pytest.approx(0.8, abs=1e-3)
    assert pred.log_prob_crash() == pytest.approx(math.log(0.2))
    assert pred.log_density(math.inf) == -math.inf
    assert pred.log_density(-0.5) == -math.inf
    # (a, b] convention, bottom edge belongs to the first bar
    assert pred.log_density(1.0) == pytest.approx(math.log(0.08))
    assert pred.log_density(0.0) == pytest.approx(math.log(0.08))


def test_bar_cdf_survival_quantile():
    pred = _bar(0.2)
    assert pred.cdf(-1.0) == 0.0
    assert pred.cdf(2.0) == pytest.approx(0.8 * (0.1 + 0.2))
    assert pred.cdf(8.0) == pytest.approx(0.8)
    assert pred.cdf(math.inf) == 1.0
    assert pred.survival(2.0) == pytest.approx(1 - pred.cdf(2.0))
    assert pred.survival(100.0) == pytest.approx(0.2)
    assert pred.quantile(0.9) == math.inf
    assert pred.quantile(0.8 * 0.1) == pytest.approx(1.0)
    q = pred.quantile(np.array([0.1, 0.5]))
    assert np.allclose(pred.cdf(q), [0.1, 0.5])


def test_bar_conditioned_above_renormalizes_and_keeps_crash():
    pred = _bar(0.2)
    cond = pred.conditioned_above(2.0)
    assert cond.p_crash == pytest.approx(0.2)
    assert cond.probs.sum() == pytest.approx(0.8)
    assert cond.edges[0] == 2.0
    assert _integrate(cond, 2.0, 8.0) == pytest.approx(0.8, abs=1e-3)
    surv = pred.value_survival(2.0)
    assert surv == pytest.approx((0.2 + 0.3 + 0.2) / 1.0)
    assert cond.log_density(3.5) == pytest.approx(pred.log_density(3.5) - math.log(surv))
    assert cond.log_density(1.5) == -math.inf
    # on an edge and below the support
    assert pred.conditioned_above(3.0).edges[0] == 3.0
    assert np.allclose(pred.conditioned_above(-5.0).probs, pred.probs)


def test_bar_truncate_below_conditions_mixed_measure():
    pred = _bar(0.2)
    trunc = pred.truncate_below(4.0)
    s = 0.8 * 0.2
    assert trunc.p_crash == pytest.approx(0.2 / (0.2 + s))
    assert trunc.probs.sum() == pytest.approx(s / (0.2 + s))


def test_bar_sample_matches_moments():
    pred = _bar(0.2)
    rng = np.random.default_rng(0)
    draws = pred.sample(rng, 400_000)
    crash = np.isinf(draws)
    assert crash.mean() == pytest.approx(0.2, abs=3e-3)
    mean, var = pred.mean_var()
    assert draws[~crash].mean() == pytest.approx(mean, rel=5e-3)
    assert draws[~crash].var() == pytest.approx(var, rel=1e-2)
    cond = pred.conditioned_above(2.5)
    d2 = cond.sample(rng, 100_000)
    assert np.all(d2[np.isfinite(d2)] >= 2.5)


def test_bar_validation():
    with pytest.raises(ValueError):
        BarPrediction(np.array([0.0, 1.0]), np.array([0.5]), 0.2)
    with pytest.raises(ValueError):
        BarPrediction(np.array([0.0, 0.0, 1.0]), np.array([0.5, 0.5]), 0.0)


# ------------------------------------------------------------------------- Gaussian
def test_gaussian_density_and_conditioning():
    g = GaussianPrediction(2.0, 4.0, 0.25)
    assert _integrate(g, -30, 30) == pytest.approx(0.75, abs=1e-4)
    assert g.log_density(3.0) == pytest.approx(math.log(0.75) + norm.logpdf(3.0, 2.0, 2.0))
    c = g.conditioned_above(3.0)
    assert c.p_crash == 0.25
    assert _integrate(c, 3.0, 40) == pytest.approx(0.75, abs=1e-4)
    assert c.log_density(2.0) == -math.inf
    assert c.cdf(3.0) == pytest.approx(0.0)
    assert c.survival(3.0) == pytest.approx(1.0)
    assert g.cdf(2.0) == pytest.approx(0.75 * 0.5)
    t = g.truncate_below(3.0)
    s = 0.75 * norm.sf(3.0, 2.0, 2.0)
    assert t.p_crash == pytest.approx(0.25 / (0.25 + s))


def test_gaussian_sample_matches_moments():
    rng = np.random.default_rng(1)
    for g in (GaussianPrediction(2.0, 4.0, 0.1), GaussianPrediction(2.0, 4.0, 0.1, lower=3.0)):
        d = g.sample(rng, 300_000)
        crash = np.isinf(d)
        assert crash.mean() == pytest.approx(0.1, abs=3e-3)
        m, v = g.mean_var()
        assert d[~crash].mean() == pytest.approx(m, rel=5e-3)
        assert d[~crash].var() == pytest.approx(v, rel=2e-2)
        assert np.all(d[~crash] >= g.lower)
    q = GaussianPrediction(0.0, 1.0, 0.5).quantile(np.array([0.25, 0.6]))
    assert q[0] == pytest.approx(0.0, abs=1e-12) and q[1] == math.inf


def test_gaussian_degenerate_point_mass():
    g = GaussianPrediction(5.0, 0.0)
    assert g.mean_var() == (5.0, 0.0)
    assert g.log_density(4.0) == -math.inf
    assert np.all(g.sample(np.random.default_rng(0), 10) == 5.0)


# ------------------------------------------------------------------ conditioned LL
def test_conditioned_ll_hand_computed_gaussian():
    g = GaussianPrediction(10.0, 4.0, 0.1)
    y, j = 11.0, 9.0
    expected = math.log(0.9) + norm.logpdf(y, 10, 2) - norm.logsf(j, 10, 2)
    assert conditioned_log_likelihood(g, y, j) == pytest.approx(expected)
    assert conditioned_log_likelihood(g, y, j) == pytest.approx(
        g.conditioned_above(j).log_density(y))
    assert conditioned_log_likelihood(g, math.inf, j) == pytest.approx(math.log(0.1))
    assert conditioned_log_likelihood(g, y) == pytest.approx(math.log(0.9) + norm.logpdf(y, 10, 2))


def test_conditioned_ll_hand_computed_bar():
    pred = _bar(0.2)
    # y = 3.5 in bar (3, 4] of mass 0.3 * 0.8 and width 1, survival above 2 is 0.7
    expected = math.log(0.8) + math.log(0.3) - math.log(0.7)
    assert conditioned_log_likelihood(pred, 3.5, 2.0) == pytest.approx(expected)
    assert conditioned_log_likelihood(pred, 3.5, 2.0) == pytest.approx(
        pred.conditioned_above(2.0).log_density(3.5))


def test_conditioned_ll_floors():
    floor = 1e-10
    lf = math.log(floor)
    pred = _bar(0.0)
    d = conditioned_ll_details(pred, 100.0, None, floor)       # outside the support
    assert d.clamped_value and d.ll == pytest.approx(lf)
    d = conditioned_ll_details(pred, math.inf, None, floor)    # crash with p_crash 0
    assert d.crashed and d.clamped_crash and d.ll == pytest.approx(lf)
    d = conditioned_ll_details(_bar(1.0), 2.0, None, floor)     # value with p_crash 1
    assert d.ll == pytest.approx(2 * lf)
    g = GaussianPrediction(0.0, 1.0)
    d = conditioned_ll_details(g, 50.0, 49.0, floor)           # no surviving mass: skip
    assert d.conditioning_skipped
    assert d.ll == pytest.approx(lf)


# ------------------------------------------------------------------------ registry
class _Dummy(PredictorBase):
    name = "dummy"

    def __init__(self, args, spec, scale=1.0):
        self.args, self.spec, self.scale = args, spec, scale

    def predict(self, query, context):
        return GaussianPrediction(self.scale, 1.0)


def test_registry_register_and_resolve(monkeypatch):
    monkeypatch.setattr(base, "_FAMILIES", {})
    monkeypatch.setattr(base, "LAZY_FAMILIES", dict(base.LAZY_FAMILIES))
    register_predictor_family("dummy", _Dummy)
    with pytest.raises(ValueError):
        register_predictor_family("dummy", _Dummy)
    p = get_predictor("dummy/a/b", SPEC, scale=3.0)
    assert p.args == ("a", "b") and p.spec is SPEC and p.scale == 3.0
    assert isinstance(p, base.Predictor)
    assert p.predict_many([None, None], []) == [GaussianPrediction(3.0, 1.0)] * 2
    assert {"dummy", "segment-gp", "tabpfn", "segment-tabpfn"} <= set(predictor_families())
    register_predictor_family("lazydummy", f"{__name__}:_Dummy")
    assert get_predictor("lazydummy/x", SPEC).args == ("x",)
    with pytest.raises(KeyError):
        get_predictor("nosuchfamily/1", SPEC)


def test_registry_builtin_segment_gp():
    p = get_predictor("segment-gp", SPEC, cold_refit_growth=0.05)
    assert p.name == "segment-gp" and p.cold_refit_growth == 0.05
    assert p.supports_unrun and p.context_aware
    with pytest.raises(ValueError):
        get_predictor("segment-gp/extra", SPEC)
