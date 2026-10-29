"""Exact GP port, MES acquisition, truncated sampling and the Segment-GP predictor."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import truncnorm

from esbopfn.core import ProblemSpec, Trajectory
from esbopfn.predictors.base import GaussianPrediction
from esbopfn.predictors.gp import (
    DEFAULT_LENGTHSCALE_BOUNDS,
    DEFAULT_NOISE_BOUNDS,
    DEFAULT_OUTPUTSCALE_BOUNDS,
    ExactGP,
    mes_acquisition,
    min_value_entropy,
    sample_positive,
)
from esbopfn.predictors.segment_gp import SegmentGP, SegmentLayout


def _sine(n=40, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 1))
    y = np.sin(2 * np.pi * X[:, 0]) + noise * rng.standard_normal(n)
    return X, y


# ------------------------------------------------------------------------------- GP
def test_gp_fits_noisy_sine():
    X, y = _sine()
    gp = ExactGP(np.zeros(1), np.ones(1)).fit(X, y)
    Xq = np.linspace(0.05, 0.95, 50)[:, None]
    mean, var = gp.predict(Xq)
    assert np.max(np.abs(mean - np.sin(2 * np.pi * Xq[:, 0]))) < 0.15
    assert np.all(var >= 0) and np.all(np.sqrt(var) < 0.2)
    _, var_n = gp.predict(Xq, with_noise=True)
    assert np.all(var_n > var)
    assert gp.noise_variance == pytest.approx(math.exp(2 * gp.log_noise) * gp.y_std**2)


def test_gp_is_deterministic_and_scales_inputs():
    X, y = _sine(seed=3)
    a = ExactGP(np.zeros(1), np.ones(1)).fit(X, y).predict(X[:5])
    b = ExactGP(np.zeros(1), np.ones(1)).fit(X, y).predict(X[:5])
    assert np.array_equal(a[0], b[0])
    # the same data in a scaled box gives the same posterior
    c = ExactGP(np.array([10.0]), np.array([30.0])).fit(10 + 20 * X, y).predict(10 + 20 * X[:5])
    assert np.allclose(a[0], c[0], atol=1e-8)


def test_gp_hyperparameters_stay_in_bounds():
    rng = np.random.default_rng(2)
    X = rng.random((30, 2))
    for y in (rng.standard_normal(30) * 1e6,                      # pure noise
              X[:, 0] * 0.0 + np.linspace(0, 1, 30) ** 8,        # nearly flat with a spike
              np.sin(200 * X[:, 0])):                              # very short length scale
        gp = ExactGP(np.zeros(2), np.ones(2), iters=60, lr=0.5).fit(X, y)
        ls = np.exp(gp.log_lengthscale)
        assert np.all(ls >= DEFAULT_LENGTHSCALE_BOUNDS[0] - 1e-12)
        assert np.all(ls <= DEFAULT_LENGTHSCALE_BOUNDS[1] + 1e-12)
        lo, hi = DEFAULT_OUTPUTSCALE_BOUNDS
        assert lo - 1e-12 <= math.exp(gp.log_outputscale) <= hi + 1e-12
        lo, hi = DEFAULT_NOISE_BOUNDS
        assert lo - 1e-12 <= math.exp(gp.log_noise) <= hi + 1e-12


def test_gp_known_noise():
    X, y = _sine(20)
    noise = np.full(20, np.nan)
    noise[:5] = 1e4          # five points that should barely matter
    gp = ExactGP(np.zeros(1), np.ones(1)).fit(X, y, noise=noise)
    mean, _ = gp.predict(X[:5])
    assert np.all(np.isfinite(mean))


def test_cold_refit_growth_counts():
    rng = np.random.default_rng(0)
    X = rng.random((40, 1))
    y = np.sin(6 * X[:, 0])
    exact = ExactGP(np.zeros(1), np.ones(1), cold_refit_growth=0.0)
    for n in range(10, 15):
        exact.fit(X[:n], y[:n])
    assert exact.n_cold_fits == 5 and exact.n_warm_fits == 0

    gp = ExactGP(np.zeros(1), np.ones(1), cold_refit_growth=0.25)
    kinds = []
    for n in range(8, 21):
        gp.fit(X[:n], y[:n])
        kinds.append("w" if gp.last_fit_warm else "c")
    # cold at 8, warm while n < 10, cold at 10, warm while n < 12.5, cold at 13, ... 16.25, 17, 21.25
    assert "".join(kinds) == "cwcwwcwwwcwww"
    assert gp.n_cold_fits == 4 and gp.n_warm_fits == 9

    other = ExactGP(np.zeros(1), np.ones(1), cold_refit_growth=0.25)
    other.fit(X[:20], y[:20], warm=gp)
    assert other.last_fit_warm


def test_warm_fit_runs_fifteen_percent_of_the_steps(monkeypatch):
    calls = {"n": 0}
    orig = ExactGP._neg_log_ml

    def counting(self, p):
        calls["n"] += 1
        return orig(self, p)

    monkeypatch.setattr(ExactGP, "_neg_log_ml", counting)
    X, y = _sine(12)
    gp = ExactGP(np.zeros(1), np.ones(1), iters=100, cold_refit_growth=0.5)
    gp.fit(X[:10], y[:10])
    assert calls["n"] == 100
    gp.fit(X[:11], y[:11])
    assert calls["n"] == 115


# ----------------------------------------------------------------------------- MES
def test_mes_prefers_unexplored_low_region():
    # cost falls toward x = 0.5, everything right of it is unexplored
    X = np.linspace(0, 0.5, 11)[:, None]
    y = 2 + 0.3 * np.sin(8 * X[:, 0]) + 2 * (0.5 - X[:, 0])
    gp = ExactGP(np.zeros(1), np.ones(1)).fit(X, y)
    cand = np.linspace(0, 1, 201)[:, None]
    scores = mes_acquisition(gp, cand, np.random.default_rng(0), incumbent=float(y.min()))
    assert cand[np.argmax(scores), 0] > 0.6
    assert scores[cand[:, 0] <= 0.5].max() < 0.1 * scores.max()
    # directly: a candidate with the same mean but larger spread scores higher
    s = min_value_entropy(np.array([1.0, 1.0]), np.array([0.1, 1.0]), np.random.default_rng(0))
    assert s[1] > s[0]


def test_sample_positive_bounds():
    rng = np.random.default_rng(0)
    d = sample_positive(np.full(10_000, 0.0), np.full(10_000, 1.0), 1.5, rng)
    assert np.all(d >= 1.5)
    assert d.mean() == pytest.approx(truncnorm.mean(1.5, np.inf), abs=0.02)
    far = sample_positive(0.0, 1.0, 40.0, rng)          # far tail, no rejection loop
    assert 40.0 <= far < 41.0
    assert sample_positive(2.0, 0.0, 1.0, rng) == 2.0
    assert sample_positive(0.5, 0.0, 1.0, rng) == 1.0
    a = sample_positive(np.zeros(5), np.ones(5), np.zeros(5), np.random.default_rng(7))
    b = sample_positive(np.zeros(5), np.ones(5), np.zeros(5), np.random.default_rng(7))
    assert np.array_equal(a, b)


# ----------------------------------------------------------------------- Segment-GP
SPEC = ProblemSpec("toy", 2, np.zeros(2), np.ones(2), 100, ("x",), (), True, False)


def _linear_traj(theta, crash_step=None):
    """Stage cost ``1 + theta_0`` per step, so ``J_t = t (1 + theta_0)``."""
    theta = np.asarray(theta, dtype=float)
    J = np.arange(1, SPEC.T + 1) * (1.0 + theta[0])
    if crash_step is not None:
        J[crash_step - 1:] = np.inf
    return Trajectory(theta, J.astype(np.float32).astype(float), np.zeros((SPEC.T, 1)), None,
                      crash_step, crash_step is None)


def _context(rng, n=12):
    trials = []
    for i in range(n):
        th = rng.random(2)
        t_obs = SPEC.T if i % 3 else int(rng.choice([40, 70]))
        trials.append(_linear_traj(th).cut(t_obs))
    return trials


def test_segment_layout_boundaries_exclude_crashes():
    rng = np.random.default_rng(0)
    ctx = _context(rng)
    crashed = _linear_traj(rng.random(2), crash_step=55).cut(100)
    layout = SegmentLayout.from_context([*ctx, crashed], SPEC.T)
    assert list(layout.edges) == [40, 70, 100]
    assert all(t is not crashed for t in layout.trials)
    assert layout.segments_after(50) == [(50, 70), (70, 100)]
    assert layout.segments_after(0) == [(0, 40), (40, 70), (70, 100)]


def test_segment_gp_matches_sum_of_linear_segments():
    rng = np.random.default_rng(1)
    ctx = _context(rng)
    pred = SegmentGP(SPEC)
    theta = np.array([0.4, 0.6])
    truth = 100 * 1.4
    query = _linear_traj(theta).cut(55)      # running trial between boundaries
    out = pred.predict(query, ctx)
    assert isinstance(out, GaussianPrediction) and out.p_crash == 0.0
    assert out.mean == pytest.approx(truth, rel=0.03)
    assert out.var > 0
    unrun = pred.predict(_linear_traj(theta).cut(0), ctx)
    assert unrun.mean == pytest.approx(truth, rel=0.03)
    done = pred.predict(_linear_traj(theta).cut(100), ctx)
    assert done.mean == pytest.approx(truth, rel=1e-6) and done.var == 0.0
    # batch path equals the single path
    batch = pred.predict_finals([query, _linear_traj(theta).cut(0)], ctx)
    assert batch[0].mean == pytest.approx(out.mean) and batch[1].mean == pytest.approx(unrun.mean)


def test_segment_gp_ignores_crashed_context():
    rng = np.random.default_rng(2)
    ctx = _context(rng)
    query = _linear_traj(np.array([0.3, 0.3])).cut(40)
    base_pred = SegmentGP(SPEC).predict(query, ctx)
    crashed = [_linear_traj(rng.random(2), crash_step=s).cut(100) for s in (20, 63, 90)]
    with_crash = SegmentGP(SPEC).predict(query, [*ctx, *crashed])
    assert with_crash.mean == pytest.approx(base_pred.mean)
    assert with_crash.var == pytest.approx(base_pred.var)


def test_segment_gp_passes_cold_refit_growth():
    rng = np.random.default_rng(3)
    ctx = _context(rng, 15)
    pred = SegmentGP(SPEC, cold_refit_growth=0.5)
    q = _linear_traj(np.array([0.5, 0.5])).cut(40)
    pred.predict(q, ctx[:12])
    pred.predict(q, ctx)
    chain = pred._chains[(70, 100)]
    assert chain.cold_refit_growth == 0.5
    assert chain.n_cold_fits + chain.n_warm_fits == 2
    pred.reset()
    assert not pred._chains
