"""Strategies, stop rules and the optimization loop on a toy problem with fake predictors."""

from __future__ import annotations

import math

import numpy as np
import pytest

from esbopfn.core import Instance, ProblemSpec, Trajectory
from esbopfn.predictors.base import BarPrediction, GaussianPrediction
from esbopfn.problems.base import Problem, finish
from esbopfn.strategies import (
    STRATEGY_IDS,
    RunRecord,
    Strategy,
    make_strategy,
    run_optimization,
    strategy_rng,
)
from esbopfn.strategies import rules
from esbopfn.strategies.base import incumbent, j_plus
from esbopfn.strategies.gp_based import (
    DecPFNH,
    crash_virtual_objective,
    sample_virtual,
)
from esbopfn.strategies.loop import _replay_stop
from esbopfn.strategies.pure_loop import acquisition, bar_ei, bar_pi, candidate_set, top_complete
from esbopfn.util.seeds import substream

T_TOY = 30
CRASH_STEP = 10
SPEC = ProblemSpec(name="toy", d=2, theta_lo=np.zeros(2), theta_hi=np.array([1.0, 2.0]), T=T_TOY,
                   channels=("y",), rate_channels=(), can_crash=True, has_success=False)
CENTER = np.array([0.3, 1.2])


def _rate(theta: np.ndarray) -> float:
    z = (np.asarray(theta) - CENTER) / SPEC.extent
    return 1.0 + 20.0 * float(z @ z)


def toy_sim(theta, instance, rng, t_max=None):
    theta = np.asarray(theta, dtype=np.float64)
    j = _rate(theta) * (1.0 + 0.05 * rng.standard_normal(T_TOY)) ** 2
    crash = CRASH_STEP if theta[0] > 0.85 else None
    n = crash if crash else T_TOY
    x = np.cumsum(j[:n])[:, None]
    return finish(theta, j[:n], x, None, T_TOY, crash)


TOY = Problem(spec=SPEC, prior=None, simulator=toy_sim)  # type: ignore[arg-type]
INST = Instance(phi={}, phi_hat={}, noise_std={}, seed=7)


def _bar(mean: float, half: float = 0.5, n: int = 20, p_crash: float = 0.0) -> BarPrediction:
    edges = np.linspace(mean - half * mean, mean + half * mean, n + 1)
    return BarPrediction(edges, np.full(n, (1 - p_crash) / n), p_crash)


class FakeDecPFN:
    """Context-blind, extrapolates the observed rate to the horizon."""

    name = "fake-decpfn"
    supports_unrun = False
    context_aware = False

    def __init__(self) -> None:
        self.calls = 0

    def predictable_steps(self, spec):
        return np.array([5, 10, 15, 20, 25])

    def predict(self, query, context):
        self.calls += 1
        mean = query.J_obs * T_TOY / query.t_obs
        return _bar(mean, p_crash=0.3 if query.theta[0] > 0.85 else 0.0)


class FakeCross(FakeDecPFN):
    """Cross-trajectory fake: unrun predictions from the true rate, prefix predictions as above."""

    supports_unrun = True
    context_aware = True

    def predict_unrun(self, thetas, context):
        return [_bar(_rate(th) * T_TOY, p_crash=0.3 if th[0] > 0.85 else 0.0) for th in thetas]


def _run(sid: str, seed: int = 3, n_max: int = 12, t_budget: int | None = None, **params):
    strategy = make_strategy(sid, TOY, strategy_rng(seed), **params)
    return strategy, run_optimization(TOY, INST, strategy, seed, n_max=n_max, t_budget=t_budget,
                                      record_curves=True)


def _full(seed: int, n: int, theta) -> Trajectory:
    return TOY.simulate(np.asarray(theta), INST, substream(seed, n))


# ----------------------------------------------------------------------------- rules
def test_exceed_rule():
    assert rules.exceed(5.0, 5.0) and rules.exceed(6.0, 5.0)
    assert not rules.exceed(4.9, 5.0) and not rules.exceed(1e9, math.inf)


def test_naive_rule_counts_crash_mass():
    pred = BarPrediction(np.array([0.0, 10.0]), np.array([0.5]), p_crash=0.5)
    # Pr[J >= 5] = 0.25 + 0.5
    assert rules.prob_at_least(pred, 5.0) == pytest.approx(0.75)
    assert rules.naive(pred, 5.0, tau=0.7) and not rules.naive(pred, 5.0, tau=0.8)
    assert not rules.naive(pred, math.inf, tau=0.1)
    assert rules.naive(pred, math.inf, tau=0.99, crash_threshold=0.4)


def test_prob_geq_exact_and_matched():
    a = BarPrediction(np.array([0.0, 1.0]), np.array([1.0]))
    b = BarPrediction(np.array([0.5, 1.5]), np.array([1.0]))
    # P(A >= B) for U(0,1), U(0.5,1.5): 0.5 * 0.5 * 0.5 = 0.125
    assert rules.prob_geq(a, b) == pytest.approx(0.125)
    assert rules.prob_geq(b, a) == pytest.approx(0.875)
    crash_a = BarPrediction(np.array([0.0, 1.0]), np.array([0.6]), p_crash=0.4)
    assert rules.prob_geq(crash_a, b) == pytest.approx(0.4 + 0.6 * 0.125)
    # Monte Carlo check against a Gaussian reference
    rng = np.random.default_rng(0)
    g = GaussianPrediction(0.8, 0.04, p_crash=0.1)
    xa, xb = a.sample(rng, 200_000), g.sample(rng, 200_000)
    assert rules.prob_geq(a, g) == pytest.approx(np.mean(xa >= xb), abs=0.01)
    # matched: incumbent truncated at J^- before the comparison
    run = BarPrediction(np.array([0.0, 2.0]), np.array([1.0]))
    inc = BarPrediction(np.array([0.0, 2.0]), np.array([1.0]))
    # truncated incumbent U(1, 2): P(U(0,2) >= U(1,2)) = 0.25
    assert rules.prob_geq(run, inc.truncate_below(1.0)) == pytest.approx(0.25)
    assert rules.matched(run, inc, 1.0, tau=0.2) and not rules.matched(run, inc, 1.0, tau=0.3)
    assert rules.matched(_bar(1.0, p_crash=0.95), inc, 1.0, tau=0.99, crash_threshold=0.9)


# ------------------------------------------------------------------------ acquisition
def test_ei_pi_closed_forms_match_numeric_integration():
    rng = np.random.default_rng(1)
    edges = np.sort(rng.uniform(0, 10, 41))
    p_crash = 0.2
    probs = rng.random(40)
    probs *= (1 - p_crash) / probs.sum()
    pred = BarPrediction(edges, probs, p_crash)
    y = np.linspace(edges[0], edges[-1], 400_001)
    dens = np.exp(pred.log_density(y))
    for J_minus in (edges[0] - 1, 3.3, edges[17], 7.9, edges[-1] + 1):
        gain = np.clip(J_minus - y, 0, None)
        assert bar_ei(pred, J_minus) == pytest.approx(np.trapezoid(dens * gain, y), abs=1e-3)
        assert bar_pi(pred, J_minus) == pytest.approx(np.trapezoid(dens * (y < J_minus), y),
                                                      abs=1e-3)
    assert bar_pi(pred, 1e9) == pytest.approx(1 - p_crash)
    g = GaussianPrediction(2.0, 1.0, p_crash=0.5)
    assert acquisition([g], 2.0, "pi")[0] == pytest.approx(0.25)


def test_candidate_set_shape_and_clipping():
    rng = np.random.default_rng(2)
    best = np.array([[0.0, 0.0], [1.0, 2.0]])
    c = candidate_set(SPEC, rng, best)
    assert c.shape == (2000, 2)
    assert np.all(c >= SPEC.theta_lo) and np.all(c <= SPEC.theta_hi)
    local = c[400:]
    near = np.minimum(np.abs(local - best[0]).max(1), np.abs(local - best[1]).max(1))
    assert np.all(near <= 0.6 * SPEC.extent.max())
    assert np.any(local[:, 0] == 0.0)  # clipped onto the boundary
    assert candidate_set(SPEC, rng, np.zeros((0, 2))).shape == (2000, 2)


# ------------------------------------------------------------- virtual observations
def test_crash_virtual_objective_values():
    v = crash_virtual_objective(np.array([1.0, 5.0, 9.0]), np.array([1.0, 0.5, 0.0]), 4.0, 8.0)
    np.testing.assert_allclose(v, [7.0, 6.5, 8.0])


def test_sample_virtual_respects_truncation_and_crash():
    rng = np.random.default_rng(3)
    g = GaussianPrediction(0.0, 1.0)
    assert all(sample_virtual(g, 1.5, rng) >= 1.5 for _ in range(200))
    bar = _bar(10.0, p_crash=0.5)
    draws = np.array([sample_virtual(bar, 12.0, rng) for _ in range(2000)])
    assert np.all(draws >= 12.0)
    assert np.mean(np.isinf(draws)) == pytest.approx(0.5, abs=0.05)


def test_hybrid_picks_lower_variance():
    dec = _bar(10.0, half=0.1)  # value variance ~ (2^2)/12
    _, dvar = dec.mean_var()
    assert DecPFNH.choose(GaussianPrediction(10.0, dvar / 2), dec) == "segment-gp"
    assert DecPFNH.choose(GaussianPrediction(10.0, dvar * 2), dec) == "decpfn"
    assert DecPFNH.choose(GaussianPrediction(10.0, 0.0), dec) == "decpfn"
    assert DecPFNH.choose(GaussianPrediction(10.0, 1.0), None) == "segment-gp"
    # crash mass does not enter the comparison (conditioned on J != inf)
    crashy = _bar(10.0, half=0.1, p_crash=0.6)
    assert DecPFNH.choose(GaussianPrediction(10.0, dvar * 1.01), crashy) == "decpfn"


def test_incumbent_and_j_plus():
    def trial(J_total, t_obs, crash=None):
        J = np.linspace(J_total / T_TOY, J_total, T_TOY).astype(np.float32)
        if crash:
            J[crash - 1:] = np.inf
        tr = Trajectory(np.zeros(2), J, J[:, None], None, crash, crash is None)
        return tr.cut(t_obs)
    hist = [trial(50, T_TOY), trial(30, 10), trial(40, T_TOY), trial(20, T_TOY, crash=5),
            trial(70, T_TOY)]
    assert incumbent(hist)[1] == pytest.approx(40)
    assert j_plus(hist) == pytest.approx(70)
    assert incumbent([])[1] == math.inf


# ------------------------------------------------------------------------- loop
class AlwaysStop(Strategy):
    name = "always-stop"

    def propose(self, history):
        return np.array([0.3, 1.2])

    def decision_steps(self):
        return np.arange(1, T_TOY)

    def should_stop(self, running, history):
        return True


def test_initial_design_runs_without_stops():
    s = AlwaysStop(SPEC, np.random.default_rng(0))
    rec = run_optimization(TOY, INST, s, seed=0, n_max=6)
    assert [t.source for t in rec.trials[:2]] == ["init", "init"]
    for t in rec.trials[: SPEC.n_init()]:
        assert t.complete or t.crashed
    assert any(t.complete for t in rec.trials[:2])
    assert all(t.t_obs == 1 for t in rec.trials[2:])


def test_budget_accounting():
    _, rec = _run("esrs", n_max=7, t_budget=10 * T_TOY)
    assert len(rec.trials) <= 7
    _, rec = _run("bo", n_max=50, t_budget=3 * T_TOY + 5, n_candidates=200)
    # BO runs full trials: after 3 complete trials 90 > 95 - 30, so the loop ends
    used = np.cumsum([t.t_obs for t in rec.trials])
    assert all(u <= 3 * T_TOY + 5 - T_TOY for u in used[:-1])
    assert used[-1] > 3 * T_TOY + 5 - T_TOY or len(rec.trials) == 50
    assert rec.steps_used == used[-1]


def test_exceed_stops_exactly_when_reaching_incumbent():
    seed = 5
    _, rec = _run("esrs", seed=seed, n_max=20)
    assert any(t.early_stopped for t in rec.trials)
    for n, t in enumerate(rec.trials):
        full = _full(seed, n, t.theta)
        J_minus = rec.incumbent_J[n - 1] if n else math.inf
        end = full.crash_step or T_TOY
        if t.source == "init":
            assert t.t_obs == end
            continue
        hits = np.flatnonzero(full.J[: end - 1] >= J_minus)
        expect = int(hits[0]) + 1 if hits.size else end
        assert t.t_obs == expect
        assert t.J_true == pytest.approx(full.J_total)
        np.testing.assert_allclose(t.J, full.J[: t.t_obs])


def test_replay_crash_precedes_rules():
    s = AlwaysStop(SPEC, np.random.default_rng(0))
    traj = toy_sim(np.array([0.9, 1.0]), INST, np.random.default_rng(0))
    s.on_trial_start([], armed=False)
    assert _replay_stop(s, traj, [], False, 1.0, T_TOY)[0] == CRASH_STEP
    assert _replay_stop(s, traj, [], True, math.inf, T_TOY)[0] == 1


def test_seeds_reproduce_and_json_round_trip():
    def strip(rec: RunRecord):
        d = RunRecord.from_json(rec.to_json())
        for t in d.trials:
            t.propose_s = t.stop_rule_s = t.end_s = 0.0
        d.timing, d.wall_s = {}, 0.0
        return d
    _, a = _run("esbo-gp", seed=11, n_max=7, n_candidates=200)
    _, b = _run("esbo-gp", seed=11, n_max=7, n_candidates=200)
    assert strip(a) == strip(b)
    _, c = _run("esbo-gp", seed=12, n_max=7, n_candidates=200)
    assert strip(a) != strip(c)
    back = RunRecord.from_json(a.to_json())
    assert back == a
    assert back.trials[0].J is not None
    compact = run_optimization(TOY, INST, make_strategy("esrs", TOY, strategy_rng(0)), 0, n_max=4)
    assert compact.trials[0].J is None
    assert '"J":null' in compact.to_json()
    # equal seeds share the initial design across strategies
    _, e = _run("esrs", seed=11, n_max=7)
    assert [t.theta for t in e.trials[:2]] == [t.theta for t in a.trials[:2]]


def test_registry_instantiates_every_id():
    for sid in STRATEGY_IDS:
        kw = {}
        if sid in ("decpfn", "decpfn-h"):
            kw["predictor"] = FakeDecPFN()
        elif "xdecpfn" in sid or "ftpfn" in sid:
            kw["predictor"] = FakeCross()
        elif sid == "esbo-t-tabpfn":
            kw["predictor"] = FakeCross()
        s = make_strategy(sid, TOY, np.random.default_rng(0), **kw)
        assert s.name == sid
        assert s.strategy_spec.display_name
    assert make_strategy("decpfn", TOY, np.random.default_rng(0)).model == "decpfn/toy/binned/1M"
    s = make_strategy("ftpfn-pi", TOY, np.random.default_rng(0), tau=0.9, crash_threshold=0.5)
    assert (s.tau, s.crash_threshold, s.acq) == (0.9, 0.5, "pi")
    assert s.model == "ftpfn/toy/default/default"
    with pytest.raises(TypeError):
        make_strategy("esrs", TOY, np.random.default_rng(0), tau=0.5)
    with pytest.raises(KeyError):
        make_strategy("nope", TOY, np.random.default_rng(0))


# ------------------------------------------------------------------ end to end
def test_bo_end_to_end_no_early_stopping():
    _, rec = _run("bo", n_max=8, n_candidates=200)
    assert len(rec.trials) == 8
    assert not any(t.early_stopped for t in rec.trials)
    assert rec.timing["gp_acq"] > 0


def test_esbo_gp_end_to_end():
    s, rec = _run("esbo-gp", n_max=10, n_candidates=200)
    assert len(rec.trials) == 10
    assert rec.best[1] < math.inf
    assert any(t.early_stopped for t in rec.trials)
    J = {i: t for i, t in enumerate(rec.trials)}
    for i, v in s.last_virtual.items():
        assert v >= J[i].J_obs or J[i].crashed


def test_oracle_uses_true_total():
    s, rec = _run("esbo-oracle", seed=5, n_max=12, n_candidates=200)
    early = [i for i, t in enumerate(rec.trials[:-1]) if t.early_stopped and
             math.isfinite(t.J_true)]
    assert early
    for i in early:
        assert s.last_virtual[i] == pytest.approx(rec.trials[i].J_true)


def test_decpfn_matched_end_to_end():
    fake = FakeDecPFN()
    s, rec = _run("decpfn", seed=4, n_max=10, n_candidates=200, predictor=fake)
    assert fake.calls > 0
    for t in rec.trials:
        assert t.t_obs >= 1
    # init trials recorded predictions for later reuse, only the incumbent curve is kept
    inc = rec.incumbent_index[-1]
    assert set(s.curves) <= {inc, len(rec.trials) - 1}
    matched_stops = 0
    for n, t in enumerate(rec.trials):
        if t.source != "strategy" or not t.early_stopped:
            continue
        full = _full(4, n, t.theta)
        by_exceed = full.J[t.t_obs - 1] >= rec.incumbent_J[n - 1]
        assert by_exceed or t.t_obs in {5, 10, 15, 20, 25}
        matched_stops += not by_exceed
    assert matched_stops > 0


def test_decpfn_hybrid_end_to_end():
    s, rec = _run("decpfn-h", seed=4, n_max=10, n_candidates=200, predictor=FakeDecPFN())
    assert len(rec.trials) == 10
    assert set(s.last_sources.values()) <= {"segment-gp", "decpfn"}


@pytest.mark.parametrize("sid", ["xdecpfn-pi", "ftpfn-ei"])
def test_pure_loop_end_to_end(sid):
    s, rec = _run(sid, seed=2, n_max=15, predictor=FakeCross(), n_uniform=40, n_local=160)
    assert len(rec.trials) == 15
    assert "gp_acq" not in rec.timing
    for t in rec.trials:
        if t.source == "strategy" and t.early_stopped:
            full = _full(2, rec.trials.index(t), t.theta)
            n = rec.trials.index(t)
            stopped_by_exceed = full.J[t.t_obs - 1] >= rec.incumbent_J[n - 1]
            assert t.t_obs in set(s.decision_steps().tolist()) or stopped_by_exceed
    # the proposals concentrate near the optimum of the fake's unrun predictions
    best_theta = np.array(rec.trials[rec.best[0]].theta)
    assert np.abs(best_theta - CENTER).max() < 0.5


def test_top_complete_order():
    def tr(J_total):
        J = np.full(T_TOY, J_total, dtype=np.float32)
        return Trajectory(np.array([J_total, 0.0]), J, J[:, None], None, None, True).cut(T_TOY)
    hist = [tr(v) for v in (5, 1, 4, 2, 3, 0.5)]
    np.testing.assert_allclose(top_complete(hist, 3)[:, 0], [0.5, 1, 2])


def test_esrs_is_uniform_random_plus_exceed():
    seed = 9
    _, rec = _run("esrs", seed=seed, n_max=15)
    rng = strategy_rng(seed)
    proposed = [t.theta for t in rec.trials if t.source == "strategy"]
    expect = [list(SPEC.theta_lo + rng.random(2) * SPEC.extent) for _ in proposed]
    np.testing.assert_allclose(proposed, expect)
    assert rec.timing == {}


def test_real_problem_smoke():
    from esbopfn.problems import get_problem

    boiler = get_problem("boiler")
    strategy = make_strategy("esrs", boiler, strategy_rng(0))
    rec = run_optimization(boiler, boiler.nominal_instance(), strategy, 0, n_max=5)
    assert len(rec.trials) == 5 and rec.trials[0].t_obs == boiler.spec.T
