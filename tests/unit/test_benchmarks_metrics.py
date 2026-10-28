"""Benchmark metrics against hand-computed values."""

import math
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from esbopfn.benchmarks import metrics as M


def _rows(task, lls, crashed=False, problem="cartpole", model="m", n=0, extra=()):
    fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    rows = [{"problem": problem, "model": model, "task_id": task, "instance": 0, "n_ctx": n,
             "crashed": crashed, "cutoff_frac": f, "ll": ll} for f, ll in zip(fracs, lls)]
    rows += [{"problem": problem, "model": model, "task_id": task, "instance": 0, "n_ctx": n,
              "crashed": crashed, "cutoff_frac": float("nan"), "ll": ll} for ll in extra]
    return rows


def test_trial_scores_mean_of_five_cutoffs():
    df = pd.DataFrame(_rows("a", [-1, -2, -3, -4, -5], extra=[-100])
                      + _rows("b", [0, 0, 0, 0, 0], crashed=True))
    s = M.trial_scores(df)
    assert list(s.task_id) == ["a"]
    assert s.score.iloc[0] == pytest.approx(-3.0) and s.n_cutoffs.iloc[0] == 5


def test_median_med_d_ll_norm_pooled():
    scores = pd.DataFrame({
        "problem": ["p1"] * 5 + ["p2"],
        "model": ["decpfn/p1/binned/1M"] * 3 + ["tabpfn/50", "other/p1"] + ["decpfn/p2/binned/1M"],
        "n_ctx": [0, 0, 0, 2, 0, 0], "score": [1.0, 2.0, 3.0, 10.0, 100.0, 7.0]})
    med = M.median_scores(scores)
    row = med[(med.problem == "p1") & (med.model == "decpfn/p1/binned/1M")].iloc[0]
    assert row["median"] == 2.0 and row["n"] == 3
    md = M.med_d(scores, ["decpfn/p1/binned/1M", "tabpfn/50", "decpfn/p2/binned/1M"])
    assert md["p1"] == pytest.approx(2.5) and md["p2"] == pytest.approx(7.0)
    normed = M.ll_norm(scores, md)
    assert normed.ll_norm.tolist() == pytest.approx([-1.5, -0.5, 0.5, 7.5, 97.5, 0.0])
    pooled = M.pooled_median(normed[normed.model.str.startswith("decpfn")])
    assert pooled.series.tolist() == ["decpfn/{p}/binned/1M"]
    assert pooled["median"].iloc[0] == pytest.approx(-0.25)   # median of -1.5, -0.5, 0.5, 0.0
    assert pooled.n_problems.iloc[0] == 2


def test_collapse_context_blind():
    s = pd.DataFrame({"model": ["decpfn/x/binned/1M", "xdecpfn/x/default/default"],
                      "n_ctx": [8, 8], "score": [0.0, 0.0]})
    out = M.collapse_context_blind(s)
    assert out.n_ctx.tolist() == [M.COLLAPSED_N, 8]


def test_pareto_front():
    pts = pd.DataFrame({"seconds": [1, 2, 3, 4, 0.5], "median": [0, 1, 0.5, 2, -1]})
    front = M.pareto_front(pts)
    assert list(zip(front.seconds, front["median"])) == [(0.5, -1), (1, 0), (2, 1), (4, 2)]


def test_stop_curve_and_steps_saved():
    rows = pd.DataFrame({
        "task_id": ["c"] * 4 + ["o"] * 4,
        "crashed": [True] * 4 + [False] * 4,
        "crash_step": [10] * 4 + [0] * 4,
        "cutoff_step": [2, 4, 6, 8] * 2,
        "p_crash": [0.1, 0.5, 0.3, 0.9, 0.2, 0.2, 0.6, 0.1]})
    tasks = pd.DataFrame({"task_id": ["c", "o", "q"], "crashed": [True, False, False],
                          "crash_step": [10, 0, 0]})
    curve = M.stop_curve(rows, tasks)
    assert curve.tau.tolist() == pytest.approx([0.9, 0.6, 0.5, 0.3, 0.2, 0.1])
    assert curve.mean_steps_saved.tolist() == pytest.approx([2, 2, 6, 6, 6, 8])
    assert curve.false_stop_rate.tolist() == pytest.approx([0, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert curve.n_crashed.iloc[0] == 1 and curve.n_ok.iloc[0] == 2
    assert M.steps_saved_at(curve, 0.1) == pytest.approx(2.0)
    assert M.steps_saved_at(curve, 0.5) == pytest.approx(8.0)


def test_ladder_curve_and_cost_rank():
    df = pd.DataFrame({"problem": "p", "model": "m", "task_id": ["a", "a", "b"],
                       "n_ctx": 0, "cutoff_frac": [0.1, 0.3, 0.1], "ll": [-1.0, -3.0, -4.0]})
    c = M.ladder_curve(df)
    assert c["mean"].iloc[0] == pytest.approx(-3.0) and c["n"].iloc[0] == 2
    r = M.cost_rank(np.array([3.0, 1.0, np.nan, 2.0]))
    assert r[0] == 0.0 and r[1] == 1.0 and np.isnan(r[2]) and r[3] == 0.5


def _record(trials, inc, timing=None):
    return NS(trials=[NS(t_obs=t, crashed=c) for t, c in trials], incumbent_J=inc,
              timing=timing or {})


def test_incumbent_axes_and_crashes():
    rec = _record([(10, False), (5, True), (10, False)], [5.0, 5.0, 3.0])
    assert M.incumbent_vs_trials(rec, 5).tolist() == [5, 5, 3, 3, 3]
    grid = np.array([0, 9, 10, 14, 15, 25, 30.0])
    assert M.incumbent_vs_time(rec, grid).tolist() == [math.inf, math.inf, 5, 5, 5, 3, 3]
    counts = M.crash_count_vs_time(rec, grid)
    assert np.isnan(counts[:2]).all() and counts[2:].tolist() == [0, 0, 1, 1, 1]
    rec2 = _record([(5, True)], [math.inf])
    mean = M.mean_crash_count([rec, rec2], grid)
    assert np.isnan(mean[0]) and mean[1] == 1.0   # only rec2 has finished a trial at 9
    assert mean[2:4].tolist() == pytest.approx([0.5, 0.5])   # rec2 crashed at 5
    assert M.final_incumbent(rec) == 3.0
    assert M.time_grid(60000, 300)[[0, -1]].tolist() == [0.0, 60000.0]
    assert M.regret_at_fraction(rec, 1.0, 5.0, 30.0, 0.5) == pytest.approx(1.0)


def test_j_best_j_esrs_scaled_regret():
    finals = pd.DataFrame({"instance": [0, 0, 0, 1, 1],
                           "strategy": ["esrs", "bo", "x", "esrs", "bo"],
                           "J_final": [10.0, 4.0, 2.0, math.inf, 7.0]})
    jb = M.j_best(finals, ["esrs", "bo"])
    assert jb.to_dict() == {0: 4.0, 1: 7.0}
    assert M.j_best(finals).to_dict() == {0: 2.0, 1: 7.0}
    je = M.j_esrs(finals)
    assert je[0] == 10.0 and math.isinf(je[1])
    r = M.scaled_regret(np.array([6.0, math.inf, 4.0]), 4.0, 10.0)
    assert r[0] == pytest.approx(1 / 3) and math.isinf(r[1]) and r[2] == 0.0
    assert np.isnan(M.scaled_regret(np.array([5.0]), 4.0, 4.0)).all()
    mat = M.scaled_regret(np.array([[6.0, 4.0], [12.0, 8.0]]), np.array([4.0, 2.0]),
                          np.array([10.0, 12.0]))
    np.testing.assert_allclose(mat, [[1 / 3, 0.0], [1.0, 0.6]])


def test_problem_median_mean_log10_timing():
    med = M.problem_median(np.array([[1.0, 2.0], [3.0, math.inf], [np.nan, np.nan]]))
    assert med[0] == 2.0 and math.isinf(med[1])
    assert M.cross_problem_mean([np.array([1.0, 2.0]), np.array([3.0, 4.0])]).tolist() == [2, 3]
    assert M.log10_display(np.array([1e-6, 0.01, 1.0])).tolist() == pytest.approx([-4, -2, 0])
    recs = [_record([], [], {"gp_acq": 2.0, "segment_gp": 4.0}),
            _record([], [], {"gp_acq": 4.0, "model_acq": 2.0})]
    split = M.timing_split(recs)
    assert split == pytest.approx({"GP acquisition": 3.0, "Segment-GP forecast": 2.0,
                                   "Transformer acquisition": 1.0})
