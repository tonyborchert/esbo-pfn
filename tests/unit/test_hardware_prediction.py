"""Hardware prediction benchmark: task selection, context gating, cutoff snapping, kinds."""

from __future__ import annotations

import math

import numpy as np
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.hardware import prediction_bench as PB
from esbopfn.hardware import records
from esbopfn.hardware.records import HwRun, HwTrial
from esbopfn.predictors.base import GaussianPrediction
from esbopfn.problems import get_problem

SPEC = get_problem("furuta").spec


def hw(i: int, outcome: str, t: int) -> HwTrial:
    J = list(np.arange(1, t + 1, dtype=float))
    x = [[0.0, 0.1]] * t
    return HwTrial(index=i, theta=[float(i)] * 8, source="init", T=600, t_obs=t,
                   outcome=outcome, crash_step=t if outcome == records.CRASHED else None,
                   success=outcome == records.COMPLETE,
                   J_obs=math.inf if outcome == records.CRASHED else float(t), J=J, x=x,
                   xdot=x, u=None, kick_sign=1.0, incumbent_index=-1, incumbent_J=math.inf)


C, E, X, B = records.COMPLETE, records.EARLY_STOPPED, records.CRASHED, records.BUDGET_CUT
RUN = HwRun("rig", "esrs", 0, [hw(0, C, 600), hw(1, E, 40), hw(2, C, 600), hw(3, X, 25),
                               hw(4, B, 300), hw(5, C, 600), hw(6, X, 1)])


class FakePredictor:
    """Gaussian around the true J_T of the queries above, records every call."""

    context_aware = True

    def __init__(self, steps=None, fail_without_context: bool = False):
        self.steps = steps
        self.fail = fail_without_context
        self.calls: list[tuple[int, int, int]] = []

    def predictable_steps(self, spec):
        return np.arange(1, spec.T) if self.steps is None else np.asarray(self.steps)

    def predict(self, query, context):
        if self.fail and not context:
            raise ValueError("no context")
        self.calls.append((int(query.theta[0]), int(query.t_obs), len(context)))
        return GaussianPrediction(mean=600.0, var=100.0, p_crash=0.1)


def test_query_selection():
    qs = PB.select_queries([RUN])
    # not the first trial, not early stopped or budget cut, at least two steps
    assert [i for _, i in qs] == [2, 3, 5]
    assert PB.eligible_counts([RUN], (0, 1, 2, 3, 5, 6)) == {0: 3, 1: 3, 2: 3, 3: 2, 5: 1,
                                                             6: 0}


def test_cutoffs_and_snapping():
    assert PB.raw_cutoffs(600) == [60, 180, 300, 420, 540]
    assert PB.raw_cutoffs(25) == [2, 8, 12, 18, 22]           # round half to even, as before
    assert PB.raw_cutoffs(2) == [1, 1, 1, 1, 1]
    grid = np.arange(9, 600, 9)
    assert PB.snap(60, grid) == 54 and PB.snap(9, grid) == 9 and PB.snap(8, grid) is None


def test_context_gating_and_scores():
    pred = FakePredictor()
    df = PB.score([RUN], pred, SPEC, predictor_name="fake", rig="rig", ladder=(0, 1, 2, 3, 5))
    got = df.groupby("n_ctx").query_index.unique().map(sorted).to_dict()
    assert got == {0: [2, 3, 5], 1: [2, 3, 5], 2: [2, 3, 5], 3: [3, 5], 5: [5]}
    # the context is exactly the n preceding trials, whatever their outcome
    assert set(df[df.query_index == 5].n_ctx_used) == {0, 1, 2, 3, 5}
    assert (5, 300, 5) in pred.calls
    crashed = df[df.crashed]
    assert set(crashed.query_index) == {3}
    np.testing.assert_allclose(crashed.ll, math.log(0.1))
    ok = df[~df.crashed]
    assert np.all(np.isfinite(ok.ll)) and np.all(ok.target_J == 600.0)
    s = PB.summarize(df, PB.eligible_counts([RUN], (0, 1, 2, 3, 5)))
    assert s["n_scored"]["0"] == 2 and s["n_scored_all"]["0"] == 3 and s["n_scored"]["5"] == 1
    assert s["n_queries"] == {"0": 3, "1": 3, "2": 3, "3": 2, "5": 1}


def test_snapping_drops_queries_before_the_first_step():
    pred = FakePredictor(steps=np.arange(9, 600, 9))
    df = PB.score([RUN], pred, SPEC, predictor_name="fake", rig="rig", ladder=(0,))
    q3 = df[df.query_index == 3]                     # t = 25: cutoffs 2, 8, 12, 18, 22
    assert sorted(q3.step) == [9, 18, 18]
    assert sorted(q3.cutoff_step) == [12, 18, 22]
    assert set(df[df.query_index == 2].step) == {54, 180, 297, 414, 540}


def test_segment_gp_style_failures_are_counted():
    pred = FakePredictor(fail_without_context=True)
    df = PB.score([RUN], pred, SPEC, predictor_name="fake", rig="rig", ladder=(0, 1))
    assert set(df.n_ctx) == {1} and df.attrs["failures"] == {0: 15}


def test_full_history_ablation():
    pred = FakePredictor()
    df, summary = PB.run_bench([RUN], pred, SPEC, rig="rig", predictor_name="fake",
                               model="fake", bench="ablation")
    assert set(df.n_ctx) == {PB.FULL}
    assert dict(zip(df.query_index, df.n_ctx_used)) == {2: 2, 3: 3, 5: 5}
    assert summary["n_queries"] == {"full": 3} and "full" in summary["median_ll"]


def test_context_blind_predictions_are_reused():
    pred = FakePredictor()
    pred.context_aware = False
    PB.score([RUN], pred, SPEC, predictor_name="fake", rig="rig", ladder=(0, 1, 2))
    assert len(pred.calls) == 15 and all(n == 0 for _, _, n in pred.calls)


def test_kind_runs_on_imported_runs(pfn_home):
    from esbopfn.benchmarks.kinds import discover_kinds
    from esbopfn.benchmarks.kinds.hardware_prediction import run, unit_key
    from esbopfn.hardware.runner import unit_key as opt_key

    kinds = discover_kinds()
    assert {"hardware_optimization", "hardware_prediction"} <= set(kinds)
    db = Db()
    key = opt_key("with_weight", "esrs", 0, {}, provenance="thesis")

    def write(key, blob):
        records.write_trials(blob / records.TRIALS_FILE, RUN.trials)

    db.run("hardware_optimization", key, write)
    pkey = unit_key("with_weight", "ladder", "fake", "fake", ladder=[0, 1], logs="thesis")
    blob = db.run("hardware_prediction", pkey,
                  lambda k, b: run(k, b, predictor=FakePredictor()))
    import json

    summary = json.loads((blob / PB.SUMMARY_FILE).read_text())
    assert summary["n_queries"] == {"0": 3, "1": 3}
    import pandas as pd

    assert list(pd.read_parquet(blob / PB.SCORES_FILE).columns) == list(PB.SCORE_COLUMNS)
    with pytest.raises(RuntimeError, match="pfn hw run"):
        kinds["hardware_optimization"].run(key, blob)
