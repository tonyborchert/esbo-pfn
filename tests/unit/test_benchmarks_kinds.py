"""Benchmark kinds on toy budgets: instances, fields, task tables, prediction, optimization."""

import math

import numpy as np
import pandas as pd
import pytest

from esbopfn.benchmarks import instances as inst
from esbopfn.benchmarks import tasks as T
from esbopfn.benchmarks.db import Db
from esbopfn.benchmarks.kinds import discover_kinds
from esbopfn.benchmarks.runner import PlannedUnit, run_units
from esbopfn.benchmarks.db import key_hash


def test_screen_stats():
    seq = [True] * 3 + [False] + [True, False] + [True] * 3
    assert inst.screen_stats(seq, 3) == {"pass_count": 2, "theta_screened": 4, "trials_used": 9}


def test_instances_deterministic_and_append_only():
    a = inst.build_instances("boiler", 2, root_seed=11)
    b = inst.build_instances("boiler", 3, root_seed=11)
    c = inst.build_instances("boiler", 3, root_seed=11, reuse=a)
    strip = [{k: v for k, v in r.items() if k != "seconds"} for r in b]
    assert [{k: v for k, v in r.items() if k != "seconds"} for r in a] == strip[:2]
    assert [{k: v for k, v in r.items() if k != "seconds"} for r in c] == strip
    assert all(r["screen"]["pass_count"] >= 25 for r in b)
    assert inst.build_instances("boiler", 1, root_seed=12)[0]["phi"] != a[0]["phi"]


def _instances_unit(db, problem="boiler", count=1):
    key = inst.instances_key(problem, count)
    kind = discover_kinds()["instances"]
    return key, run_units([PlannedUnit("instances", key, key_hash(key))], db, {"instances": kind})


def test_instances_kind_idempotent(pfn_home):
    db = Db()
    key, first = _instances_unit(db)
    assert first == (1, 0)
    assert _instances_unit(db)[1] == (0, 1)
    loaded = inst.load_instances("boiler")
    assert len(loaded) == 1 and set(loaded[0].phi) == {"heating_rate", "cooling_rate"}


def test_field_toy_run(pfn_home):
    from esbopfn.benchmarks.kinds.fields import field_key, read_field

    db = Db()
    _instances_unit(db)
    key = field_key("boiler", "oracle65", 0, 0, n_trials=3)
    db.run("fields", key, discover_kinds()["fields"].run)
    run = read_field(db.blob_dir("fields", key), key)
    assert run.n_trials == 3 and run.J.shape == (3, 701) and run.x.shape == (3, 701, 1)
    assert (run.t_obs[:2] == 701).all()          # initial design runs in full
    traj = run.trajectory(2)
    assert traj.J_total == pytest.approx(float(run.J[2, -1]))
    ctx = run.context(2)
    assert [t.t_obs for t in ctx] == [701, 701] and ctx[0].complete
    crashed, target = run.target(1)
    assert not crashed and target == pytest.approx(float(run.J[1, -1]))


def _trial_table(n_instances=60, n_seeds=2, n_trials=65, horizon=100):
    rows = []
    for s in range(n_seeds):
        for i in range(n_instances):
            for t in range(n_trials):
                crash = (i + t) % 2 == 0
                rows.append({"run_id": f"i{i:04d}_s{s:02d}", "instance": i, "seed": s,
                             "trial_index": t, "T": horizon, "t_obs": horizon,
                             "crash_step": 7 if crash else 0,
                             "J_T": math.nan if crash else float(i + t)})
    return pd.DataFrame(rows, columns=T.TRIAL_COLUMNS)


def test_build_tasks_counts_and_digest():
    trials = _trial_table()
    df = T.build_tasks(trials, "cartpole")
    assert sorted(df.n_ctx.unique()) == list(T.CONTEXTS)
    for n in T.CONTEXTS:
        cell = df[df.n_ctx == n]
        assert cell.selected.sum() == 50 and cell.crash_selected.sum() == 50
        assert not cell[cell.selected].crashed.any() and cell[cell.crash_selected].crashed.all()
        assert (cell[cell.selected].seed == 0).sum() == 30     # seed-major: all 30 of seed 0 first
    q = df[(df.n_ctx == 8)].iloc[0]
    assert q.query_index == 8 and q.task_id == "cartpole:i0000_s00:n8"
    h = T.digest(df)
    assert T.digest(df.sample(frac=1.0, random_state=0)) == h
    assert T.digest(T.build_tasks(trials.sample(frac=1.0, random_state=1), "cartpole")) == h
    changed = df.copy()
    changed.loc[changed.index[~changed.crashed][0], "target_J"] += 1.0
    assert T.digest(changed) != h
    no_crash = T.build_tasks(trials, "boiler", can_crash=False)
    assert no_crash.crash_selected.sum() == 0
    with pytest.raises(ValueError):
        T.build_tasks(_trial_table(n_instances=10), "cartpole")


def test_build_ladder_tasks():
    trials = _trial_table(n_instances=60, n_seeds=1, n_trials=6)
    df = T.build_ladder_tasks(trials, "cartpole", rungs=(0, 1, 2, 5), k=20)
    assert df.run_id.nunique() == 20 and len(df) == 80
    assert (df.query_index == 5).all() and not df.crashed.any()
    # last trial index 5 crashes when the instance index is odd
    assert (df.instance % 2 == 0).all()
    with pytest.raises(ValueError):
        T.build_ladder_tasks(trials, "cartpole", rungs=(0, 6), k=5)


# ------------------------------------------------------------------------------ prediction
class _FakePredictor:
    """Bar over [0, 100] with p_crash = min(0.9, t / 100), context recorded."""

    name = "fakebench"
    context_aware = True
    supports_unrun = False

    def __init__(self):
        self.contexts = []

    def predictable_steps(self, spec):
        return np.arange(10, spec.T, 10)

    def predict_many(self, queries, context):
        from esbopfn.predictors.base import BarPrediction

        self.contexts.append(len(context))
        out = []
        for q in queries:
            p = min(0.9, q.t_obs / 100.0)
            out.append(BarPrediction(edges=np.array([0.0, 100.0]), probs=np.array([1 - p]),
                                     p_crash=p))
        return out


def _synthetic_field(db, problem="boiler"):
    from esbopfn.benchmarks.kinds.fields import field_key, write_field
    from esbopfn.core import Trajectory
    from esbopfn.problems.base import cached_spec

    spec = cached_spec(problem)
    keys = []
    for s in range(2):
        for i in range(2):
            trajs = []
            for t in range(3):
                J = (np.arange(1, spec.T + 1) * 0.1).astype(np.float32)
                crash = 50 if (i == 1 and t in (0, 2)) else None
                if crash:
                    J[crash - 1:] = np.inf
                trajs.append(Trajectory(theta=np.array([100.0 + t]), J=J,
                                        x=np.zeros((spec.T, 1)), xdot=np.zeros((spec.T, 1)),
                                        crash_step=crash, success=crash is None))
            key = field_key(problem, "oracle65", i, s)
            db.run("fields", key, lambda k, d, tr=trajs: write_field(
                d, {"problem": problem}, spec, tr, [spec.T, 300, spec.T]))
            keys.append(key)
    return keys


def test_prediction_unit_with_fake_predictor(pfn_home):
    from esbopfn.benchmarks.kinds.prediction import cutoff_steps
    from esbopfn.benchmarks.kinds.fields import load_field_runs
    from esbopfn.predictors.base import register_predictor_family

    fake = _FakePredictor()
    register_predictor_family("fakebench", lambda args, spec, **kw: fake, replace=True)
    db = Db()
    keys = _synthetic_field(db)
    runs = load_field_runs("boiler", "oracle65", keys=keys)
    df = T.build_tasks(T.trial_table(runs), "boiler", contexts=(0, 2), k=2, k_crash=2)
    h = T.save_tasks(df, "boiler", "tasks-oracle65")
    key = {"problem": "boiler", "model": "fakebench", "checkpoint": "none",
           "tasks": "tasks-oracle65", "tasks_digest": h, "n": [0, 2], "part": "0/1"}
    db.run("prediction", key, discover_kinds()["prediction"].run)
    out = pd.read_parquet(db.blob_dir("prediction", key) / "predictions.parquet")
    assert set(fake.contexts) == {0, 2}
    ladder = out[out.cutoff_frac.notna()]
    assert sorted(ladder.cutoff_step.unique()) == [70, 210, 350, 490, 630]
    r = ladder[(ladder.cutoff_step == 70)].iloc[0]
    # log(1 - p) + log(1 / (100 - J_70)), J_t = 0.1 t
    assert r.ll == pytest.approx(math.log(0.3) - math.log(100 - 7.0), rel=1e-5)
    assert r.dist == "bar" and len(r.edges) == 2
    crashed = out[out.crashed]
    assert len(crashed) and crashed.cutoff_step.max() < 50 and crashed.cutoff_frac.isna().all()
    assert crashed.ll.iloc[0] == pytest.approx(math.log(crashed.p_crash.iloc[0]))
    stop_rows = out[out.cutoff_frac.isna() & ~out.crashed]
    assert stop_rows.edges.isna().all() and len(stop_rows) > 0
    assert cutoff_steps(np.array([125, 250]), 1000, (0.1,)) == {0.1: (125, True)}
    # part splitting covers the same tasks
    parts = []
    for i in range(2):
        k = {**key, "part": f"{i}/2"}
        db.run("prediction", k, discover_kinds()["prediction"].run)
        parts.append(pd.read_parquet(db.blob_dir("prediction", k) / "predictions.parquet"))
    assert sorted(pd.concat(parts).task_id.unique()) == sorted(out.task_id.unique())


def test_prediction_timing_with_fake(pfn_home):
    from esbopfn.benchmarks.kinds.prediction_timing import read_timing
    from esbopfn.predictors.base import register_predictor_family

    register_predictor_family("fakebench", lambda args, spec, **kw: _FakeSingle(), replace=True)
    db = Db()
    keys = _synthetic_field(db)
    from esbopfn.benchmarks.kinds.fields import load_field_runs

    df = T.build_tasks(T.trial_table(load_field_runs("boiler", "oracle65", keys=keys)), "boiler",
                       contexts=(0, 2), k=2, k_crash=2)
    h = T.save_tasks(df, "boiler", "tasks-oracle65")
    key = {"problem": "boiler", "model": "fakebench", "checkpoint": "none",
           "tasks": "tasks-oracle65", "tasks_digest": h, "n_ctx": 2, "queries": 2,
           "repeats": 3, "warmup": 1, "cutoff_frac": 0.5}
    db.run("prediction_timing", key, discover_kinds()["prediction_timing"].run)
    t = read_timing(db.blob_dir("prediction_timing", key))
    assert t["n_calls"] == 6 and t["seconds_median"] >= 0 and "device" in t


class _FakeSingle(_FakePredictor):
    def predict(self, query, context):
        return self.predict_many([query], context)[0]


def test_optimization_unit_esrs_and_bo(pfn_home):
    from esbopfn.benchmarks.kinds.optimization import read_run, strategy_label, strategy_params

    db = Db()
    _instances_unit(db)
    kind = discover_kinds()["optimization"]
    recs = {}
    for sid in ("esrs", "bo"):
        key = {"problem": "boiler", "instance": 0, "strategy": sid,
               "params": strategy_params(sid, "boiler"), "seed": 0, "budget": "capped",
               "instances_seed": inst.BENCH_SEED, "n_max": 4}
        db.run("optimization", key, kind.run)
        recs[sid] = read_run(db.blob_dir("optimization", key))
    assert len(recs["esrs"].trials) == 4 and len(recs["bo"].trials) == 4
    # equal seeds give equal initial designs across strategies
    assert recs["esrs"].trials[0].theta == recs["bo"].trials[0].theta
    assert all(t.complete for t in recs["bo"].trials)
    assert len(recs["bo"].incumbent_J) == 4
    assert strategy_label("decpfn", {"tau": 0.99, "crash_threshold": 0.9}) == \
        "decpfn@crash_threshold=0.9,tau=0.99"
    p = strategy_params("decpfn", "cartpole", {"tau": 0.99})
    assert p["model"] == "decpfn/cartpole/binned/1M" and p["checkpoint"] == "missing"


def test_landscape_units(pfn_home):
    db = Db()
    _instances_unit(db)
    kind = discover_kinds()["landscapes"]
    key = {"unit": "slice", "problem": "boiler", "instance": 0, "slice_seed": 1, "grid": 5,
           "repeats": 2, "instances_seed": inst.BENCH_SEED}
    db.run("landscapes", key, kind.run)
    z = np.load(db.blob_dir("landscapes", key) / "landscape.npz")
    assert z["cost_mean"].shape == (5,) and np.isfinite(z["cost_mean"]).all()
    assert (z["p_crash"] == 0).all()
    key = {"unit": "prior_samples", "problem": "boiler", "instance": 0, "n": 3,
           "max_unsuccessful": 1, "instances_seed": inst.BENCH_SEED}
    db.run("landscapes", key, kind.run)
    z = np.load(db.blob_dir("landscapes", key) / "samples.npz")
    assert z["J"].shape == (3, 701)


def test_make_slice_inside_box():
    from esbopfn.benchmarks.kinds.landscapes import make_slice
    from esbopfn.util.seeds import substream

    lo, hi = np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0])
    c, u, v, e = make_slice(lo, hi, substream(0, "s"))
    assert abs(np.dot(u, v)) < 1e-12 and e > 0
    for s in (-e, e):
        for t in (-e, e):
            th = c + s * u + t * v
            assert (th >= lo - 1e-9).all() and (th <= hi + 1e-9).all()
