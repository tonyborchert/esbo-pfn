"""Benchmark specs of the simulation plot groups: loading and unit enumeration."""

from collections import Counter

import pytest

from esbopfn.benchmarks import groups
from esbopfn.benchmarks.db import Db
from esbopfn.benchmarks.runner import plan, select_groups

SIM_GROUPS = ("priors", "single_trajectory", "tabpfn", "cross_trajectory", "prediction_summary",
              "optimization", "generation_speed")
FIELD_RUNS_65 = (4 + 4 + 7 + 8 + 10) * 50


@pytest.mark.parametrize("name", SIM_GROUPS)
def test_spec_loads_and_enumerates(name, pfn_home):
    cfg = groups.load(name)
    assert cfg["name"] == name and cfg["kinds"]
    units = groups.units(cfg)
    kinds = Counter(k for k, _ in units)
    assert set(kinds) <= set(cfg["kinds"])
    ranks = [groups.kind_rank(k) for k, _ in units]
    assert ranks == sorted(ranks)
    if "instances" in cfg["kinds"]:
        assert kinds["instances"] == 5
    if name in ("single_trajectory", "tabpfn", "prediction_summary"):
        assert kinds["fields"] == FIELD_RUNS_65
        assert kinds["prediction"] == 0          # no frozen task table yet
    if name == "cross_trajectory":
        assert kinds["fields"] == FIELD_RUNS_65 + 3 * 100
    if name == "optimization":
        assert kinds["optimization"] == 5 * 21 * 20
        assert kinds["optimization_timing"] == 5 * 7 * 3
        keys = [k for kind, k in units if kind == "optimization"]
        assert {k["instance"] for k in keys} == set(range(20))
        variant = [k for k in keys if k["strategy"] == "decpfn" and k["params"].get("tau") == 0.99]
        assert variant and all(k["params"]["model"].startswith("decpfn/") for k in variant)
    if name == "priors":
        assert kinds["landscapes"] == 5 * 3 + 5
    if name == "generation_speed":
        assert kinds["generation_speed"] == 5


def test_participants_resolve():
    cfg = groups.load("prediction_summary")
    parts = groups.participants(cfg, "cartpole")
    assert "decpfn/cartpole/binned/1M" in parts and "segment-gp" in parts and len(parts) == 8


def test_runner_selects_spec_groups(pfn_home):
    sel = select_groups(["generation_speed"])
    units = plan(sel, Db())
    assert [u.kind for u in units] == ["generation_speed"] * 5
    assert plan(select_groups(["priors"]), Db(), kinds=["instances"])[0].kind == "instances"
    with pytest.raises(KeyError):
        select_groups(["no_such_group"])


def test_prediction_units_appear_once_tasks_are_frozen(pfn_home):
    from esbopfn.benchmarks import tasks as T
    from esbopfn.benchmarks.instances import BENCH_SEED

    df = T.build_tasks(_trials(), "boiler", can_crash=False)
    T.save_tasks(df, "boiler", T.table_name("tasks", "oracle65", BENCH_SEED))
    cfg = groups.load("prediction_summary")
    cfg["problems"] = ["boiler"]
    keys = [k for kind, k in groups.units(cfg) if kind == "prediction"]
    models = {k["model"] for k in keys}
    assert "decpfn/boiler/binned/1M" in models and "tabpfn/50" in models
    tab = next(k for k in keys if k["model"] == "tabpfn/50")
    assert tab["n"] == [2, 4, 8, 16, 32, 64] and tab["tasks_digest"] == T.digest(df)
    timing = [k for kind, k in groups.units(cfg) if kind == "prediction_timing"]
    assert len(timing) == 5 * 7 + 6 + 5 + 6


def _trials():
    import pandas as pd

    from esbopfn.benchmarks import tasks as T

    rows = [{"run_id": f"i{i:04d}_s00", "instance": i, "seed": 0, "trial_index": t, "T": 701,
             "t_obs": 701, "crash_step": 0, "J_T": float(i + t)}
            for i in range(50) for t in range(65)]
    return pd.DataFrame(rows, columns=T.TRIAL_COLUMNS)
