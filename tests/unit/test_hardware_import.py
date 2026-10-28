"""Import of the thesis rig logs and hardware prediction results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.benchmarks.import_hardware import (CELL_MAP, THESIS_CELL_PARAMS, import_rig_logs,
                                                import_hwpred_results, thesis_key)
from esbopfn.hardware import records
from esbopfn.hardware.store import load_runs
from esbopfn.hardware.system import stage_cost

FIXTURE = (Path(__file__).resolve().parents[1] / "hardware" / "fixtures" / "hwplay_tree"
           / "runs_holdcost5_newbox_reset_fixed")
IMPL = Path(__file__).resolve().parents[3] / "impl"     # the old checkout next to this repo
REAL_TREE = IMPL / "prediction" / "hwplay" / "runs_holdcost5_newbox_reset_fixed_different_plant"
REAL_PRED = IMPL / "dynformer" / "experiments" / "hwpred" / "results"


def test_cell_map():
    assert CELL_MAP == {"random_stenger": "esrs", "plain_gp": "bo", "stenger": "esbo-gp",
                        "mtr0.99_k06binned": "decpfn", "htmr0.99_k06binned": "decpfn-h",
                        "qc_pi_0.99": "xdecpfn-pi"}
    assert THESIS_CELL_PARAMS["mtr0.99_k06binned"]["matched_threshold"] == 0.99
    assert THESIS_CELL_PARAMS["htmr0.99_k06binned"]["fail_threshold"] == 0.9


def test_import_fixture(pfn_home):
    keys = import_rig_logs(FIXTURE)                 # rig from the tree name
    assert len(keys) == 1                           # qc_pi_0.99_n10k is dropped
    key = keys[0]
    assert key == thesis_key("without_weight", "esbo-gp", 0)
    assert key["provenance"] == "thesis" and key["params"]["cell"] == "stenger"
    assert key["n_max"] == 180 and key["t_budget"] == 36000
    db = Db()
    assert db.has("hardware_optimization", key)
    blob = db.blob_dir("hardware_optimization", key)
    recs = records.read_trials(blob)
    old = [json.loads(line) for line in
           (FIXTURE / "lqr_hardware" / "stenger" / "seed0" / "trials.jsonl").read_text()
           .splitlines()]
    assert [r.index for r in recs] == [o["index"] for o in old] == [15, 16]
    early, crash = recs
    assert early.outcome == "early_stopped" and early.crash_step is None
    assert early.t_obs == old[0]["t"] == len(early.J) == len(early.x) == len(early.xdot)
    assert early.J == old[0]["J"] and early.J_obs == old[0]["J"][-1]
    np.testing.assert_allclose(early.x, np.asarray(old[0]["Y"])[:, :2])
    np.testing.assert_allclose(early.xdot, np.asarray(old[0]["Y"])[:, 2:4])
    assert early.source == "strategy" and early.u is None
    assert early.kick_sign == old[0]["recovery"]["kick_sign"]
    assert early.timings["propose_s"] == old[0]["next_theta_seconds"]
    assert early.timings["reset_s"] == old[0]["reset_seconds"]
    assert early.timings["total_s"] == pytest.approx(
        old[0]["prepare_seconds"] + old[0]["trial_seconds"] + old[0]["next_theta_seconds"]
        + old[0]["observe_seconds"])
    assert early.reset["kind"] == "fast" and "settle_brake" in early.reset["phase_s"]
    # crashed: J_obs inf, last J entry without the old crash tail penalty
    assert crash.outcome == "crashed" and crash.crash_step == crash.t_obs == old[1]["t"]
    assert math.isinf(crash.J_obs) and crash.extra["crash_tail_J"] == old[1]["J"][-1]
    expected = old[1]["J"][-2] + float(stage_cost(old[1]["Y"][-1][1]))
    assert crash.J[-1] == pytest.approx(expected, rel=1e-6)
    traj = crash.trajectory()
    assert traj.crash_step == crash.t_obs and np.isinf(traj.J[crash.t_obs - 1])
    assert crash.trial().crashed and not early.trial().crashed
    run = records.read_run(blob)
    assert run["provenance"] == "thesis" and run["source"]["cell"] == "stenger"
    # idempotent
    assert import_rig_logs(FIXTURE, "without_weight") == keys
    assert len(load_runs("without_weight", db, provenance="thesis")) == 1


def test_import_needs_a_known_rig(tmp_path, pfn_home):
    with pytest.raises(ValueError):
        import_rig_logs(tmp_path)
    with pytest.raises(KeyError):
        import_rig_logs(FIXTURE, "no_such_rig")


@pytest.mark.skipif(not REAL_TREE.exists(), reason="thesis rig logs not present")
def test_import_real_tree_one_cell(pfn_home):
    keys = import_rig_logs(REAL_TREE, cells=["qc_pi_0.99"], seeds=[0])
    assert len(keys) == 1 and keys[0]["rig"] == "with_weight"
    runs = load_runs("with_weight", provenance="thesis")
    (run,) = runs
    assert run.method == "xdecpfn-pi" and len(run.trials) == 95
    # the thesis' best cost on the rig with weight came from this run
    assert min(t.J_obs for t in run.trials if t.success) == pytest.approx(256.917, abs=1e-3)
    assert sum(t.crashed for t in run.trials) == 3


@pytest.mark.skipif(not REAL_PRED.exists(), reason="thesis hwpred results not present")
def test_import_hwpred_results(pfn_home):
    keys = import_hwpred_results(REAL_PRED)
    assert len(keys) == 14
    db = Db()
    (unit,) = db.query("hardware_prediction", rig="with_weight", bench="ladder",
                       predictor="xdecpfn")
    summary = json.loads((unit.blob_dir / "summary.json").read_text())
    # thesis hardware results, with weight, XDecPFN at n = 0 and n = 90 (all queries)
    assert summary["median_ll_all"]["0"] == pytest.approx(-5.095, abs=1e-3)
    assert summary["median_ll_all"]["90"] == pytest.approx(-0.809, abs=1e-3)
    assert summary["seconds_per_prediction"]["0"] > 0
    (abl,) = db.query("hardware_prediction", rig="with_weight", bench="ablation",
                      predictor="decpfn-nostates")
    s = json.loads((abl.blob_dir / "summary.json").read_text())
    assert s["median_ll"]["full"] == pytest.approx(-4.129, abs=1e-3)
    assert s["n_scored"]["full"] == 508
