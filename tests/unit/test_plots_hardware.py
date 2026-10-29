"""The hardware and README plot groups on a tiny synthetic benchmark database."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.benchmarks.kinds.hardware_prediction import unit_key as pred_key
from esbopfn.hardware import records
from esbopfn.hardware.prediction_bench import FULL, SCORE_COLUMNS, summarize, write_blob
from esbopfn.hardware.records import HwTrial
from esbopfn.hardware.runner import unit_key as opt_key
from esbopfn.plots import discover_groups, group_config
from esbopfn.plots import hardware as H
from esbopfn.plots import readme_art as R
from esbopfn.predictors.base import BarPrediction

RIGS = ("without_weight", "with_weight")
METHODS = {"esbo-gp": (0.0, 2.0), "decpfn-h": (10.0, 3.0), "xdecpfn-pi": (30.0, 0.05)}
SEEDS = (0, 1)
#: (outcome, final cost before the method / seed / rig offsets)
SEQUENCE = (("ok", 400.0), ("crash", None), ("ok", 330.0), ("early", None), ("ok", 300.0),
            ("ok", 320.0))
THETA = [80.0, 1.0, 12.0, 5.0, 1.0, 1.0, 1.0, 2.0]


def _trial(i: int, kind: str, J_final: float | None, propose: float, reset: float) -> HwTrial:
    T = 600
    t_obs = {"ok": T, "crash": 100, "early": 200}[kind]
    J = np.linspace(1.0, J_final if J_final else 150.0, t_obs)
    arm = 0.5 * np.sin(np.linspace(0, 6, t_obs))
    pend = np.linspace(np.pi, 0.0, t_obs)
    x = np.stack([arm, pend], 1).tolist()
    outcome = {"ok": records.COMPLETE, "crash": records.CRASHED,
               "early": records.EARLY_STOPPED}[kind]
    theta = list(THETA)
    theta[0] = 60.0 + 5.0 * i
    return HwTrial(index=i, theta=theta, source="init", T=T, t_obs=t_obs, outcome=outcome,
                   crash_step=t_obs if kind == "crash" else None, success=kind == "ok",
                   J_obs=math.inf if kind == "crash" else float(J[-1]), J=J.tolist(), x=x,
                   xdot=x, u=None, kick_sign=1.0, incumbent_index=-1, incumbent_J=math.inf,
                   timings={"propose_s": propose, "reset_s": reset, "control_s": 5.0,
                            "total_s": propose + reset + 5.0})


def _scores(rig: str, predictor: str, ll_fn, secs_fn, ladder) -> pd.DataFrame:
    rows = []
    for n in ladder:
        for seed in SEEDS:
            for q in range(1, 5):
                ll = ll_fn(n)
                if ll is None:
                    continue
                rows.append({"rig": rig, "method": "esbo-gp", "seed": seed, "query_index": q,
                             "n_ctx": n, "n_ctx_used": max(n, 0), "cutoff_frac": 0.5,
                             "cutoff_step": 300, "step": 300, "predictor": predictor,
                             "ll": ll, "p_crash": 0.1, "crashed": False, "target_J": 300.0,
                             "query_t": 600, "infer_s": secs_fn(n)})
    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))


LADDER = [0, 1, 2, 5]
LADDER_FNS = {
    "segment-gp": (lambda n: None if n == 0 else -6.0 + 0.5 * n, lambda n: 0.1 * (n + 1)),
    "decpfn": (lambda n: -3.5, lambda n: 0.003),
    "xdecpfn": (lambda n: -4.0 + 0.5 * n, lambda n: 0.01 * (n + 1)),
}
ABLATION = {"decpfn": -4.0, "decpfn-nostates": -3.9, "decpfn-nodeadtime": -3.95,
            "decpfn-nodeadtime-nostates": -4.1}


@pytest.fixture(scope="module")
def hw_db(tmp_path_factory) -> Db:
    root = tmp_path_factory.mktemp("hwdb")
    db = Db(root / "db.sqlite")
    for r_off, rig in enumerate(RIGS):
        for m, (m_off, propose) in METHODS.items():
            for s in SEEDS:
                reset = 9.0 if m == "decpfn-h" else 12.0
                trials = [_trial(i, kind, None if J is None else J - m_off + 5.0 * s
                                 + 10.0 * r_off, propose, reset)
                          for i, (kind, J) in enumerate(SEQUENCE)]
                key = opt_key(rig, m, s, {}, provenance="thesis")

                def write(key, blob, trials=trials):
                    records.write_run(blob, {"key": key})
                    records.write_trials(blob / records.TRIALS_FILE, trials)

                db.run("hardware_optimization", key, write)
        for p, (ll_fn, secs_fn) in LADDER_FNS.items():
            df = _scores(rig, p, ll_fn, secs_fn, LADDER)
            key = pred_key(rig, "ladder", p, f"{p}/thesis", ladder=LADDER, logs="thesis",
                           provenance="thesis")
            db.run("hardware_prediction", key,
                   lambda k, b, df=df: write_blob(b, df, summarize(df)))
        for p, v in ABLATION.items():
            df = _scores(rig, p, lambda n, v=v: v, lambda n: math.nan, [FULL])
            key = pred_key(rig, "ablation", p, f"{p}/thesis", logs="thesis",
                           provenance="thesis")
            db.run("hardware_prediction", key,
                   lambda k, b, df=df: write_blob(b, df, summarize(df)))
    series = {"figure": "fig3_cross", "problem": "pooled", "series": "xdecpfn/{p}/thesis/default",
              "variant": "thesis", "provenance": "thesis"}

    def write_series(key, blob):
        (blob / "series.json").write_text(json.dumps(
            {**key, "data": {"n_ctx": [0, 64], "y": [0.5, 1.25]}}))

    db.run("thesis_series", series, write_series)
    return db


def _cfg() -> dict:
    return {**group_config("hardware"), "usetex": False}


NAMES = ["hwplay_optimization__without_weight", "hwplay_optimization__with_weight",
         "hwplay_regret_violin", "hwplay_cost_1000", "hwplay_best_controller__500s__without_weight",
         "hwplay_best_controller__500s__with_weight", "hwplay_acq_wait", "hwpred_benchmark",
         "hwpred_ablation_bars"]


def test_groups_registered():
    groups = discover_groups()
    assert "hardware" in groups and "readme" in groups


@pytest.mark.parametrize("fmt,theme", [("pdf", "white"), ("png", "white"), ("pdf", "black"),
                                       ("png", "black")])
def test_render_hardware(hw_db, tmp_path, fmt, theme):
    res = H.render(_cfg(), hw_db, tmp_path, fmt, theme)
    for name in NAMES:
        assert (tmp_path / f"{name}.{fmt}").stat().st_size > 0, name
        assert (tmp_path / f"{name}.md").read_text().startswith(f"# {name}")
    if fmt != "pdf" or theme != "white":
        return
    # one known number per figure
    opt = res["optimization"]["without_weight"]
    assert opt["J_best"] == pytest.approx(270.0)            # xdecpfn-pi seed 0, trial 4
    assert opt["reset_ref_s"] == pytest.approx(9.0)          # decpfn-h resets
    assert opt["crash_counts"]["esbo-gp"] == {0: 1, 1: 1}
    assert "J_best = 270.000" in (tmp_path / f"{NAMES[0]}.md").read_text()
    assert res["optimization"]["with_weight"]["J_best"] == pytest.approx(280.0)
    vio = res["regret_violin"]["without_weight"]["methods"]["xdecpfn-pi"][1.0]
    assert vio["excluded_seeds"] == [0]                      # seed 0 defines J_best
    assert vio["values"] == pytest.approx([math.log10(5.0)])
    cost = res["cost_window"]["without_weight"]["methods"]["esbo-gp"]
    assert np.nanmin(cost["median"]) == pytest.approx(302.5)
    bc = res["best_controller"]["without_weight"]
    assert bc["xdecpfn-pi"]["J"] == pytest.approx(270.0)     # lower middle seed of two
    acq = res["acq_wait"]["with_weight"]
    assert acq["decpfn-h"]["mean"] == pytest.approx(3.0)
    assert "esrs" not in acq
    pred = res["pred_benchmark"]["per_rig"]["without_weight"]
    assert pred["decpfn"]["points"][0][2] == pytest.approx(-3.5)     # pooled point
    assert pred["decpfn"]["points"][0][0] is None
    front = res["pred_benchmark"]["drawn"]["without_weight"]["series"]["xdecpfn"]
    assert [p[0] for p in front] == LADDER                  # every rung improves
    abl = res["ablation"]["with_weight"]
    assert abl["decpfn-nostates"][0] == pytest.approx(-3.9)


def test_pareto_and_needs(hw_db):
    pts = [(0, 0.01, -4.0, 1), (1, 0.02, -4.5, 1), (2, 0.03, -3.0, 1)]
    assert [p[0] for p in H._pareto(pts)] == [0, 2]
    needs = H.needs(_cfg(), hw_db)
    # the fixture's rig runs carry no thesis cell parameters, so only prediction units match
    assert {k for k, _ in needs} == {"hardware_prediction"}
    assert len(needs) == 2 * (3 + 4)


def test_empty_database_renders_nothing(tmp_path):
    db = Db(tmp_path / "empty.sqlite")
    assert H.needs(_cfg(), db) == []
    assert H.render(_cfg(), db, tmp_path / "hw", "png", "white") == {}
    assert R.render_all({}, db, tmp_path / "readme") == {}


class FakePredictor:
    """Cost falls along the first controller coordinate, crash probability with it."""

    def predict_unrun(self, thetas, context):
        edges = np.linspace(0.0, 1000.0, 201)
        out = []
        for th in np.asarray(thetas):
            mu = 600.0 - 4.0 * (th[0] - 40.0)
            p = np.exp(-0.5 * ((edges[:-1] + 2.5 - mu) / 40.0) ** 2)
            pc = float(np.clip((90.0 - th[0]) / 100.0, 0.0, 0.9))
            out.append(BarPrediction(edges, p / p.sum() * (1.0 - pc), pc))
        return out


def test_render_readme(hw_db, tmp_path, monkeypatch):
    calls = []

    def fake_get_predictor(model_id, spec, **kw):
        calls.append(model_id)
        return FakePredictor()

    monkeypatch.setattr(R, "get_predictor", fake_get_predictor)
    cfg = {"hero": {"method": "esbo-gp", "trials": [[0, 4, 1.0], [1, 2, 0.4], [0, 2, 0.25]],
                    "extra_context": [1, [0, 5]], "n_slice": 9},
           "strip": {"problems": ["boiler", "pt2_deadtime"], "n_traj": 3}}
    res = R.render_all(cfg, hw_db, tmp_path, variants=("light", "dark"))
    for name in ("hero_3d", "problems_strip", "results_cards"):
        for suffix in ("", "_dark"):
            for ext in ("png", "svg"):
                assert (tmp_path / f"{name}{suffix}.{ext}").stat().st_size > 0
        assert (tmp_path / f"{name}.md").exists()
    assert calls == ["xdecpfn/furuta/thesis/default"]
    hero = res["hero"]
    assert hero.dim == 0 and hero.theta_name == "mu"         # the fake varies along theta 0 only
    assert hero.slice_x[hero.next_index] == pytest.approx(90.0)
    assert [t["stopped"] for t in hero.trials] == [False, True, True]
    assert res["cards"]["prediction"][0]["value"] == pytest.approx(1.25)
    hw = {r["method"]: r for r in res["cards"]["hardware"]["without_weight"]}
    assert hw["esbo-gp"]["crashes"] == pytest.approx(1.0)
    assert "optimization esrs" in res["cards"]["missing"]


def test_sidecar_helper(tmp_path):
    from esbopfn.plots import _hw_common as C

    p = C.write_sidecar(Path(tmp_path), "x", shown=["a"], metric=["b"], data=["c"],
                        numbers=["d"])
    text = p.read_text()
    assert "## Numbers" in text and "- d" in text
    assert C.fmt_num(math.inf) == "inf" and C.fmt_list([1, 2], 1) == "[1.0, 2.0]"
