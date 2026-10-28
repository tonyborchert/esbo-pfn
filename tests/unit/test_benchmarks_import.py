"""Thesis result importers on small fixtures and on the local thesis files when present."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from esbopfn.benchmarks import import_thesis as I
from esbopfn.benchmarks.db import Db
from esbopfn.benchmarks.import_timing import import_timing_bundle
from esbopfn.benchmarks.kinds.thesis_series import read_series

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "thesis"
THESIS_ROOT = Path(__file__).resolve().parents[3]


def test_name_maps():
    assert I.map_model("decpfn/epochs100/headbinned", "cartpole") == "decpfn/cartpole/binned/1M"
    assert I.map_model("decpfn/epochs10/headlinear/states0") == "decpfn-nostates/{p}/linear/100k"
    assert I.map_model("lcpfn_zscore/epochs100/headbinned", "pt2_deadtime") == \
        "lcpfn/pt2_deadtime/binned/1M"
    assert I.map_model("lcpfn/epochs100/headbinned") == "lcpfn/{p}/binned/1M/prezscore"
    assert I.map_model("dynformer_c1r2/cellnmatched", "boiler") == "xdecpfn/boiler/thesis/default"
    assert I.map_model("dynformer_c1r2/celllnjw128", "furuta") == "xdecpfn/furuta/thesis/default"
    assert I.map_model("dynformer_fr2", "cartpole") == "ftpfn/cartpole/thesis/default"
    assert I.map_model("tabpfn/resolution50/typeraw") == "tabpfn/50"
    assert I.map_model("tabpfn/resolution50/typels") is None
    assert I.map_model("stengertpfn/modesplit") == "segment-tabpfn/split"
    assert I.map_model("stenger") == "segment-gp"
    assert I.map_problem("furuta_lqr_rigreset") == "furuta" and I.map_problem("furuta_lqr") is None
    assert I.map_cell("MTr 0.9", "cartpole")[0] == "decpfn"
    assert I.map_cell("HMTr 0.9", "pt2_deadtime")[2] == "decpfn-h"
    assert I.map_cell("QCn PI abs 0.99", "boiler")[:1] == ("xdecpfn-pi",)
    assert I.map_cell("QC PI abs 0.99", "boiler") is None
    assert I.map_cell("QC PI abs 0.99", "furuta")[1]["model"] == "xdecpfn/furuta/thesis/default"
    assert I.map_cell("Random uniform+cost", "boiler")[0] == "esrs"
    assert I.map_cell("RandomESBO", "furuta")[0] == "esrs"
    assert I.map_cell("Plain GP", "boiler")[0] == "bo"
    assert I.map_cell("Stenger ESBO", "boiler")[0] == "esbo-gp"
    assert I.map_cell("ESBO-TabPFN", "boiler")[0] == "esbo-t-tabpfn"
    assert I.map_cell("QF EI abs xgrid 0.99", "boiler")[0] == "ftpfn-ei"
    assert I.map_cell("Oracle BO", "boiler")[0] == "esbo-oracle"
    assert I.map_cell("MTr 0.99 ft0.99", "boiler")[1]["crash_threshold"] == 0.99


def test_fig1_csv(pfn_home):
    db = Db()
    n = I.import_fig1_csv(db, FIX / "fig1_cross.csv", "fig3_cross", "thesis", per_problem=False)
    assert n == 1
    (unit,) = db.query("thesis_series", figure="fig3_cross")
    assert unit.key["series"] == "ftpfn/{p}/thesis/default" and unit.key["provenance"] == "thesis"
    s = read_series(unit.blob_dir)
    assert s["data"]["n_ctx"] == [0, 2] and s["data"]["y"] == [0.17, 1.09]
    assert I.import_fig1_csv(db, FIX / "fig1_cross.csv", "fig3_cross", "thesis", False) == 0


def test_ladder_parquet(pfn_home):
    from esbopfn.benchmarks.kinds.prediction import PREDICTION_COLUMNS, load_predictions
    from esbopfn.benchmarks.metrics import ladder_curve

    db = Db()
    assert I.import_ladder_parquet(db, FIX / "cartpole__dynformer_fr2.parquet")
    (unit,) = db.query("prediction_ladder")
    assert unit.key["model"] == "ftpfn/cartpole/thesis/default" and unit.key["n"] == [0, 2]
    df = pd.read_parquet(unit.blob_dir / "predictions.parquet")
    assert list(df.columns) == PREDICTION_COLUMNS
    assert set(df.run_id) == {"i0000_s00", "i0003_s01"} and df.instance.max() == 3
    assert df.cutoff_fallback[df.cutoff_frac == 0.1].all()       # 125 > 100 on T = 1000
    curve = ladder_curve(load_predictions([unit]))
    assert curve.set_index("n_ctx")["mean"].to_dict() == pytest.approx({0: -7.0, 2: -6.0})


def test_timing_bundle(pfn_home):
    from esbopfn.benchmarks.kinds.prediction_timing import read_timing

    db = Db()
    assert import_timing_bundle(db, FIX / "timing_bundle") == 2
    units = {u.key["model"]: u for u in db.query("prediction_timing")}
    assert set(units) == {"decpfn/boiler/binned/1M", "xdecpfn/furuta/thesis/default"}
    t = read_timing(units["xdecpfn/furuta/thesis/default"].blob_dir)
    assert t["seconds_median"] == 0.05 and t["device"].startswith("cuda")
    assert units["xdecpfn/furuta/thesis/default"].key["source_model"] == \
        "dynformer_c1r2/cellf_s20260931"


def test_instances_json(pfn_home):
    from esbopfn.benchmarks.instances import load_instances

    db = Db()
    assert I.import_instances_json(db, FIX / "instances" / "boiler.json")
    (inst,) = load_instances("boiler", "thesis")
    assert inst.phi == {"heating_rate": 1.869, "cooling_rate": 0.1402}
    assert inst.noise_std == {"temperature": 0.0221}


def test_raw_optplay_run(pfn_home):
    from esbopfn.benchmarks.kinds.optimization import read_run

    db = Db()
    assert I.import_optplay_tree(db, FIX / "optplay" / "pt2_deadtime@g0.05@bcapped") == 1
    (unit,) = db.query("optimization")
    k = unit.key
    assert (k["problem"], k["instance"], k["strategy"], k["cell"]) == \
        ("pt2_deadtime", 3, "bo", "Plain GP")
    rec = read_run(unit.blob_dir)
    t = rec.trials
    assert [x.t_obs for x in t] == [12, 501, 200, 501]
    assert t[0].crashed and math.isinf(t[0].J_obs) and math.isinf(t[0].J_true)
    assert t[1].complete and t[1].J_obs == pytest.approx(1002.0)
    assert t[2].early_stopped and t[2].J_true == pytest.approx(1503.0)
    assert rec.incumbent_index == [-1, 1, 1, 3] and rec.incumbent_J[-1] == pytest.approx(501.0)
    assert rec.timing == {"gp_acq": 3.0} and rec.seed == 7
    assert [x.source for x in t] == ["init", "init", "init", "strategy"]   # n_init = 3 on PT2


def test_real_thesis_import(pfn_home):
    """Imports the local thesis result files (skipped when the old tree is not next to the repo)."""
    if not (THESIS_ROOT / "impl" / "dynformer" / "predplay").is_dir():
        pytest.skip("thesis result files not found")
    counts = I.import_thesis_results(THESIS_ROOT, Db())
    assert counts["instances"] == 5 and counts["tasks"] == 5
    assert counts["prediction_ladder"] == 7 and counts["prediction_timing"] > 300
    assert counts["thesis_series"] > 500
    db = Db()
    med = read_series(db.query("thesis_series", figure="fig3_med_D", variant="thesis")[0].blob_dir)
    assert dict(zip(med["data"]["problem"], med["data"]["med_D"]))["cartpole"] == \
        pytest.approx(-7.6728, abs=1e-3)
    ft = read_series(db.query("thesis_series", figure="fig3_cross", variant="thesis",
                              series="ftpfn/{p}/thesis/default")[0].blob_dir)
    assert ft["data"]["y"][ft["data"]["n_ctx"].index(64)] == pytest.approx(1.59, abs=5e-3)
    info = json.loads((pfn_home / "benchmarks" / "tasks" / "cartpole" / "tasks-thesis.json")
                      .read_text())
    assert info["n_selected"] == 350 and info["thesis_sha"] == "63de6db1acbaedec"
