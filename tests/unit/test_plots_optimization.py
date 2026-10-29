"""Plot group ``optimization`` on a small synthetic benchmark database."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.plots import common as C
from esbopfn.plots import group_config
from esbopfn.plots import optimization as O

_spec = importlib.util.spec_from_file_location(
    "plots_fakedb", Path(__file__).resolve().parents[1] / "fixtures" / "plots" / "fakedb.py")
fakedb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fakedb)

PROBLEMS = ["boiler", "cartpole"]
N_INST = 3
#: final cost per strategy entry (same on every instance), ESRS 10 and the best 1 give
#: scaled regret (J - 1) / 9.
FINALS = {
    ("esrs", ()): 10.0,
    ("bo", ()): 5.5,                     # scaled 0.5 -> log10 -0.301
    ("decpfn", ()): 2.8,                 # scaled 0.2
    ("xdecpfn-pi", ()): 1.0,             # the best -> floor
    ("ftpfn-ei", ()): 1.9,               # scaled 0.1 -> -1
    ("decpfn", (("crash_threshold", 0.9),)): 3.7,      # scaled 0.3
    ("xdecpfn-ei", (("tau", 0.9),)): 4.6,               # scaled 0.4
}
TIMING = {"gp_acq": 2.0, "segment_gp": 1.5, "model_acq": 0.0, "model_forecast": 0.25}


def _record(problem: str, strategy: str, params: dict, final: float, crashed_first: bool,
            timing: dict | None = None) -> dict:
    from esbopfn.problems.base import cached_spec

    spec = cached_spec(problem)
    t_budget = int(spec.t_budget())
    step = t_budget // 4
    incs = [math.inf, 1.5 * final, 1.2 * final, final]
    trials = []
    for k in range(4):
        crashed = crashed_first and k == 0
        trials.append(dict(theta=[0.0] * spec.d, t_obs=step, J_obs=math.inf if crashed else 1.0,
                           crashed=crashed, success=not crashed, complete=not crashed,
                           J_true=1.0, source="init" if k < 2 else "strategy", propose_s=0.0,
                           stop_rule_s=0.0, end_s=0.0, J=None))
    return dict(problem=problem, strategy=strategy, params=params, seed=0, instance_seed=0,
                T=int(spec.T), n_init=2, n_max=100, t_budget=t_budget, trials=trials,
                incumbent_index=[-1, 1, 2, 3], incumbent_J=incs, timing=timing or {},
                wall_s=1.0, version=1)


def _key(problem: str, inst: int, strategy: str, overrides: tuple, budget: str) -> dict:
    from esbopfn.strategies import default_params

    params = default_params(strategy, problem)
    params.update(dict(overrides))
    return {"problem": problem, "instance": inst, "strategy": strategy, "params": params,
            "seed": 0, "budget": budget, "instances_seed": 1}


def _write_run(db: Db, kind: str, key: dict, rec: dict) -> None:
    import json

    text = json.dumps(rec).replace("Infinity", "Infinity")
    fakedb.add_unit(db, kind, key, {"run.json": text})


def build_native(db: Db) -> None:
    for p in PROBLEMS:
        for i in range(N_INST):
            for (sid, over), final in FINALS.items():
                rec = _record(p, sid, dict(over), final, crashed_first=sid == "esrs")
                _write_run(db, "optimization", _key(p, i, sid, over, "capped"), rec)
        for sid in ("bo", "decpfn-h"):
            rec = _record(p, sid, {}, 1.0, False, timing=TIMING)
            _write_run(db, "optimization_timing", _key(p, 0, sid, (), "timing"), rec)


def build_thesis(db: Db, problem: str = "pt2_deadtime") -> None:
    inst = list(range(N_INST))
    for series, final in (("esrs", 10.0), ("bo", 5.5), ("xdecpfn-pi", 1.0)):
        fakedb.add_series(db, "optplay_finals", problem, series,
                          {"instance": inst, "incumbent_trials": [final] * N_INST,
                           "incumbent_time": [final] * N_INST,
                           "incumbent_capped": [final] * N_INST})
        fakedb.add_series(db, "optplay_crashes", problem, series,
                          {"time": [0.0, 500.0, 1000.0], "mean_crash_count": [None, 1.0, 2.5],
                           "mean_crash_frac": [None, 0.5, 0.5], "n_contributing": [0, 3, 3]},
                          meta={"t_budget": 1000})
    fakedb.add_series(db, "optplay_reference", problem, "J_best_all_cells",
                      {"instance": inst, "J_best": [1.0] * N_INST})


def _cfg(problems: list[str] = PROBLEMS) -> dict:
    cfg = copy.deepcopy(group_config("optimization"))
    cfg["problems"] = problems
    cfg["optimization"]["instances"] = N_INST
    cfg["figures"]["crashes"]["problems"] = [p for p in problems if p != "boiler"]
    cfg["usetex"] = False
    return cfg


@pytest.fixture
def native_db(pfn_home):
    db = Db()
    build_native(db)
    return db


def _ctx(db: Db, cfg: dict, tmp_path: Path) -> C.Context:
    return C.Context(cfg=cfg, db=db, out_dir=tmp_path, fmt="png", theme="white")


@pytest.mark.parametrize("theme", ["white", "black"])
@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_group_renders(native_db, tmp_path, fmt, theme):
    out = tmp_path / "fig"
    O.GROUP.render(_cfg(), native_db, out, fmt, theme)
    for spec in O.FIGURES:
        assert (out / f"{spec.name}.{fmt}").stat().st_size > 0, spec.name
        side = (out / f"{spec.name}.md").read_text()
        assert "Thesis asset:" in side and spec.name in side
    assert not (out / "skipped.txt").exists()


def test_skips_without_units(pfn_home, tmp_path):
    out = tmp_path / "fig"
    O.GROUP.render(_cfg(), Db(), out, "png", "white")
    skipped = (out / "skipped.txt").read_text().splitlines()
    assert len(skipped) == len(O.FIGURES)
    assert all("pfn bench run optimization" in line for line in skipped)


def test_thesis_fallback(pfn_home, tmp_path):
    db = Db()
    build_thesis(db)
    out = tmp_path / "fig"
    O.GROUP.render(_cfg(["pt2_deadtime"]), db, out, "png", "white")
    skipped = (out / "skipped.txt").read_text()
    assert "optplay_cross:" in skipped and "optplay_timing:" in skipped
    side = (out / "optplay_violin__cross_full.md").read_text()
    assert "source: thesis import" in side and "50 % budget not shown" in side
    assert "BO 100% -0.301" in side
    assert "XDecPFN-PI 100% -4.000 (n=3, 3 at floor)" in side
    assert "ESRS 2.50" in (out / "optplay_crashes.md").read_text()


def _hlines(ax) -> set[float]:
    return {round(float(line.get_ydata()[0]), 3) for line in ax.lines
            if len(line.get_ydata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]}


def test_violin_medians(native_db, tmp_path):
    with C.style.thesis_style("white", usetex=False):
        res = O.fig_violin_cross_full(_ctx(native_db, _cfg(), tmp_path))
    meds = _hlines(res.fig.axes[0])
    plt.close(res.fig)
    # 100 %: BO (5.5 - 1) / 9 = 0.5, FT-PFN-EI 0.1, XDecPFN-PI at the floor
    assert {-0.301, -1.0, -4.0, 0.0} <= meds
    # 50 %: after two of four trials the incumbent is 1.5 J, BO (8.25 - 1) / (10 - 1)
    assert "BO 50% -0.094" in "\n".join(res.sidecar.numbers)


def test_variant_violins(native_db, tmp_path):
    with C.style.thesis_style("white", usetex=False):
        dec = O.fig_decpfn_thresholds(_ctx(native_db, _cfg(), tmp_path))
        qx = O.fig_qx_variants(_ctx(native_db, _cfg(), tmp_path))
    assert round(math.log10(0.3), 3) in _hlines(dec.fig.axes[0])
    assert round(math.log10(0.4), 3) in _hlines(qx.fig.axes[0])
    plt.close("all")


def test_cross_curves(native_db, tmp_path):
    with C.style.thesis_style("white", usetex=False):
        res = O.fig_cross(_ctx(native_db, _cfg(), tmp_path))
        sheet = O.fig_cross_sheet(_ctx(native_db, _cfg(), tmp_path))
    time_ax = res.fig.axes[1]
    ys = {line.get_label(): line.get_ydata() for line in time_ax.lines}
    assert ys["BO"][-1] == pytest.approx(math.log10(0.5))
    assert len([a for a in sheet.fig.axes if a.get_title()]) == 1 + len(PROBLEMS)
    plt.close("all")


def test_crash_counts(native_db, tmp_path):
    with C.style.thesis_style("white", usetex=False):
        res = O.fig_crashes(_ctx(native_db, _cfg(), tmp_path))
    ys = {line.get_label(): line.get_ydata() for line in res.fig.axes[0].lines}
    assert ys["ESRS"][-1] == pytest.approx(1.0) and ys["BO"][-1] == pytest.approx(0.0)
    plt.close("all")


def test_timing_bars(native_db, tmp_path):
    with C.style.thesis_style("white", usetex=False):
        res = O.fig_timing(_ctx(native_db, _cfg(), tmp_path))
    ax = res.fig.axes[0]
    tops = [p.get_y() + p.get_height() for p in ax.patches]
    assert max(tops) == pytest.approx(sum(TIMING.values()))
    assert "BO: boiler 3.8 s" in "\n".join(res.sidecar.numbers)
    plt.close("all")


def test_cell_matching():
    cell = O._cell({"id": "decpfn", "params": {"tau": 0.99}})
    assert O.matches(_key("boiler", 0, "decpfn", (("tau", 0.99),), "capped"), cell, "boiler")
    assert not O.matches(_key("boiler", 0, "decpfn", (), "capped"), cell, "boiler")
    assert not O.matches(_key("boiler", 0, "decpfn", (("tau", 0.99), ("crash_threshold", 0.9)),
                              "capped"), cell, "boiler")
