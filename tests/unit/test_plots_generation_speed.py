"""Plot group ``generation_speed`` on a synthetic database and model tree."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.plots import common as c
from esbopfn.plots import group_config
from esbopfn.plots.generation_speed import GROUP, fmt_seconds, fmt_share

_FAKEDB = Path(__file__).resolve().parents[1] / "fixtures" / "plots" / "fakedb.py"
_spec = importlib.util.spec_from_file_location("plots_fakedb", _FAKEDB)
fakedb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fakedb)


def _cfg() -> dict:
    return {**group_config("generation_speed"), "usetex": False}


def _add_speed(db: Db) -> None:
    for i, p in enumerate(c.PROBLEMS):
        fakedb.add_unit(db, "generation_speed",
                        {"problem": p, "count": 2000, "sets": 10, "budget_s": 120.0, "seed": 0},
                        {"speed.json": {"problem": p, "traj_s_per_10k": 0.27 + i,
                                        "traj_acceptance": 0.88, "system_acceptance": 0.015,
                                        "ftpfn_s_per_10k": 60.0 * (i + 1),
                                        "xdecpfn_s_per_10k": 3600.0 * 6, "single_trajectories": 2000,
                                        "xdecpfn_sets": 10, "ftpfn_sets": 10,
                                        "system_candidates": 100, "host": "test"}})


def _add_model(home: Path) -> None:
    d = home / "models" / "xdecpfn" / "cartpole" / "default" / "default" / "0"
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(b"x")
    (d / "metrics.json").write_text(json.dumps({"n_params": 11_000_000, "steps_done": 100,
                                                "flops_per_step": 2e15, "train_time_s": 1926.0}))


def test_formatting():
    assert fmt_seconds(0.27) == ("0.27", "s")
    assert fmt_seconds(21 * 60) == ("21", "min")
    assert fmt_seconds(5.7 * 3600) == ("5.7", "h")
    assert fmt_share(0.0136) == "0.0136" and fmt_share(0.65) == "0.65"


@pytest.mark.parametrize("theme", ["white", "black"])
@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_tables(pfn_home, tmp_path, fmt, theme):
    db = Db()
    _add_speed(db)
    _add_model(pfn_home)
    out = tmp_path / "out"
    GROUP.render(_cfg(), db, out, fmt, theme)
    md = (out / "prior_cost_table.md").read_text()
    tex = (out / "prior_cost_table.tex").read_text()
    assert "| Boiler | 0.27 s | 0.88 | 0.015 | 1.0 min | 6.0 h |" in md
    assert r"Boiler & $0.27$\,s & $0.88$ & $0.015$ & $1.0$\,min & $6.0$\,h \\" in tex
    assert r"\cmidrule(lr){2-3}" in tex
    mmd = (out / "model_cost_table.md").read_text()
    assert "| XDecPFN | 132,400 trajectory sets | 11.0 M | 2.00e17 | 32 min |" in mmd
    assert r"$2.00\times10^{17}$" in (out / "model_cost_table.tex").read_text()
    assert "| LC-PFN" not in mmd   # not trained here


def test_skips_without_units(pfn_home, tmp_path):
    out = tmp_path / "out"
    GROUP.render(_cfg(), Db(), out, "png", "white")
    skipped = (out / "skipped.txt").read_text()
    assert "prior_cost_table" in skipped and "generation_speed units" in skipped
    assert "model_cost_table" in skipped
