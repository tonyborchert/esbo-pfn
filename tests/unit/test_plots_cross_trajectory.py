"""Plot group ``cross_trajectory`` on a synthetic database (examples excluded)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.plots import common, group_config

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "plots" / "fakedb.py"
FIGS = ["context_ladder", "ftpfn_xdecpfn_stop_sheet"]


def _fakedb():
    spec = importlib.util.spec_from_file_location("plots_fakedb", FIXTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg():
    return {**group_config("cross_trajectory"), "usetex": False, "only": FIGS}


def _ladder_rows(rungs, ll_of_n):
    import pandas as pd

    rows = []
    for n in rungs:
        for q in range(4):
            for f in (0.1, 0.3, 0.5, 0.7, 0.9):
                rows.append({"task_id": f"q{q}", "run_id": f"i{q:04d}_s00", "instance": q, "seed": 0,
                                 "n_ctx": n, "query_index": 180, "crashed": False, "crash_step": 0,
                                 "target_J": 1.0, "cutoff_frac": f, "cutoff_fallback": False,
                                 "cutoff_step": int(f * 100), "J_obs": 0.5,
                                 "ll": ll_of_n(n) + (0.2 if q % 2 else -0.2), "p_crash": 0.0,
                                 "dist": "bar", "edges": None, "probs": None, "mean": 1.0, "var": 0.1,
                                 "seconds": 0.01})
    return pd.DataFrame(rows)


def _fill() -> Db:
    fake = _fakedb()
    db = Db()
    rungs = {"pt2_deadtime": [0, 1, 2, 4, 8, 16, 33, 67, 134],
             "cartpole": [0, 1, 2, 5, 11, 22, 45, 90, 180],
             "furuta": [0, 1, 2, 5, 11, 22, 45, 90, 180]}
    for p, rs in rungs.items():
        for m, off in (("xdecpfn", 0.0), ("ftpfn", -1.0)):
            fake.add_prediction(db, "prediction_ladder", p, f"{m}/{p}/default/default",
                                _ladder_rows(rs, lambda n, o=off: -3.0 + o + 0.01 * n))
    # imported stop curves (one problem), per context size
    for m in ("xdecpfn", "ftpfn"):
        n_ctx, fs, saved = [], [], []
        for n in (0, 2, 64):
            n_ctx += [n] * 3
            fs += [0.0, 0.04, 0.3]
            saved += [0.0, 10.0 + n, 50.0 + n]
        fake.add_series(db, "fig2_stop_curves", "cartpole", f"{m}/cartpole/thesis/default",
                        {"n_ctx": n_ctx, "tau": [1.0, 0.5, 0.1] * 3, "false_stop_rate": fs,
                         "mean_steps_saved": saved, "n_crashed": [50] * 9, "n_ok": [80] * 9,
                         "stop_rate_crashed": [0.0, 0.5, 1.0] * 3})
    return db


@pytest.mark.parametrize("theme", ["white", "black"])
@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_renders(pfn_home, tmp_path, theme, fmt):
    from esbopfn.plots.cross_trajectory import GROUP

    db = _fill()
    out = tmp_path / "out"
    GROUP.render(_cfg(), db, out, fmt, theme)
    for name in FIGS:
        assert (out / f"{name}.{fmt}").stat().st_size > 0
        assert "Thesis asset:" in (out / f"{name}.md").read_text()
    assert "source: thesis import" in (out / "ftpfn_xdecpfn_stop_sheet.md").read_text()


def test_skips_without_data(pfn_home, tmp_path):
    from esbopfn.plots.cross_trajectory import GROUP

    out = tmp_path / "out"
    GROUP.render(_cfg(), Db(), out, "png", "white")
    skipped = (out / "skipped.txt").read_text()
    assert all(name in skipped for name in FIGS)
    assert "prediction_ladder units" in skipped


def _draw(name: str, db: Db, tmp_path: Path):
    from esbopfn.plots import style
    from esbopfn.plots.cross_trajectory import FIGURES

    spec = next(f for f in FIGURES if f.name == name)
    ctx = common.Context(cfg=_cfg(), db=db, out_dir=tmp_path, fmt="png", theme="white")
    with style.thesis_style("white", usetex=False):
        return spec.draw(ctx)


def test_ladder_known_mean(pfn_home, tmp_path):
    res = _draw("context_ladder", _fill(), tmp_path)
    cart = res.fig.axes[1]
    xdec = cart.lines[0]
    x, y = (np.asarray(v, float) for v in xdec.get_data())
    assert list(x) == list(range(9))                    # equidistant rungs
    assert np.isclose(y[-1], -3.0 + 0.01 * 180)         # mean over cutoffs, then queries
    assert np.isclose(cart.lines[1].get_ydata()[0], -4.0)
    assert [t.get_text() for t in cart.get_xticklabels()] == ["0", "2", "11", "45", "180"]


def test_stop_sheet_per_context(pfn_home, tmp_path):
    res = _draw("ftpfn_xdecpfn_stop_sheet", _fill(), tmp_path)
    cart = res.fig.axes[1]
    assert len(cart.lines) == 6 and not res.fig.axes[0].lines
    alphas = [ln.get_alpha() for ln in cart.lines[:3]]
    assert alphas == sorted(alphas)                    # fainter for less context
    assert np.isclose(max(cart.lines[2].get_ydata()), 114.0)
    assert any("n=64 74.0" in s for s in res.sidecar.numbers)
