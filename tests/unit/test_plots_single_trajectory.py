"""Plot group ``single_trajectory`` on a synthetic database (examples excluded)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esbopfn.benchmarks import metrics
from esbopfn.benchmarks.db import Db
from esbopfn.plots import common, group_config

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "plots" / "fakedb.py"
FIGS = ["lcpfn_decpfn_heads_benchmark", "lcpfn_decpfn_stop_sheet", "gridsearch_panels"]


def _fakedb():
    spec = importlib.util.spec_from_file_location("plots_fakedb", FIXTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(only=FIGS):
    return {**group_config("single_trajectory"), "usetex": False, "only": list(only)}


def _arms():
    from esbopfn.plots.single_trajectory import HEADS, SERIES, SIZES
    return [(f, h, s) for f, *_ in SERIES for h in HEADS for s in SIZES]


def _fill(home: Path) -> Db:
    fake = _fakedb()
    db = Db()
    rng = np.random.default_rng(0)
    # thesis aggregates of the heads sheet: value = index / 10 on boiler and cartpole
    for p in ("boiler", "cartpole"):
        for i, (f, h, s) in enumerate(_arms()):
            fake.add_series(db, "fig3_per_problem", p, f"{f}/{p}/{h}/{s}",
                            {"n_ctx": [-1], "y": [i / 10.0], "n": [50]})
    # native stop-grid rows of DecPFN 1M binned on cartpole (overrides nothing in the import)
    rows = fake.prediction_rows(rng, [0, 2], 10, -5.0)
    fake.add_prediction(db, "prediction", "cartpole", "decpfn/cartpole/binned/1M", rows)
    # thesis stop curve of LC-PFN 1M on pt2
    fake.add_series(db, "fig2_stop_curves", "pt2_deadtime", "lcpfn/pt2_deadtime/binned/1M",
                    {"n_ctx": [-1, -1, -1], "tau": [1.0, 0.5, 0.1],
                     "false_stop_rate": [0.0, 0.05, 0.5], "mean_steps_saved": [0.0, 12.0, 30.0],
                     "n_crashed": [5, 5, 5], "n_ok": [10, 10, 10],
                     "stop_rate_crashed": [0.0, 0.5, 1.0]})
    # HPO grid
    for m in ("decpfn", "lcpfn"):
        for size, t in (("100k", 5.0), ("1M", 50.0)):
            d = home / "hpo" / m / "cartpole" / size
            d.mkdir(parents=True)
            pd.DataFrame({"stage": ["stage1", "stage2", "stage2"], "name": ["a", "b", "c"],
                          "head": ["linear", "binned", "gaussian"], "d_model": [256, 256, 512],
                          "n_layers": [1, 2, 3], "val_nll": [9.0, 6.0, 6.5],
                          "train_time_s": [t, t, 2 * t], "n_params": [1e6, 2e6, 2e7]}
                         ).to_csv(d / "trials.csv", index=False)
    return db


@pytest.mark.parametrize("theme", ["white", "black"])
@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_renders(pfn_home, tmp_path, theme, fmt):
    from esbopfn.plots.single_trajectory import GROUP

    db = _fill(pfn_home)
    out = tmp_path / "out"
    GROUP.render(_cfg(), db, out, fmt, theme)
    for name in FIGS:
        assert (out / f"{name}.{fmt}").stat().st_size > 0
        side = (out / f"{name}.md").read_text()
        assert "Thesis asset:" in side
    assert "source: thesis import" in (out / "lcpfn_decpfn_heads_benchmark.md").read_text()
    assert not (out / "skipped.txt").exists()


def test_skips_without_data(pfn_home, tmp_path):
    from esbopfn.plots.single_trajectory import GROUP

    out = tmp_path / "out"
    GROUP.render(_cfg(), Db(), out, "png", "white")
    skipped = (out / "skipped.txt").read_text()
    for name in FIGS:
        assert name in skipped
    assert "pfn hpo" in skipped and "pfn bench run single_trajectory" in skipped


def _draw(name: str, db: Db, tmp_path: Path):
    from esbopfn.plots import style
    from esbopfn.plots.single_trajectory import FIGURES

    spec = next(f for f in FIGURES if f.name == name)
    ctx = common.Context(cfg=_cfg(), db=db, out_dir=tmp_path, fmt="png", theme="white")
    with style.thesis_style("white", usetex=False):
        return spec.draw(ctx)


def test_heads_centered_values(pfn_home, tmp_path):
    db = _fill(pfn_home)
    res = _draw("lcpfn_decpfn_heads_benchmark", db, tmp_path)
    pts = np.concatenate([c.get_offsets() for c in res.fig.axes[0].collections])
    n = len(_arms())
    assert len(pts) == 2 * n
    boiler = pts[pts[:, 0] < 0.5, 1]           # slot 0, thesis values only
    mean = np.mean(np.arange(n) / 10.0)
    assert np.isclose(boiler.max(), (n - 1) / 10.0 - mean)
    assert f"boiler {mean:.3f}" in res.sidecar.numbers[0]
    # cartpole DecPFN 1M binned comes from the native unit (scores around -5), not the import
    assert "native" in res.sidecar.data[0] and "thesis" in res.sidecar.data[0]


def test_stop_sheet_native_and_thesis(pfn_home, tmp_path):
    db = _fill(pfn_home)
    rows = common.prediction_frame(db, "prediction", ["decpfn/{p}/binned/1M"], ["cartpole"])
    want = metrics.steps_saved_at(metrics.stop_curve(rows), 0.1)
    res = _draw("lcpfn_decpfn_stop_sheet", db, tmp_path)
    pt2, cart, _ = res.fig.axes[:3]
    assert len(cart.lines) == 1 and len(pt2.lines) == 1
    x, y = (np.asarray(v, float) for v in cart.lines[0].get_data())
    assert np.isclose(y[x <= 0.1].max(), want)
    assert np.isclose(max(pt2.lines[0].get_ydata()), 30.0)   # whole curve kept, axis zoomed
    assert pt2.get_xlim() == (0.0, 0.1)


def test_gridsearch_point(pfn_home, tmp_path):
    db = _fill(pfn_home)
    res = _draw("gridsearch_panels", db, tmp_path)
    ax100k = res.fig.axes[0]
    pts = np.concatenate([c.get_offsets() for c in ax100k.collections])
    assert len(pts) == 4                       # stage-2 rows only, two models
    assert any(np.allclose(p, [5.0, -6.0]) for p in pts)
    assert ax100k.get_xlim()[1] == 25.0
