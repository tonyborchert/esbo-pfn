"""Shared plot helpers (`esbopfn.plots.common`) and group discovery."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.plots import common, discover_groups

FAKEDB = Path(__file__).resolve().parents[1] / "fixtures" / "plots" / "fakedb.py"


def load_fakedb():
    spec = importlib.util.spec_from_file_location("plots_fakedb", FAKEDB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fakedb = load_fakedb()

GROUPS = {"priors", "single_trajectory", "tabpfn", "cross_trajectory", "prediction_summary",
          "optimization", "generation_speed"}


def test_groups_discovered():
    groups = discover_groups()
    assert GROUPS <= set(groups)
    assert "common" not in groups and "examples" not in groups


def _line_figure(ctx: common.Context) -> common.Output:
    fig, axes = common.panel_sheet(1, 3)
    for ax in axes.flat:
        ax.plot([0, 1], [0, 2], color=common.model_color("decpfn/cartpole/binned/1M", ctx.theme))
        common.finish(ax, xlabel="x", ylabel=common.better("LL", True))
    prov = common.Provenance()
    prov.notes.append("synthetic")
    return common.Output(common.Sidecar("simulation/x.pdf", label="fig:x", shown=["a line"],
                                        numbers=["y(1) = 2"], provenance=prov), fig=fig)


def _missing_figure(ctx: common.Context) -> common.Output:
    raise common.MissingData(["prediction units of decpfn/{p}/binned/1M"], "run `pfn bench run x`")


def _broken_figure(ctx: common.Context) -> common.Output:
    raise ValueError("boom")


@pytest.mark.parametrize("theme", ["white", "black"])
@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_render_group_writes_and_skips(pfn_home, tmp_path, theme, fmt):
    figures = [common.FigureSpec("line", _line_figure), common.FigureSpec("gone", _missing_figure)]
    report = common.render_group("demo", figures, {"usetex": False}, Db(), tmp_path, fmt, theme)
    assert (tmp_path / f"line.{fmt}").stat().st_size > 0
    side = (tmp_path / "line.md").read_text()
    assert "writing/thesis/assets_generated/simulation/x.pdf" in side and "y(1) = 2" in side
    assert "gone" in report.skipped and not (tmp_path / "gone.md").exists()
    assert "prediction units" in (tmp_path / "skipped.txt").read_text()


def test_render_group_failure_does_not_stop_others(pfn_home, tmp_path):
    figures = [common.FigureSpec("bad", _broken_figure), common.FigureSpec("line", _line_figure)]
    with pytest.raises(RuntimeError, match="bad"):
        common.render_group("demo", figures, {"usetex": False}, Db(), tmp_path, "png", "white")
    assert (tmp_path / "line.png").exists()


def test_render_group_only(pfn_home, tmp_path):
    figures = [common.FigureSpec("bad", _broken_figure), common.FigureSpec("line", _line_figure)]
    report = common.render_group("demo", figures, {"usetex": False, "only": ["line"]}, Db(),
                                 tmp_path, "png", "white")
    assert list(report.written) == ["line"]


def test_text_outputs(pfn_home, tmp_path):
    def table(ctx):
        return common.Output(common.Sidecar("tab.tex"), texts={"t.md": "| a |\n", "t.tex": "a\\\\"})

    common.render_group("demo", [common.FigureSpec("t", table)], {"usetex": False}, Db(),
                        tmp_path, "pdf", "white")
    text = (tmp_path / "t.md").read_text()
    assert text.startswith("# t") and text.rstrip().endswith("| a |")  # table after the sidecar
    assert (tmp_path / "t.tex").exists()


def test_panel_sheet_geometry():
    with plt.rc_context({"text.usetex": False}):
        fig, axes = common.panel_sheet(2, 3, n=5)
    assert axes[1, 2] is None
    w_pt = fig.get_size_inches()[0] * 72
    h_pt = fig.get_size_inches()[1] * 72
    box = axes[0, 0].get_position()
    assert box.width * w_pt == pytest.approx(common.box_width_pt(3))
    assert box.height * h_pt == pytest.approx(common.BOX_H_PT)
    gap = (axes[0, 1].get_position().x0 - box.x1) * w_pt
    assert gap == pytest.approx(common.GAP_PT)
    plt.close(fig)
    fig, axes = common.panel_sheet(1, 2, as_cols=3)
    centers = [(a.get_position().x0 + a.get_position().x1) / 2 for a in axes.flat]
    lefts = axes[0, 0].get_position().x0
    span_mid = (lefts + axes[0, 1].get_position().x1) / 2
    assert np.mean(centers) == pytest.approx(span_mid)
    plt.close(fig)


def test_thesis_series_and_timing(pfn_home):
    db = Db()
    fakedb.add_series(db, "fig3_cross", "pooled", "tabpfn/50", {"n_ctx": [2, 4], "y": [0.1, 0.4]})
    fakedb.add_timing(db, "cartpole", "xdecpfn/cartpole/thesis/default", 4, 0.02)
    fakedb.add_timing(db, "cartpole", "lcpfn/cartpole/binned/1M/prezscore", 0, 0.004)
    fakedb.add_timing(db, "boiler", "xdecpfn/boiler/thesis/default", 4, 0.04)
    fakedb.add_timing(db, "boiler", "xdecpfn/boiler/default/default", 4, 0.5, provenance=None)
    prov = common.Provenance()
    blobs = common.thesis_series(db, "fig3_cross", prov=prov)
    assert len(blobs) == 1 and blobs[0].frame.y.tolist() == [0.1, 0.4]
    assert prov.thesis_import and "source: thesis import" in prov.lines()[0]
    times = common.timing_table(db)
    assert common.seconds_for(times, "xdecpfn/cartpole/default/default", "cartpole", 4) == 0.02
    assert common.seconds_for(times, "xdecpfn/boiler/default/default", "boiler", 4) == 0.5
    assert common.seconds_for(times, "lcpfn/cartpole/binned/1M", "cartpole", 64) == 0.004
    assert common.seconds_for(times, "xdecpfn/{p}/default/default", None, 4) == pytest.approx(0.26)
    assert np.isnan(common.seconds_for(times, "ftpfn/cartpole/default/default", "cartpole", 4))


def test_prediction_frame_thesis_fallback(pfn_home):
    db = Db()
    rows = fakedb.prediction_rows(np.random.default_rng(0), [0], 4, -3.0, stop_grid=False)
    fakedb.add_prediction(db, "prediction_ladder", "cartpole", "ftpfn/cartpole/thesis/default",
                          rows)
    df = common.prediction_frame(db, "prediction_ladder", ["ftpfn/{p}/default/default"],
                                 ["cartpole", "furuta"])
    assert set(df.model) == {"ftpfn/cartpole/default/default"}
    assert common.prediction_frame(db, "prediction_ladder", ["ftpfn/{p}/default/default"],
                                   ["cartpole"], native_only=True).empty


def test_draw_front_staircase():
    pts = pd.DataFrame({"seconds": [0.1, 0.2, 0.3, 0.4], "median": [0.0, 0.5, 0.4, 1.0],
                        "n_ctx": [2, 4, 8, 16]})
    with plt.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots()
        front = common.draw_front(ax, pts, "#000000")
    assert front.n_ctx.tolist() == [2, 4, 16]
    line = ax.lines[0]
    assert line.get_drawstyle() == "steps-post"
    assert [t.get_text() for t in ax.texts] == ["2", "4", "16"]
    plt.close(fig)


def test_labels_and_colors():
    assert common.model_label("decpfn-nostates/cartpole/binned/1M") == "DecPFN-no-St"
    assert common.model_label("segment-tabpfn/split") == "Segment-TabPFN"
    assert common.model_color("segment-tabpfn/rate") == common.SEGMENT_TABPFN_RATE_COLOR
    assert common.model_color("decpfn-nostates/x/binned/1M") == common.model_color("decpfn")
    assert common.thesis_model_id("xdecpfn/furuta/default/default") == \
        "xdecpfn/furuta/thesis/default"
    assert common.problem_label("threetank_pi") == "three-tank"
