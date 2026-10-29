"""Plot group tabpfn (tabpfn_panel) on synthetic databases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.plots import common as C
from esbopfn.plots import discover_groups, group_config
from esbopfn.plots import tabpfn as TP
from esbopfn.plots.style import thesis_style

_spec = importlib.util.spec_from_file_location(
    "plots_fakedb", Path(__file__).resolve().parents[1] / "fixtures" / "plots" / "fakedb.py")
fakedb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fakedb)


def build(db: Db) -> None:
    for t in TP.TRAJECTORY + TP.SEGMENT:
        ns = [0, 2, 4, 64]
        ys = [-17.0, -0.5, 0.2, 0.75] if t == "tabpfn/50" else [-15.0, -0.8, 0.1, 0.5]
        fakedb.add_series(db, "fig3_cross", "pooled", t, {"n_ctx": ns, "y": ys})
        for p in C.PROBLEMS:
            for n in ns:
                fakedb.add_timing(db, p, t, n, 0.1 * (1 + n / 8))


def cfg() -> dict:
    return {**group_config("tabpfn"), "usetex": False}


@pytest.mark.parametrize("theme", ["white", "black"])
@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_renders(pfn_home, tmp_path, theme, fmt):
    build(Db())
    discover_groups()["tabpfn"].render(cfg(), Db(), tmp_path, fmt, theme)
    assert (tmp_path / f"tabpfn_panel.{fmt}").stat().st_size > 0
    text = (tmp_path / "tabpfn_panel.md").read_text()
    assert "tabpfn_panel.pdf" in text and "source: thesis import" in text


def test_known_number_and_floors(pfn_home, tmp_path):
    build(Db())
    ctx = C.Context(cfg=cfg(), db=Db(), out_dir=tmp_path, fmt="png", theme="white")
    with thesis_style("white", usetex=False):
        out = TP.draw_tabpfn_panel(ctx)
    ys = np.concatenate([np.asarray(ln.get_ydata(), float) for ln in out.fig.axes[0].lines])
    assert np.isclose(ys, 0.75).any()          # tabpfn/50 n=64 at its pooled value
    assert not np.isclose(ys, -17.0).any()     # n = 0 below the floor n >= 2
    seg = [s for s in out.sidecar.numbers if "segment-tabpfn/split" in s][0]
    assert "n=2 -0.800" in seg and "(not drawn)" in seg   # Segment-TabPFN floor n >= 4


def test_skips_without_data(pfn_home, tmp_path):
    discover_groups()["tabpfn"].render(cfg(), Db(), tmp_path, "png", "white")
    assert "tabpfn_panel: missing" in (tmp_path / "skipped.txt").read_text()
