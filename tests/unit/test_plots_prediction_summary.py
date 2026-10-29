"""Plot group prediction_summary (fig3_cross, fig3_sheet, fig2_sheet) on synthetic databases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.plots import common as C
from esbopfn.plots import discover_groups, group_config
from esbopfn.plots import prediction_summary as PS
from esbopfn.plots.style import thesis_style

_spec = importlib.util.spec_from_file_location(
    "plots_fakedb", Path(__file__).resolve().parents[1] / "fixtures" / "plots" / "fakedb.py")
fakedb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fakedb)

PROBLEMS = list(C.PROBLEMS)
AWARE_N = [0, 4, 64]
BASELINE_N = [4, 64]
#: ll per model on every problem: every participant scores 0 except DecPFN (2.0), so med_D = 0
#: and DecPFN's pooled normalized median is exactly 2.0.
DECPFN_LL = 2.0


def score_rows(model: str, n_values: list[int], ll: float, crash_aware: bool, T: int = 100,
               n_tasks: int = 4) -> pd.DataFrame:
    """Deterministic prediction rows: five cutoff rows of value ``ll`` per non-crashed query, and
    for crash-aware models stop-grid rows (crashed queries reach p_crash 1 at step 50)."""
    rows = []
    for n in n_values:
        for q in range(n_tasks):
            crashed = crash_aware and q == 0
            tid = f"n{n}_q{q}"
            base = dict(task_id=tid, run_id=f"i{q:04d}_s00", instance=q, seed=0, n_ctx=n,
                        query_index=n, crashed=crashed, crash_step=80 if crashed else 0,
                        target_J=float("nan") if crashed else 1.0, cutoff_fallback=False,
                        J_obs=0.5, dist="gaussian", edges=None, probs=None, mean=1.0, var=0.1,
                        seconds=0.01)
            if not crashed:
                for f in (0.1, 0.3, 0.5, 0.7, 0.9):
                    rows.append({**base, "cutoff_frac": f, "cutoff_step": int(f * T), "ll": ll,
                                 "p_crash": 0.0})
            if crash_aware:
                for step in (10, 30, 50, 70):
                    p = 1.0 if (crashed and step >= 50) else 0.0
                    rows.append({**base, "cutoff_frac": float("nan"), "cutoff_step": step,
                                 "ll": float("nan"), "p_crash": p})
    return pd.DataFrame(rows)


def build_native(db: Db) -> None:
    for p in PROBLEMS:
        for tmpl in PS.DEFAULT_PARTICIPANTS:
            mid = tmpl.replace("{p}", p)
            fam = C.model_family(mid)
            blind = fam in ("decpfn", "decpfn-nostates", "lcpfn")
            aware_crash = fam not in ("tabpfn", "segment-tabpfn", "segment-gp")
            ns = AWARE_N if aware_crash else BASELINE_N
            ll = DECPFN_LL if fam == "decpfn" else 0.0
            fakedb.add_prediction(db, "prediction", p, mid, score_rows(mid, ns, ll, aware_crash))
            for n in ([0] if blind else ns):
                fakedb.add_timing(db, p, mid, n, 0.002 * (1 + n / 16), provenance=None)


def build_thesis(db: Db) -> None:
    for tmpl in PS.DEFAULT_PARTICIPANTS:
        thesis_id = C.thesis_model_id(tmpl)
        blind = C.model_family(tmpl) in ("decpfn", "decpfn-nostates", "lcpfn")
        ns = [-1] if blind else [2, 8, 64]
        ys = [0.1 * (i + 1) for i in range(len(ns))]
        if C.model_family(tmpl) == "ftpfn":
            ys = [0.5, 1.0, 1.25]
        fakedb.add_series(db, "fig3_cross", "pooled", thesis_id, {"n_ctx": ns, "y": ys})
        for p in PROBLEMS:
            fakedb.add_series(db, "fig3_per_problem", p, thesis_id.replace("{p}", p),
                              {"n_ctx": ns, "y": [y + 1.0 for y in ys]})
            for n in ([0] if blind else ns):
                fakedb.add_timing(db, p, thesis_id.replace("{p}", p), n, 0.003 * (1 + n / 8))
    fakedb.add_series(db, "fig3_med_D", "pooled", "med_D",
                      {"problem": PROBLEMS, "med_D": [1.0] * len(PROBLEMS)})
    for p in C.CRASH_PROBLEMS:
        for tmpl, ns in (("decpfn/{p}/binned/1M", [-1]), ("xdecpfn/{p}/thesis/default", [0, 64])):
            n_col, fs, saved = [], [], []
            for n in ns:
                n_col += [n] * 3
                fs += [0.0, 0.05, 0.2]
                saved += [0.0, 30.0, 60.0]
            fakedb.add_series(db, "fig2_stop_curves", p, tmpl.replace("{p}", p),
                              {"n_ctx": n_col, "tau": [1.0] * len(fs), "false_stop_rate": fs,
                               "mean_steps_saved": saved})


def cfg() -> dict:
    return {**group_config("prediction_summary"), "usetex": False}


def ctx(tmp_path: Path, theme: str = "white") -> C.Context:
    return C.Context(cfg=cfg(), db=Db(), out_dir=tmp_path, fmt="png", theme=theme)


def scatter_points(fig) -> np.ndarray:
    pts = [c.get_offsets() for ax in fig.axes for c in ax.collections]
    return np.concatenate([np.asarray(p) for p in pts if len(p)]) if pts else np.empty((0, 2))


@pytest.mark.parametrize("theme", ["white", "black"])
@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_renders_native(pfn_home, tmp_path, theme, fmt):
    build_native(Db())
    group = discover_groups()["prediction_summary"]
    group.render(cfg(), Db(), tmp_path, fmt, theme)
    for name in ("fig3_cross", "fig3_sheet", "fig2_sheet"):
        assert (tmp_path / f"{name}.{fmt}").stat().st_size > 0
        text = (tmp_path / f"{name}.md").read_text()
        assert "Thesis asset" in text and "source: native" in text
    assert not (tmp_path / "skipped.txt").exists()


def test_native_known_number(pfn_home, tmp_path):
    build_native(Db())
    with thesis_style("white", usetex=False):
        out = PS.draw_fig3_cross(ctx(tmp_path))
        pts = scatter_points(out.fig)
    # DecPFN (pooled, context-blind) sits at its known normalized median
    assert np.any(np.isclose(pts[:, 1], DECPFN_LL)) and np.any(np.isclose(pts[:, 0], 0.002))
    assert any("DecPFN (decpfn/{p}/binned/1M): pooled 2.000" in s for s in out.sidecar.numbers)
    with thesis_style("white", usetex=False):
        out2 = PS.draw_fig2_sheet(ctx(tmp_path))
    # one crashed query of four, stopped at step 50 of crash step 80: 30 steps saved, no false stop
    assert any(s.endswith("DecPFN: 30.0 steps saved at false-stop share <= 0.1")
               for s in out2.sidecar.numbers)


def test_thesis_fallback(pfn_home, tmp_path):
    build_thesis(Db())
    GROUP = discover_groups()["prediction_summary"]
    GROUP.render(cfg(), Db(), tmp_path, "png", "white")
    text = (tmp_path / "fig3_cross.md").read_text()
    assert "source: thesis import" in text and "fig2_stop_curves" not in text
    assert "DecPFN-no-St" in (tmp_path / "fig2_sheet.md").read_text()   # noted as missing
    with thesis_style("white", usetex=False):
        out = PS.draw_fig3_cross(ctx(tmp_path))
        pts = scatter_points(out.fig)
        sheet = PS.draw_fig3_sheet(ctx(tmp_path))
    # FT-PFN n=64 at the imported pooled value 1.25, timing 0.003 * 9 (same on every problem)
    assert np.any(np.isclose(pts[:, 0], 0.027) & np.isclose(pts[:, 1], 1.25))
    # per-problem panel: imported y + 1 minus med_D 1 = the pooled value again
    lines = [ln for ax in sheet.fig.axes[1:2] for ln in ax.lines]
    assert any(np.isclose(np.asarray(ln.get_ydata(), float), 1.25).any() for ln in lines)


def test_skips_without_data(pfn_home, tmp_path):
    GROUP = discover_groups()["prediction_summary"]
    GROUP.render(cfg(), Db(), tmp_path, "png", "white")
    skipped = (tmp_path / "skipped.txt").read_text()
    for name in ("fig3_cross", "fig3_sheet", "fig2_sheet"):
        assert f"{name}: missing" in skipped
        assert not (tmp_path / f"{name}.png").exists()
