"""Live-inference example figures (plots/examples.py) with fake predictors and synthetic runs."""

from __future__ import annotations

import numpy as np
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.benchmarks.kinds.fields import FieldRun
from esbopfn.plots import common as c
from esbopfn.plots import examples as ex
from esbopfn.predictors.base import BarPrediction, GaussianPrediction
from esbopfn.training.registry import ModelNotTrained

T = 1000
P_CRASH = 0.37


class FakePredictor:
    """Gaussian at the cutoff with a known crash probability, bars for unrun controllers."""

    def __init__(self, p_crash: float = P_CRASH):
        self.p_crash = p_crash
        self.calls: list[int] = []

    def predictable_steps(self, spec):
        return np.arange(10, spec.T, 10)

    def predict(self, query, context):
        self.calls.append(len(context))
        return GaussianPrediction(mean=1.2 * float(query.J_obs) + 500.0, var=200.0 ** 2,
                                  p_crash=self.p_crash)

    def predict_unrun(self, thetas, context):
        edges = np.linspace(0.0, 10.0, 11)
        out = []
        for th in np.asarray(thetas):
            probs = np.full(10, 0.07)
            probs[int(abs(th[0]) * 9) % 10] += 0.2
            p = float(np.clip(abs(th[1]), 0, 0.9))
            out.append(BarPrediction(edges, probs / probs.sum() * (1 - p), p))
        return out


def fake_run(n: int = 12, d: int = 4, crash_last: bool = False) -> FieldRun:
    rng = np.random.default_rng(0)
    J = np.cumsum(rng.uniform(0, 10, size=(n, T)), axis=1).astype(np.float32)
    crash = np.zeros(n, np.int32)
    if crash_last:
        crash[-1] = 500
        J[-1, 499:] = np.inf
    key = {"problem": "cartpole", "field": "oracle65", "instance": 8, "seed": 0}
    return FieldRun(key=key, theta=rng.uniform(size=(n, d)), t_obs=np.full(n, T, np.int32),
                    crash_step=crash, success=crash == 0, J=J,
                    x=np.zeros((n, T, 2), np.float32), xdot=np.zeros((n, T, 2), np.float32))


@pytest.fixture
def fakes(pfn_home, monkeypatch):
    preds: dict[str, FakePredictor] = {}

    def build(model_id, problem):
        return preds.setdefault(model_id, FakePredictor())

    run = fake_run()
    monkeypatch.setattr(ex, "_build", build)
    monkeypatch.setattr(ex, "_checkpoint", lambda m: "fake")
    monkeypatch.setattr(ex, "example_trial", lambda db, cache, problem="cartpole":
                        ex.ExampleTrial(run, 11, "test", "i0008_s00", "thesis", ["synthetic"]))
    g = 6
    coords = np.linspace(-1, 1, g)
    tt, ss = np.meshgrid(coords, coords, indexing="ij")
    thetas = np.stack([ss, tt, 0 * ss, 0 * ss], axis=-1)
    real = {"thetas": thetas, "p_crash": np.abs(tt), "cost_mean": ss + 2.0,
            "cost_mean_nocrash": ss + 2.0, "extent": np.asarray(1.0)}
    monkeypatch.setattr(ex, "_instances_seed", lambda db, p: "thesis")
    monkeypatch.setattr(ex, "pick_plant", lambda db, cfg, seed, cache, prov: (3, real, "test"))
    loop = fake_run(n=64)
    monkeypatch.setattr(ex, "loop_run", lambda *a, **k: loop)
    return preds


SPECS = [ex.LCPFN_DECPFN_EXAMPLE, ex.FTPFN_XDECPFN_EXAMPLE, ex.LANDSCAPE_FURUTA]


@pytest.mark.parametrize("theme", ["white", "black"])
@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_examples_render(fakes, tmp_path, fmt, theme):
    out = tmp_path / "figs"
    report = c.render_group("t", SPECS, {"usetex": False}, Db(), out, fmt, theme)
    assert set(report.written) == {s.name for s in SPECS} and not report.skipped
    for s in SPECS:
        assert (out / f"{s.name}.{fmt}").stat().st_size > 0
        md = (out / f"{s.name}.md").read_text()
        assert "Thesis asset" in md and "$PFN_HOME/figures/_cache/t/" in md


def _texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts] + [t.get_text() for t in fig.texts]


def test_crash_text_and_context(fakes, tmp_path):
    ctx = c.Context({"usetex": False}, Db(), tmp_path, "png", "white")
    with __import__("esbopfn.plots.style", fromlist=["x"]).thesis_style("white", usetex=False):
        out = ex.draw_ftpfn_xdecpfn_example(ctx)
    texts = _texts(out.fig)
    assert sum("Crash: 37%" in t for t in texts) == 6
    assert any("Context: 10" in t for t in texts)
    # context sizes 0, 5, 10 reach each model, the query is never in its own context
    for model in ("ftpfn/cartpole/default/default", "xdecpfn/cartpole/default/default"):
        assert fakes[model].calls == [0, 5, 10]
    draws = ex.random_context(fake_run(), 11, (0, 5, 10))
    assert 11 not in draws[10] and set(draws[5]) <= set(draws[10])
    assert any("crash 37.0%" in n for n in out.sidecar.numbers)


def test_landscape_numbers(fakes, tmp_path):
    ctx = c.Context({"usetex": False}, Db(), tmp_path, "png", "white")
    with __import__("esbopfn.plots.style", fromlist=["x"]).thesis_style("white", usetex=False):
        out = ex.draw_landscape_furuta(ctx)
    # 8 model panels + real + key, the real crash mean is mean(|t|) over the grid
    assert len(out.fig.axes) == 10
    real = [n for n in out.sidecar.numbers if n.startswith("real")][0]
    assert f"{100 * np.mean(np.abs(np.linspace(-1, 1, 6))):.0f}%" in real


def test_missing_models_skip(fakes, tmp_path, monkeypatch):
    def not_trained(model_id, problem):
        raise ModelNotTrained(model_id)

    monkeypatch.setattr(ex, "_build", not_trained)
    out = tmp_path / "figs"
    report = c.render_group("t", SPECS, {"usetex": False}, Db(), out, "png", "white")
    assert set(report.skipped) == {s.name for s in SPECS}
    text = (out / "skipped.txt").read_text()
    assert "lcpfn/cartpole/binned/1M" in text and "decpfn/cartpole/binned/1M" in text
    assert "pfn train" in text


def test_median_task_and_colors():
    import pandas as pd

    df = pd.DataFrame({"n_ctx": [64] * 4, "selected": [True] * 4,
                       "crashed": [False, False, False, True],
                       "target_J": [3.0, 1.0, 2.0, 0.5], "run_id": ["a", "b", "c", "d"],
                       "query_index": [64] * 4})
    assert ex.median_task(df).run_id == "c"
    rgb = ex.landscape_colors(np.array([[1.0, 2.0]]), np.array([[0.0, 1.0]]))
    assert np.allclose(rgb[0, 0], ex.BLUE_RGB) and np.allclose(rgb[0, 1], ex.RED_RGB)
