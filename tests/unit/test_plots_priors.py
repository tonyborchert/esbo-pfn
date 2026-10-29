"""Plot group ``priors`` on a synthetic database."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from esbopfn.benchmarks.db import Db
from esbopfn.plots import common as c
from esbopfn.plots import group_config
from esbopfn.plots.priors import GROUP, draw_landscapes, draw_samples, landscape_rgb
from esbopfn.problems.base import cached_spec

_FAKEDB = Path(__file__).resolve().parents[1] / "fixtures" / "plots" / "fakedb.py"
_spec = importlib.util.spec_from_file_location("plots_fakedb", _FAKEDB)
fakedb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fakedb)

PROBLEMS = list(c.PROBLEMS)
KNOWN_J = 123.5


def _cfg() -> dict:
    return {**group_config("priors"), "usetex": False}


def _add_samples(db: Db, seed) -> None:
    rng = np.random.default_rng(0)
    for p in PROBLEMS:
        spec = cached_spec(p)
        T, n = spec.T, 10
        J = np.cumsum(rng.random((n, T)), axis=1).astype(np.float32)
        J[0] = np.linspace(0, KNOWN_J, T)
        crash = np.zeros(n, np.int32)
        crash[1] = T // 2
        J[1, T // 2 - 1:] = np.inf
        fakedb.add_unit(db, "landscapes", {"unit": "prior_samples", "problem": p, "instance": 0,
                                           "n": 10, "max_unsuccessful": 5,
                                           "instances_seed": seed},
                        {"samples.npz": {"theta": rng.random((n, spec.d)), "J": J,
                                         "success": crash == 0, "crash_step": crash}})


def _add_slices(db: Db, seed, problems=PROBLEMS, instances=(0, 1)) -> None:
    for p in problems:
        d = cached_spec(p).d
        for i in instances:
            if d == 1:
                G = 8
                thetas = np.linspace(0, 1, G)[:, None]
                shape = (G,)
            else:
                G = 4
                thetas = np.zeros((G, G, d))
                shape = (G, G)
            cost = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
            p_crash = np.zeros(shape)
            p_crash.flat[-1] = 1.0
            fakedb.add_unit(db, "landscapes",
                            {"unit": "slice", "problem": p, "instance": i, "slice_seed": 1,
                             "grid": G, "repeats": 2, "instances_seed": seed},
                            {"landscape.npz": {"coords": np.linspace(-1, 1, G), "thetas": thetas,
                                               "p_crash": p_crash, "cost_mean": cost,
                                               "cost_mean_nocrash": cost,
                                               "center": np.zeros(d), "u": np.ones(d),
                                               "v": np.ones(d), "extent": np.asarray(1.0)}})


@pytest.fixture
def db(pfn_home):
    from esbopfn.benchmarks.groups import instances_seed

    db = Db()
    seed = instances_seed(group_config("priors"))
    _add_samples(db, seed)
    _add_slices(db, seed)
    return db


@pytest.mark.parametrize("theme", ["white", "black"])
@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_group_renders(db, tmp_path, fmt, theme):
    out = tmp_path / "out"
    GROUP.render(_cfg(), db, out, fmt, theme)
    for name in ("priors_samples", "priors_landscapes"):
        assert (out / f"{name}.{fmt}").stat().st_size > 0
        text = (out / f"{name}.md").read_text()
        assert "Thesis asset" in text and "priors/" in text
    assert not (out / "skipped.txt").exists()


def test_samples_known_curve(db, tmp_path):
    ctx = c.Context(cfg=_cfg(), db=db, out_dir=tmp_path, fmt="png", theme="white")
    with c.style.thesis_style("white", usetex=False):
        res = draw_samples(ctx)
    ax = res.fig.axes[0]
    assert ax.lines[0].get_ydata()[-1] == pytest.approx(KNOWN_J, rel=1e-5)
    # the crashed curve stops before its crash step and gets a red cross
    T = cached_spec("boiler").T
    assert len(ax.lines[1].get_xdata()) == T // 2 - 1
    assert any(line.get_marker() == "x" for line in ax.lines)
    plt.close(res.fig)


def test_landscape_colors_and_partial(pfn_home, tmp_path):
    db = Db()
    _add_slices(db, 20261029, problems=["cartpole"], instances=(0,))
    ctx = c.Context(cfg=_cfg(), db=db, out_dir=tmp_path, fmt="png", theme="white")
    with c.style.thesis_style("white", usetex=False):
        res = draw_landscapes(ctx)
    img = res.fig.axes[0].images[0].get_array()
    # cell 0 is the cheapest (full blue), the last cell crashes always (blue 0 + full red)
    assert np.allclose(img[0, 0], [0.20, 0.55, 0.98])
    assert np.allclose(img[-1, -1], [0.98, 0.20, 0.20])
    assert "cartpole instance 0" in res.sidecar.numbers[0]
    plt.close(res.fig)
    assert np.allclose(landscape_rgb(np.array([1.0, np.nan]), np.array([0.0, 0.0]))[1], 0)


def test_missing_slices_skip(pfn_home, tmp_path):
    db = Db()
    _add_samples(db, 20261029)
    out = tmp_path / "out"
    GROUP.render(_cfg(), db, out, "png", "white")
    assert (out / "priors_samples.png").exists()
    assert not (out / "priors_landscapes.png").exists()
    assert "pfn bench run priors" in (out / "skipped.txt").read_text()


def test_samples_without_instances_skip(pfn_home, tmp_path):
    out = tmp_path / "out"
    GROUP.render(_cfg(), Db(), out, "png", "white")
    skipped = (out / "skipped.txt").read_text()
    assert "priors_samples" in skipped and "instances" in skipped
