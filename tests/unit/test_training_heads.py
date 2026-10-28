"""Crash-aware heads and input normalization."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from esbopfn.predictors.base import BarPrediction, GaussianPrediction
from esbopfn.training.heads import BarHead, GaussianHead, make_head
from esbopfn.training.normalization import (
    Calib, bar_edges, channel_stats, make_calib, strictly_increasing, time_stats)


def test_bar_head_nll_matches_thesis_formula():
    edges = np.array([-1.0, 0.0, 0.5, 2.0])
    head = BarHead(edges, target_mean=10.0, target_std=2.0)
    logits = torch.tensor([[0.3, -1.0, 0.7, 0.1]])
    a = torch.softmax(logits, -1)[0].numpy()
    w_inf = a[-1]
    w = a[:-1] / (1 - w_inf)
    y = torch.tensor([[10.0 + 2.0 * 0.25]])  # z = 0.25, bin 2 = [0, 0.5)
    nll = head.nll(logits[:, None], y).item()
    expect = -math.log(1 - w_inf) - math.log(w[1]) + math.log(0.5)
    assert nll == pytest.approx(expect, rel=1e-5)
    assert head.nll(logits[:, None], y, raw_units=True).item() == pytest.approx(expect + math.log(2.0), rel=1e-5)
    crash = head.nll(logits[:, None], torch.tensor([[math.inf]])).item()
    assert crash == pytest.approx(-math.log(w_inf), rel=1e-5)


def test_bar_head_clamps_out_of_support_targets():
    head = BarHead(np.array([0.0, 1.0, 2.0, 3.0]))
    z = torch.tensor([-5.0, 0.0, 0.999, 1.0, 2.999, 3.0, 50.0])
    assert head.bin_index(z).tolist() == [0, 0, 0, 1, 2, 2, 2]
    out = torch.zeros(7, 4)
    nll = head.nll(out, z)
    assert torch.isfinite(nll).all()
    assert nll[0].item() == pytest.approx(nll[1].item())


def test_bar_head_density_integrates_and_matches_prediction():
    rng = np.random.default_rng(0)
    edges = np.sort(rng.normal(size=11))
    head = BarHead(edges, 3.0, 0.5)
    out = torch.from_numpy(rng.normal(size=(1, 11))).float()
    pred = head.predictions(out)[0]
    assert isinstance(pred, BarPrediction)
    assert np.allclose(pred.edges, edges * 0.5 + 3.0)
    assert pred.probs.sum() + pred.p_crash == pytest.approx(1.0)
    y = float(pred.edges[3] + 0.3 * (pred.edges[4] - pred.edges[3]))
    nll = head.nll(out, torch.tensor([y]), raw_units=True).item()
    assert -pred.log_density(y) == pytest.approx(nll, rel=1e-4)


def test_gaussian_head_nll_and_prediction():
    head = GaussianHead(target_mean=5.0, target_std=2.0)
    out = torch.tensor([[0.5, math.log(0.25), -1.0]])
    w_inf = 1 / (1 + math.exp(1.0))
    y = 5.0 + 2.0 * 1.0
    expect = -math.log(1 - w_inf) + 0.5 * math.log(2 * math.pi * 0.25) + (1.0 - 0.5) ** 2 / 0.5
    assert head.nll(out, torch.tensor([y])).item() == pytest.approx(expect, rel=1e-5)
    assert head.nll(out, torch.tensor([math.inf])).item() == pytest.approx(-math.log(w_inf), rel=1e-5)
    pred = head.predictions(out)[0]
    assert isinstance(pred, GaussianPrediction)
    assert pred.mean == pytest.approx(6.0) and pred.var == pytest.approx(0.25 * 4.0)
    assert pred.p_crash == pytest.approx(w_inf)
    raw = head.nll(out, torch.tensor([y]), raw_units=True).item()
    assert -pred.log_density(y) == pytest.approx(raw, rel=1e-5)


def test_edges_equal_mass_linear_and_strict():
    s = np.concatenate([np.zeros(500), np.linspace(1, 2, 500), [np.inf]])
    b = bar_edges(s, 10, "binned")
    assert len(b) == 11 and np.all(np.diff(b) > 0) and b[0] == 0.0 and b[-1] == 2.0
    lin = bar_edges(s, 4, "linear")
    assert np.allclose(lin, [0, 0.5, 1, 1.5, 2])
    assert np.all(np.diff(strictly_increasing(np.array([1.0, 1.0, 1.0, 2.0]))) > 0)


def test_time_and_channel_stats():
    m, s = time_stats(1, 999)
    t = np.arange(1, 1000)
    assert m == pytest.approx(t.mean()) and s == pytest.approx(t.std())
    v = np.array([[1.0, 5.0], [3.0, 5.0], [np.inf, 5.0]])
    mean, std = channel_stats(v)
    assert np.allclose(mean, [2.0, 5.0]) and np.allclose(std, [1.0, 1.0])


def test_calib_round_trip_and_head():
    rng = np.random.default_rng(1)
    vals = rng.normal(3.0, 2.0, size=(200, 5, 2))
    calib = make_calib(model="decpfn", problem="boiler", head="binned", features=["J", "dJ", "t"],
                       feature_values=vals, feature_valid=None, time_range=(1, 10), J_index=0,
                       edge_samples=rng.normal(3.0, 2.0, 1000), grid=np.arange(1, 11), T=11,
                       n_bin=20)
    again = Calib.from_dict(calib.to_dict())
    assert np.allclose(again.input_mean, calib.input_mean) and np.allclose(again.edges, calib.edges)
    raw = np.concatenate([vals[0], np.arange(1, 6)[:, None]], axis=1)
    z = calib.normalize_inputs(raw)
    assert np.allclose(z * calib.input_std + calib.input_mean, raw, atol=1e-5)
    assert np.allclose(calib.raw_edges()[[0, -1]],
                       calib.edges[[0, -1]] * calib.target_std + calib.target_mean)
    assert calib.log_jacobian == pytest.approx(math.log(calib.target_std))
    assert isinstance(make_head(again), BarHead) and make_head(again).n_outputs == 21
    g = Calib.from_dict({**calib.to_dict(), "head": "gaussian", "edges": None})
    assert isinstance(make_head(g), GaussianHead)
