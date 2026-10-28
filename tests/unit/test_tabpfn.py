"""Checkpoint handling, Trajectory-TabPFN and Segment-TabPFN.

The model tests skip when the TabPFN v3 checkpoint is not in ``$PFN_HOME/checkpoints/tabpfn``
(``pfn checkpoints download tabpfn``).
"""

from __future__ import annotations

import numpy as np
import pytest

from esbopfn.cli import main
from esbopfn.core import ProblemSpec, Trajectory
from esbopfn.predictors.base import BarPrediction, GaussianPrediction, get_predictor
from esbopfn.predictors.checkpoints import (
    TABPFN_FILENAME,
    CheckpointMissing,
    download_tabpfn,
    tabpfn_checkpoint,
    tabpfn_status,
)

SPEC = ProblemSpec("toy", 2, np.zeros(2), np.ones(2), 100, ("x",), (), True, False)


# ---------------------------------------------------------------------- checkpoints
def test_checkpoint_missing_has_instructions(pfn_home):
    with pytest.raises(CheckpointMissing) as info:
        tabpfn_checkpoint()
    assert info.value.path == pfn_home / "checkpoints" / "tabpfn" / TABPFN_FILENAME
    assert "pfn checkpoints download tabpfn" in info.value.instructions
    assert not tabpfn_status().present
    with pytest.raises(CheckpointMissing):
        get_predictor("tabpfn/50", SPEC)
    with pytest.raises(CheckpointMissing):
        get_predictor("segment-tabpfn/rate", SPEC)


def test_download_from_local_file_and_cli(pfn_home, tmp_path, capsys):
    fake = tmp_path / "weights.ckpt"
    fake.write_bytes(b"0123456789")
    assert main(["checkpoints", "status"]) == 1
    assert main(["checkpoints", "download", "tabpfn", "--from", str(fake)]) == 0
    path = tabpfn_checkpoint()
    assert path.read_bytes() == b"0123456789"
    status = tabpfn_status(verify=True)
    assert status.present and status.size == 10 and status.matches_thesis is False
    assert download_tabpfn() == path          # already present, nothing happens
    assert main(["checkpoints", "status"]) == 0
    assert "present" in capsys.readouterr().out


def test_registry_rejects_bad_ids():
    with pytest.raises(ValueError):
        get_predictor("tabpfn/abc", SPEC)
    with pytest.raises(ValueError):
        get_predictor("segment-tabpfn/anchored", SPEC)


# --------------------------------------------------------------------------- models
def _checkpoint_or_skip():
    try:
        return tabpfn_checkpoint()
    except CheckpointMissing as exc:
        pytest.skip(f"TabPFN checkpoint missing: {exc.path}")


def _traj(theta, crash_step=None):
    """Monotone costs: stage cost ``1 + 2 theta_0`` per step."""
    theta = np.asarray(theta, dtype=float)
    J = np.arange(1, SPEC.T + 1) * (1.0 + 2.0 * theta[0])
    if crash_step is not None:
        J[crash_step - 1:] = np.inf
    return Trajectory(theta, J, np.zeros((SPEC.T, 1)), None, crash_step, crash_step is None)


def _context(n=10, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t_obs = SPEC.T if i % 2 else int(rng.choice([30, 60]))
        out.append(_traj(rng.random(2)).cut(t_obs))
    out.append(_traj(rng.random(2), crash_step=45).cut(SPEC.T))
    return out


def test_trajectory_tabpfn_shapes_and_means():
    _checkpoint_or_skip()
    pred = get_predictor("tabpfn/25", SPEC, device="cpu")
    assert pred.name == "tabpfn/25"
    assert list(pred.predictable_steps(SPEC)) == [4 * i for i in range(1, 25)]
    ctx = _context()
    thetas = [np.array([0.2, 0.5]), np.array([0.8, 0.5])]
    queries = [_traj(thetas[0]).cut(40), _traj(thetas[1]).cut(0)]
    preds = pred.predict_many(queries, ctx)
    assert len(preds) == 2
    for p, th in zip(preds, thetas):
        assert isinstance(p, BarPrediction) and p.p_crash == 0.0
        assert p.probs.sum() == pytest.approx(1.0)
        mean, var = p.mean_var()
        assert mean == pytest.approx(100 * (1 + 2 * th[0]), rel=0.15)
        assert var > 0
    assert preds[0].mean_var()[0] < preds[1].mean_var()[0]
    single = pred.predict(queries[0], ctx)
    assert single.mean_var()[0] == pytest.approx(preds[0].mean_var()[0], rel=1e-3)
    done = pred.predict(_traj(thetas[0]).cut(100), ctx)
    assert isinstance(done, GaussianPrediction) and done.var == 0.0


@pytest.mark.parametrize("mode", ["split", "rate"])
def test_segment_tabpfn_modes(mode):
    _checkpoint_or_skip()
    pred = get_predictor(f"segment-tabpfn/{mode}", SPEC, device="cpu")
    ctx = _context(12, seed=1)
    th = np.array([0.5, 0.5])
    queries = [_traj(th).cut(45), _traj(th).cut(0), _traj(th).cut(100)]
    out = pred.predict_many(queries, ctx)
    assert all(isinstance(p, GaussianPrediction) and p.p_crash == 0.0 for p in out)
    assert out[0].mean == pytest.approx(200.0, rel=0.15)
    assert out[1].mean == pytest.approx(200.0, rel=0.15)
    assert out[0].var > 0
    assert out[2].mean == pytest.approx(200.0) and out[2].var == 0.0
