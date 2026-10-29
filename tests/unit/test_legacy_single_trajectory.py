"""Imported thesis DecPFN and LC-PFN checkpoints (``predictors/{decpfn,lcpfn}/legacy.py``).

The fixtures in ``tests/fixtures/legacy_single_trajectory`` were written by
``tools/export_thesis_single_trajectory.py parity`` with the old code base: one rollout of the old
simulator per case (golden case of ``tests/golden``, nominal plant, noise off), the old model
inputs of a prefix and the old predictions. The feature tests need nothing else. The model tests
need the export directory (``~/Downloads/thesis_single_trajectory_export``) and skip without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from esbopfn.core import Instance
from esbopfn.predictors import get_predictor
from esbopfn.predictors.base import BarPrediction, GaussianPrediction
from esbopfn.problems import get_problem
from esbopfn.problems.pt2_deadtime import sim as pt2_sim
from esbopfn.training.registry import ModelNotTrained

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures" / "legacy_single_trajectory"
GOLDEN = ROOT / "golden"
EXPORT = Path("~/Downloads/thesis_single_trajectory_export").expanduser()
needs_export = pytest.mark.skipif(not (EXPORT / "manifest.json").is_file(),
                                  reason="thesis single-trajectory export not present")

#: Old DecPFN token stride per problem (from the checkpoints).
RESOLUTION = {"cartpole": 15, "furuta": 9, "boiler": 10, "threetank_pi": 7, "pt2_deadtime": 8}


def _fixtures(prefix: str) -> list:
    return [pytest.param(p, id=p.stem) for p in sorted(FIX.glob(f"{prefix}*.npz"))]


def _parse(path: Path) -> tuple[str, str, str, str]:
    """``(family, problem, variant, size)`` of a fixture name ``<family>_<problem>_<head>_<size>``."""
    family, rest = path.stem.split("_", 1)
    problem, head, size = rest.rsplit("_", 2)
    variant = f"thesis-{head}" + ("-linearnorm" if family == "lcpfn" else "")
    return family, problem, variant, size


def _new_trial(problem: str, data):
    """The golden rollout of the new simulator matching the fixture, cut at its ``t_obs``."""
    gold = np.load(GOLDEN / f"{problem}.npz")
    P = get_problem(problem)
    k = int(data["case"])
    phi = dict(zip(map(str, gold["phi_names"]), gold["phi_values"].tolist()))
    extras = dict(zip(map(str, gold["extra_names"]), gold["extra_values"].tolist()))
    inst = Instance(phi=phi, phi_hat=dict(phi) if P.prior.uses_phi_hat else {},
                    noise_std={n: 0.0 for n in P.prior.noise_names}, extras=extras)
    np.testing.assert_allclose(gold[f"theta_{k}"], data["theta"])
    rng = np.random.default_rng(int(gold[f"seed_{k}"]))
    if problem == "pt2_deadtime":     # the old port's delay line (see test_problems_golden)
        traj = pt2_sim.rollout(data["theta"], inst, rng, P.spec.T, 11)
    else:
        traj = P.simulate(data["theta"], inst, rng)
    return P, traj.cut(int(data["t_obs"]))


def _leaf(path: Path) -> Path:
    return EXPORT.joinpath(*_parse(path))


def _predictor(path: Path, spec):
    """Legacy predictor built straight from the export leaf (no model tree)."""
    import json

    import torch

    from esbopfn.predictors.decpfn.legacy import LegacyDecPFN, LegacyDecPFNPredictor
    from esbopfn.predictors.lcpfn.legacy import LegacyLCPFN, LegacyLCPFNPredictor

    leaf = _leaf(path)
    arch = json.loads((leaf / "arch.json").read_text())
    calib = json.loads((leaf / "calib.json").read_text())
    state = torch.load(leaf / "state_dict.pt", map_location="cpu", weights_only=True)
    if arch["family"] == "lcpfn":
        model = LegacyLCPFN(arch)
        model.load_state_dict(state, strict=True)
        return LegacyLCPFNPredictor(model, arch, calib, spec)
    model = LegacyDecPFN(arch, calib)
    model.load_state_dict(state, strict=True)
    return LegacyDecPFNPredictor(model, arch, calib, spec)


def _compare(pred, data, tol: float) -> None:
    assert abs(pred.p_crash - float(data["p_crash"])) < tol
    if "probs" in data:
        assert isinstance(pred, BarPrediction)
        np.testing.assert_allclose(pred.probs, data["probs"].astype(np.float64), rtol=0, atol=tol)
        np.testing.assert_allclose(pred.edges[[0, -1]], data["edges_ends"][:2], rtol=1e-12)
        assert pred.edges.size == int(data["edges_ends"][2])
    else:
        assert isinstance(pred, GaussianPrediction)
        np.testing.assert_allclose([pred.mean, pred.var], [data["mean"], data["var"]], rtol=tol)


# --------------------------------------------------------------------------- features (no model)
@pytest.mark.parametrize("path", _fixtures("decpfn"))
def test_features_match_old_code(path: Path) -> None:
    from esbopfn.predictors.decpfn.legacy import legacy_features, legacy_token_steps

    data = np.load(path)
    family, problem, _, _ = _parse(path)
    P, trial = _new_trial(problem, data)
    np.testing.assert_allclose(trial.traj.J[:trial.t_obs], data["J"], rtol=1e-5)
    res = RESOLUTION[problem]
    seq_len = len(np.arange(0, P.spec.T - 1, res)) + 1
    grid = legacy_token_steps(P.spec.T, res, seq_len)
    np.testing.assert_array_equal(grid[grid < trial.t_obs], data["grid"])
    feats = legacy_features(problem, trial, res, seq_len, states=family == "decpfn")
    old = data["features"][:, :-1]                  # the last old column is the crash flag
    assert feats.shape == old.shape
    np.testing.assert_allclose(feats, old, rtol=1e-4, atol=1e-6)


def test_model_not_trained(tmp_path: Path) -> None:
    for mid in ("decpfn/cartpole/thesis-binned/1M", "decpfn-nostates/cartpole/thesis-binned/1M",
                "lcpfn/cartpole/thesis-binned-linearnorm/1M"):
        with pytest.raises(ModelNotTrained, match="import thesis-models"):
            get_predictor(mid, "cartpole", home=tmp_path)


# --------------------------------------------------------------------------- models (export)
@needs_export
def test_every_export_loads_strictly() -> None:
    import json

    import torch

    from esbopfn.benchmarks.import_models import single_trajectory_leaves
    from esbopfn.predictors.decpfn.legacy import LegacyDecPFN
    from esbopfn.predictors.lcpfn.legacy import LegacyLCPFN

    leaves = single_trajectory_leaves(EXPORT)
    manifest = json.loads((EXPORT / "manifest.json").read_text())
    assert len(leaves) == len(manifest["exported"]) > 0
    for leaf in leaves:
        family = leaf.parts[-4]
        arch = json.loads((leaf / "arch.json").read_text())
        state = torch.load(leaf / "state_dict.pt", map_location="cpu", weights_only=True)
        if family == "lcpfn":
            LegacyLCPFN(arch).load_state_dict(state, strict=True)
        else:
            calib = json.loads((leaf / "calib.json").read_text())
            LegacyDecPFN(arch, calib).load_state_dict(state, strict=True)


@needs_export
@pytest.mark.parametrize("path", _fixtures(""))
def test_network_parity(path: Path) -> None:
    """Old model inputs through the legacy network: the old predictions to 1e-6."""
    import torch

    data = np.load(path)
    family, problem, _, _ = _parse(path)
    pr = _predictor(path, get_problem(problem).spec)
    if family == "lcpfn":
        k = max(1, min(int(data["k_obs"]), pr.seq_len - 1))
        y = torch.as_tensor(data["ynorm"].astype(np.float32))[:, None]
        with torch.no_grad():
            out = pr.model(y, k).double().numpy()[-1, 0]
    else:
        out = pr.outputs([data["features"][:, :-1]])[0]
    _compare(pr.prediction(out), data, 1e-6)


@needs_export
@pytest.mark.parametrize("path", _fixtures(""))
def test_adapter_parity(path: Path) -> None:
    """New simulator rollout through the legacy predictor: the old predictions to 1e-5."""
    data = np.load(path)
    family, problem, _, _ = _parse(path)
    P, trial = _new_trial(problem, data)
    pr = _predictor(path, P.spec)
    if family == "lcpfn":
        k = pr.kappa(trial)
        np.testing.assert_allclose(pr.inputs(trial, k), data["ynorm"], rtol=1e-5, atol=1e-6)
    _compare(pr.predict(trial, []), data, 1e-5)


@needs_export
def test_import_thesis_skips_existing(tmp_path: Path) -> None:
    """A second import skips the models already imported, ``overwrite`` replaces them."""
    from esbopfn.benchmarks.import_models import import_thesis

    export = tmp_path / "export"
    leaf = export / "decpfn" / "boiler" / "thesis-binned" / "1M"
    leaf.parent.mkdir(parents=True)
    leaf.symlink_to(EXPORT / "decpfn" / "boiler" / "thesis-binned" / "1M")
    home = tmp_path / "home"
    first = import_thesis(export, home=home)
    assert [m.skipped for m in first] == [False]
    second = import_thesis(export, home=home)
    assert [(m.model_id, m.skipped) for m in second] == [("decpfn/boiler/thesis-binned/1M", True)]
    third = import_thesis(export, home=home, overwrite=True)
    assert [m.skipped for m in third] == [False]


@needs_export
def test_import_and_registry(tmp_path: Path) -> None:
    from esbopfn.benchmarks.import_models import import_single_trajectory_leaf
    from esbopfn.predictors.lcpfn.legacy import LegacyLCPFNPredictor
    from esbopfn.problems.prior import sample_instance, sample_theta
    from esbopfn.training.registry import load_model
    from esbopfn.util.seeds import substream

    home = tmp_path / "home"
    ids = ["decpfn/cartpole/thesis-binned/1M", "decpfn-nostates/cartpole/thesis-gaussian/100k",
           "lcpfn/cartpole/thesis-binned-linearnorm/1M",
           "lcpfn/cartpole/thesis-gaussian-linearnorm/100k"]
    for mid in ids:
        m = import_single_trajectory_leaf(EXPORT / mid, home)
        assert m.model_id == mid and m.step in (800, 8000)
        cfg = load_model(m.path).train_cfg
        assert cfg["provenance"] == "thesis" and cfg["legacy"] is True
    with pytest.raises(FileExistsError):
        import_single_trajectory_leaf(EXPORT / ids[0], home)

    P = get_problem("cartpole")
    rng = substream(11)
    inst = sample_instance(P, rng)
    trajs = [P.simulate(sample_theta(P.spec, rng), inst, rng) for _ in range(12)]
    queries = []
    for tr in trajs:
        queries.append(tr.cut(tr.T))                                  # complete or crashed
        cut = (tr.crash_step - 1) if tr.crash_step else 500
        queries += [tr.cut(1), tr.cut(max(1, cut // 2)), tr.cut(max(1, cut))]
    for mid in ids:
        pr = get_predictor(mid, "cartpole", home=home)
        steps = pr.predictable_steps(P.spec)
        assert steps[0] == 1 and steps[-1] < P.spec.T and np.all(np.diff(steps) > 0)
        preds = pr.predict_many(queries, [])
        for q, p in zip(queries, preds):
            assert 0.0 <= p.p_crash <= 1.0
            if q.crashed:
                assert p.p_crash == 1.0
            elif isinstance(p, BarPrediction):
                assert np.isclose(p.probs.sum() + p.p_crash, 1.0, atol=1e-6)
                assert np.all(np.isfinite(p.probs))
            else:
                assert np.isfinite(p.mean) and p.var >= 0.0
        one = pr.predict(queries[2], [])
        assert type(one) is type(preds[2])
        if isinstance(pr, LegacyLCPFNPredictor):
            curve = pr.predict_curve(queries[2])
            assert len(curve.steps) == len(curve.predictions) > 0
            assert curve.steps[-1] == P.spec.T
    with pytest.raises(ValueError):
        get_predictor(ids[0], "furuta", home=home)
