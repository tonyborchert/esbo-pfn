"""Export the thesis DecPFN and LC-PFN checkpoints of the old code base into a neutral format.

Runs in the OLD environment, because the LC-PFN checkpoints are pickled whole modules that only
unpickle with the old class paths::

    OLD=/Users/tonyborchert/Documents/Uni/bachelorarbeit/impl/prediction
    PYTHONPATH=$OLD:$OLD/util $OLD/.venv/bin/python tools/export_thesis_single_trajectory.py \\
        export --models ~/Downloads/dynformer_timing_bundle/repo/prediction/models \\
        --out ~/Downloads/thesis_single_trajectory_export

The old tree is ``<models>/<decpfn|lcpfn>/<dataset>/transfer/<10|100>/<head>[_nostates][_zscore]/0/
model.pt``. Every leaf becomes ``<out>/<family>/<problem>/<variant>/<size>/`` with

- ``state_dict.pt``: plain tensors (DecPFN: network plus ``head.*``, LC-PFN: the module without
  its ``criterion.*`` buffers),
- ``arch.json``: every hyperparameter needed to rebuild the old module,
- ``calib.json``: normalization statistics, bin edges (model space and raw units), the old
  normalization spec and the flags the old inference reads,
- ``source.json``: old path, ``SOURCE.txt`` content (if any), sha256 and modification time.

Names: family ``decpfn`` (``_nostates`` leaves → ``decpfn-nostates``), ``lcpfn``. Variant
``thesis-<head>``. LC-PFN leaves with ``_zscore`` are the thesis LC-PFN. A plain LC-PFN leaf
(linear input normalization) is exported as ``thesis-<head>-linearnorm`` only when the zscore leaf
of the same problem, size and head is absent. Size ``10`` epochs → ``100k``, ``100`` → ``1M``.
Problems ``boiler threetank_pi pt2_deadtime cartpole`` (dataset ``<p>/universal``) and ``furuta``
(dataset ``furuta/lqr-new-realistic/rigreset(1,False)``). ``furuta/lqr`` is skipped.

``parity --models DIR --out FIXTURES`` writes the parity fixtures of
``tests/unit/test_legacy_single_trajectory.py``: one rollout of the old simulator per case (nominal
plant, noise off, as ``tests/golden/make_golden.py``), the old model inputs of a prefix and the
old predictions.

This file must not import ``esbopfn``. It only uses the old code base and numpy/torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("export_thesis_single_trajectory")

#: Old dataset directory (relative to ``<models>/<family>/``) -> problem name of the new package.
DATASETS = {
    "boiler/universal": "boiler",
    "threetank_pi/universal": "threetank_pi",
    "pt2_deadtime/universal": "pt2_deadtime",
    "cartpole/universal": "cartpole",
    "furuta/lqr-new-realistic/rigreset(1,False)": "furuta",
}
SIZES = {"10": "100k", "100": "1M"}
HEADS = ("binned", "gaussian", "linear")

#: Old ``(y, Y)`` channel names per problem (``j``/``J`` follow, ``failed`` is the pad channel).
Y_NAMES = {
    "boiler": ["temperature", "u"],
    "threetank_pi": ["V2", "V3", "V4"],
    "pt2_deadtime": ["y"],
    "cartpole": ["x_cart_rel_goal", "phi_pole", "action"],
    "furuta": ["phi_arm", "phi_pend", "phidot_arm", "phidot_pend", "target"],
}
#: Names of the old per-step ``y`` block. Cartpole carries velocities, the others the
#: per-step increments of ``Y`` (``y_0 = 0``).
y_NAMES = {
    "boiler": ["d_temperature", "d_u"],
    "threetank_pi": ["d_V2", "d_V3", "d_V4"],
    "pt2_deadtime": ["d_y"],
    "cartpole": ["xdot_cart", "phidot_pole", "d_action"],
    "furuta": ["d_phi_arm", "d_phi_pend", "d_phidot_arm", "d_phidot_pend", "d_target"],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _list(t) -> list:
    return None if t is None else torch.as_tensor(t).detach().cpu().double().tolist()


def _write(out: Path, state: dict, arch: dict, calib: dict, source: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in state.items()}, out / "state_dict.pt")
    for name, data in (("arch", arch), ("calib", calib), ("source", source)):
        (out / f"{name}.json").write_text(json.dumps(data, indent=1, sort_keys=True))


def _source(path: Path, leaf: Path) -> dict:
    txt = None
    for cand in (leaf / "SOURCE.txt", leaf.parent / "SOURCE.txt"):
        if cand.is_file():
            txt = cand.read_text()
            break
    return {"old_path": str(path), "source_txt": txt, "sha256": _sha256(path),
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
            "bytes": path.stat().st_size}


# --------------------------------------------------------------------------- DecPFN
def export_decpfn(path: Path, problem: str, dataset: str, head: str, nostates: bool, size: str,
                  out_root: Path) -> Path:
    from predictors.decpfn.architecture import load

    ck = torch.load(path, weights_only=True, map_location="cpu")
    model, hd = load(path)                   # the old loader must accept the checkpoint
    cfg = dict(ck["config"])
    if bool(cfg["extra_inputs"]) == nostates:
        raise ValueError(f"{path}: extra_inputs={cfg['extra_inputs']} contradicts the leaf name")
    if ck.get("drop_y"):
        raise ValueError(f"{path}: drop_y={ck['drop_y']} is not supported")
    names = (y_NAMES[problem] + Y_NAMES[problem] if cfg["extra_inputs"] else []) + ["j", "J"]
    n_feat = int(ck["input_bin_edges"].shape[0])
    if n_feat != len(names):
        raise ValueError(f"{path}: {n_feat} input features, expected {names}")
    layer = model.encoder.layers[0]
    family = "decpfn-nostates" if nostates else "decpfn"
    state = dict(model.state_dict())
    state.update({f"head.{k}": v for k, v in hd.state_dict().items()})
    arch = {
        "class": "LegacyDecPFN", "family": family, "problem": problem, "old_dataset": dataset,
        "head": head, "old_output_type": cfg["output_type"],
        "is_gaussian_head": bool(ck["is_gaussian_head"]),
        "d_model": int(cfg["embed_size"]), "n_layers": int(cfg["n_layers"]),
        "n_heads": int(cfg["n_heads"]), "d_ff": int(cfg["hidden_mult"]) * int(cfg["embed_size"]),
        "activation": cfg["activation"], "dropout": float(cfg["dropout"]),
        "norm_first": bool(layer.norm_first), "layer_norm_eps": float(layer.norm1.eps),
        "final_norm": True, "n_bins": int(cfg["n_bins"]),
        "n_outputs": int(hd.linear.out_features), "n_features": n_feat,
        "features": names, "feature_aggregation": "y and j summed between grid points",
        "extra_inputs": bool(cfg["extra_inputs"]), "add_diffed_up": bool(cfg["add_diffed_up"]),
        "resolution": int(cfg["resolution"]), "sequence_length": int(ck["sequence_length"]),
        "grid_rule": "0-based steps arange(0, T-1, resolution) + [T-1], the first L-1 are tokens",
        "time_encoding": "linspace(-sqrt(3), sqrt(3), L-1)[position]",
        "input_norm": cfg["input_norm"], "old_config": cfg,
    }
    hp = dict(ck["head_params"])
    calib = {
        "input_norm": cfg["input_norm"],
        "input_mean": _list(ck.get("input_mean")), "input_std": _list(ck.get("input_std")),
        "input_bin_edges": _list(ck["input_bin_edges"]),
        "output_bin_edges": _list(ck["output_bin_edges"]),
        "edges": None if ck["is_gaussian_head"] else _list(hp["bin_edges"]),
        "target_mean": hp.get("target_mean"), "target_std": hp.get("target_std"),
        "target_range": None if hp.get("target_range") is None else list(hp["target_range"]),
        "failure": "logit 0 (bar heads) or sigmoid of output 2 (gaussian head)",
    }
    out = out_root / family / problem / f"thesis-{head}" / size
    _write(out, state, arch, calib, _source(path, path.parent))
    return out


# --------------------------------------------------------------------------- LC-PFN
def _lcpfn_edges_raw(m) -> list[float]:
    spec = m.norm_spec
    b = m.criterion.borders.detach().cpu().double().numpy()
    return list((b[1:] - spec["low"]) / (spec["high"] - spec["low"]) * spec["cap"])


def export_lcpfn(path: Path, problem: str, dataset: str, head: str, zscore: bool, size: str,
                 out_root: Path) -> Path:
    m = torch.load(path, weights_only=False, map_location="cpu")
    for attr in ("norm_spec", "input_norm_edges", "seq_len", "fail_threshold", "output_type"):
        if getattr(m, attr, None) is None:
            raise ValueError(f"{path}: model lacks {attr}")
    if m.output_type != head:
        raise ValueError(f"{path}: output_type {m.output_type} != leaf head {head}")
    input_norm = getattr(m, "input_norm", None)
    if zscore != (input_norm == "zscore") or m.input_norm_edges.numel() == 2 and not zscore:
        raise ValueError(f"{path}: input_norm {input_norm} contradicts the leaf name")
    enc = m.transformer_encoder
    layer = enc.layers[0]
    if enc.norm is not None or layer.pre_norm or m.input_ln is not None or m.pos_encoder is not None \
            or m.style_encoder is not None or m.global_att_embeddings is not None or m.full_attention:
        raise ValueError(f"{path}: unexpected LC-PFN architecture")
    act = layer.activation
    act_name = getattr(act, "__name__", type(act).__name__).lower()
    state = {k: v for k, v in m.state_dict().items() if not k.startswith("criterion.")}
    x_norm = [[float(n.mean), float(n.std)] for n in list(m.encoder)[:2]]
    y_norm = [float(m.y_encoder[0].mean), float(m.y_encoder[0].std)]
    variant = f"thesis-{head}" + ("" if zscore else "-linearnorm")
    arch = {
        "class": "LegacyLCPFN", "family": "lcpfn", "problem": problem, "old_dataset": dataset,
        "head": head, "d_model": int(m.ninp), "n_layers": len(enc.layers),
        "n_heads": int(layer.self_attn.num_heads), "d_ff": int(layer.linear1.out_features),
        "decoder_hidden": int(m.decoder[0].out_features), "n_out": int(m.decoder[2].out_features),
        "activation": act_name, "pre_norm": False, "layer_norm_eps": float(layer.norm1.eps),
        "dropout": float(layer.dropout.p), "seq_len": int(m.seq_len),
        "resolution": int(getattr(m, "resolution", 0) or 0),
        "grid_rule": "0-based steps linspace(0, T-1, seq_len).astype(int)",
        "x_encoding": x_norm, "y_encoding": y_norm,
        "attention_mask": "every position sees the observed prefix and itself",
        "input_norm": input_norm,
    }
    crit = m.criterion
    calib = {
        "norm_spec": dict(m.norm_spec), "input_norm": input_norm,
        "input_norm_edges": _list(m.input_norm_edges),
        "borders": _list(crit.borders), "fail_threshold": float(m.fail_threshold),
        "edges_raw": None if head == "gaussian" else _lcpfn_edges_raw(m),
        "target_mean": getattr(crit, "target_mean", None),
        "target_std": getattr(crit, "target_std", None),
        "to_cost_scale": float(m.norm_spec["cap"] / (m.norm_spec["high"] - m.norm_spec["low"])),
        "mask_crashed": getattr(m, "mask_crashed", None),
        "criterion": type(crit).__name__,
    }
    out = out_root / "lcpfn" / problem / variant / size
    _write(out, state, arch, calib, _source(path, path.parent))
    return out


# --------------------------------------------------------------------------- walk
def leaves(models: Path):
    """``(family, dataset, problem, epochs, leaf_name, path)`` of every exportable leaf."""
    for family in ("decpfn", "lcpfn"):
        for dataset, problem in DATASETS.items():
            root = models / family / dataset / "transfer"
            if not root.is_dir():
                log.warning("missing %s", root)
                continue
            for epochs_dir in sorted(root.iterdir()):
                if epochs_dir.name not in SIZES:
                    continue
                for leaf in sorted(epochs_dir.iterdir()):
                    path = leaf / "0" / "model.pt"
                    if path.is_file():
                        yield family, dataset, problem, epochs_dir.name, leaf.name, path


def export(models: Path, out: Path) -> dict:
    found = list(leaves(models))
    have = {(f, p, e, n) for f, _, p, e, n, _ in found}
    done, skipped, failed = [], [], []
    for family, dataset, problem, epochs, name, path in found:
        parts = name.split("_")
        head, flags = parts[0], set(parts[1:])
        size = SIZES[epochs]
        if head not in HEADS or flags - {"nostates", "zscore"} \
                or (family == "decpfn" and "zscore" in flags) \
                or (family == "lcpfn" and "nostates" in flags):
            skipped.append({"path": str(path), "reason": f"unknown leaf {name}"})
            continue
        try:
            if family == "decpfn":
                o = export_decpfn(path, problem, dataset, head, "nostates" in flags, size, out)
            else:
                zscore = "zscore" in flags
                if not zscore and (family, problem, epochs, f"{head}_zscore") in have:
                    skipped.append({"path": str(path), "reason": "zscore leaf present"})
                    continue
                o = export_lcpfn(path, problem, dataset, head, zscore, size, out)
        except Exception as exc:  # noqa: BLE001  (record and continue)
            log.exception("failed %s", path)
            failed.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        done.append({"path": str(path), "out": str(o.relative_to(out))})
        log.info("exported %s", o.relative_to(out))
    manifest = {"models": str(models), "exported": done, "skipped": skipped, "failed": failed,
                "torch": torch.__version__, "python": sys.version.split()[0],
                "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# --------------------------------------------------------------------------- parity fixtures
#: ``(name, family dir, problem, leaf, epochs, golden case, cutoff)``.
PARITY = [
    ("decpfn_cartpole_binned_1M", "decpfn", "cartpole", "binned", "100", 1, 400),
    ("decpfn-nostates_cartpole_binned_1M", "decpfn", "cartpole", "binned_nostates", "100", 1, 400),
    ("decpfn_furuta_binned_1M", "decpfn", "furuta", "binned", "100", 2, 300),
    ("decpfn_boiler_binned_1M", "decpfn", "boiler", "binned", "100", 1, 350),
    ("decpfn_threetank_pi_binned_1M", "decpfn", "threetank_pi", "binned", "100", 0, 200),
    ("decpfn_pt2_deadtime_binned_1M", "decpfn", "pt2_deadtime", "binned", "100", 1, 250),
    ("decpfn_cartpole_gaussian_1M", "decpfn", "cartpole", "gaussian", "100", 1, 400),
    ("lcpfn_cartpole_binned_1M", "lcpfn", "cartpole", "binned", "100", 1, 400),
    ("lcpfn_cartpole_gaussian_1M", "lcpfn", "cartpole", "gaussian", "100", 1, 400),
    ("lcpfn_furuta_binned_1M", "lcpfn", "furuta", "binned", "100", 2, 300),
]


def _old_trial(problem: str, case: int):
    """The old Trial (with cost) of golden case ``case``, exactly as ``tests/golden/make_golden.py``."""
    if problem == "cartpole":
        from experiments.cartpole import benchmark_goals, default_weighting
        from experiments.cartpole.objective import CartPoleCost
        from experiments.cartpole.system import CartPolePIDMujocoSystem

        seed, theta = [(0, [2.95, 288.45, 23.39, 1463.05]), (1, [2.7, 252.04, 53.2, 463.7]),
                       (4, [2.57, 48.01, 40.99, 730.9])][case]
        phi = np.array([1.0, 0.1, 0.5, 10.0], dtype=np.float64)
        tr = CartPolePIDMujocoSystem().recorded_trial(phi, np.array(theta), benchmark_goals,
                                                      np.zeros(4), seed=seed)
        return CartPoleCost(default_weighting).with_cost(tr), theta, seed
    if problem == "furuta":
        from experiments.furuta.objective import FurutaHoldCost
        from experiments.furuta.system import default_phi
        from experiments.furuta.tick_reset import FurutaRealisticTickResetSystem

        thetas = ([60.0, 1.0, 12.0, 5.0, 1.0, 1.0, 1.0, 3.0], [85.0, 1.15, 11.6, 5.4, 0.6, 1.4, 0.7, 2.0],
                  [45.0, 0.75, 12.4, 4.6, 1.4, 0.6, 1.3, 1.6], [70.0, 0.9, 12.2, 4.8, 0.8, 1.2, 1.1, 3.4])
        sysm = FurutaRealisticTickResetSystem()
        phi10 = np.concatenate([default_phi, [2.0 / 120.0]])
        tr = sysm.recorded_trial(phi10, np.array(thetas[case]), noise_stds=np.zeros(3),
                                 phi_hat=phi10, seed=case, kick_voltage=0.3)
        return FurutaHoldCost(sysm.frequency, threshold_deg=5.0).with_cost(tr), thetas[case], case
    if problem == "boiler":
        from experiments.boiler.objective import BoilerObjective
        from experiments.boiler.system import DEFAULT_PHI, BoilerBangBangSystem

        theta = [[20.0], [120.0], [400.0]][case]
        tr = BoilerBangBangSystem().recorded_trial(DEFAULT_PHI, np.array(theta), noise_stds=[0.0],
                                                   seed=0)
        return BoilerObjective().with_cost(tr), theta, 0
    if problem == "threetank_pi":
        from experiments.threetank_pi.objective import ThreeTankObjective
        from experiments.threetank_pi.system import DEFAULT_PHI, ThreeTankPISystem

        theta = [[2.0, 0.05], [0.5, 0.01], [3.0, 0.1]][case]
        tr = ThreeTankPISystem().recorded_trial(DEFAULT_PHI, np.array(theta), noise_stds=[0.0], seed=0)
        return ThreeTankObjective().with_cost(tr), theta, 0
    if problem == "pt2_deadtime":
        from experiments.pt2_deadtime.objective import PT2Objective
        from experiments.pt2_deadtime.system import DEFAULT_PHI, PT2DeadTimePIDSystem

        theta = [list(np.log10([5.0, 2.0, 1.0])), [0.27, -0.46, -0.92]][case]
        tr = PT2DeadTimePIDSystem().recorded_trial(DEFAULT_PHI, np.array(theta), noise_stds=[0.0],
                                                   seed=0)
        return PT2Objective().with_cost(tr), theta, 0
    raise KeyError(problem)


def parity(models: Path, out: Path) -> None:
    from predictors.decpfn.architecture import load as load_decpfn
    from predictors.decpfn.infer import DecPFNPredictor
    from predictors.lcpfn.custom import predictor as lcpfn_predictor

    rev = {v: k for k, v in DATASETS.items()}
    out.mkdir(parents=True, exist_ok=True)
    for name, family, problem, leaf, epochs, case, cutoff in PARITY:
        path = models / family / rev[problem] / "transfer" / epochs / leaf / "0" / "model.pt"
        trial, theta, seed = _old_trial(problem, case)
        if trial.did_fail() and cutoff >= trial.t:
            raise ValueError(f"{name}: cutoff {cutoff} is not before the crash")
        partial = trial.cutoff(cutoff)
        rec = {"theta": np.asarray(theta, np.float64), "seed": np.int64(seed),
               "case": np.int64(case), "t_obs": np.int64(cutoff), "T": np.int64(trial.T),
               "J": np.asarray(trial.J, np.float64)[:cutoff]}
        if family == "decpfn":
            pr = DecPFNPredictor(load_decpfn(path))
            cfg = pr.model.config
            L = pr.model.sequence_length
            cols = ["y", "Y", "j", "J", "failed"] if cfg.extra_inputs else ["j", "J", "failed"]
            grid = np.arange(0, partial.T - 1, cfg.resolution)
            grid = grid[grid < partial.t][:L - 1]
            feats = partial.get_formatted(cols, subsample=grid, add_diffed_up=bool(cfg.add_diffed_up))
            rec["features"] = feats.astype(np.float64)
            rec["grid"] = grid.astype(np.int64)
        else:
            pr = lcpfn_predictor(str(path))
            ynorm, k_obs, grid = pr._input_curve(partial)
            rec["ynorm"] = ynorm.astype(np.float64)
            rec["k_obs"] = np.int64(k_obs)
            rec["grid"] = np.asarray(grid, np.int64)
        from model.predictor import GaussianPrediction, HistogramPrediction

        pred = pr.predict(partial, predict_only_last=True) if family == "lcpfn" else pr.predict(partial)
        if isinstance(pred, HistogramPrediction):
            p = float(np.asarray(pred._failure_prob).reshape(-1)[-1])
            probs = np.asarray(pred._probs, np.float64)
            probs = probs.reshape(-1, probs.shape[-1])[-1]
            rec["probs"] = (probs * (1.0 - p)).astype(np.float32)
            rec["edges_ends"] = np.array([pred._edges[0], pred._edges[-1], pred._edges.size])
        elif isinstance(pred, GaussianPrediction):
            p = float(np.asarray(pred._fail).reshape(-1)[-1])
            rec["mean"] = np.float64(pred._mean.reshape(-1)[-1])
            rec["var"] = np.float64(pred._var.reshape(-1)[-1])
        else:
            raise TypeError(type(pred))
        rec["p_crash"] = np.float64(p)
        rec["old_path"] = np.array(str(path))
        np.savez_compressed(out / f"{name}.npz", **rec)
        log.info("parity fixture %s (p_crash %.6f)", name, p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="export every DecPFN and LC-PFN leaf")
    e.add_argument("--models", type=Path, required=True, help="old models/ directory")
    e.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("parity", help="write the parity fixtures")
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    if args.cmd == "export":
        m = export(args.models.expanduser(), args.out.expanduser())
        log.info("exported %d, skipped %d, failed %d", len(m["exported"]), len(m["skipped"]),
                 len(m["failed"]))
        return 1 if m["failed"] else 0
    parity(args.models.expanduser(), args.out.expanduser())
    return 0


if __name__ == "__main__":
    sys.exit(main())
