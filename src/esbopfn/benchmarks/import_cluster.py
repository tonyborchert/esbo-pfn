"""Import the thesis artifacts that were copied from the cluster (``pfn import thesis-results
--raw DIR``) and have no native counterpart in the database schema of the other importers.

==============================================  ==============================================
old file (below a ``--raw`` directory)          stored as
==============================================  ==============================================
``{decpfn,lcpfn}/cartpole/universal/grid/*``    ``$PFN_HOME/hpo/<model>/cartpole/<size>/
and ``.../greedy/<10|100>/<k>_<param>/``        trials.csv`` (the table ``pfn hpo`` writes,
``trials.csv`` (old ``prediction/models``)      read by fig:app:hpo-gridsearch)
``priors_sheet_cache/priors_sheet.pkl``         ``landscapes`` slice units, ``instances_seed:
(old ``landscape_visualization/out``)           "thesis"`` (fig:app:prior-landscapes)
==============================================  ==============================================

HPO tables. The old sweeps (``util/optim``) write one ``trials.csv`` per tree with the columns
``status, embed_size, n_layers, output_type, lr, wd, dropout, epochs, tflops, training_time,
val_accuracies[(a, b)]``. ``val_accuracies[(20, 99)]`` is the validation NLL at the grid cutoffs
between 20 % and 99 % (the HPO objective), so it maps to ``val_nll``. Stage 2 is ``grid/<e>``
plus the separately swept LC-PFN Gaussian head (``grid/<e>gauss``) and the DecPFN extension
``grid/100ext``. The superseded ``grid/100gauss_loginnode`` is not read. Stage 1 is
``greedy/<e>/{1_lr,2_wd,3_dropout}``. The parameter count is recovered from ``tflops`` as the old
figure did (``n_params = tflops 1e12 / (6 steps batch length)``). The LC-PFN Gaussian trees ran
weeks after the main grids on a different node, so the thesis figure drew them at the training
time of the same architecture's binned cell (linear if the binned one failed) times the measured
Gaussian/binned factor (LC-PFN ``grid/_control_head`` at 100 epochs, the DecPFN grid ratio at 10
epochs). ``train_time_s`` holds that drawn value, ``train_time_measured_s`` the measured one and
``time_basis`` how it was obtained.

Landscapes. ``priors_sheet.pkl`` holds per problem three panels (``kind`` ``line`` for the 1-D
Boiler, ``image`` otherwise) with ``coords, cost_mean, cost_mean_nocrash, p_crash, extent, u, v,
instance_index, source``. Each panel becomes one ``landscapes`` slice unit keyed by its
``instance`` (the old optplay instance index), ``row`` (position in the thesis figure) and
``source``. The old cache has no slice center, so ``center`` and the per-cell ``thetas`` of the
2-D panels are NaN.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from esbopfn.benchmarks.db import Db

log = logging.getLogger(__name__)

PROVENANCE = "thesis"

# ------------------------------------------------------------------------------ HPO tables
#: Old epoch budget -> size of this package.
HPO_SIZES = {10: "100k", 100: "1M"}
#: Constants of the old sweeps (cartpole/universal, T = 1000, resolution 15, batch 128, 80 steps
#: per epoch). Sequence lengths of the old trainers: DecPFN ``len(arange(0, T - 1, 15))``,
#: LC-PFN ``ceil((T - 1) / 15) + 1``.
_BATCH = 128
_EPOCH_STEPS = 80
_LENGTH = {"decpfn": len(np.arange(0, 999, 15)), "lcpfn": int(math.ceil(999 / 15)) + 1}
_Y = "val_accuracies[(20, 99)]"
_Y_ALL = "val_accuracies[(0, 99)]"
#: Wall-clock cap of the stage-2 grids (thesis appendix, hyperparameter search).
_CAPS = {10: 25.0, 100: 250.0}
HPO_COLUMNS = ("stage", "name", "head", "d_model", "n_layers", "lr", "weight_decay", "dropout",
               "time_limit_s", "val_nll", "val_nll_0_99", "train_time_s", "n_params",
               "steps_done", "stopped_by", "status", "train_time_measured_s", "time_basis",
               "source", "provenance")


def _read_tree(path: Path) -> pd.DataFrame | None:
    f = path / "trials.csv"
    if not f.is_file():
        return None
    df = pd.read_csv(f)
    df["_tree"] = str(path)
    return df


def _gauss_factor(root: Path, epochs: int) -> tuple[float, str]:
    """Training-time factor Gaussian/binned head, as the thesis figure took it."""
    ctrl = _read_tree(root / "lcpfn" / "cartpole" / "universal" / "grid" / "_control_head")
    if epochs == 100 and ctrl is not None:
        ok = ctrl[ctrl.status == "completed"].set_index("output_type")["training_time"]
        if {"binned", "gaussian"} <= set(ok.index):
            return (float(ok["gaussian"] / ok["binned"]),
                    "LC-PFN grid/_control_head, both heads back to back in one job")
    dec = [_read_tree(root / "decpfn" / "cartpole" / "universal" / "grid" / str(epochs))]
    if epochs == 100:
        dec.append(_read_tree(root / "decpfn" / "cartpole" / "universal" / "grid" / "100ext"))
    frames = [d for d in dec if d is not None]
    if not frames:
        return 1.0, "no reference, measured time kept"
    d = pd.concat(frames)
    d = d[d.status == "completed"]
    wide = d.set_index(["embed_size", "n_layers", "output_type"])["training_time"]
    wide = wide.unstack("output_type").dropna(subset=["binned", "gaussian"])
    return (float((wide["gaussian"] / wide["binned"]).mean()),
            f"DecPFN grid/{epochs}, mean Gaussian/binned over {len(wide)} architectures")


def _hpo_rows(df: pd.DataFrame, model: str, epochs: int, stage: str) -> list[dict[str, Any]]:
    steps = epochs * _EPOCH_STEPS
    rows = []
    for _, r in df.iterrows():
        done = str(r.get("status")) == "completed"
        tflops = float(r.get("tflops", math.nan))
        n_params = tflops * 1e12 / (6 * steps * _BATCH * _LENGTH[model])
        tree = Path(str(r["_tree"]))
        rows.append({
            "stage": stage, "name": f"{tree.parent.name}/{tree.name}/{int(r['trial'])}",
            "head": r.get("output_type"), "d_model": int(r["embed_size"]),
            "n_layers": int(r["n_layers"]), "lr": float(r["lr"]), "weight_decay": float(r["wd"]),
            "dropout": float(r["dropout"]),
            "time_limit_s": _CAPS[epochs] if stage == "stage2" else None,
            "val_nll": float(r[_Y]) if done else math.nan,
            "val_nll_0_99": float(r[_Y_ALL]) if done and _Y_ALL in r else math.nan,
            "train_time_s": float(r["training_time"]) if done else math.nan,
            "n_params": int(round(n_params)) if np.isfinite(n_params) else None,
            "steps_done": steps if done else None, "stopped_by": None if done else "failed",
            "status": r.get("status"),
            "train_time_measured_s": float(r["training_time"]) if done else math.nan,
            "time_basis": "measured", "source": str(tree / "trials.csv"),
            "provenance": PROVENANCE})
    return rows


def _project_gauss(rows: list[dict[str, Any]], factor: float, how: str) -> None:
    """Draw each LC-PFN Gaussian cell at its architecture's binned (else linear) time x factor."""
    base: dict[tuple[int, int], tuple[float, str]] = {}
    for head in ("binned", "linear"):
        for r in rows:
            if (r["head"] == head and np.isfinite(r["train_time_measured_s"])
                    and not r["source"].endswith(("gauss/trials.csv",))):
                base.setdefault((r["d_model"], r["n_layers"]), (r["train_time_measured_s"], head))
    for r in rows:
        if r["head"] != "gaussian" or not np.isfinite(r["train_time_measured_s"]):
            continue
        hit = base.get((r["d_model"], r["n_layers"]))
        if hit is None:
            r["time_basis"] = "measured (no bar-head cell of this architecture to project from)"
            continue
        t0, head = hit
        r["train_time_s"] = t0 * factor
        r["time_basis"] = f"projected = {head} {t0:.1f} s x {factor:.4f} ({how})"


def import_hpo_tables(models_root: Path | str, home: Path | str | None = None,
                      problem: str = "cartpole", overwrite: bool = False) -> int:
    """Old HPO trees below ``models_root`` (``<models>/{decpfn,lcpfn}/<problem>/universal/``) ->
    ``$PFN_HOME/hpo/<model>/<problem>/<size>/trials.csv``. Returns the tables written.

    An existing table is kept unless ``overwrite``. A table written by ``pfn hpo`` (no
    ``provenance`` column) is never replaced."""
    from esbopfn.config import paths

    root = Path(models_root)
    written = 0
    for model in ("decpfn", "lcpfn"):
        base = root / model / problem / "universal"
        if not (base / "grid").is_dir():
            continue
        for epochs, size in HPO_SIZES.items():
            grid = [_read_tree(base / "grid" / str(epochs))]
            if model == "lcpfn":
                grid.append(_read_tree(base / "grid" / f"{epochs}gauss"))
            if model == "decpfn" and epochs == 100:
                grid.append(_read_tree(base / "grid" / "100ext"))
            grid = [g for g in grid if g is not None]
            if not grid:
                continue
            rows: list[dict[str, Any]] = []
            for step in sorted((base / "greedy" / str(epochs)).glob("*_*")):
                df = _read_tree(step)
                if df is not None:
                    rows += _hpo_rows(df, model, epochs, "stage1")
            stage2: list[dict[str, Any]] = []
            for df in grid:
                stage2 += _hpo_rows(df, model, epochs, "stage2")
            if model == "lcpfn":
                factor, how = _gauss_factor(root, epochs)
                _project_gauss(stage2, factor, how)
            rows += stage2
            out = paths(home).root / "hpo" / model / problem / size / "trials.csv"
            if out.is_file():
                head = out.read_text().splitlines()[0].split(",")
                if "provenance" not in head:
                    log.warning("%s was written by `pfn hpo`, not replaced", out)
                    continue
                if not overwrite:
                    continue
            out.parent.mkdir(parents=True, exist_ok=True)
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=HPO_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in HPO_COLUMNS})
            out.write_text(buf.getvalue())
            written += 1
    return written


# ------------------------------------------------------------------------------ landscapes
_OLD_PROBLEMS = {"boiler": "boiler", "threetank_pi": "threetank_pi",
                 "pt2_deadtime": "pt2_deadtime", "cartpole": "cartpole",
                 "furuta_lqr_rigreset": "furuta"}


def import_landscape_pickle(db: Db, path: Path | str, overwrite: bool = False) -> int:
    """The thesis ``priors_sheet.pkl`` -> ``landscapes`` slice units. Returns units written."""
    from esbopfn.benchmarks.import_thesis import store

    path = Path(path)
    with open(path, "rb") as fh:
        sheet = pickle.load(fh)  # noqa: S301 -- the thesis cache, written by the old code
    n = 0
    for col in sheet.get("columns", []):
        problem = _OLD_PROBLEMS.get(str(col.get("problem")))
        if problem is None:
            continue
        d = int(col["D"])
        for row, panel in enumerate(col.get("panels", [])):
            line = panel.get("kind") == "line"
            if line:
                thetas = np.asarray(panel["thetas"], float).reshape(-1, 1)
                coords = thetas[:, 0]
            else:
                coords = np.asarray(panel["coords"], float)
                thetas = np.full((len(coords), len(coords), d), np.nan)
            arrays = {
                "coords": coords, "thetas": thetas,
                "p_crash": np.asarray(panel["p_crash"], float),
                "cost_mean": np.asarray(panel["cost_mean"], float),
                "cost_mean_nocrash": np.asarray(panel.get("cost_mean_nocrash",
                                                          panel["cost_mean"]), float),
                "center": np.full(d, np.nan),
                "u": np.asarray(panel.get("u") or ([1.0] if line else np.full(d, np.nan)), float),
                "v": np.asarray(panel.get("v") or ([0.0] if line else np.full(d, np.nan)), float),
                "extent": np.asarray(float(panel.get("extent", math.nan))),
            }
            for k in ("cost_std", "p_crash_se"):
                if k in panel:
                    arrays[k] = np.asarray(panel[k], float)
            key = {"unit": "slice", "problem": problem, "instance": int(panel["instance_index"]),
                   "slice_seed": row, "grid": int(len(coords)),
                   "repeats": int(sheet.get("repeats", 6)), "instances_seed": PROVENANCE,
                   "provenance": PROVENANCE, "row": row,
                   "source": str(panel.get("source") or "optplay_screened")}
            meta = {"source": str(path), "old_problem": col.get("problem"),
                    "dataset": col.get("dataset"), "phi": panel.get("phi"),
                    "axis_labels": list(panel.get("axis_labels") or []),
                    "screen": sheet.get("screen"), "seed": sheet.get("seed")}

            def write(dir_: Path, a: dict[str, np.ndarray] = arrays,
                      m: dict[str, Any] = meta) -> None:
                np.savez(dir_ / "landscape.npz", **a)
                (dir_ / "meta.json").write_text(json.dumps(m, indent=1, default=str))

            n += store(db, "landscapes", key, write, overwrite)
    return n


def import_raw_dir(db: Db, raw: Path | str, home: Path | str | None = None,
                   overwrite: bool = False) -> dict[str, int]:
    """Every HPO tree and landscape cache below one ``--raw`` directory."""
    raw = Path(raw)
    counts: dict[str, int] = {}
    # Only the directory itself and its children are searched (a raw optplay tree holds
    # hundreds of thousands of files).
    roots = {p.parent.parent.parent.parent
             for pat in ("*/cartpole/universal/grid", "*/*/cartpole/universal/grid")
             for p in raw.glob(pat)}
    for models_root in sorted(roots):
        counts["hpo_tables"] = counts.get("hpo_tables", 0) + import_hpo_tables(
            models_root, home=home, overwrite=overwrite)
    pkls = {p for pat in ("priors_sheet_cache/priors_sheet.pkl",
                          "*/priors_sheet_cache/priors_sheet.pkl",
                          "*/*/priors_sheet_cache/priors_sheet.pkl")
            for p in raw.glob(pat)}
    for pkl in sorted(pkls):
        counts["landscapes"] = counts.get("landscapes", 0) + import_landscape_pickle(
            db, pkl, overwrite)
    return counts
