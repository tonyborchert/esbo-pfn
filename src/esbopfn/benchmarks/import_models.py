"""Import the thesis checkpoints (XDecPFN, FT-PFN, DecPFN, LC-PFN) into the model tree.

The thesis checkpoints were written by an earlier implementation whose architecture differs from
the models of this package (:mod:`esbopfn.predictors.xdecpfn.legacy`,
:mod:`esbopfn.predictors.ftpfn.legacy`). They are imported unchanged as ``variant=thesis`` runs,

- ``xdecpfn/<problem>/thesis/default`` from ``f_s20260931_seg<k>_final.pt`` plus its ln-J
  calibration ``calib_c1r2_<key>_lnJ.json``,
- ``ftpfn/<problem>/thesis/default`` from ``final.pt`` (its calibration is stored inside),

and served by the legacy predictors through the usual registry
(``get_predictor("xdecpfn/cartpole/thesis/default", "cartpole")``).

The DecPFN and LC-PFN checkpoints come from the neutral export written by
``tools/export_thesis_single_trajectory.py`` (run in the old environment, see
``docs/LEGACY_MODELS.md``). Every export leaf ``<family>/<problem>/<variant>/<size>/`` becomes the
model id of the same name, ``decpfn/<p>/thesis-<head>/<size>``,
``decpfn-nostates/<p>/thesis-<head>/<size>`` and ``lcpfn/<p>/thesis-<head>[-linearnorm]/<size>``,
served by :mod:`esbopfn.predictors.decpfn.legacy` and :mod:`esbopfn.predictors.lcpfn.legacy`.
Their ``config.yaml`` carries ``provenance: thesis`` and ``legacy: true``.

Entry point of ``pfn models import-thesis --from DIR``::

    import_thesis(from_dir, *extra_dirs, home=None, problems=None, overwrite=False)
        -> list[ImportedModel]

Every directory is searched for the export leaves (``*/*/thesis-*/*/arch.json``), for the layouts
of the timing bundle
(``[checkpoints/]fr2/<problem>/final.pt``, ``[checkpoints/]c1r2_qloop/<problem>/f_*_seg*_final.pt``,
``[checkpoints/]c1r2_calib/qloop/calib_c1r2_<key>_lnJ.json``) and for checkpoints and calibrations
lying directly in it (the Furuta XDecPFN folder). Per model and problem the finished checkpoint
with the most steps wins. The two single-file importers are public as well.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Problem keys of the old code -> problem names of this package.
OLD_PROBLEMS = {"furuta": "furuta", "furuta180sg": "furuta", "furuta_lqr_rigreset": "furuta",
                "cartpole": "cartpole", "pt2_deadtime": "pt2_deadtime",
                "threetank_pi": "threetank_pi", "boiler": "boiler"}
#: Steps of the thesis models (planning record): XDecPFN 16,550 on every problem.
THESIS_STEPS = {"xdecpfn": 16550, "ftpfn": 109375}


@dataclass
class ImportedModel:
    model_id: str
    path: Path
    source: Path
    step: int
    note: str = ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _problem(old: str) -> str:
    try:
        return OLD_PROBLEMS[old]
    except KeyError:
        raise ValueError(f"unknown problem key {old!r} in a thesis checkpoint") from None


def _target(model: str, problem: str, home: Path | str | None, overwrite: bool) -> Path:
    from esbopfn.training.registry import is_trained, model_dir

    out = model_dir(f"{model}/{problem}/thesis/default", home)
    if is_trained(out) and not overwrite:
        raise FileExistsError(f"{out} already holds an imported model (overwrite=True replaces it)")
    return out


def _note(model: str, problem: str, step: int) -> str:
    want = THESIS_STEPS[model]
    if step == want:
        return ""
    return (f"trained for {step} steps, the thesis model of {problem} has {want} "
            "(only this sweep cell was available)")


def import_xdecpfn_checkpoint(ckpt: Path | str, calib: Path | str, problem: str | None = None,
                              home: Path | str | None = None,
                              overwrite: bool = False) -> ImportedModel:
    """Import one old XDecPFN checkpoint with its calibration as ``xdecpfn/<p>/thesis/default``."""
    from esbopfn.predictors.xdecpfn.legacy import LegacyXDecPFN
    from esbopfn.training.registry import save_model

    ckpt, calib = Path(ckpt).expanduser(), Path(calib).expanduser()
    ck = _load(ckpt)
    if ck.get("mcfg", {}).get("arm") != "noinj1q":
        raise ValueError(f"{ckpt} is not an XDecPFN (noinj1q) checkpoint")
    recorded = Path(str(ck.get("calib") or ck.get("args", {}).get("calib", ""))).name
    if recorded and recorded != calib.name:
        raise ValueError(f"{ckpt} was trained with {recorded}, got {calib.name}")
    cal = json.loads(calib.read_text())
    problem = problem or _problem(str(ck["problem"]))
    LegacyXDecPFN(ck["mcfg"]).load_state_dict(ck["model"], strict=True)
    step = int(ck["step"])
    note = _note("xdecpfn", problem, step)
    out = _target("xdecpfn", problem, home, overwrite)
    model_cfg = {"class": "LegacyXDecPFN", "mcfg": dict(ck["mcfg"]), "k": int(ck["k"]),
                 "T_tok": int(ck.get("T_tok", 0)), "old_problem": str(ck["problem"]),
                 "n_grid": ck.get("n_grid")}
    train_cfg = {"model": "xdecpfn", "problem": problem, "variant": "thesis", "size": "default",
                 "provenance": "thesis", "source": str(ckpt), "sha256": _sha256(ckpt),
                 "calib_source": str(calib), "step": step, "done": bool(ck.get("done", False)),
                 "seed": ck.get("seed"), "args": ck.get("args", {}), "note": note}
    metrics = {"steps_done": step, "n_params": ck.get("params"), "loss_ema": ck.get("loss_ema"),
               "flops_per_step": ck.get("flops_per_step"),
               "counters": ck.get("counters"), "elapsed_total_s": ck.get("elapsed_total_s")}
    save_model(out, ck["model"], model_cfg, cal, train_cfg, metrics)
    return ImportedModel(f"xdecpfn/{problem}/thesis/default", out, ckpt, step, note)


def import_ftpfn_checkpoint(ckpt: Path | str, problem: str | None = None,
                            home: Path | str | None = None,
                            overwrite: bool = False) -> ImportedModel:
    """Import one old FT-PFN ``final.pt`` as ``ftpfn/<p>/thesis/default``."""
    from esbopfn.predictors.ftpfn.legacy import LegacyFTPFN
    from esbopfn.training.registry import save_model

    ckpt = Path(ckpt).expanduser()
    ck = _load(ckpt)
    if "cfg" not in ck or "null_ctx" not in ck.get("model", {}):
        raise ValueError(f"{ckpt} is not an FT-PFN checkpoint")
    cal = ck.get("calib_dict")
    if cal is None and (ckpt.parent / "calib.json").is_file():
        cal = json.loads((ckpt.parent / "calib.json").read_text())
    if cal is None:
        raise ValueError(f"{ckpt} carries no calibration and no calib.json lies beside it")
    problem = problem or _problem(str(ck["problem"]))
    LegacyFTPFN(ck["cfg"]).load_state_dict(ck["model"], strict=True)
    step = int(ck["step"])
    note = _note("ftpfn", problem, step)
    out = _target("ftpfn", problem, home, overwrite)
    model_cfg = {"class": "LegacyFTPFN", "cfg": dict(ck["cfg"]), "old_problem": str(ck["problem"]),
                 "contexts": ck.get("contexts")}
    train_cfg = {"model": "ftpfn", "problem": problem, "variant": "thesis", "size": "default",
                 "provenance": "thesis", "source": str(ckpt), "sha256": _sha256(ckpt),
                 "step": step, "attn": ck.get("attn"), "note": note}
    save_model(out, ck["model"], model_cfg, cal, train_cfg, {"steps_done": step})
    return ImportedModel(f"ftpfn/{problem}/thesis/default", out, ckpt, step, note)


# --------------------------------------------------------------------------- DecPFN, LC-PFN
#: Export families -> model families of this package.
SINGLE_TRAJECTORY_FAMILIES = ("decpfn", "decpfn-nostates", "lcpfn")
#: Optimizer steps of one old epoch (10,240 examples at batch 128).
OLD_EPOCH_STEPS = 80
OLD_EPOCHS = {"100k": 10, "1M": 100}


def single_trajectory_leaves(root: Path) -> list[Path]:
    """Export leaves ``<family>/<problem>/thesis-*/<size>/`` below ``root`` (sorted)."""
    out = []
    for fam in SINGLE_TRAJECTORY_FAMILIES:
        out += [p.parent for p in sorted(root.glob(f"{fam}/*/thesis-*/*/arch.json"))
                if (p.parent / "state_dict.pt").is_file()]
    return out


def _legacy_note(family: str, variant: str) -> str:
    if family == "lcpfn" and variant.endswith("-linearnorm"):
        return ("LC-PFN with linear input normalization (the z-scored input leaf of this "
                "problem, size and head was not in the export)")
    return ""


def import_single_trajectory_leaf(leaf: Path | str, home: Path | str | None = None,
                                  overwrite: bool = False) -> ImportedModel:
    """Import one export leaf of ``tools/export_thesis_single_trajectory.py`` as
    ``<family>/<problem>/<variant>/<size>``. The legacy model class must load it strictly."""
    import torch

    from esbopfn.training.registry import is_trained, model_dir, save_model

    leaf = Path(leaf).expanduser()
    family, problem, variant, size = leaf.parts[-4:]
    if family not in SINGLE_TRAJECTORY_FAMILIES or not variant.startswith("thesis"):
        raise ValueError(f"{leaf} is not a DecPFN or LC-PFN export leaf")
    arch = json.loads((leaf / "arch.json").read_text())
    calib = json.loads((leaf / "calib.json").read_text())
    source = json.loads((leaf / "source.json").read_text())
    state = torch.load(leaf / "state_dict.pt", map_location="cpu", weights_only=True)
    if arch.get("family") != family or arch.get("problem") != problem:
        raise ValueError(f"{leaf}: arch.json names {arch.get('family')}/{arch.get('problem')}")
    if family == "lcpfn":
        from esbopfn.predictors.lcpfn.legacy import LegacyLCPFN

        LegacyLCPFN(arch).load_state_dict(state, strict=True)
    else:
        from esbopfn.predictors.decpfn.legacy import LegacyDecPFN

        LegacyDecPFN(arch, calib).load_state_dict(state, strict=True)
    model_id = f"{family}/{problem}/{variant}/{size}"
    out = model_dir(model_id, home)
    if is_trained(out) and not overwrite:
        raise FileExistsError(f"{out} already holds an imported model (overwrite=True replaces it)")
    epochs = OLD_EPOCHS.get(size)
    steps = None if epochs is None else epochs * OLD_EPOCH_STEPS
    note = _legacy_note(family, variant)
    train_cfg = {"model": family, "problem": problem, "variant": variant, "size": size,
                 "provenance": "thesis", "legacy": True, "head": arch.get("head"),
                 "old_dataset": arch.get("old_dataset"), "source": source.get("old_path"),
                 "sha256": source.get("sha256"), "source_txt": source.get("source_txt"),
                 "export_dir": str(leaf), "epochs": epochs,
                 "examples": None if epochs is None else epochs * 10240,
                 "input_norm": arch.get("input_norm"), "note": note}
    if "old_config" in arch:
        train_cfg["old_config"] = arch["old_config"]
    save_model(out, state, arch, calib, train_cfg, {"steps_done": steps, "epochs": epochs})
    return ImportedModel(model_id, out, Path(str(source.get("old_path"))), steps or 0, note)


# --------------------------------------------------------------------------- discovery
def _candidates(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """FT-PFN checkpoints, XDecPFN checkpoints and ln-J calibrations below ``root``."""
    bases = [root, root / "checkpoints"]
    ft, xd, cal = [], [], []
    for b in bases:
        ft += sorted(b.glob("fr2/*/final.pt"))
        xd += sorted(b.glob("c1r2_qloop/*/f_*_seg*_final.pt"))
        cal += sorted(b.glob("c1r2_calib/qloop/calib_c1r2_*_lnJ.json"))
    xd += sorted(root.glob("f_*_seg*_final.pt"))
    cal += sorted(root.glob("calib_c1r2_*_lnJ.json"))
    return ft, xd, cal


def import_thesis(from_dir: Path | str, *extra_dirs: Path | str, home: Path | str | None = None,
                  problems: Iterable[str] | None = None,
                  overwrite: bool = False) -> list[ImportedModel]:
    """Import every thesis checkpoint found in the given directories.

    ``problems`` restricts the import, ``overwrite`` replaces models imported before. Returns the
    imported models (one per model id).
    """
    want = set(problems) if problems is not None else None
    ft: list[Path] = []
    xd: list[Path] = []
    cal: dict[str, Path] = {}
    for d in (from_dir, *extra_dirs):
        root = Path(d).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a directory")
        f, x, c = _candidates(root)
        ft += f
        xd += x
        for p in c:
            cal.setdefault(p.name, p)
    best: dict[tuple[str, str], tuple[int, Path]] = {}
    for kind, paths in (("ftpfn", ft), ("xdecpfn", xd)):
        for p in paths:
            ck = _load(p)
            if kind == "xdecpfn" and not ck.get("done", True):
                continue
            prob = _problem(str(ck["problem"]))
            if want is not None and prob not in want:
                continue
            step = int(ck.get("step", 0))
            if (kind, prob) not in best or step > best[(kind, prob)][0]:
                best[(kind, prob)] = (step, p)
    done: list[ImportedModel] = []
    for d in (from_dir, *extra_dirs):
        for leaf in single_trajectory_leaves(Path(d).expanduser()):
            if want is not None and leaf.parts[-3] not in want:
                continue
            done.append(import_single_trajectory_leaf(leaf, home, overwrite))
    for (kind, prob), (_, path) in sorted(best.items()):
        if kind == "ftpfn":
            done.append(import_ftpfn_checkpoint(path, prob, home, overwrite))
            continue
        name = Path(str(_load(path)["calib"])).name
        if name not in cal:
            log.warning("no calibration %s for %s, skipped", name, path)
            continue
        done.append(import_xdecpfn_checkpoint(path, cal[name], prob, home, overwrite))
    for m in done:
        log.info("imported %s from %s%s", m.model_id, m.source, f" ({m.note})" if m.note else "")
    return done
