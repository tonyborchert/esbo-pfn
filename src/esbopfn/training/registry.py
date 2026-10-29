"""The trained-model tree under ``$PFN_HOME/models`` and its model ids.

A model id is ``<model>/<problem>/<variant>/<size>[/<run>]``, e.g. ``decpfn/cartpole/binned/1M``,
``lcpfn-last/furuta/binned/100k/1`` or ``xdecpfn/cartpole/default/default``. The run defaults to
``0``. The directory of an id is ``$PFN_HOME/models/<model>/<problem>/<variant>/<size>/<run>/``
and holds

- ``model.pt``: ``{"state_dict": ..., "model_cfg": {...}}``
- ``config.yaml``: resolved hyperparameters, dataset id, seed, git hash, provenance
- ``calib.json``: bin edges and z-score statistics
- ``metrics.json``: loss curve, validation results, timing

A model counts as trained once ``model.pt`` exists (it is written last).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from esbopfn.config import paths, repo_root
from esbopfn.training.common import write_json
from esbopfn.util.yaml import dump_yaml, load_yaml

DEFAULT_RUN = "0"
MODEL_FILE = "model.pt"
CONFIG_FILE = "config.yaml"
CALIB_FILE = "calib.json"
METRICS_FILE = "metrics.json"


class ModelNotTrained(FileNotFoundError):
    """Raised when a model id has no trained model directory. ``hint`` says how to train it."""

    def __init__(self, model_id: str, hint: str = "") -> None:
        self.model_id = model_id
        self.hint = hint or train_hint(model_id)
        super().__init__(f"model {model_id!r} is not trained. {self.hint}")


@dataclass(frozen=True)
class ModelId:
    model: str
    problem: str
    variant: str
    size: str
    run: str = DEFAULT_RUN

    @classmethod
    def parse(cls, model_id: str | ModelId) -> ModelId:
        if isinstance(model_id, ModelId):
            return model_id
        parts = [p for p in str(model_id).strip().strip("/").split("/") if p]
        if len(parts) not in (4, 5):
            raise ValueError(
                f"model id {model_id!r} must be <model>/<problem>/<variant>/<size>[/<run>]"
            )
        return cls(*parts)

    def __str__(self) -> str:
        return "/".join([self.model, self.problem, self.variant, self.size, self.run])

    @property
    def short(self) -> str:
        """The id without the run when the run is the default."""
        base = "/".join([self.model, self.problem, self.variant, self.size])
        return base if self.run == DEFAULT_RUN else f"{base}/{self.run}"


@dataclass
class ModelFiles:
    path: Path
    state_dict: dict[str, Any]
    model_cfg: dict[str, Any]
    calib: dict[str, Any]
    train_cfg: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)


def train_hint(model_id: str | ModelId) -> str:
    try:
        mid = ModelId.parse(model_id)
    except ValueError:
        return "Train it with `pfn train`."
    cmd = f"pfn train {mid.model} --problem {mid.problem}"
    if mid.variant not in ("default",):
        cmd += f" --head {mid.variant}"
    if mid.size not in ("default",):
        cmd += f" --size {mid.size}"
    if mid.run != DEFAULT_RUN:
        cmd += f" --run {mid.run}"
    return f"Train it with `{cmd}`."


def models_root(home: Path | str | None = None) -> Path:
    return paths(home).root / "models"


def model_dir(model_id: str | ModelId, home: Path | str | None = None) -> Path:
    """Directory of ``model_id`` (not created)."""
    mid = ModelId.parse(model_id)
    return models_root(home) / mid.model / mid.problem / mid.variant / mid.size / mid.run


def is_trained(path: Path) -> bool:
    return (Path(path) / MODEL_FILE).is_file()


def resolve(model_id: str | ModelId, home: Path | str | None = None, hint: str = "") -> Path:
    """Directory of a trained ``model_id``, raising :class:`ModelNotTrained` when absent."""
    path = model_dir(model_id, home)
    if not is_trained(path):
        raise ModelNotTrained(ModelId.parse(model_id).short, hint)
    return path


def git_hash() -> str | None:
    root = repo_root()
    if root is None:
        return None
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def save_model(
    out_dir: Path,
    state_dict: dict[str, Any],
    model_cfg: dict[str, Any],
    calib: dict[str, Any],
    train_cfg: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Write a model directory. ``model.pt`` is written last so a partial directory never counts
    as trained. ``metrics`` is merged into an existing ``metrics.json`` (from :func:`fit`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(train_cfg)
    cfg.setdefault("git_hash", git_hash())
    dump_yaml(cfg, out_dir / CONFIG_FILE)
    write_json(out_dir / CALIB_FILE, calib)
    metrics_path = out_dir / METRICS_FILE
    merged = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    merged.update(metrics)
    write_json(metrics_path, merged)
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    tmp = out_dir / (MODEL_FILE + ".tmp")
    torch.save({"state_dict": cpu_state, "model_cfg": model_cfg}, tmp)
    tmp.replace(out_dir / MODEL_FILE)
    return out_dir


def load_model(path: Path | str) -> ModelFiles:
    """Read a model directory written by :func:`save_model`."""
    path = Path(path)
    blob = torch.load(path / MODEL_FILE, map_location="cpu", weights_only=False)
    calib = json.loads((path / CALIB_FILE).read_text()) if (path / CALIB_FILE).exists() else {}
    train_cfg = load_yaml(path / CONFIG_FILE) if (path / CONFIG_FILE).exists() else {}
    mpath = path / METRICS_FILE
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
    return ModelFiles(path, blob["state_dict"], blob["model_cfg"], calib, train_cfg or {}, metrics)


def list_models(home: Path | str | None = None) -> list[ModelId]:
    """Ids of every trained model below the models root, sorted."""
    root = models_root(home)
    if not root.is_dir():
        return []
    out = []
    for f in root.glob(f"*/*/*/*/*/{MODEL_FILE}"):
        rel = f.parent.relative_to(root).parts
        out.append(ModelId(*rel))
    return sorted(out, key=str)


# ----------------------------------------------------------------------------- hyperparameters
def hparams_base(model: str) -> str:
    """The hyperparameter file a model reads (variants inherit their base model's rows)."""
    from esbopfn.config import configs_dir

    if (configs_dir() / "hparams" / f"{model}.yaml").is_file():
        return model
    for f in sorted((configs_dir() / "hparams").glob("*.yaml")):
        if model in (load_yaml(f).get("inherited_by") or []):
            return f.stem
    raise KeyError(f"no hyperparameter file for model {model!r}")


def load_hparams_file(model: str, path: Path | str | None = None) -> dict[str, Any]:
    """The parsed ``configs/hparams/<base>.yaml`` of ``model``, or ``path`` when given."""
    from esbopfn.config import configs_dir

    if path is None:
        path = configs_dir() / "hparams" / f"{hparams_base(model)}.yaml"
    return load_yaml(path)


def hparams_row(model: str, size: str | None = None, head: str | None = None,
                path: Path | str | None = None) -> dict[str, Any]:
    """The resolved hyperparameters of one (size, head) row: the ``fixed`` block of the default
    file, overridden by the file's own ``fixed`` block and then by the row. ``path`` may be any
    file with ``configs`` rows, e.g. an HPO ``hparams_candidate.yaml``."""
    default = load_hparams_file(model)
    data = load_hparams_file(model, path) if path is not None else default
    rows = [r for r in data.get("configs") or []
            if (size is None or str(r.get("size")) == str(size))
            and (head is None or r.get("head") == head)]
    if len(rows) != 1:
        src = path or f"configs/hparams/{hparams_base(model)}.yaml"
        raise KeyError(f"{src} has {len(rows)} rows for size={size} head={head}, expected one")
    out: dict[str, Any] = {**(default.get("fixed") or {}), **(data.get("fixed") or {}), **rows[0]}
    out["source"] = str(path) if path is not None else f"configs/hparams/{hparams_base(model)}.yaml"
    return out


# ----------------------------------------------------------------------------- predictor ids
#: Predictor families of trained models whose id prefix is not a module name under
#: ``esbopfn.predictors`` (``decpfn`` and ``lcpfn`` are found by convention).
PREDICTOR_FAMILIES: dict[str, str] = {
    "decpfn-nostates": "esbopfn.predictors.decpfn:make_nostates_predictor",
    "decpfn-nodeadtime": "esbopfn.predictors.decpfn:make_nodeadtime_predictor",
    "decpfn-nodeadtime-nostates":
        "esbopfn.predictors.decpfn:make_nodeadtime_nostates_predictor",
    "lcpfn-last": "esbopfn.predictors.lcpfn:make_last_predictor",
}


def register_predictor_families() -> None:
    """Make :data:`PREDICTOR_FAMILIES` resolvable through ``predictors.get_predictor``."""
    from esbopfn.predictors.base import LAZY_FAMILIES, register_predictor_family

    for prefix, target in PREDICTOR_FAMILIES.items():
        if LAZY_FAMILIES.get(prefix) != target:
            register_predictor_family(prefix, target, replace=True)
