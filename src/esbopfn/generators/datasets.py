"""The dataset catalog ``configs/datasets.yaml``: ids, sizes, seeds, chunking and locations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from esbopfn.config import configs_dir, paths
from esbopfn.util.yaml import load_yaml

KINDS = ("traj", "sets", "contexts", "screen", "fields")
PROBLEM_TOKEN = "{p}"


class UnknownDataset(KeyError):
    """Raised for an id that is not in the catalog."""


@dataclass(frozen=True)
class DatasetDef:
    """One concrete dataset (template already expanded for its problem)."""

    id: str
    kind: str
    problem: str
    name: str
    count: int = 0
    seed: int = 0
    chunk: int = 1
    options: dict[str, Any] = field(default_factory=dict)
    prefix_of: str | None = None
    placeholder: bool = False
    description: str = ""

    @property
    def n_chunks(self) -> int:
        return -(-self.count // self.chunk) if self.count else 0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "problem": self.problem, "name": self.name,
                "count": self.count, "seed": self.seed, "chunk": self.chunk,
                "options": dict(self.options), "prefix_of": self.prefix_of,
                "placeholder": self.placeholder, "description": self.description}


def split_id(dataset_id: str) -> tuple[str, str, str]:
    """``<kind>/<problem>/<name>`` -> its three parts."""
    parts = dataset_id.strip("/").split("/")
    if len(parts) != 3 or not all(parts):
        raise UnknownDataset(f"dataset ids look like <kind>/<problem>/<name>, got {dataset_id!r}")
    return parts[0], parts[1], parts[2]


@dataclass
class DatasetCatalog:
    """All datasets of ``datasets.yaml`` expanded per problem, plus the model -> datasets map."""

    problems: tuple[str, ...]
    datasets: dict[str, DatasetDef]
    models: dict[str, tuple[str, ...]]

    def resolve(self, dataset_id: str) -> DatasetDef:
        if dataset_id not in self.datasets:
            kind, problem, _ = split_id(dataset_id)
            known = sorted(i for i in self.datasets if i.startswith(f"{kind}/{problem}/"))
            raise UnknownDataset(f"unknown dataset {dataset_id!r}"
                                 + (f", known for {kind}/{problem}: {', '.join(known)}" if known else ""))
        return self.datasets[dataset_id]

    def storage(self, dataset_id: str) -> DatasetDef:
        """The dataset whose directory holds ``dataset_id`` (itself unless it is a prefix)."""
        d = self.resolve(dataset_id)
        while d.prefix_of is not None:
            d = self.resolve(d.prefix_of)
        return d

    def ids(self, problem: str | None = None, kind: str | None = None) -> list[str]:
        return [i for i, d in self.datasets.items()
                if (problem is None or d.problem == problem) and (kind is None or d.kind == kind)]

    def for_model(self, model: str, problem: str) -> list[str]:
        """Dataset ids a model needs for training on ``problem``."""
        if model not in self.models:
            raise KeyError(f"no datasets listed for model {model!r}, known: {', '.join(self.models)}")
        return [t.replace(PROBLEM_TOKEN, problem) for t in self.models[model]]


def _expand(key: str, entry: dict[str, Any], problems: tuple[str, ...]) -> list[DatasetDef]:
    targets = tuple(entry.get("problems", problems)) if PROBLEM_TOKEN in key else (None,)
    out = []
    for p in targets:
        did = key.replace(PROBLEM_TOKEN, p) if p is not None else key
        kind, problem, name = split_id(did)
        prefix = entry.get("prefix_of")
        if prefix is not None and p is not None:
            prefix = prefix.replace(PROBLEM_TOKEN, p)
        desc = str(entry.get("description", "")).replace(PROBLEM_TOKEN, problem)
        out.append(DatasetDef(
            id=did, kind=str(entry.get("kind", kind)), problem=problem, name=name,
            count=int(entry.get("count", 0)), seed=int(entry.get("seed", 0)),
            chunk=int(entry.get("chunk", 1)), options=dict(entry.get("options") or {}),
            prefix_of=prefix, placeholder=bool(entry.get("placeholder", False)), description=desc))
    return out


def load_catalog(path: Path | str | None = None) -> DatasetCatalog:
    """Parse ``datasets.yaml`` (the shipped one by default)."""
    raw = load_yaml(path if path is not None else configs_dir() / "datasets.yaml")
    problems = tuple(raw.get("problems", ()))
    datasets: dict[str, DatasetDef] = {}
    for key, entry in (raw.get("datasets") or {}).items():
        for d in _expand(key, entry or {}, problems):
            if d.id in datasets:
                raise ValueError(f"dataset {d.id} defined twice")
            datasets[d.id] = d
    # prefixes inherit kind, seed and chunk of their parent
    for did, d in list(datasets.items()):
        if d.prefix_of is not None:
            if d.prefix_of not in datasets:
                raise ValueError(f"{did}: prefix_of {d.prefix_of!r} is not a dataset")
            parent = datasets[d.prefix_of]
            if d.count > parent.count:
                raise ValueError(f"{did}: a prefix cannot be larger than {parent.id}")
            datasets[did] = DatasetDef(
                id=did, kind=parent.kind, problem=d.problem, name=d.name, count=d.count,
                seed=parent.seed, chunk=parent.chunk, options=dict(parent.options),
                prefix_of=d.prefix_of, placeholder=parent.placeholder,
                description=d.description)
    for d in datasets.values():
        if d.kind not in KINDS:
            raise ValueError(f"{d.id}: unknown kind {d.kind!r}")
        if not d.placeholder and (d.count <= 0 or d.chunk <= 0):
            raise ValueError(f"{d.id}: count and chunk must be positive")
    models = {m: tuple(v) for m, v in (raw.get("models") or {}).items()}
    return DatasetCatalog(problems=problems, datasets=datasets, models=models)


def dataset_dir(dataset: DatasetDef | str, home: Path | str | None = None,
                catalog: DatasetCatalog | None = None) -> Path:
    """``$PFN_HOME/datasets/<problem>/<kind>/<name>`` of the dataset that stores ``dataset``."""
    cat = catalog or load_catalog()
    did = dataset.id if isinstance(dataset, DatasetDef) else dataset
    owner = cat.storage(did)
    return paths(home).root / "datasets" / owner.problem / owner.kind / owner.name
