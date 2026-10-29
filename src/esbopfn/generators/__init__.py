"""Generated training data: the C++ generators driven chunk by chunk into pools.

``generate(dataset_id, workers, seed, resume)`` runs the binary for every chunk that is not
finished yet (the binary skips finished chunks itself, so an interrupted run resumes where it
stopped), ``status(dataset_id)`` reports progress, ``open_dataset(dataset_id)`` returns a
:class:`~esbopfn.generators.pools.Pool` or :class:`~esbopfn.generators.pools.ContextStore`.

Chunk seeds: chunk ``k`` of a dataset with seed ``S`` is generated from
``esbopfn.util.seeds.derive_seed(S, k)``, computed inside the binaries by a port of numpy's
SeedSequence (:func:`chunk_seed` is the Python side of that contract). The data therefore depend
on ``(S, chunk size)`` only, never on the number of workers, processes or shards.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from esbopfn.generators import binaries
from esbopfn.generators.datasets import (
    DatasetCatalog,
    DatasetDef,
    dataset_dir,
    load_catalog,
)
from esbopfn.generators.pools import (
    CHUNK_LOG,
    FINISHED_FILE,
    META_FILE,
    SHARDS_DIR,
    ContextStore,
    Pool,
    chunk_units,
    n_chunks,
    read_chunk_log,
    write_json_atomic,
)
from esbopfn.util.seeds import derive_seed

log = logging.getLogger(__name__)

__all__ = ["DatasetStatus", "chunk_seed", "generate", "open_dataset", "parse_shard", "status",
           "chunks_of_shard", "format_chunk_list"]


class NotGeneratable(RuntimeError):
    """Raised for placeholder datasets that the benchmark units write (``fields/...``)."""


def chunk_seed(seed: int, chunk: int) -> int:
    """Seed of chunk ``chunk``, identical to what the generator binaries derive."""
    return derive_seed(int(seed), int(chunk))


def parse_shard(text: str | None) -> tuple[int, int] | None:
    """``"i/n"`` -> ``(i, n)`` with ``0 <= i < n``."""
    if text is None:
        return None
    i, n = (int(v) for v in str(text).split("/"))
    if not 0 <= i < n:
        raise ValueError(f"shard must be i/n with 0 <= i < n, got {text!r}")
    return i, n


def chunks_of_shard(total: int, shard: tuple[int, int] | None) -> list[int]:
    """Chunk indices handled by shard ``i/n``: every n-th chunk starting at i."""
    if shard is None:
        return list(range(total))
    i, n = shard
    return list(range(i, total, n))


def format_chunk_list(chunks: list[int]) -> str:
    """Compact ``a-b,c`` form of sorted chunk indices (what ``--chunks`` accepts)."""
    parts: list[str] = []
    start = prev = None
    for c in sorted(chunks):
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            parts.append(f"{start}-{prev}" if prev != start else str(start))
            start = prev = c
    if start is not None:
        parts.append(f"{start}-{prev}" if prev != start else str(start))
    return ",".join(parts)


@dataclass
class DatasetStatus:
    """Progress of one dataset."""

    id: str
    kind: str
    problem: str
    path: str
    count: int
    chunk: int
    chunks_total: int
    chunks_done: int
    units_done: int
    finished: bool
    placeholder: bool = False
    exists: bool = False
    bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fraction(self) -> float:
        return self.chunks_done / self.chunks_total if self.chunks_total else 0.0


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _done_chunks(owner: DatasetDef, directory: Path) -> dict[int, int]:
    if owner.kind == "contexts":
        done = {}
        for p in (directory / SHARDS_DIR).glob("chunk_*"):
            meta = p / "shard.json"
            if meta.is_file():
                done[int(p.name.split("_")[1])] = int(json.loads(meta.read_text())["examples"])
        return done
    return read_chunk_log(directory / CHUNK_LOG)


def status(dataset_id: str, home: Path | str | None = None,
           catalog: DatasetCatalog | None = None) -> DatasetStatus:
    """Chunks and units finished for ``dataset_id`` (prefix datasets count their own chunks)."""
    cat = catalog or load_catalog()
    d = cat.resolve(dataset_id)
    directory = dataset_dir(d, home, cat)
    if d.placeholder:
        return DatasetStatus(id=d.id, kind=d.kind, problem=d.problem, path=str(directory),
                             count=0, chunk=0, chunks_total=0, chunks_done=0, units_done=0,
                             finished=(directory / FINISHED_FILE).is_file(), placeholder=True,
                             exists=directory.exists(), bytes=_dir_bytes(directory))
    owner = cat.storage(dataset_id)
    total = n_chunks(d.count, owner.chunk)
    logged = _done_chunks(owner, directory)
    done = [k for k in range(total)
            if logged.get(k) == chunk_units(owner.count, owner.chunk, k)]
    units = sum(min(chunk_units(owner.count, owner.chunk, k),
                    max(0, d.count - k * owner.chunk)) for k in done)
    finished = len(done) == total
    return DatasetStatus(id=d.id, kind=d.kind, problem=d.problem, path=str(directory),
                         count=d.count, chunk=owner.chunk, chunks_total=total,
                         chunks_done=len(done), units_done=units, finished=finished,
                         exists=directory.exists(),
                         bytes=_dir_bytes(directory) if d.prefix_of is None else 0)


def _kind_args(owner: DatasetDef) -> list[str]:
    o = owner.options
    args: list[str] = []
    if o.get("dead_time") is False:
        args.append("--no-dead-time")
    if owner.kind == "sets":
        args += ["--resolution", str(o.get("resolution", "grid")), "--labels", str(o.get("labels", 128))]
    if owner.kind == "contexts":
        args += ["--n-unrun", str(o.get("n_unrun", 16))]
    if "n_max" in o:
        args += ["--n-max", str(o["n_max"])]
    return args


def _pilot_rho(owner: DatasetDef, workers: int, home: Path | str | None) -> float:
    args = ["--workers", str(workers)]
    if owner.options.get("dead_time") is False:
        args.append("--no-dead-time")
    res = binaries.run_json(owner.problem, "pilot", *args, home=home)
    return float(res["rho_unsuccessful"])


def _write_meta(owner: DatasetDef, directory: Path, seed: int, extra: dict[str, Any],
                home: Path | str | None) -> dict[str, Any]:
    from esbopfn.problems.registry import get_spec

    spec_hash = get_spec(owner.problem).spec_hash()
    meta_path = directory / META_FILE
    meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    for key in ("seed", "chunk", "problem", "kind"):
        want = {"seed": seed, "chunk": owner.chunk, "problem": owner.problem, "kind": owner.kind}[key]
        if key in meta and meta[key] != want:
            raise RuntimeError(f"{directory} was generated with {key}={meta[key]}, requested {want}"
                               " (use --no-resume to start over)")
    if "problem_spec_hash" in meta and meta["problem_spec_hash"] != spec_hash:
        raise RuntimeError(f"{directory} was generated for a different problem spec"
                           " (use --no-resume to start over)")
    meta.update({
        "dataset_id": owner.id, "kind": owner.kind, "problem": owner.problem, "seed": seed,
        "count": owner.count, "chunk": owner.chunk, "options": owner.options,
        "generator": binaries.BINARY_OF[owner.problem],
        "generator_version": binaries.version(owner.problem, home),
        "problem_spec_hash": spec_hash,
        "chunk_seed": "derive_seed(seed, chunk)",
    })
    meta.update(extra)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(meta_path, meta)
    return meta


def generate(dataset_id: str, workers: int = 1, seed: int | None = None, resume: bool = True,
             shard: tuple[int, int] | None = None, home: Path | str | None = None,
             catalog: DatasetCatalog | None = None, count: int | None = None) -> DatasetStatus:
    """Generate (or continue) ``dataset_id`` with the C++ binary.

    A prefix dataset generates the first chunks of its parent. ``shard=(i, n)`` restricts the run
    to every n-th chunk, so n independent jobs together produce the whole dataset. With
    ``resume=False`` an existing directory is removed first. ``count`` generates only the chunks
    holding the first ``count`` units (a quick partial dataset, completed by a later run without
    it, since the chunks do not depend on each other).
    """
    cat = catalog or load_catalog()
    d = cat.resolve(dataset_id)
    if d.placeholder:
        raise NotGeneratable(f"{d.id} is written by the `fields` benchmark units "
                             "(pfn bench run --kind fields), not by pfn data gen")
    owner = cat.storage(dataset_id)
    directory = dataset_dir(owner, home, cat)
    if not resume and directory.exists():
        if d.prefix_of is not None:
            raise RuntimeError(f"{d.id} is a prefix of {owner.id}, refusing to delete that pool")
        log.info("removing %s", directory)
        shutil.rmtree(directory)
    seed = owner.seed if seed is None else int(seed)
    needed = n_chunks(d.count if count is None else max(1, min(int(count), d.count)), owner.chunk)
    chunks = [k for k in chunks_of_shard(n_chunks(owner.count, owner.chunk), shard) if k < needed]
    before = status(dataset_id, home, cat)
    if not chunks:
        return before
    extra: dict[str, Any] = {}
    args = ["--out", str(directory), "--count", str(owner.count), "--seed", str(seed),
            "--chunk", str(owner.chunk), "--workers", str(workers),
            "--chunks", format_chunk_list(chunks), *_kind_args(owner)]
    if owner.kind == "traj":
        meta_path = directory / META_FILE
        old = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        rho = old.get("rho_unsuccessful")
        if rho is None:
            rho = _pilot_rho(owner, workers, home)
        extra = {"rho_unsuccessful": rho}
        args += ["--rho", repr(float(rho))]
    _write_meta(owner, directory, seed, extra, home)
    sub = {"traj": "traj", "sets": "sets", "contexts": "contexts", "screen": "screen"}[owner.kind]
    cmd = binaries.command(owner.problem, sub, *args, home=home)
    log.info("generating %s: %d chunk(s) on %d worker(s)", d.id, len(chunks), workers)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:2])} exited with {proc.returncode}")
    st = status(dataset_id, home, cat)
    log.info("%s: %d/%d chunks done after %.1f s", d.id, st.chunks_done, st.chunks_total,
             time.monotonic() - t0)
    return st


def open_dataset(dataset_id: str, home: Path | str | None = None,
                 catalog: DatasetCatalog | None = None) -> Pool | ContextStore:
    """Open the pool or context store of a dataset, limited to its own count for prefixes."""
    cat = catalog or load_catalog()
    d = cat.resolve(dataset_id)
    directory = dataset_dir(d, home, cat)
    if d.kind == "contexts":
        return ContextStore(directory, limit=d.count if d.prefix_of else None)
    limit = None
    if d.prefix_of is not None:
        pool = Pool.open(directory)
        limit = d.count * pool.rows_per_unit
    return Pool.open(directory, limit=limit)

