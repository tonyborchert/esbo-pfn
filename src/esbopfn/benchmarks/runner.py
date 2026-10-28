"""Plan and run the benchmark units that plot groups need, optionally one shard at a time."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from esbopfn.benchmarks.db import Db, key_hash
from esbopfn.benchmarks.kinds import BenchmarkKind, discover_kinds
from esbopfn.plots import PlotGroup, discover_groups, group_config

log = logging.getLogger(__name__)

_SHARD_RE = re.compile(r"^(\d+)/(\d+)$")


@dataclass(frozen=True)
class Shard:
    """Shard ``index`` of ``count`` (0-based)."""

    index: int = 0
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1 or not 0 <= self.index < self.count:
            raise ValueError(f"invalid shard {self.index}/{self.count}")


def parse_shard(text: str | None) -> Shard:
    """Parse ``i/n``. None means the whole set."""
    if not text:
        return Shard()
    m = _SHARD_RE.match(text.strip())
    if not m:
        raise ValueError(f"--shard must look like i/n, got {text!r}")
    return Shard(int(m.group(1)), int(m.group(2)))


@dataclass(frozen=True)
class PlannedUnit:
    kind: str
    key: dict[str, Any]
    hash: str


def select_groups(names: list[str] | None) -> dict[str, PlotGroup]:
    """The named groups (all plot modules if ``names`` is empty), raising ``KeyError`` on
    unknown names. A name without a plot module but with a benchmark spec
    (``configs/benchmarks/<name>.yaml``) selects the spec's units."""
    from esbopfn.benchmarks.groups import spec_group, spec_names

    groups = discover_groups()
    if not names:
        return groups
    specs = set(spec_names())
    missing = [n for n in names if n not in groups and n not in specs]
    if missing:
        raise KeyError(f"unknown plot groups {missing}, known: {sorted(set(groups) | specs)}")
    return {n: groups[n] if n in groups else spec_group(n) for n in names}


def plan(groups: dict[str, PlotGroup], db: Db, shard: Shard = Shard(),
         kinds: list[str] | None = None) -> list[PlannedUnit]:
    """Deduplicated units the groups need (only ``kinds`` if given), sorted by the dependency
    order of the kinds and then by hash, restricted to one shard."""
    from esbopfn.benchmarks.groups import kind_rank

    seen: dict[tuple[str, str], PlannedUnit] = {}
    for name, group in groups.items():
        for kind, key in group.needs(group_config(name), db):
            if kinds and kind not in kinds:
                continue
            h = key_hash(key)
            seen.setdefault((kind, h), PlannedUnit(kind, key, h))
    units = [seen[k] for k in sorted(seen, key=lambda k: (kind_rank(k[0]), k[0], k[1]))]
    return [u for i, u in enumerate(units) if i % shard.count == shard.index]


def run_units(
    units: list[PlannedUnit], db: Db, kinds: dict[str, BenchmarkKind] | None = None
) -> tuple[int, int]:
    """Run every unit not yet done. Returns ``(ran, skipped)``."""
    kinds = discover_kinds() if kinds is None else kinds
    ran = skipped = 0
    for unit in units:
        if unit.kind not in kinds:
            raise KeyError(f"no benchmark kind {unit.kind!r} registered")
        if db.has(unit.kind, unit.key):
            skipped += 1
            continue
        log.info("running %s %s", unit.kind, unit.hash[:12])
        db.run(unit.kind, unit.key, kinds[unit.kind].run)
        ran += 1
    return ran, skipped
