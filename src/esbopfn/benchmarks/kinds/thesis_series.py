"""Kind ``thesis_series``: aggregated series of the thesis figures, imported where the raw data
behind them is not available locally (see ``docs/BENCHMARKS.md``, importing the thesis results).

Units are only written by ``pfn import thesis-results``. Key ``{figure, problem, series,
variant, provenance: "thesis"}``, ``problem = "pooled"`` for cross-problem series.
Blob ``series.json``: ``{figure, problem, series, variant, source, columns[], data{column:
[values]}, meta{}}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from esbopfn.benchmarks.kinds import BenchmarkKind, Key


def enumerate_units(cfg: dict[str, Any]) -> list[Key]:
    return []


def run(key: Key, blob_dir: Path) -> None:
    raise RuntimeError("thesis_series units are written by `pfn import thesis-results`")


def read_series(blob_dir: Path) -> dict[str, Any]:
    return json.loads((Path(blob_dir) / "series.json").read_text())


KIND = BenchmarkKind(name="thesis_series", enumerate=enumerate_units, run=run)
