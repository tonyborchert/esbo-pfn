"""Kind ``prediction_ladder``: one predictor on the context-ladder tasks of one problem (thesis
fig:sim:context-sweep).

Same scoring as :mod:`esbopfn.benchmarks.kinds.prediction` (five cutoffs, conditioned LL), on the
ladder table built from field ``oracle181``: the query is the last trial of a run, the context the
first ``n`` trials for every rung ``n``. No stop-grid rows. The figure plots the mean over the
cutoffs, then the mean over the 50 queries (:func:`esbopfn.benchmarks.metrics.ladder_curve`).

Key ``{problem, model, checkpoint, tasks, tasks_digest, n, part}`` with ``tasks`` naming the
ladder table. Blob: ``predictions.parquet`` and ``meta.json`` with the columns of ``prediction``
(imported thesis units leave ``dist``, ``edges``, ``probs``, ``mean``, ``var`` empty).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from esbopfn.benchmarks.kinds import BenchmarkKind, Key
from esbopfn.benchmarks.kinds.prediction import run_scoring, scoring_keys


def enumerate_units(cfg: dict[str, Any]) -> list[Key]:
    return scoring_keys(cfg, "prediction_ladder", "ladder")


def run(key: Key, blob_dir: Path) -> None:
    run_scoring(key, blob_dir, "ladder", stop_grid=False)


KIND = BenchmarkKind(name="prediction_ladder", enumerate=enumerate_units, run=run)
