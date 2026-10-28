"""Import the thesis prediction timing bundle (``results_timing_bundle/predplay_timing.json``,
falling back to the ``predplay`` list of ``summary.json``) as native ``prediction_timing`` units.

Each record holds one (model, problem, n) cell measured with one call per query on the
workstation (``device`` as recorded). Key ``{problem, model, n_ctx, provenance: "thesis",
source_model}``, blob ``timing.json`` with ``seconds_median, seconds_mean, seconds_p90, n_calls,
device, host, cpu, torch, torch_threads, measured_at`` plus ``cutoff_frac``. The XDecPFN records
were measured on the architecture twin ``dynformer_c1r2/cellf_s20260931`` (same architecture as the
thesis XDecPFN) and are stored under the thesis XDecPFN id with ``source_model`` naming the twin.
The thesis figures replaced non-monotone cells by power-law fits at plot time, the records here
are the measurements.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from esbopfn.benchmarks.db import Db

log = logging.getLogger(__name__)

TIMING_FIELDS = ("seconds_median", "seconds_mean", "seconds_p90", "n_calls", "device", "host",
                 "cpu", "torch", "torch_threads", "measured_at", "cutoff_frac")


def read_records(bundle: Path) -> list[dict[str, Any]]:
    path = Path(bundle) / "predplay_timing.json"
    if path.exists():
        data = json.loads(path.read_text())
        return list(data["records"] if isinstance(data, dict) else data)
    path = Path(bundle) / "summary.json"
    if path.exists():
        return list(json.loads(path.read_text()).get("predplay", []))
    return []


def import_records(db: Db, records: list[dict[str, Any]], source: Path,
                   overwrite: bool = False) -> int:
    from esbopfn.benchmarks.import_thesis import PROVENANCE, map_model, map_problem, store

    n = 0
    seen: set[tuple[str, str, int]] = set()
    for rec in records:
        problem = map_problem(str(rec.get("problem")))
        model = map_model(str(rec.get("model_id")), problem) if problem else None
        if problem is None or model is None:
            continue
        cell = (problem, model, int(rec["n_ctx"]))
        if cell in seen:
            log.info("timing record %s duplicates %s, kept the first", rec.get("model_id"), cell)
            continue
        seen.add(cell)
        key = {"problem": problem, "model": model, "n_ctx": int(rec["n_ctx"]),
               "provenance": PROVENANCE, "source_model": str(rec["model_id"])}
        blob = {k: rec.get(k) for k in TIMING_FIELDS}
        blob["source"] = str(source)
        n += store(db, "prediction_timing", key,
                   lambda d, b=blob: (d / "timing.json").write_text(json.dumps(b, indent=1)),
                   overwrite)
    return n


def import_timing_bundle(db: Db, bundle: Path | str, overwrite: bool = False) -> int:
    bundle = Path(bundle)
    return import_records(db, read_records(bundle), bundle, overwrite)
