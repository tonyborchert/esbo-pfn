"""``pfn status``: datasets, models, benchmark units and the Slurm queue."""

from __future__ import annotations

import argparse
import getpass
import shutil
import subprocess

from esbopfn.benchmarks.db import Db
from esbopfn.config import paths


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("status", help="show datasets, models, benchmark units, queue")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    tree = paths()
    print(f"PFN_HOME {tree.root}")
    datasets_dir = tree.root / "datasets"
    datasets = sorted(p for p in datasets_dir.glob("*/*") if p.is_dir())
    print(f"\ndatasets: {len(datasets)}")
    for d in datasets:
        state = "finished" if any(d.glob("*.finished")) or (d / "finished").exists() else "partial"
        print(f"  {d.relative_to(datasets_dir)}  ({state})")
    models = sorted((tree.root / "models").glob("**/model.pt"))
    print(f"\nmodels: {len(models)}")
    for m in models:
        print(f"  {m.parent.relative_to(tree.root / 'models')}")
    counts = Db(tree.root / "benchmarks" / "db.sqlite").counts()
    print(f"\nbenchmark units: {sum(counts.values())}")
    for (kind, status), n in counts.items():
        print(f"  {kind:<24}{status:<10}{n:>8}")
    squeue = shutil.which("squeue")
    if squeue:
        print("\nslurm queue:")
        out = subprocess.run(
            [squeue, "-u", getpass.getuser(), "-o", "%.12i %.24j %.8T %.10M %R"],
            capture_output=True, text=True, check=False,
        )
        print(out.stdout.rstrip() or out.stderr.rstrip())
    return 0
