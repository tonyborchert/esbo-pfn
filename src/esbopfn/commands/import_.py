"""``pfn import thesis-results|thesis-models``: bring the thesis results and checkpoints in."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("import", help="import thesis results or checkpoints")
    sub = p.add_subparsers(dest="what", required=True)
    r = sub.add_parser("thesis-results", help="thesis benchmark results into the database")
    r.add_argument("--from", dest="root", required=True, type=Path,
                   help="the bachelorarbeit directory (holding impl/ and writing/)")
    r.add_argument("--raw", action="append", default=[], type=Path,
                   help="directory holding raw optplay trees <problem>@g0.05@bcapped[@timing]/")
    r.add_argument("--overwrite", action="store_true", help="replace units already imported")
    r.set_defaults(func=run_results)
    m = sub.add_parser("thesis-models", help="thesis checkpoints (XDecPFN, FT-PFN, DecPFN and LC-PFN export)")
    m.add_argument("--from", dest="dirs", required=True, action="append", type=Path,
                   help="directory with the checkpoints, repeatable")
    m.add_argument("--problem", action="append", default=None)
    m.add_argument("--overwrite", action="store_true",
                   help="replace models imported before (default: skip them)")
    m.set_defaults(func=run_models)
    # Other importers add their subcommands through ``register_import(sub)`` in their module.
    import importlib

    for module in ("esbopfn.benchmarks.import_hardware",):
        try:
            hook = getattr(importlib.import_module(module), "register_import", None)
        except ImportError:
            continue
        if hook is not None:
            hook(sub)


def run_results(args: argparse.Namespace) -> int:
    from esbopfn.benchmarks.import_thesis import import_thesis_results

    counts = import_thesis_results(args.root, raw_dirs=list(args.raw), overwrite=args.overwrite)
    for kind in sorted(counts):
        print(f"{kind:<24}{counts[kind]:>8}")
    return 0


def run_models(args: argparse.Namespace) -> int:
    from esbopfn.benchmarks.import_models import import_thesis

    first, *rest = args.dirs
    for m in import_thesis(first, *rest, problems=args.problem, overwrite=args.overwrite):
        if m.skipped:
            print(f"{m.model_id:<40} skipped, already imported (--overwrite replaces it)")
        else:
            print(f"{m.model_id:<40} step {m.step:>7}  {m.note}")
    return 0
