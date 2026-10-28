"""``pfn plot``: render the figures of plot groups from the benchmark database."""

from __future__ import annotations

import argparse
from pathlib import Path

from esbopfn.benchmarks.db import Db
from esbopfn.benchmarks.runner import select_groups
from esbopfn.config import paths
from esbopfn.plots import group_config


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("plot", help="render plot groups")
    p.add_argument("groups", nargs="*", help="plot groups (default: all)")
    p.add_argument("--format", choices=("pdf", "png"), default="pdf")
    p.add_argument("--theme", choices=("white", "black"), default="white")
    p.add_argument("--out", type=Path, default=None, help="output dir (default figures/<group>)")
    p.add_argument("--list", action="store_true", help="list the registered groups")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    groups = select_groups(list(args.groups))
    if args.list or not groups:
        print("\n".join(sorted(groups)) if groups else "no plot groups registered")
        return 0
    db = Db()
    for name, group in groups.items():
        out_dir = (args.out / name) if args.out else paths().figures / name
        out_dir.mkdir(parents=True, exist_ok=True)
        group.render(group_config(name), db, out_dir, args.format, args.theme)
        print(f"{name}: {out_dir}")
    return 0
