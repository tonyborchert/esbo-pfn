"""``pfn bench plan|run|kinds``: list or compute the benchmark units of plot groups."""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from esbopfn.benchmarks.db import Db
from esbopfn.benchmarks.kinds import discover_kinds
from esbopfn.benchmarks.runner import parse_shard, plan, run_units, select_groups

log = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("bench", help="plan or run benchmark units")
    p.add_argument("action", choices=("plan", "run", "kinds"))
    p.add_argument("groups", nargs="*", help="plot groups (default: all)")
    p.add_argument("--group", action="append", default=[], help="plot group, repeatable")
    p.add_argument("--shard", default=None, help="i/n: only every n-th unit starting at i")
    p.add_argument("--kind", action="append", default=[],
                   help="only units of this kind, repeatable (run dependency stages separately)")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if args.action == "kinds":
        kinds = discover_kinds()
        print("\n".join(sorted(kinds)) if kinds else "no benchmark kinds registered")
        return 0
    groups = select_groups(list(args.groups) + list(args.group))
    if not groups:
        print("no plot groups registered")
        return 0
    db = Db()
    shard, kinds = parse_shard(args.shard), list(args.kind) or None
    units = plan(groups, db, shard, kinds)
    if args.action == "plan":
        missing = [u for u in units if not db.has(u.kind, u.key)]
        total, todo = Counter(u.kind for u in units), Counter(u.kind for u in missing)
        for kind in sorted(total):
            print(f"{kind:<24}{todo[kind]:>8} missing of {total[kind]}")
        print(f"{'total':<24}{len(missing):>8} missing of {len(units)}")
        return 0
    ran, skipped = run_units(units, db)
    # Units whose inputs were produced in this pass (prediction units need a frozen task table,
    # which needs the whole field) appear in the next plan.
    for _ in range(4):
        later = [u for u in plan(groups, db, shard, kinds) if not db.has(u.kind, u.key)]
        if not later:
            break
        more, _skip = run_units(later, db)
        ran += more
    print(f"ran {ran}, skipped {skipped} (already done)")
    return 0
