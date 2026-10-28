"""``pfn config``: print the resolved output paths and the cluster profile."""

from __future__ import annotations

import argparse

from esbopfn.cluster.profile import load_profile
from esbopfn.config import configs_dir, paths, profile_name


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("config", help="print resolved paths and profile")
    p.add_argument("--profile", default=None, help="profile name (default $PFN_PROFILE or local)")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    tree = paths()
    name = args.profile or profile_name()
    profile = load_profile(name)
    print(f"PFN_HOME     {tree.root}")
    for sub, path in tree.as_dict().items():
        print(f"  {sub:<13}{path}")
    print(f"configs      {configs_dir()}")
    print(f"PFN_PROFILE  {profile.name} (scheduler: {profile.scheduler})")
    if profile.root:
        resolved = profile.resolved_root()
        print(f"  {'root':<13}{profile.root} -> {resolved if resolved else 'unresolved'}")
    for kind, res in profile.defaults.items():
        print(f"  {kind:<13}cpus={res.cpus} mem={res.mem} time={res.time} gpus={res.gpus}")
    return 0
