"""``pfn hpo <lcpfn|decpfn> --problem P --size S``: the LC-PFN/DecPFN hyperparameter search."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("hpo", help="hyperparameter search of LC-PFN or DecPFN")
    p.add_argument("model", choices=["lcpfn", "decpfn"])
    p.add_argument("--problem", default="cartpole")
    p.add_argument("--size", required=True, choices=["100k", "1M"])
    p.add_argument("--stage", default="all", choices=["1", "2", "all"])
    p.add_argument("--shard", default=None, help="i/n: run every n-th pending trial")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--hparams", type=Path, default=None,
                   help="file whose fixed block and hpo section define the search")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from esbopfn.generators import parse_shard
    from esbopfn.training.hpo import HPO

    hpo = HPO(args.model, args.problem, args.size, seed=args.seed, device=args.device,
              hparams_file=args.hparams)
    hpo.run(args.stage, parse_shard(args.shard))
    trials, best, done = hpo.stage1()
    state = "finished" if done else f"{len(trials)} trials in the current step"
    print(f"{hpo.root}: stage 1 {state}, best so far {best}")
    cand = hpo.root / "hparams_candidate.yaml"
    if cand.exists():
        print(f"candidate: {cand} (pfn train {args.model} --hparams {cand} ...)")
    return 0
