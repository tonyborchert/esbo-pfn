"""``pfn train <model> --problem P``: train a model into ``$PFN_HOME/models``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)

#: Models with an output head and a training size axis (the rest train one default variant).
AXES_MODELS = ("lcpfn", "lcpfn-last", "decpfn", "decpfn-nostates", "decpfn-nodeadtime",
               "decpfn-nodeadtime-nostates")
HEADS = ("binned", "gaussian", "linear")
SIZES = ("100k", "1M")
#: Models that exist for some problems only.
MODEL_PROBLEMS = {"decpfn-nodeadtime": ("furuta",), "decpfn-nodeadtime-nostates": ("furuta",)}


def register(subparsers: argparse._SubParsersAction) -> None:
    from esbopfn.training import TRAINERS

    p = subparsers.add_parser("train", help="train a model (see docs/TRAINING.md)")
    p.add_argument("model", choices=[*sorted(TRAINERS), "all"])
    p.add_argument("--problem", required=True, help="problem name or 'all'")
    p.add_argument("--head", default=None, choices=[*HEADS, "all"],
                   help="output head of LC-PFN/DecPFN models (default binned)")
    p.add_argument("--size", default=None, choices=[*SIZES, "all"],
                   help="training size of LC-PFN/DecPFN models (default 1M)")
    p.add_argument("--hparams", type=Path, default=None,
                   help="hyperparameter file overriding configs/hparams (e.g. an HPO candidate)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default="0", help="run label, the last model id component")
    p.add_argument("--steps", type=int, default=None, help="override the number of steps")
    p.add_argument("--time-limit", type=float, default=None, help="cap the training time [s]")
    p.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps, cpu")
    p.set_defaults(func=run)


def requests(args: argparse.Namespace) -> list:
    from esbopfn.problems.registry import PROBLEM_NAMES
    from esbopfn.training import TRAINERS, TrainRequest

    models = sorted(TRAINERS) if args.model == "all" else [args.model]
    problems = list(PROBLEM_NAMES) if args.problem == "all" else [args.problem]
    out = []
    for m in models:
        allowed = MODEL_PROBLEMS.get(m)
        for prob in problems:
            if allowed is not None and prob not in allowed:
                if args.model != "all" and args.problem != "all":
                    raise SystemExit(f"{m} exists for {', '.join(allowed)} only")
                continue
            if m in AXES_MODELS:
                heads = HEADS if args.head == "all" else [args.head or "binned"]
                sizes = SIZES if args.size == "all" else [args.size or "1M"]
            else:
                heads, sizes = [None], [None]
            for h in heads:
                for s in sizes:
                    out.append(TrainRequest(model=m, problem=prob, head=h, size=s,
                                            hparams=args.hparams, seed=args.seed, run=args.run,
                                            steps=args.steps, time_limit=args.time_limit,
                                            device=args.device))
    return out


def run(args: argparse.Namespace) -> int:
    from esbopfn.training import get_trainer
    from esbopfn.training.single_trajectory_data import DataMissing

    failed = 0
    for req in requests(args):
        label = "/".join(x for x in (req.model, req.problem, req.head, req.size) if x)
        try:
            path = get_trainer(req.model)(req)
        except (DataMissing, FileNotFoundError, KeyError, ValueError) as exc:
            failed += 1
            log.error("%s: %s", label, exc)
            continue
        print(f"{label}: {path}")
    return 1 if failed else 0
