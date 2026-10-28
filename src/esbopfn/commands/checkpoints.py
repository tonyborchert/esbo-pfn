"""``pfn checkpoints status|download <name>``: third-party checkpoints (the TabPFN regressor)."""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("checkpoints", help="show or download third-party checkpoints")
    sub = p.add_subparsers(dest="checkpoints_command", metavar="<action>")
    st = sub.add_parser("status", help="show which checkpoints are present")
    st.add_argument("--verify", action="store_true", help="compare SHA-256 with the thesis file")
    st.set_defaults(func=run_status)
    dl = sub.add_parser("download", help="download or copy a checkpoint into PFN_HOME")
    dl.add_argument("name", choices=["tabpfn"])
    dl.add_argument("--from", dest="source", default=None,
                    help="copy this local file instead of downloading")
    dl.set_defaults(func=run_download)
    p.set_defaults(func=lambda args: (p.print_help(), 2)[1])


def run_status(args: argparse.Namespace) -> int:
    from esbopfn.predictors.checkpoints import CHECKPOINTS, tabpfn_instructions

    missing = 0
    for status_fn, _ in CHECKPOINTS.values():
        status = status_fn(verify=args.verify)
        print(status.line())
        if not status.present:
            missing += 1
            print(tabpfn_instructions(status.path))
    return 1 if missing else 0


def run_download(args: argparse.Namespace) -> int:
    from esbopfn.predictors.checkpoints import CHECKPOINTS

    _, download_fn = CHECKPOINTS[args.name]
    try:
        path = download_fn(source=args.source)
    except (RuntimeError, FileNotFoundError) as exc:
        print(exc)
        return 1
    print(f"{args.name} checkpoint at {path}")
    return 0
