"""The ``pfn`` entry point.

Subcommands are discovered: every module in :mod:`esbopfn.commands` exposes
``register(subparsers)``, which adds its parser and sets ``func`` (taking the parsed namespace and
returning an exit code). Adding a command means adding a module there, this file stays unchanged.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import pkgutil
from collections.abc import Sequence

import esbopfn
import esbopfn.commands


def discover_commands() -> list[str]:
    """Fully qualified names of the command modules, sorted."""
    return sorted(
        f"{esbopfn.commands.__name__}.{m.name}"
        for m in pkgutil.iter_modules(esbopfn.commands.__path__)
        if not m.name.startswith("_")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pfn", description="esbo-pfn command line")
    parser.add_argument("--version", action="version", version=f"esbo-pfn {esbopfn.__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more log output")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name in discover_commands():
        module = importlib.import_module(name)
        register = getattr(module, "register", None)
        if register is None:
            raise RuntimeError(f"command module {name} has no register(subparsers)")
        register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    return int(func(args) or 0)
