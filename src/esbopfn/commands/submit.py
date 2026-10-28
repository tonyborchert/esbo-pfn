"""``pfn submit``: wrap a command into a Slurm job, or run it in place with ``--local``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from esbopfn.cluster.profile import JOB_KINDS, load_profile
from esbopfn.cluster.submit import (
    JobRequest,
    default_name,
    job_home,
    parse_array,
    render,
    run_local,
    sbatch,
    script_path,
)
from esbopfn.config import paths

log = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "submit",
        help="run a command as a Slurm job or locally",
        description="Example: pfn submit --kind bench --array 0-19 -- pfn bench run --group g",
    )
    p.add_argument("--profile", default=None, help="cluster profile (default $PFN_PROFILE)")
    p.add_argument("--kind", choices=JOB_KINDS, default="bench", help="job kind for defaults")
    p.add_argument("--time", default=None, help="wall clock, e.g. 02:00:00")
    p.add_argument("--cpus", type=int, default=None)
    p.add_argument("--mem", default=None, help="memory, e.g. 16G")
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument("--array", default=None, help="array range a-b, appends --shard i/n")
    p.add_argument("--after", default=None, help="job id this job depends on (afterok)")
    p.add_argument("--name", default=None, help="job name (default from the command)")
    p.add_argument("--dry-run", action="store_true", help="print the script, submit nothing")
    p.add_argument("--local", action="store_true", help="run the command here in place")
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- <command ...>")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        log.error("no command given, use: pfn submit [options] -- pfn <command ...>")
        return 2
    profile = load_profile(args.profile)
    base = profile.resources(args.kind)
    overrides = {
        k: v
        for k, v in {"time": args.time, "cpus": args.cpus, "mem": args.mem, "gpus": args.gpus}.items()
        if v is not None
    }
    req = JobRequest(
        command=tuple(command),
        kind=args.kind,
        resources=dataclasses.replace(base, **overrides),
        name=args.name or default_name(tuple(command)),
        array=parse_array(args.array) if args.array else None,
        after=args.after,
    )
    home = job_home(profile)
    tree = paths(home)
    local = args.local or profile.scheduler == "none"
    if args.dry_run:
        logs_dir = tree.root / "logs"
        print(render(req, profile, home, Path.cwd(), logs_dir), end="")
        return 0
    if local:
        return run_local(req, home)
    text = render(req, profile, home, Path.cwd(), tree.logs)
    script = script_path(tree.jobs, req.name)
    script.write_text(text)
    script.chmod(0o755)
    job_id = sbatch(script)
    log.info("submitted %s as job %s", script, job_id)
    print(job_id)
    return 0
