"""Render job scripts from a profile and submit them with ``sbatch`` or run them in place.

A job wraps one command, usually ``pfn <subcommand> ...``. With ``--array a-b`` the command gets
``--shard i/n`` appended, where ``n = b - a + 1`` and ``i = SLURM_ARRAY_TASK_ID - a``.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path

from esbopfn.cluster.profile import Profile, Resources
from esbopfn.config import pfn_home

log = logging.getLogger(__name__)

_ARRAY_RE = re.compile(r"^(\d+)-(\d+)(%\d+)?$")
_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ArraySpec:
    """A Slurm array range ``first-last`` with an optional ``%throttle``."""

    first: int
    last: int
    throttle: str = ""

    @property
    def n(self) -> int:
        return self.last - self.first + 1

    def __str__(self) -> str:
        return f"{self.first}-{self.last}{self.throttle}"


def parse_array(text: str) -> ArraySpec:
    """Parse ``a-b`` or ``a-b%k``."""
    m = _ARRAY_RE.match(text.strip())
    if not m:
        raise ValueError(f"--array must look like 0-19 or 0-19%4, got {text!r}")
    first, last = int(m.group(1)), int(m.group(2))
    if last < first:
        raise ValueError(f"--array range is empty: {text!r}")
    return ArraySpec(first, last, m.group(3) or "")


@dataclass(frozen=True)
class JobRequest:
    """Everything needed to render one job."""

    command: tuple[str, ...]
    kind: str
    resources: Resources
    name: str
    array: ArraySpec | None = None
    after: str | None = None


def default_name(command: tuple[str, ...]) -> str:
    """Job name from the command words, e.g. ``pfn bench run x`` gives ``bench-run-x``."""
    words = [w for w in command if not w.startswith("-")]
    if words and Path(words[0]).name == "pfn":
        words = words[1:]
    name = "-".join(words[:3]) or "job"
    return _NAME_RE.sub("_", name)[:60]


def job_home(profile: Profile, environ: dict[str, str] | None = None) -> Path:
    """``PFN_HOME`` of a job: ``$PFN_HOME`` if set at submit time, else the profile root, else
    ``./out``."""
    env = os.environ if environ is None else environ
    if env.get("PFN_HOME", "").strip():
        return pfn_home(env)
    root = profile.resolved_root(env)
    return root.resolve() if root is not None else pfn_home(env)


def _template() -> str:
    return resources.files("esbopfn.cluster").joinpath("templates/job.sh.tmpl").read_text()


def _launch_words(profile: Profile, command: tuple[str, ...]) -> str:
    quoted = " ".join(shlex.quote(w) for w in command)
    if command and command[0] == "pfn" and profile.launcher:
        return f"{profile.launcher} {quoted}"
    return quoted


def render(
    req: JobRequest, profile: Profile, home: Path, workdir: Path, logs_dir: Path
) -> str:
    """Return the job script text."""
    res = req.resources
    gpu = res.gpus > 0
    lines: list[str] = []
    if profile.scheduler == "slurm":
        part = "gpu" if gpu else "cpu"
        lines.append(f"#SBATCH --job-name={req.name}")
        account = profile.resolved_account(part)
        if account:
            lines.append(f"#SBATCH --account={account}")
        if profile.partitions.get(part):
            lines.append(f"#SBATCH --partition={profile.partitions[part]}")
        lines += [
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={res.cpus}",
            f"#SBATCH --mem={res.mem}",
            f"#SBATCH --time={res.time}",
        ]
        if gpu:
            lines.append(f"#SBATCH --gres=gpu:{res.gpus}")
        pattern = "%x_%A_%a" if req.array else "%x_%j"
        lines.append(f"#SBATCH --output={logs_dir}/{pattern}.out")
        if req.array:
            lines.append(f"#SBATCH --array={req.array}")
        if req.after:
            lines.append(f"#SBATCH --dependency=afterok:{req.after}")
    modules = list(profile.modules)
    if req.kind == "gen":
        modules += [m for m in profile.modules_cpp if m not in modules]
    module_block = f"module purge\nmodule load {' '.join(modules)}" if modules else ""
    pre = profile.pre.rstrip("\n")
    if gpu and profile.pre_gpu:
        pre = (pre + "\n" + profile.pre_gpu.rstrip("\n")).strip("\n")
    command = _launch_words(profile, req.command)
    shard = ""
    if req.array:
        shard = f'SHARD="$((SLURM_ARRAY_TASK_ID - {req.array.first}))/{req.array.n}"'
        command += ' --shard "$SHARD"'
    values = {
        "sbatch": "\n".join(lines),
        "profile": profile.name,
        "kind": req.kind,
        "modules": module_block,
        "pre": pre,
        "home": shlex.quote(str(home)),
        "workdir": shlex.quote(str(workdir)),
        "shard": shard,
        "command": command,
    }
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], _template())
    return re.sub(r"\n{2,}", "\n", text)


def script_path(jobs_dir: Path, name: str, now: datetime | None = None) -> Path:
    ts = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return jobs_dir / f"{ts}_{name}.sh"


def local_argv(command: tuple[str, ...]) -> list[str]:
    """Argument vector for running a command in place. ``pfn`` runs in this interpreter."""
    if command and command[0] == "pfn":
        return [sys.executable, "-m", "esbopfn", *command[1:]]
    return list(command)


def run_local(req: JobRequest, home: Path) -> int:
    """Run the command in place (each array shard in turn). Returns the first nonzero exit code."""
    env = dict(os.environ, PFN_HOME=str(home))
    shards = [None] if req.array is None else range(req.array.n)
    for i in shards:
        argv = local_argv(req.command)
        if i is not None:
            argv += ["--shard", f"{i}/{req.array.n}"]  # type: ignore[union-attr]
        log.info("running locally: %s", shlex.join(argv))
        code = subprocess.run(argv, env=env, check=False).returncode
        if code != 0:
            return code
    return 0


def sbatch(script: Path) -> str:
    """Submit a script and return the job id."""
    exe = shutil.which("sbatch")
    if exe is None:
        raise RuntimeError("sbatch not found, use --local or a profile with scheduler: none")
    out = subprocess.run(
        [exe, "--parsable", str(script)], capture_output=True, text=True, check=True
    )
    return out.stdout.strip().split(";")[0]
