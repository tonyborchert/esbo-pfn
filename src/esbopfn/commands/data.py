"""``pfn data``: build the C++ generators, generate, inspect and verify datasets.

    pfn data build [--blas accelerate|openblas|mkl] [--native]
    pfn data status [--problem P] [--json]
    pfn data gen <dataset_id> | --for train <model> --problem P | --for all [--problem P]
                 [--workers N] [--seed S] [--shard i/n] [--no-resume]
    pfn data verify <dataset_id> [--sample N] [--no-parity] [--n-ks 500]
    pfn data speed --problem P [--count 2000] [--sets 10] [--budget 120]
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from esbopfn.generators import generate, open_dataset, parse_shard, status
from esbopfn.generators.binaries import BLAS_CHOICES, build, detect_blas
from esbopfn.generators.datasets import load_catalog

log = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("data", help="generate and inspect training data")
    sub = p.add_subparsers(dest="data_command", metavar="<action>")

    b = sub.add_parser("build", help="build and install the C++ generators into $PFN_HOME/bin")
    b.add_argument("--blas", choices=BLAS_CHOICES, default=None,
                   help="LAPACK provider (default: accelerate on macOS, mkl if MKLROOT, else openblas)")
    b.add_argument("--native", action="store_true", help="compile with -march=native")
    b.add_argument("--jobs", type=int, default=None, help="parallel compile jobs")
    b.set_defaults(func=run_build)

    s = sub.add_parser("status", help="progress of every dataset")
    s.add_argument("--problem", default=None)
    s.add_argument("--json", action="store_true", help="print JSON lines")
    s.set_defaults(func=run_status)

    g = sub.add_parser("gen", help="generate datasets (resumes unfinished chunks)")
    g.add_argument("dataset", nargs="?", default=None, help="dataset id, e.g. traj/cartpole/1M")
    g.add_argument("--for", dest="for_", nargs="+", default=None, metavar="WHAT",
                   help="'train <model>' (with --problem) or 'all'")
    g.add_argument("--problem", default=None)
    g.add_argument("--workers", type=int, default=None,
                   help="threads (default: the cores available to this process)")
    g.add_argument("--seed", type=int, default=None, help="override the catalog seed")
    g.add_argument("--shard", default=None, help="i/n: generate every n-th chunk starting at i")
    g.add_argument("--no-resume", action="store_true", help="delete existing data first")
    g.set_defaults(func=run_gen)

    v = sub.add_parser("verify", help="check invariants and C++/Python parity")
    v.add_argument("dataset")
    v.add_argument("--sample", type=int, default=2000, help="rows or examples checked")
    v.add_argument("--no-parity", action="store_true", help="skip the parity checks")
    v.add_argument("--n-ks", type=int, default=500, help="prior samples of the distribution test")
    v.set_defaults(func=run_verify)

    sp = sub.add_parser("speed", help="generation-speed benchmark on one core")
    sp.add_argument("--problem", required=True)
    sp.add_argument("--count", type=int, default=2000, help="single trajectories timed")
    sp.add_argument("--sets", type=int, default=10, help="trajectory sets timed per kind")
    sp.add_argument("--budget", type=float, default=120.0, help="seconds per phase at most")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=run_speed)

    p.set_defaults(func=lambda args: (p.print_help(), 2)[1])


def run_build(args: argparse.Namespace) -> int:
    blas = args.blas or detect_blas()
    bindir = build(blas=blas, jobs=args.jobs, native=args.native)
    print(f"installed furuta_gen, cartpole_gen, plants_gen into {bindir} (BLAS {blas})")
    return 0


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def run_status(args: argparse.Namespace) -> int:
    cat = load_catalog()
    ids = cat.ids(problem=args.problem)
    for did in ids:
        st = status(did, catalog=cat)
        if args.json:
            print(json.dumps(st.to_dict()))
            continue
        if st.placeholder:
            state = "finished" if st.finished else "placeholder (benchmark phase)"
            print(f"{did:38s} {state}")
            continue
        state = "finished" if st.finished else f"{st.chunks_done}/{st.chunks_total} chunks"
        size = f"  {_fmt_bytes(st.bytes)}" if st.bytes else ""
        print(f"{did:38s} {state:18s} {st.units_done:>9d}/{st.count:<9d}{size}")
    return 0


def _targets(args: argparse.Namespace) -> list[str]:
    cat = load_catalog()
    if args.dataset is not None:
        return [args.dataset]
    if not args.for_:
        raise SystemExit("give a dataset id or --for train <model> --problem P | --for all")
    if args.for_[0] == "all":
        return [d for d in cat.ids(problem=args.problem) if not cat.resolve(d).placeholder]
    if args.for_[0] == "train" and len(args.for_) == 2:
        if not args.problem:
            raise SystemExit("--for train <model> needs --problem")
        return cat.for_model(args.for_[1], args.problem)
    raise SystemExit("--for takes 'train <model>' or 'all'")


def default_workers() -> int:
    """Cores available to this process (the Slurm allocation on the cluster)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS
        return max(1, os.cpu_count() or 1)


def run_gen(args: argparse.Namespace) -> int:
    workers = args.workers or default_workers()
    shard = parse_shard(args.shard)
    for did in _targets(args):
        st = generate(did, workers=workers, seed=args.seed, resume=not args.no_resume, shard=shard)
        state = "finished" if st.finished else f"{st.chunks_done}/{st.chunks_total} chunks"
        print(f"{did}: {state} ({st.path})")
    return 0


def run_verify(args: argparse.Namespace) -> int:
    from esbopfn.generators.pools import ContextStore
    from esbopfn.generators.verify import Report, check_contexts, check_pool, parity_report

    cat = load_catalog()
    d = cat.resolve(args.dataset)
    rep = Report()
    data = open_dataset(args.dataset, catalog=cat)
    if isinstance(data, ContextStore):
        check_contexts(data, sample=args.sample, report=rep)
    else:
        rho = data.meta.get("rho_unsuccessful")
        check_pool(data, sample=args.sample, report=rep, rho_unsuccessful=rho)
    if not args.no_parity:
        parity_report(d.problem, n_samples=args.n_ks, report=rep)
    for line in rep.lines():
        print(line)
    return 0 if rep.ok else 1


def run_speed(args: argparse.Namespace) -> int:
    from esbopfn.generators.speed import HEADER, measure, row, thesis_row

    res = measure(args.problem, count=args.count, sets=args.sets, budget_s=args.budget)
    if args.json:
        print(json.dumps(res.to_dict()))
        return 0
    print(HEADER)
    print(row(res) + "   (this machine, one core)")
    print(thesis_row(args.problem) + "   (thesis table)")
    print(f"timed: {res.single_trajectories} trajectories, {res.ftpfn_sets} FT-PFN and "
          f"{res.xdecpfn_sets} XDecPFN sets, {res.system_candidates} candidate instances")
    return 0
