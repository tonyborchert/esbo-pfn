"""``pfn hw run|status|list|pause|bench-pred|import-logs|import-pred|serve``: the rig field, the
hardware prediction benchmark and the thesis rig logs. See ``docs/HARDWARE.md``."""

from __future__ import annotations

import argparse
import json
import logging
import math

log = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("hw", help="QUBE rig: run the field, prediction benchmark, import")
    sub = p.add_subparsers(dest="hw_action", metavar="<action>", required=True)

    r = sub.add_parser("run", help="run (or resume) the rig field")
    r.add_argument("--rig", required=True, help="rig configuration (configs/rigs.yaml)")
    r.add_argument("--methods", default=None,
                   help="comma separated (default: the field's six methods)")
    r.add_argument("--seeds", default=None, help="e.g. 0-4 or 0,2 (default: 0-4)")
    r.add_argument("--trials", type=int, default=None, help="N_max (default 180)")
    r.add_argument("--steps", type=int, default=None, help="T_budget in steps (default 36000)")
    r.add_argument("--url", default="ws://localhost:8765", help="rig server")
    r.add_argument("--plant", choices=("sim",), default=None,
                   help="start an in-process simulated rig instead of connecting to --url")
    r.add_argument("--max-new-trials", type=int, default=None,
                   help="stop after this many new trials (testing)")
    r.set_defaults(func=_run)

    s = sub.add_parser("status", help="progress of the rig runs in the database")
    s.add_argument("--rig", default=None)
    s.add_argument("--provenance", default=None, choices=("native", "thesis"))
    s.set_defaults(func=_status)

    ls = sub.add_parser("list", help="rig configurations and the field defaults")
    ls.set_defaults(func=_list)

    pa = sub.add_parser("pause", help="stop a running field after its current trial")
    pa.add_argument("--clear", action="store_true", help="remove the pause flag")
    pa.set_defaults(func=_pause)

    b = sub.add_parser("bench-pred", help="hardware prediction benchmark on the rig logs")
    b.add_argument("--rig", required=True)
    b.add_argument("--models", nargs="*", default=None,
                   help="name=model_id pairs (default: configs/benchmarks/hardware.yaml)")
    b.add_argument("--ablation", action="store_true", help="the four-model DecPFN ablation")
    b.add_argument("--logs", choices=("thesis", "native"), default=None)
    b.set_defaults(func=_bench_pred)

    il = sub.add_parser("import-logs", help="import an old hwplay run tree")
    il.add_argument("root")
    il.add_argument("--rig", default=None, help="default: from the tree name")
    il.add_argument("--overwrite", action="store_true")
    il.set_defaults(func=_import_logs)

    ip = sub.add_parser("import-pred", help="import the old hwpred results directory")
    ip.add_argument("results_dir")
    ip.add_argument("--overwrite", action="store_true")
    ip.set_defaults(func=_import_pred)

    sv = sub.add_parser("serve", help="start the rig server (same as python -m "
                        "esbopfn.hardware.server)")
    from esbopfn.hardware.server.remote_env_server import add_arguments

    add_arguments(sv)
    sv.set_defaults(func=_serve)


def _bench_config() -> dict:
    import yaml

    from esbopfn.config import configs_dir

    return yaml.safe_load((configs_dir() / "benchmarks" / "hardware.yaml").read_text()) or {}


def _run(args: argparse.Namespace) -> int:
    from esbopfn.hardware.client import RigClient
    from esbopfn.hardware.rigs import field_defaults, get_rig
    from esbopfn.hardware.runner import FieldPlan, parse_seeds, run_field

    get_rig(args.rig)
    fd = field_defaults()
    plan = FieldPlan(
        rig=args.rig,
        methods=[m.strip() for m in args.methods.split(",")] if args.methods else list(fd.methods),
        seeds=parse_seeds(args.seeds) if args.seeds else list(fd.seeds),
        n_max=args.trials or fd.n_max, t_budget=args.steps or fd.t_budget)
    server = None
    url = args.url
    if args.plant == "sim":
        from esbopfn.hardware.server import serve

        server = serve("sim", port=0).start()
        url = server.url
        log.info("simulated rig at %s", url)
    client = RigClient(url)
    try:
        results = run_field(plan, client, max_new_trials=args.max_new_trials)
    finally:
        client.close()
        if server is not None:
            server.shutdown()
    for r in results:
        print(f"{r.key['method']:<12} seed {r.key['seed']}: {r.n_trials:4d} trials "
              f"({r.new_trials} new) {r.stop_reason}")
    return 0


def _status(args: argparse.Namespace) -> int:
    from esbopfn.benchmarks.db import Db
    from esbopfn.hardware import records
    from esbopfn.hardware.runner import pause_flag

    db = Db()
    filters = {}
    if args.rig:
        filters["rig"] = args.rig
    if args.provenance:
        filters["provenance"] = args.provenance
    units = db.query("hardware_optimization", status=None, **filters)
    if not units:
        print("no rig runs in the database")
    print(f"{'rig':<16}{'method':<12}{'seed':>5}{'prov':>8}{'status':>9}{'trials':>8}"
          f"{'steps':>8}{'crashed':>8}{'J-':>10}")
    for u in sorted(units, key=lambda u: (u.key["rig"], u.key["method"], u.key["seed"],
                                          u.key["provenance"])):
        trials = records.read_trials(db.blob_dir("hardware_optimization", u.key))
        inc = trials[-1].incumbent_J if trials else math.inf
        print(f"{u.key['rig']:<16}{u.key['method']:<12}{u.key['seed']:>5}"
              f"{u.key['provenance']:>8}{u.status:>9}{len(trials):>8}"
              f"{sum(t.t_obs for t in trials):>8}{sum(t.crashed for t in trials):>8}"
              f"{inc:>10.2f}")
    if pause_flag(db).exists():
        print(f"\npause flag set: {pause_flag(db)}")
    return 0


def _list(args: argparse.Namespace) -> int:
    from esbopfn.hardware.rigs import field_defaults, load_rigs

    for rig in load_rigs().values():
        print(f"{rig.name:<16}{rig.description} (mass: {rig.mass_note}, reset reference "
              f"{rig.reset_reference_method})")
    fd = field_defaults()
    print(f"\nmethods {', '.join(fd.methods)}\nseeds {list(fd.seeds)}\n"
          f"N_max {fd.n_max}, T_budget {fd.t_budget}")
    for m, p in fd.method_params.items():
        print(f"  {m}: {json.dumps(p)}")
    return 0


def _pause(args: argparse.Namespace) -> int:
    from esbopfn.benchmarks.db import Db
    from esbopfn.hardware.runner import pause_flag

    flag = pause_flag(Db())
    if args.clear:
        flag.unlink(missing_ok=True)
        print(f"pause flag cleared ({flag})")
    else:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
        print(f"pause flag set ({flag}): the field stops after the current trial")
    return 0


def _bench_pred(args: argparse.Namespace) -> int:
    from esbopfn.benchmarks.db import Db
    from esbopfn.benchmarks.kinds.hardware_prediction import NAME, resolve_models, run, unit_key

    cfg = _bench_config()
    pred = cfg.get("prediction") or {}
    methods = (cfg.get("optimization") or {}).get("methods")
    logs = args.logs or pred.get("logs", "thesis")
    bench = "ablation" if args.ablation else "ladder"
    if args.models:
        models = dict(m.split("=", 1) if "=" in m else (m.split("/")[0], m) for m in args.models)
    else:
        models = pred.get("ablation" if args.ablation else "predictors") or {}
    models = resolve_models(models)
    db = Db()
    for name, model in models.items():
        key = unit_key(args.rig, bench, name, model,
                       ladder=None if args.ablation else pred.get("ladder"),
                       cutoff_fracs=pred.get("cutoff_fracs"), methods=methods, logs=logs)
        blob = db.run(NAME, key, run)
        summary = json.loads((blob / "summary.json").read_text())
        print(f"{name}: median LL (non-crashed) {summary['median_ll']}")
    return 0


def _import_logs(args: argparse.Namespace) -> int:
    from esbopfn.benchmarks.import_hardware import import_rig_logs

    keys = import_rig_logs(args.root, args.rig, overwrite=args.overwrite)
    print(f"{len(keys)} rig runs imported or present")
    return 0


def _import_pred(args: argparse.Namespace) -> int:
    from esbopfn.benchmarks.import_hardware import import_hwpred_results

    keys = import_hwpred_results(args.results_dir, overwrite=args.overwrite)
    print(f"{len(keys)} hardware prediction units imported or present")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from esbopfn.hardware.server.remote_env_server import run_from_args

    return run_from_args(args)
