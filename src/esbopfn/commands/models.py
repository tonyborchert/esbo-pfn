"""``pfn models list|show <id>``: the trained models under ``$PFN_HOME/models``."""

from __future__ import annotations

import argparse
import json


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("models", help="list or inspect trained models")
    sub = p.add_subparsers(dest="models_command", metavar="<action>")
    ls = sub.add_parser("list", help="list trained model ids")
    ls.set_defaults(func=run_list)
    sh = sub.add_parser("show", help="show config and metrics of a model id")
    sh.add_argument("model_id")
    sh.set_defaults(func=run_show)
    p.set_defaults(func=run_list)


def run_list(args: argparse.Namespace) -> int:
    from esbopfn.training.registry import list_models, models_root

    ids = list_models()
    print(f"models under {models_root()}: {len(ids)}")
    for mid in ids:
        print(f"  {mid.short}")
    return 0


def run_show(args: argparse.Namespace) -> int:
    from esbopfn.training.registry import ModelNotTrained, load_model, resolve

    try:
        path = resolve(args.model_id)
    except (ModelNotTrained, ValueError) as exc:
        print(exc)
        return 1
    files = load_model(path)
    n_params = sum(int(v.numel()) for v in files.state_dict.values())
    res = files.metrics.get("result", {})
    print(f"id         {args.model_id}")
    print(f"path       {path}")
    print(f"params     {n_params:,}")
    print(f"model      {json.dumps(files.model_cfg)}")
    for key in ("hparams", "datasets", "seed", "steps", "git_hash", "provenance"):
        if key in files.train_cfg:
            print(f"{key:<11}{json.dumps(files.train_cfg[key], default=str)}")
    if files.calib:
        c = files.calib
        print(f"calib      head={c.get('head')} features={c.get('features')} "
              f"grid={len(c.get('grid', []))} points")
    if res:
        print(f"result     {json.dumps(res)}")
    return 0
