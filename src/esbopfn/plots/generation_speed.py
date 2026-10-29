"""Plot group ``generation_speed``: tab:app:prior-cost and tab:setup:model-cost as Markdown and
LaTeX.

- ``prior_cost_table``: per problem the time to sample 10,000 single trajectories, the trajectory
  and system acceptance, and the time for 10,000 trajectory sets of FT-PFN and XDecPFN, all on one
  CPU core, from the ``generation_speed`` units.
- ``model_cost_table``: training data, parameters, training FLOPs and wall-clock of the trained
  predictors on Cartpole (LC-PFN and DecPFN with the binned head at 100k and 1M, XDecPFN, FT-PFN),
  from ``metrics.json`` and ``model.pt`` of the model tree. Rows of models that are not trained are
  left out.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from esbopfn.plots import PlotGroup
from esbopfn.plots import common as c

GROUP_NAME = "generation_speed"

PRIOR_LABEL = {"boiler": "Boiler", "threetank_pi": "Three-Tank", "pt2_deadtime": "PT2 dead-time",
               "cartpole": "Cartpole", "furuta": "Furuta"}


# ------------------------------------------------------------------------------- formatting
def _sig(x: float, digits: int = 2) -> str:
    if not math.isfinite(x):
        return "nan"
    if x == 0:
        return "0"
    mag = math.floor(math.log10(abs(x)))
    dec = max(digits - 1 - mag, 0)
    return f"{x:.{dec}f}"


def fmt_seconds(s: float) -> tuple[str, str]:
    """``(value, unit)`` in s below a minute, min below an hour, h above."""
    if not math.isfinite(s):
        return "nan", "s"
    if s < 60:
        return _sig(s, 2), "s"
    if s < 3600:
        return _sig(s / 60, 2), "min"
    return _sig(s / 3600, 2), "h"


def fmt_share(x: float) -> str:
    if x >= 0.1:
        return f"{x:.2f}"
    out = _sig(x, 3)
    return out.rstrip("0").rstrip(".") if "." in out else out


def _md(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _tex_time(s: float) -> str:
    v, u = fmt_seconds(s)
    return f"${v}$\\,{u}"


# ------------------------------------------------------------------------------- prior cost
def load_speed(ctx: c.Context, prov: c.Provenance) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    sec = ctx.cfg.get("generation_speed") if isinstance(ctx.cfg.get("generation_speed"),
                                                          dict) else {}
    for p in list(ctx.cfg.get("problems") or c.PROBLEMS):
        units = ctx.db.query("generation_speed", problem=p)
        if not units:
            continue
        # prefer the unit of the spec's parameters, else the one with the most samples
        units.sort(key=lambda u: (int(u.key.get("count", 0)) != int(sec.get("count", -1)),
                                  -int(u.key.get("count", 0))))
        prov.add_units("generation_speed", units[:1])
        out[p] = json.loads((Path(units[0].blob_dir) / "speed.json").read_text())
    return out


def draw_prior_cost(ctx: c.Context) -> c.Output:
    prov = c.Provenance()
    problems = list(ctx.cfg.get("problems") or c.PROBLEMS)
    speed = load_speed(ctx, prov)
    if not speed:
        raise c.MissingData([f"generation_speed units of {', '.join(problems)}"],
                            "run `pfn bench run generation_speed` (needs the C++ binaries, "
                            "`pfn data build`)")
    md_rows, tex_rows, numbers = [], [], []
    for p in problems:
        r = speed.get(p)
        if r is None:
            continue
        traj = fmt_seconds(float(r["traj_s_per_10k"]))
        ft = fmt_seconds(float(r["ftpfn_s_per_10k"]))
        xd = fmt_seconds(float(r["xdecpfn_s_per_10k"]))
        ta, sa = fmt_share(float(r["traj_acceptance"])), fmt_share(float(r["system_acceptance"]))
        md_rows.append([PRIOR_LABEL.get(p, p), f"{traj[0]} {traj[1]}", ta, sa,
                        f"{ft[0]} {ft[1]}", f"{xd[0]} {xd[1]}"])
        tex_rows.append(f"        {PRIOR_LABEL.get(p, p)} & {_tex_time(r['traj_s_per_10k'])} & "
                        f"${ta}$ & ${sa}$ & {_tex_time(r['ftpfn_s_per_10k'])} & "
                        f"{_tex_time(r['xdecpfn_s_per_10k'])} \\\\")
        numbers.append(f"{p}: {r['traj_s_per_10k']:.4g} s per 10k trajectories (acceptance "
                       f"{r['traj_acceptance']:.4g}), system acceptance "
                       f"{r['system_acceptance']:.4g}, FT-PFN sets {r['ftpfn_s_per_10k']:.4g} s, "
                       f"XDecPFN sets {r['xdecpfn_s_per_10k']:.4g} s per 10k (from "
                       f"{r.get('single_trajectories')} trajectories, {r.get('ftpfn_sets')} / "
                       f"{r.get('xdecpfn_sets')} sets, host {r.get('host', '?')})")
    header = ["Problem", "Trajectories: time", "Trajectories: acceptance", "Systems: acceptance",
              "Trajectory sets: FT-PFN", "Trajectory sets: XDecPFN"]
    tex = "\n".join([
        r"\begin{tabular}{@{}l c c c c c@{}}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{\textbf{Trajectories}} & \textbf{Systems} & "
        r"\multicolumn{2}{c}{\textbf{Trajectory sets}} \\",
        r"    \cmidrule(lr){2-3} \cmidrule(lr){4-4} \cmidrule(l){5-6}",
        r"    \textbf{Problem} & \textbf{Time} & \textbf{Acceptance} & \textbf{Acceptance} & "
        r"\textbf{\gls{ftpfn}} & \textbf{\gls{xdecpfn}} \\",
        r"    \midrule",
        *tex_rows,
        r"    \bottomrule",
        r"\end{tabular}", ""])
    side = c.Sidecar(
        thesis_asset="../sections/appendix.tex", label="tab:app:prior-cost",
        shown=["Sampling cost of each prior on one CPU core: 10,000 single trajectories, the "
               "trajectory and system acceptance, 10,000 trajectory sets for FT-PFN and XDecPFN."],
        metric=["Times are scaled to 10,000 samples from the measured counts, acceptance is the "
                "accepted share of sampled trajectories (single-trajectory rejection) and system "
                "instances (degeneracy screen)."],
        numbers=numbers,
        data=["`generation_speed` units (the C++ binaries' `speed` subcommand, one core)."],
        provenance=prov)
    return c.Output(side, texts={"prior_cost_table.md": _md(md_rows, header),
                                 "prior_cost_table.tex": tex})


# ------------------------------------------------------------------------------- model cost
#: Rows of tab:setup:model-cost: (label, LaTeX label, model id candidates, training data).
MODEL_ROWS: list[tuple[str, str, list[str], str, str]] = [
    ("LC-PFN 100k", r"\gls{lcpfn} 100k", ["lcpfn/cartpole/binned/100k"],
     "10^5 trajectories", r"$10^5$ trajectories"),
    ("LC-PFN 1M", r"\gls{lcpfn} 1M", ["lcpfn/cartpole/binned/1M"],
     "10^6 trajectories", r"$10^6$ trajectories"),
    ("DecPFN 100k", r"\gls{decpfn} 100k", ["decpfn/cartpole/binned/100k"],
     "10^5 trajectories", r"$10^5$ trajectories"),
    ("DecPFN 1M", r"\gls{decpfn} 1M", ["decpfn/cartpole/binned/1M"],
     "10^6 trajectories", r"$10^6$ trajectories"),
    ("XDecPFN", r"\gls{xdecpfn}",
     ["xdecpfn/cartpole/default/default", "xdecpfn/cartpole/thesis/default"],
     "132,400 trajectory sets", r"$132{,}400$ trajectory sets"),
    ("FT-PFN", r"\gls{ftpfn}",
     ["ftpfn/cartpole/default/default", "ftpfn/cartpole/thesis/default"],
     "3,500,000 trajectory sets", r"$3{,}500{,}000$ trajectory sets"),
]


def _count_params(path: Path) -> int | None:
    try:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 -- the column stays empty
        return None
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(sd, dict):
        return None
    return int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))


def model_cost(path: Path) -> dict[str, Any]:
    """``{params, flops, wall_s}`` of a model directory (None where unknown)."""
    m: dict[str, Any] = {}
    try:
        m = json.loads((path / "metrics.json").read_text())
    except (OSError, ValueError):
        pass
    params = m.get("n_params")
    if params is None:
        params = _count_params(path / "model.pt")
    steps = m.get("steps_done")
    flops = m.get("flops_total")
    if flops is None and m.get("flops_per_step") and steps:
        flops = float(m["flops_per_step"]) * float(steps)
    wall = m.get("train_time_s", m.get("elapsed_total_s"))
    return {"params": params, "flops": flops, "wall_s": wall, "steps": steps}


def _fmt_params(n: int | None) -> str:
    return "–" if n is None else f"{n / 1e6:.1f} M"


def _fmt_flops(f: float | None, tex: bool) -> str:
    if not f:
        return "–"
    e = int(math.floor(math.log10(f)))
    m = f / 10 ** e
    return f"${m:.2f}\\times10^{{{e}}}$" if tex else f"{m:.2f}e{e}"


def _fmt_wall(s: float | None, tex: bool) -> str:
    if s is None:
        return "–"
    v, u = fmt_seconds(float(s))
    return f"${v}$\\,{u}" if tex else f"{v} {u}"


def draw_model_cost(ctx: c.Context) -> c.Output:
    from esbopfn.training.registry import is_trained, model_dir

    prov = c.Provenance()
    md_rows, tex_rows, numbers, missing = [], [], [], []
    for label, tex_label, ids, data_md, data_tex in MODEL_ROWS:
        found = next((m for m in ids if is_trained(model_dir(m))), None)
        if found is None:
            missing.append(ids[0])
            continue
        path = model_dir(found)
        prov.add_file(path)
        if "/thesis/" in found:
            prov.thesis_import = True
        cost = model_cost(path)
        md_rows.append([label, data_md, _fmt_params(cost["params"]),
                        _fmt_flops(cost["flops"], False), _fmt_wall(cost["wall_s"], False)])
        p = "–" if cost["params"] is None else f"${cost['params'] / 1e6:.1f}$\\,M"
        tex_rows.append(f"        {tex_label} & {data_tex} & {p} & "
                        f"{_fmt_flops(cost['flops'], True)} & "
                        f"{_fmt_wall(cost['wall_s'], True)} \\\\")
        numbers.append(f"{label} ({found}): params {cost['params']}, FLOPs {cost['flops']}, "
                       f"wall-clock {cost['wall_s']} s, steps {cost['steps']}")
    if not md_rows:
        raise c.MissingData([f"trained models {', '.join(missing)}"],
                            "train them with `pfn train` or import with `pfn import "
                            "thesis-models`")
    header = ["Model", "Training data", "Parameters", "Training FLOPs", "Wall-clock"]
    tex = "\n".join([
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X "
        r">{\raggedright\arraybackslash}p{4.2cm} c c c@{}}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Training data} & \textbf{Parameters} & "
        r"\textbf{Training \glspl{flop}} & \textbf{Wall-clock} \\",
        r"    \midrule",
        *tex_rows,
        r"    \bottomrule",
        r"\end{tabularx}", ""])
    side = c.Sidecar(
        thesis_asset="../sections/setup.tex", label="tab:setup:model-cost",
        shown=["Training data, parameter count, training FLOPs and wall-clock of the trained "
               "predictors on Cartpole (LC-PFN and DecPFN with the binned head)."]
        + ([f"Not trained here, left out: {', '.join(missing)}."] if missing else []),
        metric=["Parameters: `n_params` of metrics.json, else counted from model.pt.",
                "FLOPs: `flops_per_step` x `steps_done` where the model records them.",
                "Wall-clock: training time of metrics.json (`train_time_s`, or `elapsed_total_s` "
                "of imported checkpoints), on the machine that trained the model."],
        numbers=numbers, data=["The model tree `$PFN_HOME/models/`."], provenance=prov)
    return c.Output(side, texts={"model_cost_table.md": _md(md_rows, header),
                                 "model_cost_table.tex": tex})


FIGURES = [c.FigureSpec("prior_cost_table", draw_prior_cost),
           c.FigureSpec("model_cost_table", draw_model_cost)]

GROUP = PlotGroup(name=GROUP_NAME, needs=c.spec_needs(GROUP_NAME),
                  render=c.make_render(GROUP_NAME, FIGURES))
