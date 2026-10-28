"""Hyperparameter search of LC-PFN and DecPFN (thesis, hyperparameters and Appendix sweeps).

Stage 1 sweeps the regularizing hyperparameters sequentially from the start configuration of
``configs/hparams/<model>.yaml`` (learning rate, then weight decay, then dropout, each step keeping
the best value of the steps before). Stage 2 trains a grid ``d_model x n_layers`` for every head
with the stage-1 values and a wall-clock cap per run. Every head takes the best cell of its own
grid. The objective is the validation NLL of the horizon prediction at the grid cutoffs between
20 % and 99 % in raw cost units (``val_nll`` of the trainers).

Output tree ``$PFN_HOME/hpo/<model>/<problem>/<size>/``::

    stage1/<step>_<param>_<value>/   trained model dir + result.json
    stage2/<head>/d<d_model>_l<n_layers>/
    trials.csv                       every finished trial
    pareto.json                      stage-2 Pareto fronts (time, NLL) and the selected cells
    hparams_candidate.yaml           the selected configuration (pfn train --hparams FILE)
    gridsearch_panel.pdf             training wall-clock vs validation NLL, marker size ~ params

Each trial is idempotent (a finished ``result.json`` is never retrained) and resumable (training
continues from its checkpoint). ``--shard i/n`` runs every n-th trial of the current step
(finished ones are skipped).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from esbopfn.config import paths
from esbopfn.training import registry
from esbopfn.training.common import write_json
from esbopfn.util.yaml import dump_yaml

log = logging.getLogger(__name__)

MODELS = ("lcpfn", "decpfn")
RESULT_FILE = "result.json"
TRIAL_FIELDS = ("stage", "name", "head", "d_model", "n_layers", "lr", "weight_decay", "dropout",
                "time_limit_s", "val_nll", "val_nll_0_99", "train_time_s", "n_params",
                "steps_done", "stopped_by")


@dataclass(frozen=True)
class HPOTrial:
    stage: str
    name: str
    head: str
    d_model: int
    n_layers: int
    lr: float
    weight_decay: float
    dropout: float
    time_limit_s: float | None = None

    def row(self, fixed: dict[str, Any], size: str) -> dict[str, Any]:
        return {**fixed, "size": size, "head": self.head, "d_model": self.d_model,
                "n_layers": self.n_layers, "lr": self.lr, "weight_decay": self.weight_decay,
                "dropout": self.dropout}


#: ``trainer(trial, hparams_row, out_dir) -> metrics`` with at least ``val_nll``,
#: ``train_time_s`` and ``n_params``.
Trainer = Callable[[HPOTrial, dict[str, Any], Path], dict[str, Any]]


def _fmt(v: float) -> str:
    return f"{v:.1e}" if v and (abs(v) < 1e-2 or abs(v) >= 1e3) else f"{v:g}"


def default_trainer(model: str, problem: str, size: str, seed: int = 0, device: str = "auto",
                    home: Path | str | None = None) -> Trainer:
    """Train with the real trainer and read its metrics. Validation runs once at the end."""

    def train_trial(trial: HPOTrial, row: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        if model == "decpfn":
            from esbopfn.predictors.decpfn.train import train
        else:
            from esbopfn.predictors.lcpfn.train import train
        path = train(problem, trial.head, size, row, seed=seed, out_dir=out_dir,
                     time_limit=trial.time_limit_s, device=device, variant=model,
                     home=Path(home) if home is not None else None, eval_every=10**9)
        metrics = json.loads((path / registry.METRICS_FILE).read_text())
        return {**metrics["result"], "n_params": metrics.get("n_params")}

    return train_trial


class HPO:
    """The search of one (model, problem, size)."""

    def __init__(self, model: str, problem: str, size: str, home: Path | str | None = None,
                 trainer: Trainer | None = None, seed: int = 0, device: str = "auto",
                 hparams_file: Path | str | None = None):
        if model not in MODELS:
            raise ValueError(f"HPO exists for {MODELS}, got {model!r}")
        self.model, self.problem, self.size = model, problem, size
        self.home = home
        self.root = paths(home).root / "hpo" / model / problem / size
        data = registry.load_hparams_file(model, hparams_file)
        self.fixed = dict(data["fixed"])
        self.space = data["hpo"]
        self.trainer = trainer or default_trainer(model, problem, size, seed, device, home)

    # ------------------------------------------------------------------ results
    def trial_dir(self, trial: HPOTrial) -> Path:
        return self.root / trial.stage / trial.name

    def result(self, trial: HPOTrial) -> dict[str, Any] | None:
        p = self.trial_dir(trial) / RESULT_FILE
        return json.loads(p.read_text()) if p.is_file() else None

    def run_trial(self, trial: HPOTrial) -> dict[str, Any]:
        done = self.result(trial)
        if done is not None:
            return done
        out = self.trial_dir(trial)
        out.mkdir(parents=True, exist_ok=True)
        log.info("HPO %s/%s/%s: %s/%s", self.model, self.problem, self.size, trial.stage,
                 trial.name)
        metrics = self.trainer(trial, trial.row(self.fixed, self.size), out)
        res = {**asdict(trial), **metrics}
        write_json(out / RESULT_FILE, res)
        return res

    # ------------------------------------------------------------------ stage 1
    def _stage1_step(self, k: int, best: dict[str, float]) -> list[HPOTrial]:
        start = self.space["start"]
        step = self.space["stage1"][k]
        base = {"lr": best.get("lr", float(step["values"][0])),
                "weight_decay": best.get("weight_decay", float(start["weight_decay"])),
                "dropout": best.get("dropout", float(start["dropout"]))}
        out = []
        for v in step["values"]:
            hp = {**base, step["param"]: float(v)}
            out.append(HPOTrial("stage1", f"{k + 1}_{step['param']}_{_fmt(float(v))}",
                                start["head"], int(start["d_model"]), int(start["n_layers"]),
                                hp["lr"], hp["weight_decay"], hp["dropout"]))
        return out

    def stage1(self) -> tuple[list[HPOTrial], dict[str, float], bool]:
        """``(trials of the current step, best values so far, finished)``. The current step is the
        first one with a missing result, its trials depend on the best values before it."""
        best: dict[str, float] = {}
        for k, step in enumerate(self.space["stage1"]):
            trials = self._stage1_step(k, best)
            results = [self.result(t) for t in trials]
            if any(r is None for r in results):
                return trials, best, False
            i = min(range(len(trials)), key=lambda j: _nll(results[j]))
            best[step["param"]] = getattr(trials[i], step["param"])
        return [], best, True

    # ------------------------------------------------------------------ stage 2
    def stage2(self) -> list[HPOTrial]:
        trials_1, best, done = self.stage1()
        if not done:
            raise RuntimeError("stage 1 is not finished, run `pfn hpo ... --stage 1` first")
        g = self.space["stage2"]
        cap = g.get("time_limit_s", {}).get(self.size)
        return [HPOTrial("stage2", f"{h}/d{d}_l{n}", h, int(d), int(n), best["lr"],
                         best["weight_decay"], best["dropout"],
                         None if cap is None else float(cap))
                for h in g["heads"] for d in g["d_model"] for n in g["n_layers"]]

    # ------------------------------------------------------------------ driver
    def run(self, stage: str = "all", shard: tuple[int, int] | None = None) -> None:
        """Run the pending trials of ``stage`` (``1``, ``2`` or ``all``), then summarize."""
        if stage in ("1", "all"):
            while True:
                trials, _, done = self.stage1()
                if done:
                    break
                pending = [t for t in _shard(trials, shard) if self.result(t) is None]
                for t in pending:
                    self.run_trial(t)
                if shard is not None:   # later steps need every shard of this one
                    break
        if stage in ("2", "all") and self.stage1()[2]:
            for t in _shard(self.stage2(), shard):
                self.run_trial(t)   # skips finished trials
        self.summarize()

    # ------------------------------------------------------------------ outputs
    def results(self) -> list[dict[str, Any]]:
        out = []
        for f in sorted(self.root.glob(f"stage*/**/{RESULT_FILE}")):
            out.append(json.loads(f.read_text()))
        return out

    def selection(self) -> dict[str, Any] | None:
        """Per head the stage-2 Pareto front and the best cell, None before stage 2 is done."""
        if not self.stage1()[2]:
            return None
        grid = self.stage2()
        res = {t.name: self.result(t) for t in grid}
        if any(r is None for r in res.values()):
            return None
        sel: dict[str, Any] = {}
        for head in self.space["stage2"]["heads"]:
            rows = [res[t.name] for t in grid if t.head == head]
            front = pareto_front(rows)
            best = min(rows, key=_nll)
            sel[head] = {"best": _brief(best), "pareto": [_brief(r) for r in front]}
        return sel

    def candidate(self) -> dict[str, Any] | None:
        sel = self.selection()
        if sel is None:
            return None
        _, best, _ = self.stage1()
        configs = [{"size": self.size, "head": h, "n_layers": int(s["best"]["n_layers"]),
                    "d_model": int(s["best"]["d_model"]), "lr": best["lr"],
                    "weight_decay": best["weight_decay"], "dropout": best["dropout"]}
                   for h, s in sel.items()]
        return {"model": self.model, "problem": self.problem, "size": self.size,
                "provenance": f"pfn hpo {self.model} --problem {self.problem} --size {self.size}",
                "fixed": dict(self.fixed), "configs": configs}

    def summarize(self) -> None:
        rows = self.results()
        if not rows:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=TRIAL_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in TRIAL_FIELDS})
        tmp = self.root / "trials.csv.tmp"
        tmp.write_text(buf.getvalue())
        tmp.replace(self.root / "trials.csv")
        sel = self.selection()
        if sel is None:
            return
        write_json(self.root / "pareto.json", sel)
        dump_yaml(self.candidate(), self.root / "hparams_candidate.yaml")
        try:
            gridsearch_panel([r for r in rows if r["stage"] == "stage2"], self.model,
                             self.root / "gridsearch_panel.pdf",
                             title=f"{self.model} {self.problem} {self.size}")
        except Exception as exc:  # the plot must not lose finished results
            log.warning("gridsearch panel failed: %s", exc)


def _nll(r: dict[str, Any] | None) -> float:
    v = None if r is None else r.get("val_nll")
    return float(v) if v is not None and math.isfinite(float(v)) else math.inf


def _brief(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r.get(k) for k in ("name", "d_model", "n_layers", "val_nll", "train_time_s",
                                  "n_params")}


def _shard(items: list[HPOTrial], shard: tuple[int, int] | None) -> list[HPOTrial]:
    if shard is None:
        return items
    i, n = shard
    return sorted(items, key=lambda t: (t.stage, t.name))[i::n]


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows not dominated in (training time, validation NLL), both minimized, sorted by time."""
    pts = sorted(rows, key=lambda r: (float(r.get("train_time_s") or 0.0), _nll(r)))
    front, best = [], math.inf
    for r in pts:
        if _nll(r) < best:
            front.append(r)
            best = _nll(r)
    return front


HEAD_MARKERS = {"binned": "o", "linear": "s", "gaussian": "^"}


def gridsearch_panel(rows: list[dict[str, Any]], model: str, path: Path, title: str = "",
                     theme: str = "white") -> Path:
    """Training wall-clock against validation NLL of the stage-2 grid, marker size proportional
    to the parameter count, one marker shape per head (thesis fig:app:hpo-gridsearch)."""
    import matplotlib.pyplot as plt

    from esbopfn.plots.style import figsize, method_color, save, thesis_style

    with thesis_style(theme):
        fig, ax = plt.subplots(figsize=figsize(1, 2.6))
        params = [float(r.get("n_params") or 1.0) for r in rows]
        scale = 120.0 / max(params) if params else 1.0
        for head, marker in HEAD_MARKERS.items():
            sub = [r for r in rows if r.get("head") == head and math.isfinite(_nll(r))]
            if not sub:
                continue
            ax.scatter([float(r["train_time_s"]) for r in sub], [_nll(r) for r in sub],
                       s=[max(6.0, scale * float(r.get("n_params") or 1.0)) for r in sub],
                       marker=marker, facecolors="none", edgecolors=method_color(model, theme),
                       linewidths=0.8, label=head)
        ax.set_xlabel("training wall-clock [s]")
        ax.set_ylabel("validation NLL")
        if title:
            ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="best")
        out = save(fig, path, "pdf")
        plt.close(fig)
    return out
