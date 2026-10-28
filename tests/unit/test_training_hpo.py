"""HPO stage enumeration, resumability and outputs with a fake trainer."""

from __future__ import annotations

import csv
import json
import math

import pytest

from esbopfn.training import registry
from esbopfn.training.hpo import HPO, pareto_front
from esbopfn.util.yaml import load_yaml


class FakeTrainer:
    """Deterministic objective with its optimum at lr 5.6e-4, wd 1e-4, dropout 0 and per head a
    different best cell."""

    best_cell = {"linear": (384, 3), "binned": (512, 2), "gaussian": (256, 12)}

    def __init__(self, fail_on: str | None = None):
        self.calls: list[str] = []
        self.fail_on = fail_on

    def __call__(self, trial, row, out_dir):
        if self.fail_on and self.fail_on in trial.name:
            raise RuntimeError("boom")
        self.calls.append(f"{trial.stage}/{trial.name}")
        assert row["batch_size"] == 128 and row["head"] == trial.head
        bd, bl = self.best_cell[trial.head]
        nll = (abs(math.log10(row["lr"] / 5.6e-4)) + abs(math.log10(row["weight_decay"] / 1e-4))
               + row["dropout"] + (0 if trial.stage == "stage1" else
                                   abs(math.log2(row["d_model"] / bd)) + abs(math.log2(row["n_layers"] / bl))))
        return {"val_nll": nll, "train_time_s": row["d_model"] * row["n_layers"] / 1000,
                "n_params": row["d_model"] ** 2 * row["n_layers"], "steps_done": 10,
                "stopped_by": "steps"}


def test_stage1_is_sequential_and_resumable(pfn_home):
    fake = FakeTrainer()
    hpo = HPO("decpfn", "cartpole", "100k", trainer=fake)
    trials, best, done = hpo.stage1()
    assert not done and [t.lr for t in trials] == [1e-4, 1.8e-4, 3.2e-4, 5.6e-4, 1e-3]
    assert all(t.weight_decay == 1e-6 and t.dropout == 0 and t.head == "linear"
               and (t.d_model, t.n_layers) == (512, 3) for t in trials)
    hpo.run("1", shard=(0, 2))     # a shard runs only its part of the current step
    assert len(fake.calls) == 3
    hpo.run("1", shard=(1, 2))
    assert len(fake.calls) == 5
    trials, best, done = hpo.stage1()
    assert best == {"lr": 5.6e-4} and [t.weight_decay for t in trials] == [1e-6, 1e-4, 1e-2, 1e-1]
    assert all(t.lr == 5.6e-4 for t in trials)
    with pytest.raises(RuntimeError):
        hpo.stage2()
    hpo.run("1")
    _, best, done = hpo.stage1()
    assert done and best == {"lr": 5.6e-4, "weight_decay": 1e-4, "dropout": 0.0}
    n = len(fake.calls)
    assert n == 5 + 4 + 5
    hpo.run("1")                   # idempotent
    assert len(fake.calls) == n


def test_stage2_grid_candidate_and_outputs(pfn_home, tmp_path):
    fake = FakeTrainer()
    hpo = HPO("lcpfn", "cartpole", "1M", trainer=fake)
    with pytest.raises(RuntimeError):
        hpo.stage2()
    hpo.run("1")
    grid = hpo.stage2()
    assert len(grid) == 3 * 4 * 5 and all(t.time_limit_s == 250.0 for t in grid)
    # an interrupted stage 2 (a failing trial) keeps the finished results and resumes
    failing = HPO("lcpfn", "cartpole", "1M", trainer=FakeTrainer(fail_on="gaussian/d256"))
    with pytest.raises(RuntimeError):
        failing.run("2")
    done_before = sum(1 for t in grid if hpo.result(t) is not None)
    assert 0 < done_before < 60 and not (hpo.root / "hparams_candidate.yaml").exists()
    hpo.run("2")
    assert sum(1 for c in fake.calls if c.startswith("stage2")) == 60 - done_before
    root = pfn_home.resolve() / "hpo/lcpfn/cartpole/1M"
    assert hpo.root == root
    with (root / "trials.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 14 + 60
    sel = json.loads((root / "pareto.json").read_text())
    for head, (d, n) in FakeTrainer.best_cell.items():
        assert (sel[head]["best"]["d_model"], sel[head]["best"]["n_layers"]) == (d, n)
        times = [p["train_time_s"] for p in sel[head]["pareto"]]
        assert times == sorted(times)
    cand = load_yaml(root / "hparams_candidate.yaml")
    assert cand["fixed"]["batch_size"] == 128 and len(cand["configs"]) == 3
    assert (root / "gridsearch_panel.pdf").stat().st_size > 1000
    # the candidate file is accepted by pfn train --hparams
    row = registry.hparams_row("lcpfn-last", "1M", "gaussian", root / "hparams_candidate.yaml")
    assert (row["d_model"], row["n_layers"], row["lr"], row["weight_decay"]) == (256, 12, 5.6e-4, 1e-4)
    assert row["n_bin"] == 1000


def test_pareto_front():
    rows = [{"train_time_s": 1, "val_nll": 3}, {"train_time_s": 2, "val_nll": 4},
            {"train_time_s": 3, "val_nll": 1}, {"train_time_s": 0.5, "val_nll": float("nan")}]
    assert [r["val_nll"] for r in pareto_front(rows)] == [3, 1]


def test_hpo_rejects_other_models(pfn_home):
    with pytest.raises(ValueError):
        HPO("xdecpfn", "cartpole", "1M", trainer=FakeTrainer())
