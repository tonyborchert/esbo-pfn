"""Training core, registry, hyperparameter files and the train command."""

from __future__ import annotations

import json
import math

import pytest
import torch

from esbopfn.cli import main
from esbopfn.training import TRAINERS, TrainRequest, common, get_trainer, registry


def test_lr_schedule_warmup_then_cosine():
    total = 100
    f = [common.lr_factor(s, total, 0.25) for s in range(total + 1)]
    assert f[0] == 0.0 and f[25] == pytest.approx(1.0) and f[100] == pytest.approx(0.0)
    assert f[12] == pytest.approx(12 / 25)
    assert f[25 + 75 // 2] == pytest.approx(0.5 * (1 + math.cos(math.pi * 37 / 75)))
    assert all(a <= b for a, b in zip(f[:25], f[1:26]))
    assert all(a >= b for a, b in zip(f[25:], f[26:]))


def _toy(seed: int) -> torch.nn.Module:
    return common.build_model(lambda: torch.nn.Linear(3, 1), seed)


def _batches(start: int):
    w = torch.tensor([[1.0, -2.0, 0.5]])
    step = start
    while True:
        x = torch.randn(32, 3, generator=torch.Generator().manual_seed(1234 + step))
        step += 1
        yield {"x": x, "y": x @ w.T}


def _interrupted(after: int):
    def make(start: int):
        for i, b in enumerate(_batches(start)):
            if start + i >= after:
                raise KeyboardInterrupt
            yield b
    return make


def _loss(model, batch):
    return ((model(batch["x"]) - batch["y"]) ** 2).mean()


def test_build_model_is_seeded_and_leaves_global_rng():
    state = torch.get_rng_state()
    a, b, c = _toy(1), _toy(1), _toy(2)
    assert torch.equal(a.weight, b.weight) and not torch.equal(a.weight, c.weight)
    assert torch.equal(state, torch.get_rng_state())


def test_fit_trains_resumes_and_is_idempotent(tmp_path):
    cfg = common.TrainConfig(lr=0.05, steps=60, device="cpu", log_every=10, checkpoint_every=20,
                             eval_every=30)
    model = _toy(0)
    res = common.fit(model, _batches, _loss, cfg, tmp_path, evaluate=lambda m: {"w0": m.weight[0, 0]})
    m = json.loads((tmp_path / common.METRICS_NAME).read_text())
    assert res.steps_done == 60 and res.stopped_by == "steps"
    assert m["loss"][-1][1] < m["loss"][0][1] and len(m["evals"]) == 2
    again = common.fit(_toy(5), _batches, _loss, cfg, tmp_path)   # finished: nothing to do
    assert again.steps_done == 60

    # a run killed at step 25 resumes from its step-20 checkpoint to the identical result
    cfg2 = common.TrainConfig(lr=0.05, steps=60, device="cpu", checkpoint_every=20)
    full = _toy(0)
    common.fit(full, _batches, _loss, cfg2, tmp_path / "full")
    resumed = _toy(0)
    with pytest.raises(KeyboardInterrupt):
        common.fit(resumed, _interrupted(25), _loss, cfg2, tmp_path / "part")
    assert torch.load(tmp_path / "part" / common.CKPT_NAME, weights_only=False)["step"] == 20
    r = common.fit(_toy(3), _batches, _loss, cfg2, tmp_path / "part")
    assert r.steps_done == 60
    ck = torch.load(tmp_path / "part" / common.CKPT_NAME, weights_only=False)["model"]
    assert torch.allclose(ck["weight"], full.weight.detach())


def test_time_limit_truncates_without_changing_schedule(tmp_path):
    cfg = common.TrainConfig(lr=0.05, steps=10_000, device="cpu", time_limit_s=0.05)
    res = common.fit(_toy(0), _batches, _loss, cfg, tmp_path)
    assert res.stopped_by == "time_limit" and 0 < res.steps_done < 10_000


def test_to_device_handles_nested_structures():
    from esbopfn.training.single_trajectory_data import LCPFNBatch

    b = LCPFNBatch(torch.zeros(2, 3), torch.zeros(2, 1), torch.zeros(2, 2), 1)
    moved = common.to_device({"a": [b, (torch.ones(1),)]}, torch.device("cpu"))
    assert moved["a"][0].kappa == 1 and moved["a"][1][0].item() == 1.0


# ----------------------------------------------------------------------------- registry
def test_model_ids_and_dirs(pfn_home):
    mid = registry.ModelId.parse("decpfn/cartpole/binned/1M")
    assert mid.run == "0" and str(mid) == "decpfn/cartpole/binned/1M/0"
    assert mid.short == "decpfn/cartpole/binned/1M"
    assert registry.ModelId.parse("lcpfn-last/furuta/binned/100k/3").run == "3"
    with pytest.raises(ValueError):
        registry.ModelId.parse("decpfn/cartpole")
    assert registry.model_dir(mid) == pfn_home.resolve() / "models/decpfn/cartpole/binned/1M/0"
    with pytest.raises(registry.ModelNotTrained) as exc:
        registry.resolve("decpfn-nostates/cartpole/linear/100k")
    assert "pfn train decpfn-nostates --problem cartpole --head linear --size 100k" in str(exc.value)


def test_save_load_list(pfn_home):
    out = registry.model_dir("xdecpfn/cartpole/default/default")
    registry.save_model(out, {"w": torch.ones(2)}, {"d_model": 8}, {"edges": [0, 1]},
                        {"seed": 1}, {"final": 1.0})
    files = registry.load_model(registry.resolve("xdecpfn/cartpole/default/default"))
    assert files.model_cfg == {"d_model": 8} and files.calib["edges"] == [0, 1]
    assert files.train_cfg["seed"] == 1 and files.metrics["final"] == 1.0
    assert [m.short for m in registry.list_models()] == ["xdecpfn/cartpole/default/default"]
    assert main(["models", "list"]) == 0
    assert main(["models", "show", "xdecpfn/cartpole/default/default"]) == 0
    assert main(["models", "show", "decpfn/cartpole/binned/1M"]) == 1


# ----------------------------------------------------------------------------- hparams
THESIS_TABLE = [  # thesis Appendix tab:app:hpo-configs (model, size, head, N_layer, d_model, lr, wd)
    ("lcpfn", "100k", "binned", 2, 512, 5.6e-4, 1e-4),
    ("lcpfn", "100k", "gaussian", 2, 384, 5.6e-4, 1e-4),
    ("lcpfn", "100k", "linear", 3, 384, 5.6e-4, 1e-4),
    ("lcpfn", "1M", "binned", 12, 384, 3.2e-4, 1e-4),
    ("lcpfn", "1M", "gaussian", 12, 256, 3.2e-4, 1e-4),
    ("lcpfn", "1M", "linear", 6, 512, 3.2e-4, 1e-4),
    ("decpfn", "100k", "binned", 3, 1024, 5.6e-4, 1e-6),
    ("decpfn", "100k", "gaussian", 6, 512, 5.6e-4, 1e-6),
    ("decpfn", "100k", "linear", 6, 1024, 5.6e-4, 1e-6),
    ("decpfn", "1M", "binned", 6, 384, 5.6e-4, 1e-4),
    ("decpfn", "1M", "gaussian", 3, 256, 5.6e-4, 1e-4),
    ("decpfn", "1M", "linear", 12, 384, 5.6e-4, 1e-4),
]


def test_hparams_match_thesis_table():
    rows = []
    for model in ("lcpfn", "decpfn"):
        data = registry.load_hparams_file(model)
        assert data["fixed"] == {"batch_size": 128, "d_k": 16, "d_ff_factor": 2, "n_bin": 1000,
                                 "warmup_frac": 0.25, "clip_grad_norm": 1.0, "bf16": True}
        for r in data["configs"]:
            assert r["dropout"] == 0.0
            rows.append((model, r["size"], r["head"], r["n_layers"], r["d_model"], r["lr"],
                         r["weight_decay"]))
        hpo = data["hpo"]
        assert hpo["start"] == {"head": "linear", "d_model": 512, "n_layers": 3,
                                "weight_decay": 1e-6, "dropout": 0.0}
        assert [s["values"] for s in hpo["stage1"]] == [
            [1e-4, 1.8e-4, 3.2e-4, 5.6e-4, 1e-3], [1e-6, 1e-4, 1e-2, 1e-1], [0, 0.1, 0.2, 0.35, 0.5]]
        assert hpo["stage2"]["d_model"] == [256, 384, 512, 1024]
        assert hpo["stage2"]["n_layers"] == [1, 2, 3, 6, 12]
        assert hpo["stage2"]["time_limit_s"] == {"100k": 25, "1M": 250}
    assert len(rows) == 12
    assert sorted(rows) == sorted(THESIS_TABLE)


def test_hparams_inheritance_and_row():
    assert registry.hparams_base("decpfn-nostates") == "decpfn"
    assert registry.hparams_base("decpfn-nodeadtime") == "decpfn"
    assert registry.hparams_base("lcpfn-last") == "lcpfn"
    row = registry.hparams_row("decpfn-nostates", "1M", "binned")
    assert (row["n_layers"], row["d_model"], row["batch_size"]) == (6, 384, 128)
    cfg = common.train_config_from_row(row, steps=7812)
    assert cfg.clip_grad_norm == 1.0 and cfg.batch_size == 128 and cfg.warmup_frac == 0.25


def test_trainers_dict_and_train_command(pfn_home, caplog):
    assert {"lcpfn", "lcpfn-last", "decpfn", "decpfn-nostates", "decpfn-nodeadtime"} <= set(TRAINERS)
    assert callable(get_trainer("decpfn"))
    with pytest.raises(KeyError):
        get_trainer("nope")
    from esbopfn.cli import build_parser
    from esbopfn.commands.train import requests

    args = build_parser().parse_args(["train", "all", "--problem", "all"])
    reqs = requests(args)
    assert all(isinstance(r, TrainRequest) for r in reqs)
    nodead = [r for r in reqs if r.model == "decpfn-nodeadtime"]
    assert [r.problem for r in nodead] == ["furuta"]
    assert all(r.head == "binned" and r.size == "1M" for r in reqs if r.model == "decpfn")
    args = build_parser().parse_args(["train", "decpfn", "--problem", "boiler", "--head", "all",
                                      "--size", "all"])
    assert len(requests(args)) == 6
    # missing data is reported with the generation hint, not a traceback
    assert main(["train", "decpfn", "--problem", "boiler", "--size", "100k"]) == 1
    assert "pfn data gen --for train decpfn --problem boiler" in caplog.text


def test_train_rows_override(pfn_home):
    from esbopfn.cli import build_parser
    from esbopfn.commands.train import requests

    args = build_parser().parse_args(["train", "decpfn", "--problem", "boiler", "--rows", "500"])
    assert [r.rows for r in requests(args)] == [500]
    args = build_parser().parse_args(["train", "xdecpfn", "--problem", "boiler", "--rows", "500"])
    with pytest.raises(SystemExit):
        requests(args)
