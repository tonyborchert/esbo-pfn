"""Shared training core of the four own models (LC-PFN, DecPFN, XDecPFN, FT-PFN).

:func:`fit` runs AdamW with a linear warmup from 0 to the learning rate over the first
``warmup_frac`` of the optimizer steps and a cosine decay to 0 over the rest, stepped once per
optimizer step. The gradient norm is clipped (``clip_grad_norm``, None turns clipping off), CUDA
runs in bf16 autocast, ``out_dir/ckpt_last.pt`` is written every ``checkpoint_every`` steps and a
later call with the same ``out_dir`` resumes from it. ``metrics.json`` holds the loss curve and the
results of ``evaluate``. An optional wall-clock cap (``time_limit_s``, used by the HPO) stops
training early without changing the schedule, so a capped run is a truncated full run.

Randomness: the model is initialized by the caller, preferably through :func:`build_model`, which
seeds the parameter initialization from ``(seed, "init")`` inside a forked RNG state. Dropout
inside :func:`fit` draws from ``(seed, "train")`` in a forked RNG state as well, so the process
global RNG is never consumed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from esbopfn.util.seeds import derive_seed

log = logging.getLogger(__name__)

CKPT_NAME = "ckpt_last.pt"
METRICS_NAME = "metrics.json"


@dataclass
class TrainConfig:
    """Optimizer, schedule and bookkeeping settings of one training run."""

    lr: float
    weight_decay: float = 0.0
    dropout: float = 0.0
    batch_size: int = 128
    steps: int = 0                      # total optimizer steps (the schedule length)
    warmup_frac: float = 0.25           # linear warmup from 0 to lr, then cosine to 0, per step
    clip_grad_norm: float | None = 1.0  # None = no clipping (XDecPFN, FT-PFN)
    bf16: bool = True                   # bf16 autocast, applied on CUDA only
    seed: int = 0
    time_limit_s: float | None = None   # HPO wall-clock cap on the training time
    checkpoint_every: int = 500
    eval_every: int = 500
    log_every: int = 50
    device: str = "auto"                # cuda > mps > cpu

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class FitResult:
    """Outcome of :func:`fit`. ``stopped_by`` is ``steps``, ``time_limit`` or ``data``."""

    steps_done: int
    train_time_s: float
    final_metrics: dict = field(default_factory=dict)
    stopped_by: str = "steps"


# ----------------------------------------------------------------------------- helpers
def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` picks cuda, then mps, then cpu. Any other string is passed to ``torch.device``."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def lr_factor(step: int, total: int, warmup_frac: float = 0.25) -> float:
    """Multiplier of the base learning rate for the optimizer step with index ``step`` (0-based):
    linear from 0 over the warmup, then cosine to 0 at ``total``."""
    total = max(int(total), 1)
    warmup = int(round(warmup_frac * total))
    if step < warmup:
        return step / warmup
    span = max(total - warmup, 1)
    progress = min(max(step - warmup, 0) / span, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@contextlib.contextmanager
def seeded_torch(seed: int, *labels: int | str, device: torch.device | None = None):
    """Run a block with the torch RNG seeded from ``(seed, *labels)`` and restore the previous
    CPU (and CUDA) RNG state afterward."""
    cuda = device is not None and device.type == "cuda"
    devices = [device.index or 0] if cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(derive_seed(seed, *labels))
        yield


def build_model(factory: Callable[[], torch.nn.Module], seed: int) -> torch.nn.Module:
    """Construct a model with its initialization seeded from ``(seed, "init")``."""
    with seeded_torch(seed, "init"):
        return factory()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def to_device(batch: Any, device: torch.device) -> Any:
    """Move every tensor inside nested dicts, lists, tuples and dataclasses to ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if dataclasses.is_dataclass(batch) and not isinstance(batch, type):
        return dataclasses.replace(
            batch,
            **{f.name: to_device(getattr(batch, f.name), device) for f in dataclasses.fields(batch)
               if f.init},
        )
    if isinstance(batch, Mapping):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        return type(batch)(*(to_device(v, device) for v in batch))
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)
    return batch


def autocast_context(device: torch.device, bf16: bool):
    """bf16 autocast on CUDA when requested, a no-op context elsewhere."""
    if bf16 and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=_json_default))
    tmp.replace(path)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _save_ckpt(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# ----------------------------------------------------------------------------- fit
Batches = Iterator[Any] | Callable[[int], Iterator[Any]]


def fit(
    model: torch.nn.Module,
    batches: Batches,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    cfg: TrainConfig,
    out_dir: Path,
    evaluate: Callable[[torch.nn.Module], dict[str, float]] | None = None,
) -> FitResult:
    """Train ``model`` in place for ``cfg.steps`` optimizer steps.

    ``batches`` is an iterator of batches or a callable ``start_step -> iterator`` (preferred,
    since a resumed run then continues the data stream at the right place without drawing the
    skipped batches). Every batch is moved to the device with :func:`to_device` before
    ``loss_fn(model, batch)``, which returns the scalar loss. Training ends after ``cfg.steps``
    steps, when ``time_limit_s`` of training time is used up, or when the iterator is exhausted.
    ``evaluate(model)`` runs every ``eval_every`` steps and once at the end (in eval mode, without
    gradients), its results are logged to ``metrics.json`` and the last one is returned in
    :attr:`FitResult.final_metrics`. The model stays on the training device afterward.
    """
    if cfg.steps <= 0:
        raise ValueError("TrainConfig.steps must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, cfg.steps, cfg.warmup_frac)
    )
    history: dict[str, Any] = {"loss": [], "lr": [], "grad_norm": [], "evals": [], "skipped": 0}
    step, train_time = 0, 0.0
    stopped_by = "steps"
    ckpt_path = out_dir / CKPT_NAME

    with seeded_torch(cfg.seed, "train", device=device):
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            step, train_time = int(ckpt["step"]), float(ckpt["train_time_s"])
            history = ckpt["history"]
            stopped_by = ckpt.get("stopped_by", "steps")
            _set_rng_state(ckpt["rng"], device)
            log.info("resuming %s at step %d (%.1f s trained)", out_dir, step, train_time)
            if ckpt.get("finished"):
                return _finish(model, cfg, out_dir, history, step, train_time, stopped_by, None)

        stream = batches(step) if callable(batches) else _skip(batches, step)
        model.train()
        window: list[float] = []
        last_ckpt = step
        while step < cfg.steps:
            if cfg.time_limit_s is not None and train_time >= cfg.time_limit_s:
                stopped_by = "time_limit"
                break
            t0 = time.perf_counter()
            try:
                batch = next(stream)
            except StopIteration:
                stopped_by = "data"
                break
            batch = to_device(batch, device)
            with autocast_context(device, cfg.bf16):
                loss = loss_fn(model, batch)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                history["skipped"] += 1
                log.warning("non-finite loss at step %d, step skipped", step)
                scheduler.step()
                step += 1
                continue
            loss.backward()
            if cfg.clip_grad_norm is not None:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            else:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            optimizer.step()
            scheduler.step()
            step += 1
            value = float(loss.detach())
            if device.type in ("cuda", "mps"):
                _sync(device)
            train_time += time.perf_counter() - t0
            window.append(value)
            if step % cfg.log_every == 0 or step == cfg.steps:
                mean = sum(window) / len(window)
                history["loss"].append([step, mean, train_time])
                history["lr"].append([step, scheduler.get_last_lr()[0]])
                history["grad_norm"].append([step, float(gnorm)])
                window = []
                log.info("step %d/%d loss %.4f lr %.2e (%.1f s)", step, cfg.steps, mean,
                         scheduler.get_last_lr()[0], train_time)
            if evaluate is not None and cfg.eval_every and step % cfg.eval_every == 0 \
                    and step < cfg.steps:
                history["evals"].append({"step": step, **_run_eval(model, evaluate)})
            if cfg.checkpoint_every and step - last_ckpt >= cfg.checkpoint_every:
                _save_ckpt(ckpt_path, _payload(model, optimizer, scheduler, step, train_time,
                                               history, device, stopped_by, finished=False))
                last_ckpt = step
        if window:
            history["loss"].append([step, sum(window) / len(window), train_time])
        final = _run_eval(model, evaluate) if evaluate is not None else {}
        if final:
            history["evals"].append({"step": step, **final})
        # a capped run stays resumable (with a larger cap), a run that used its steps is final
        _save_ckpt(ckpt_path, _payload(model, optimizer, scheduler, step, train_time, history,
                                       device, stopped_by, finished=stopped_by != "time_limit"))
    return _finish(model, cfg, out_dir, history, step, train_time, stopped_by, final)


def _skip(it: Iterator[Any], n: int) -> Iterator[Any]:
    for _ in range(n):
        next(it, None)
    return it


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _run_eval(model: torch.nn.Module, evaluate: Callable[[torch.nn.Module], dict]) -> dict:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = {k: float(v) for k, v in evaluate(model).items()}
    finally:
        model.train(was_training)
    return result


def _payload(model, optimizer, scheduler, step, train_time, history, device, stopped_by,
             finished: bool) -> dict[str, Any]:
    return {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "step": step, "train_time_s": train_time,
        "history": history, "rng": _rng_state(device), "stopped_by": stopped_by,
        "finished": finished,
    }


def _finish(model, cfg, out_dir, history, step, train_time, stopped_by, final) -> FitResult:
    if final is None:
        final = history["evals"][-1] if history["evals"] else {}
        final = {k: v for k, v in final.items() if k != "step"}
    last_loss = history["loss"][-1][1] if history["loss"] else float("nan")
    final_metrics = {"train_loss": last_loss, **final}
    metrics = {
        "steps_done": step, "steps_planned": cfg.steps, "train_time_s": train_time,
        "stopped_by": stopped_by, "n_params": count_parameters(model),
        "final": final_metrics, **history,
    }
    write_json(out_dir / METRICS_NAME, metrics)
    return FitResult(step, train_time, final_metrics, stopped_by)


def train_config_from_row(row: Mapping[str, Any], *, steps: int, seed: int = 0,
                          time_limit_s: float | None = None, device: str = "auto",
                          eval_every: int | None = None) -> TrainConfig:
    """A :class:`TrainConfig` from a resolved hyperparameter row (``registry.hparams_row``)."""
    return TrainConfig(
        lr=float(row["lr"]), weight_decay=float(row.get("weight_decay", 0.0)),
        dropout=float(row.get("dropout", 0.0)), batch_size=int(row.get("batch_size", 128)),
        steps=int(steps), warmup_frac=float(row.get("warmup_frac", 0.25)),
        clip_grad_norm=(None if row.get("clip_grad_norm") is None
                        else float(row["clip_grad_norm"])),
        bf16=bool(row.get("bf16", True)), seed=int(seed), time_limit_s=time_limit_s,
        eval_every=int(eval_every) if eval_every is not None else max(500, steps // 10),
        checkpoint_every=500, log_every=50, device=device)
