"""Live-inference example figures (thesis fig:sim:lcpfn-decpfn-example,
fig:sim:ftpfn-xdecpfn-example and fig:sim:landscape-furuta).

The three :class:`~esbopfn.plots.common.FigureSpec` objects of this module are listed by the
``single_trajectory`` and ``cross_trajectory`` plot groups.

The example trial
    One non-crashed Cartpole query of the prediction benchmark: the task with the median total
    cost (lower median, ties broken by run and query index) among the ``selected``, non-crashed
    ``n = 64`` tasks. The native task table ``tasks-oracle65`` and its field units are used when
    they exist. Otherwise the imported thesis table ``tasks-thesis`` names the run (thesis:
    ``i0008_s00``, query 64) and that Oracle run is simulated on the imported thesis instance
    with the seed of the ``fields`` kind, cached under ``<out_dir>/_cache/examples/``. Without
    any table, run ``i0008_s00`` of the available instance list is used. If the query trial of a
    simulated run crashes, the last non-crashed trial at or before it is the query.

The Furuta landscape
    One plant (the first of instances 0-9 whose true slice has a crash share between 0.25 and
    0.85, unless ``figures.landscape_furuta.instance`` is given), the oblique 24 x 24 slice of the
    ``landscapes`` kind (slice seed 1, six repeats per grid point), a pure-loop run per predictor
    family on that plant (XDecPFN-PI, FT-PFN-EI, 64 trials) whose prefixes of 1, 4, 16 and 64
    trials are the contexts, and the models' predictions for the unrun slice controllers
    (``predict_unrun``). The true slice is read from the database when present, else computed
    with the kind's own function, and every live result is cached as npz under
    ``<out_dir>/_cache/examples/``, keyed by plant, slice, model checkpoint hash and context.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from esbopfn.plots import common as c
from esbopfn.plots import style

log = logging.getLogger(__name__)

EXAMPLE_PROBLEM = "cartpole"
EXAMPLE_N_CTX = 64
EXAMPLE_RUN = (8, 0)                         # (instance, seed) of the thesis example run
EXAMPLE_QUERY = 64
CUTOFF_FRACS = (0.35, 0.65, 0.85)
CONTEXT_CUTOFF = 0.35
CONTEXT_SIZES = (0, 5, 10)
CONTEXT_SEED = 20260920
Y_HEADROOM = 1.6
SIDE_FRAC = 0.075                            # width of the density strip right of T

LANDSCAPE_PROBLEM = "furuta"
LANDSCAPE_CONTEXTS = (1, 4, 16, 64)
LANDSCAPE_SLICE_SEED = 1
LANDSCAPE_GRID = 24
LANDSCAPE_REPEATS = 6
LANDSCAPE_SCAN = 10
LANDSCAPE_CRASH_BAND = (0.25, 0.85)
LANDSCAPE_LOOP_SEED = 0
#: Landscape colors: objective rank (blue) and crash probability (red), blended additively.
BLUE_RGB = np.array([0.20, 0.55, 0.98])
RED_RGB = np.array([0.98, 0.20, 0.20])

DENSITY_COLOR = style.RWTH_BLUE
#: Sidecar line on the live-result cache (``<out_dir>/_cache/examples/``).
CACHE_NOTE = ("Live results (simulated runs, true slice, model predictions) are cached as npz "
              "under `_cache/examples/` next to this figure, keyed by plant, slice, model "
              "checkpoint hash and context. Delete that directory to recompute them.")


# ============================================================================ models
def _build(model_id: str, problem: str) -> Any:
    """The predictor of ``model_id`` (raises when it is not trained)."""
    from esbopfn.benchmarks.kinds.prediction import build_predictor

    return build_predictor(model_id, problem)


def _checkpoint(model_id: str) -> str:
    from esbopfn.benchmarks.kinds.prediction import checkpoint_hash

    try:
        return checkpoint_hash(model_id)
    except Exception:  # noqa: BLE001 -- only a cache key
        return "unknown"


def load_model(candidates: list[str], problem: str) -> tuple[str, Any]:
    """The first trained model of ``candidates``. Raises :class:`MissingData` naming the train
    command when none is trained."""
    errors = []
    for mid in candidates:
        try:
            return mid, _build(mid, problem)
        except (FileNotFoundError, KeyError, LookupError) as exc:
            errors.append((mid, exc))
    hint = getattr(errors[0][1], "hint", "") if errors else ""
    raise c.MissingData([f"trained model {m}" for m, _ in errors],
                        hint or f"train it with `pfn train {candidates[0].split('/')[0]} "
                                f"--problem {problem}`")


def load_models(candidate_lists: list[list[str]], problem: str) -> list[tuple[str, Any]]:
    """:func:`load_model` for several models, one :class:`MissingData` naming all missing."""
    out, missing, hints = [], [], []
    for cands in candidate_lists:
        try:
            out.append(load_model(cands, problem))
        except c.MissingData as exc:
            missing += exc.missing
            hints.append(exc.hint)
    if missing:
        raise c.MissingData(missing, " ".join(h for h in hints if h))
    return out


def cross_candidates(family: str, problem: str) -> list[str]:
    return [f"{family}/{problem}/default/default", f"{family}/{problem}/thesis/default"]


# ============================================================================ the example trial
@dataclass
class ExampleTrial:
    """The query trial and the run it belongs to."""

    run: Any                    # benchmarks.kinds.fields.FieldRun
    query: int
    source: str
    run_label: str
    instances_seed: Any
    notes: list[str]

    @property
    def trajectory(self) -> Any:
        return self.run.trajectory(self.query)

    @property
    def T(self) -> int:
        return int(self.run.J.shape[1])


def _instances_seed(db: Any, problem: str) -> Any:
    """The native instance list if the database has one, else the thesis import."""
    from esbopfn.benchmarks.instances import BENCH_SEED

    native = [u for u in db.query("instances", problem=problem) if "provenance" not in u.key]
    if native:
        return BENCH_SEED
    if db.query("instances", problem=problem, provenance="thesis"):
        return "thesis"
    raise c.MissingData(f"instances of {problem}",
                        f"run `pfn bench run priors --kind instances` or `pfn import "
                        f"thesis-results`")


def median_task(tasks: Any, n_ctx: int = EXAMPLE_N_CTX) -> Any:
    """The lower-median-cost selected non-crashed task at ``n_ctx`` (ties by run, query)."""
    sel = tasks[(tasks.n_ctx == n_ctx) & tasks.selected.astype(bool) & ~tasks.crashed.astype(bool)]
    if sel.empty:
        return None
    sel = sel.sort_values(["target_J", "run_id", "query_index"], kind="stable")
    return sel.iloc[(len(sel) - 1) // 2]


def _cache_dir(out_dir: Path) -> Path:
    d = Path(out_dir) / "_cache" / "examples"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _Recorder:
    """Problem proxy keeping every simulated (uncut) trajectory."""

    def __init__(self, problem: Any) -> None:
        self.problem = problem
        self.spec = problem.spec
        self.trajectories: list[Any] = []

    def simulate(self, theta: Any, instance: Any, rng: Any, t_max: int | None = None) -> Any:
        traj = self.problem.simulate(theta, instance, rng, t_max)
        self.trajectories.append(traj)
        return traj


def _cached_field(path: Path, build: Any) -> Any:
    """``read_field`` of a cached field directory, built by ``build(path)`` on first use."""
    from esbopfn.benchmarks.kinds.fields import read_field

    if not (path / "trials.npz").exists():
        tmp = path.with_name(path.name + ".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        build(tmp)
        if path.exists():
            import shutil

            shutil.rmtree(path)
        tmp.rename(path)
    return read_field(path)


def _simulated_oracle_run(problem: str, instance: int, seed: int, instances_seed: Any,
                          cache: Path, n_trials: int = 65) -> Any:
    from esbopfn.benchmarks.instances import load_instances
    from esbopfn.benchmarks.kinds import fields
    from esbopfn.problems import get_problem

    key = fields.field_key(problem, "oracle65", instance, seed, instances_seed, n_trials)
    name = f"field_{problem}_{instances_seed}_i{instance:04d}_s{seed:02d}_n{n_trials}"

    def build(tmp: Path) -> None:
        prob = get_problem(problem)
        inst = load_instances(problem, instances_seed)[instance]
        record, trajs = fields.simulate_field(prob, inst, n_trials, fields.run_seed(key))
        meta = {k: key[k] for k in ("problem", "field", "instance", "seed")}
        meta.update(run_seed=fields.run_seed(key), instances_seed=instances_seed)
        fields.write_field(tmp, meta, prob.spec, trajs, [t.t_obs for t in record.trials])

    return _cached_field(cache / name, build)


def example_trial(db: Any, out_dir: Path, problem: str = EXAMPLE_PROBLEM) -> ExampleTrial:
    """The example query (see the module docstring for the rule)."""
    from esbopfn.benchmarks.instances import BENCH_SEED
    from esbopfn.benchmarks.kinds.fields import field_index, read_field
    from esbopfn.benchmarks.tasks import load_tasks

    notes: list[str] = []
    native = load_tasks(problem, "tasks-oracle65")
    if native is not None:
        task = median_task(native[0])
        idx = field_index(problem, "oracle65", BENCH_SEED, db)
        if task is not None and (int(task.instance), int(task.seed)) in idx:
            key, blob = idx[(int(task.instance), int(task.seed))]
            run = read_field(blob, key)
            notes.append(f"native task {task.task_id} (median-cost non-crashed n = 64 task of "
                         f"`tasks-oracle65`), field unit {blob.name[:10]}")
            return ExampleTrial(run, int(task.query_index), "native", str(task.run_id),
                                BENCH_SEED, notes)
    seed_used = _instances_seed(db, problem)
    instance, seed, query = EXAMPLE_RUN[0], EXAMPLE_RUN[1], EXAMPLE_QUERY
    source = "fallback"
    thesis = load_tasks(problem, "tasks-thesis")
    if thesis is not None:
        task = median_task(thesis[0])
        if task is not None:
            instance, seed, query = int(task.instance), int(task.seed), int(task.query_index)
            source = "thesis-table"
            notes.append(f"query named by the imported thesis task table `tasks-thesis`: run "
                         f"{task.run_id}, trial index {query}, thesis J_T = "
                         f"{float(task.target_J):.2f}")
    run = _simulated_oracle_run(problem, instance, seed, seed_used, _cache_dir(out_dir),
                                max(query + 1, 65))
    notes.append(f"Oracle run simulated live on instance {instance} of the "
                 f"{'thesis-imported' if seed_used == 'thesis' else 'native'} instance list "
                 f"(seed {seed} of the `fields` kind), cached under `_cache/examples/`")
    q = min(query, run.n_trials - 1)
    while q > 0 and int(run.crash_step[q]) > 0:
        q -= 1
    if q != query:
        notes.append(f"trial {query} crashes in the simulated run, query = trial {q}")
    return ExampleTrial(run, q, source, f"i{instance:04d}_s{seed:02d}", seed_used, notes)


# ============================================================================ example panels
def snap_cutoffs(predictor: Any, spec: Any, fracs: tuple[float, ...]) -> dict[float, int]:
    from esbopfn.benchmarks.kinds.prediction import cutoff_steps

    steps = np.asarray(predictor.predictable_steps(spec), dtype=np.int64)
    return {f: s for f, (s, _) in cutoff_steps(steps, int(spec.T), fracs).items()}


@dataclass
class PanelResult:
    model: str
    cutoff_step: int
    n_ctx: int
    p_crash: float
    mean: float
    std: float
    observed: float


def predict_panel(predictor: Any, query: Any, context: list[Any]) -> tuple[Any, PanelResult]:
    """Prediction at a cut query, conditioned on the observed cost. ``query`` is a Trial."""
    pred = predictor.predict(query, context)
    j_obs = float(query.J_obs)
    cond = pred.conditioned_above(j_obs) if math.isfinite(j_obs) else pred
    mean, var = cond.mean_var()
    return cond, PanelResult("", int(query.t_obs), len(context), float(pred.p_crash),
                             float(mean), float(math.sqrt(max(var, 0.0))), j_obs)


def predict_at(predictor: Any, spec: Any, traj: Any, frac: float, context: list[Any]
               ) -> tuple[int, Any, PanelResult]:
    """Prediction with the trial cut at ``floor(frac T)``. A predictor that cannot predict there
    is cut at its last predictable step at or before it."""
    step = int(math.floor(frac * spec.T + 1e-9))
    try:
        pred, r = predict_panel(predictor, traj.cut(step), context)
    except (ValueError, IndexError, KeyError):
        step = snap_cutoffs(predictor, spec, (frac,)).get(frac, step)
        pred, r = predict_panel(predictor, traj.cut(step), context)
    return step, pred, r


def draw_example_panel(ax: Any, J: np.ndarray, cutoff: int, pred: Any, y_max: float,
                       theme: str, lines: list[str]) -> None:
    """Trajectory over ``1..T``, the cutoff, the predictive density of ``J_T`` as a strip right
    of ``T`` and its mean. No ticks, the corner text ``lines``."""
    fg = c.fg_color(theme)
    T = J.shape[0]
    t = np.arange(1, T + 1)
    finite = np.isfinite(J)
    ax.plot(t[finite], J[finite], color=fg, lw=1.0, zorder=3)
    ax.axvline(cutoff, color=fg, ls="--", alpha=0.6, lw=1.0, zorder=2)
    side = SIDE_FRAC * T / (1 - SIDE_FRAC)
    x0 = T + 0.5
    ys = np.linspace(0.0, y_max, 600)
    with np.errstate(all="ignore"):
        dens = np.exp(np.asarray(pred.log_density(ys), dtype=float))
    dens[~np.isfinite(dens)] = 0.0
    if dens.max() > 0:
        width = 0.92 * side * dens / dens.max()
        ax.fill_betweenx(ys, x0, x0 + width, color=DENSITY_COLOR, alpha=0.5, lw=0, zorder=2)
    mean = pred.mean_var()[0]
    if np.isfinite(mean):
        ax.plot([x0, x0 + side], [mean, mean], color=DENSITY_COLOR, lw=1.8, zorder=4,
                solid_capstyle="butt")
    ax.axvline(x0, color=fg, lw=0.3, alpha=0.08)
    ax.set_xlim(0, T + side)
    ax.set_ylim(0, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_color(fg)
    ax.text(0.04, 0.95, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=5.6, linespacing=1.6, zorder=5,
            bbox=dict(facecolor=c.bg_color(theme), alpha=0.75, edgecolor="none",
                      boxstyle="round,pad=0.1"))


def six_panel_figure(groups: tuple[str, str], theme: str) -> tuple[Any, list[Any]]:
    """Six flush panels in one row, two group titles, a legend above, ``t`` once below."""
    fig_w, left, right = c.TEXT_WIDTH_IN, 0.30, 0.06
    panel_w = (fig_w - left - right) / 6
    panel_h = 1.14
    top, bottom = 0.50, 0.22
    fig_h = top + panel_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [fig.add_axes(((left + i * panel_w) / fig_w, bottom / fig_h, panel_w / fig_w,
                          panel_h / fig_h)) for i in range(6)]
    for g, name in enumerate(groups):
        x = (left + (3 * g + 1.5) * panel_w) / fig_w
        fig.text(x, (bottom + panel_h + 0.05) / fig_h, name, ha="center", va="bottom",
                 fontsize=8)
    axes[0].set_ylabel(r"$J_t(\boldsymbol{\theta})$" if c.usetex_on() else r"$J_t(\theta)$")
    fig.text((left + 3 * panel_w) / fig_w, 0.02 / fig_h, r"$t$", ha="center", va="bottom")
    fg = c.fg_color(theme)
    handles = [Line2D([], [], color=fg, ls="--", alpha=0.6, lw=1.0),
               Line2D([], [], color=fg, lw=1.0),
               Patch(facecolor=DENSITY_COLOR, alpha=0.5, lw=0),
               Line2D([], [], color=DENSITY_COLOR, lw=1.8)]
    labels = ["cutoff", "true graph", "predictive density", "mean prediction"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.2)
    return fig, axes


def _y_max(J: np.ndarray) -> float:
    fin = J[np.isfinite(J)]
    j_t = float(fin[-1]) if fin.size else 1.0
    return max(Y_HEADROOM * j_t, float(fin.max()) * 1.05 if fin.size else 1.0)


def _panel_line(r: PanelResult, label: str) -> str:
    return (f"{label}: cutoff step {r.cutoff_step}, n = {r.n_ctx}, crash {100 * r.p_crash:.1f}%, "
            f"mean {r.mean:.1f}, std {r.std:.1f}, observed {r.observed:.1f}")


def _crash_text(p: float) -> str:
    return f"Crash: {100 * p:.0f}{c.pct()}"


def draw_lcpfn_decpfn_example(ctx: c.Context) -> c.Output:
    from esbopfn.problems.base import cached_spec

    (lc_id, lc), (dec_id, dec) = load_models([[f"lcpfn/{EXAMPLE_PROBLEM}/binned/1M"],
                                              [f"decpfn/{EXAMPLE_PROBLEM}/binned/1M"]],
                                             EXAMPLE_PROBLEM)
    ex = example_trial(ctx.db, ctx.out_dir)
    spec = cached_spec(EXAMPLE_PROBLEM)
    traj = ex.trajectory
    J = np.asarray(traj.J, dtype=float)
    y_max = _y_max(J)
    context = ex.run.context(min(EXAMPLE_N_CTX, ex.query))
    fig, axes = six_panel_figure(("LC-PFN", "DecPFN"), ctx.theme)
    results: list[str] = []
    k = 0
    for mid, model, label in ((lc_id, lc, "LC-PFN"), (dec_id, dec, "DecPFN")):
        for f in CUTOFF_FRACS:
            step, pred, r = predict_at(model, spec, traj, f, context)
            r.model = mid
            draw_example_panel(axes[k], J, step, pred, y_max, ctx.theme, [_crash_text(r.p_crash)])
            results.append(_panel_line(r, f"{label} {int(round(100 * f))}%"))
            k += 1
    sc = c.Sidecar(
        thesis_asset="simulation/prediction/lcpfn_decpfn/lcpfn_decpfn_example.pdf",
        label="fig:sim:lcpfn-decpfn-example",
        shown=["Six flush panels on one Cartpole trial: LC-PFN (left triple) and DecPFN (right "
               "triple), 1M binned, each at cutoffs 35, 65 and 85 % of T (snapped to the model's "
               "last predictable step at or before the cutoff).",
               f"Models: `{lc_id}`, `{dec_id}`. Context: the first "
               f"{min(EXAMPLE_N_CTX, ex.query)} trials of the run (context-blind models ignore "
               "it).",
               "Black: the true cumulative cost J_t over the whole trial, dashed: the cutoff, "
               "blue strip right of T: predictive density of J_T conditioned on J_T >= J_t, "
               "dark blue bar: its mean. Corner text: the predicted crash probability."],
        metric=["No ticks. x = step t, y = J_t, one shared y limit "
                f"{y_max:.1f} (1.6 J_T or the trajectory maximum)."],
        numbers=results,
        data=[f"Run {ex.run_label}, query trial {ex.query}, T = {ex.T}, true J_T = "
              f"{float(J[-1]):.2f}."] + ex.notes + [CACHE_NOTE])
    return c.Output(sc, fig=fig)


def random_context(run: Any, query: int, sizes: tuple[int, ...], seed: int = CONTEXT_SEED
                   ) -> dict[int, list[int]]:
    """Nested uniform draws (one permutation, prefixes) from the run's other trials."""
    pool = np.array([i for i in range(run.n_trials) if i != query])
    perm = np.random.default_rng(seed).permutation(pool)
    return {n: sorted(int(i) for i in perm[:n]) for n in sizes}


def full_trial(run: Any, i: int) -> Any:
    traj = run.trajectory(i)
    return traj.cut(int(traj.J.shape[0]))


def draw_ftpfn_xdecpfn_example(ctx: c.Context) -> c.Output:
    from esbopfn.problems.base import cached_spec

    (ft_id, ft), (xd_id, xd) = load_models([cross_candidates("ftpfn", EXAMPLE_PROBLEM),
                                            cross_candidates("xdecpfn", EXAMPLE_PROBLEM)],
                                           EXAMPLE_PROBLEM)
    ex = example_trial(ctx.db, ctx.out_dir)
    spec = cached_spec(EXAMPLE_PROBLEM)
    traj = ex.trajectory
    J = np.asarray(traj.J, dtype=float)
    y_max = _y_max(J)
    draws = random_context(ex.run, ex.query, CONTEXT_SIZES)
    fig, axes = six_panel_figure(("FT-PFN", "XDecPFN"), ctx.theme)
    results: list[str] = []
    k = 0
    for mid, model, label in ((ft_id, ft, "FT-PFN"), (xd_id, xd, "XDecPFN")):
        for n in CONTEXT_SIZES:
            context = [full_trial(ex.run, i) for i in draws[n]]
            step, pred, r = predict_at(model, spec, traj, CONTEXT_CUTOFF, context)
            r.model = mid
            draw_example_panel(axes[k], J, step, pred, y_max, ctx.theme,
                               [_crash_text(r.p_crash), f"Context: {n}"])
            results.append(_panel_line(r, f"{label} n={n}"))
            k += 1
    sc = c.Sidecar(
        thesis_asset="simulation/prediction/ftpfn_xdecpfn/ftpfn_xdecpfn_example.pdf",
        label="fig:sim:ftpfn-xdecpfn-example",
        shown=["Six flush panels on the Cartpole trial of `lcpfn_decpfn_example`: FT-PFN (left "
               "triple) and XDecPFN (right triple), all cut at 35 % of T (snapped per model), "
               "columns = context size n = 0, 5, 10.",
               f"Models: `{ft_id}`, `{xd_id}`.",
               "Context: trials of the same run drawn uniformly without the query, nested (one "
               f"permutation with seed {CONTEXT_SEED}, sizes are prefixes), each given in full "
               "length. Drawn trials: "
               + "; ".join(f"n={n}: {draws[n]}" for n in CONTEXT_SIZES) + ".",
               "Black: true J_t, dashed: cutoff, blue strip: predictive density of J_T "
               "conditioned on J_T >= J_t, dark blue bar: its mean. Corner text: crash "
               "probability and context size."],
        metric=[f"No ticks. x = step t, y = J_t, shared y limit {y_max:.1f}."],
        numbers=results,
        data=[f"Run {ex.run_label}, query trial {ex.query}, T = {ex.T}, true J_T = "
              f"{float(J[-1]):.2f}."] + ex.notes + [CACHE_NOTE])
    return c.Output(sc, fig=fig)


# ============================================================================ landscape
def landscape_colors(cost: np.ndarray, p_crash: np.ndarray) -> np.ndarray:
    """Additive RGB blend: blue = cost rank (1 = cheapest), red = crash probability."""
    from esbopfn.benchmarks.metrics import cost_rank

    rank = np.nan_to_num(cost_rank(np.asarray(cost, dtype=float)), nan=0.0)
    crash = np.clip(np.nan_to_num(np.asarray(p_crash, dtype=float)), 0.0, 1.0)
    rgb = BLUE_RGB * rank[..., None] + RED_RGB * crash[..., None]
    return np.clip(rgb, 0.0, 1.0)


def _slice_key(instance: int, instances_seed: Any) -> dict[str, Any]:
    return {"unit": "slice", "problem": LANDSCAPE_PROBLEM, "instance": int(instance),
            "slice_seed": LANDSCAPE_SLICE_SEED, "grid": LANDSCAPE_GRID,
            "repeats": LANDSCAPE_REPEATS, "instances_seed": instances_seed}


def true_slice(db: Any, instance: int, instances_seed: Any, cache: Path,
               prov: c.Provenance) -> dict[str, np.ndarray]:
    """The ``landscapes`` slice unit of the plant (DB first, else computed into the cache)."""
    from esbopfn.benchmarks.kinds import landscapes

    key = _slice_key(instance, instances_seed)
    if db.has("landscapes", key):
        unit = db.get("landscapes", key)
        prov.add_units("landscapes", [unit])
        path = Path(unit.blob_dir) / "landscape.npz"
    else:
        d = cache / f"slice_{LANDSCAPE_PROBLEM}_{instances_seed}_i{instance:04d}_s" \
                    f"{LANDSCAPE_SLICE_SEED}_g{LANDSCAPE_GRID}_r{LANDSCAPE_REPEATS}"
        path = d / "landscape.npz"
        if not path.exists():
            d.mkdir(parents=True, exist_ok=True)
            landscapes.run_slice(key, d)
        prov.add_file(path.relative_to(cache.parent.parent))
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def pick_plant(db: Any, cfg: dict[str, Any], instances_seed: Any, cache: Path,
               prov: c.Provenance) -> tuple[int, dict[str, np.ndarray], str]:
    sec = ((cfg.get("figures") or {}).get("landscape_furuta") or {})
    if "instance" in sec:
        i = int(sec["instance"])
        return i, true_slice(db, i, instances_seed, cache, prov), "set in the group config"
    lo, hi = LANDSCAPE_CRASH_BAND
    first = None
    for i in range(LANDSCAPE_SCAN):
        sl = true_slice(db, i, instances_seed, cache, c.Provenance())
        share = float(np.mean(sl["p_crash"]))
        first = first or (i, sl)
        if lo <= share <= hi:
            true_slice(db, i, instances_seed, cache, prov)
            return i, sl, (f"first of instances 0-{LANDSCAPE_SCAN - 1} with a true crash share "
                           f"in [{lo}, {hi}] ({share:.2f})")
    assert first is not None
    true_slice(db, first[0], instances_seed, cache, prov)
    return first[0], first[1], "no instance in the crash band, instance 0"


def loop_run(strategy_id: str, model_id: str, predictor: Any, instance: int,
             instances_seed: Any, n: int, cache: Path) -> Any:
    """A pure-loop run of ``n`` trials on the plant, as a cached field (full trajectories)."""
    from esbopfn.benchmarks.instances import load_instances
    from esbopfn.benchmarks.kinds import fields
    from esbopfn.problems import get_problem
    from esbopfn.strategies import make_strategy, run_optimization, strategy_rng

    ck = _checkpoint(model_id)
    name = (f"loop_{strategy_id}_{LANDSCAPE_PROBLEM}_{instances_seed}_i{instance:04d}"
            f"_s{LANDSCAPE_LOOP_SEED}_n{n}_{ck}")

    def build(tmp: Path) -> None:
        prob = get_problem(LANDSCAPE_PROBLEM)
        inst = load_instances(LANDSCAPE_PROBLEM, instances_seed)[instance]
        rec = _Recorder(prob)
        strat = make_strategy(strategy_id, prob.spec, strategy_rng(LANDSCAPE_LOOP_SEED),
                              model=model_id, predictor=predictor)
        record = run_optimization(rec, inst, strat, LANDSCAPE_LOOP_SEED, n_max=n,
                                  t_budget=n * prob.spec.T)
        meta = {"problem": LANDSCAPE_PROBLEM, "strategy": strategy_id, "model": model_id,
                "instance": instance, "seed": LANDSCAPE_LOOP_SEED,
                "instances_seed": instances_seed}
        fields.write_field(tmp, meta, prob.spec, rec.trajectories,
                           [t.t_obs for t in record.trials])

    return _cached_field(cache / name, build)


def model_slice(predictor: Any, model_id: str, run: Any, n: int, thetas: np.ndarray,
                tag: str, cache: Path) -> dict[str, np.ndarray]:
    """Predicted mean total cost and crash probability of the unrun slice controllers given the
    first ``n`` trials of ``run`` (cached)."""
    h = hashlib.sha256(np.ascontiguousarray(thetas).tobytes()).hexdigest()[:10]
    path = cache / f"model_{model_id.replace('/', '-')}_{_checkpoint(model_id)}_{tag}_{h}_n{n}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    shape = thetas.shape[:-1]
    preds = predictor.predict_unrun(thetas.reshape(-1, thetas.shape[-1]), run.context(n))
    mean = np.array([p.mean_var()[0] for p in preds], dtype=float).reshape(shape)
    pc = np.array([p.p_crash for p in preds], dtype=float).reshape(shape)
    np.savez_compressed(path, cost_mean=mean, p_crash=pc)
    return {"cost_mean": mean, "p_crash": pc}


def _image(ax: Any, rgb: np.ndarray, theme: str) -> None:
    ax.imshow(rgb, origin="lower", aspect="auto", interpolation="nearest", extent=(0, 1, 0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color(c.fg_color(theme))


def _color_key(ax: Any, theme: str) -> None:
    n = 48
    q = np.linspace(0.0, 1.0, n)
    cost = np.tile((1.0 - q)[:, None], (1, n))   # cheapest at the top
    crash = np.tile(q[None, :], (n, 1))
    ax.imshow(landscape_colors(cost, crash), origin="lower", extent=(0, 1, 0, 1), aspect="auto",
              interpolation="bilinear")
    ax.set_xticks([0, 1], ["0", "1"], fontsize=5.5)
    ax.set_yticks([0, 1], ["worst", "best"], fontsize=5.5)
    ax.set_xlabel("crash prob.", fontsize=5.8, labelpad=1)
    ax.set_ylabel("objective", fontsize=5.8, labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def _panel_stats(name: str, cost: np.ndarray, pc: np.ndarray) -> str:
    v = cost[np.isfinite(cost)]
    rng = f"J in [{v.min():.0f}, {v.max():.0f}]" if v.size else "no finite cost"
    return f"{name}: {rng}, mean crash probability {100 * float(np.mean(pc)):.0f}%"


def draw_landscape_furuta(ctx: c.Context) -> c.Output:
    prov = c.Provenance()
    (xd_id, xd), (ft_id, ft) = load_models([cross_candidates("xdecpfn", LANDSCAPE_PROBLEM),
                                            cross_candidates("ftpfn", LANDSCAPE_PROBLEM)],
                                           LANDSCAPE_PROBLEM)
    seed_used = _instances_seed(ctx.db, LANDSCAPE_PROBLEM)
    cache = _cache_dir(ctx.out_dir)
    instance, real, why = pick_plant(ctx.db, ctx.cfg, seed_used, cache, prov)
    thetas = np.asarray(real["thetas"], dtype=float)
    n_max = max(LANDSCAPE_CONTEXTS)
    tag = f"{seed_used}_i{instance:04d}_s{LANDSCAPE_SLICE_SEED}"
    rows = []
    for mid, model, sid, label in ((xd_id, xd, "xdecpfn-pi", "XDecPFN"),
                                   (ft_id, ft, "ftpfn-ei", "FT-PFN")):
        run = loop_run(sid, mid, model, instance, seed_used, n_max, cache)
        rows.append((label, mid, sid, run,
                     [model_slice(model, mid, run, n, thetas, f"{tag}_{sid}", cache)
                      for n in LANDSCAPE_CONTEXTS]))
    real_cost = real.get("cost_mean_nocrash", real["cost_mean"])

    ncols = len(LANDSCAPE_CONTEXTS) + 1
    W, left, top, bottom = c.TEXT_WIDTH_IN, 0.26, 0.22, 0.04
    cell = (W - left - 0.02) / ncols
    H = top + 2 * cell + bottom
    fig = plt.figure(figsize=(W, H))

    def rect(r: int, col: int) -> tuple[float, float, float, float]:
        return ((left + col * cell) / W, (bottom + (1 - r) * cell) / H, cell / W, cell / H)

    numbers: list[str] = []
    for r, (label, mid, sid, run, panels) in enumerate(rows):
        for col, (n, pan) in enumerate(zip(LANDSCAPE_CONTEXTS, panels)):
            ax = fig.add_axes(rect(r, col))
            _image(ax, landscape_colors(pan["cost_mean"], pan["p_crash"]), ctx.theme)
            if r == 0:
                ax.set_title(f"$n={n}$", fontsize=6.5, pad=4)
            numbers.append(_panel_stats(f"{label} n={n}", pan["cost_mean"], pan["p_crash"]))
        fig.text(left * 0.4 / W, (bottom + (1.5 - r) * cell) / H, label, rotation=90,
                 ha="center", va="center", fontsize=6.8)
    ax = fig.add_axes(rect(0, ncols - 1))
    _image(ax, landscape_colors(real_cost, real["p_crash"]), ctx.theme)
    ax.set_title("real", fontsize=6.5, pad=4)
    numbers.append(_panel_stats("real", real_cost, real["p_crash"]))
    kx, ky, kw, kh = rect(1, ncols - 1)
    pad_l, pad_b, pad_t = 0.30 / W, 0.22 / H, 0.12 / H
    key = fig.add_axes((kx + pad_l, ky + pad_b, kw - pad_l, kh - pad_b - pad_t))
    key.set_box_aspect(1)
    key.set_anchor("NE")
    _color_key(key, ctx.theme)

    data = [f"Plant: instance {instance} of the "
            f"{'thesis-imported' if seed_used == 'thesis' else 'native'} Furuta instance list "
            f"({why}). Slice: `landscapes` unit slice seed {LANDSCAPE_SLICE_SEED}, "
            f"{LANDSCAPE_GRID} x {LANDSCAPE_GRID} grid, {LANDSCAPE_REPEATS} repeats per point, "
            f"extent {float(real['extent']):.4g}."]
    for label, mid, sid, run, _ in rows:
        data.append(f"{label} context: prefixes of one `{sid}` run (seed {LANDSCAPE_LOOP_SEED}, "
                    f"{run.n_trials} trials, {int((run.crash_step > 0).sum())} crashed) on the "
                    f"same plant with `{mid}` (checkpoint {_checkpoint(mid)}), cached under "
                    "`_cache/examples/`.")
    sc = c.Sidecar(
        thesis_asset="simulation/prediction/ftpfn_xdecpfn/landscape_furuta_BEST_NO_DELETE.pdf",
        label="fig:sim:landscape-furuta",
        shown=["2 x 5 sheet on one 2-D oblique slice of one Furuta plant: XDecPFN (top row) and "
               "FT-PFN (bottom row) at context sizes n = "
               + ", ".join(str(n) for n in LANDSCAPE_CONTEXTS)
               + ", the real landscape top right, the color key bottom right.",
               "Each pixel is one slice controller. Blue: per-panel rank of the mean total cost "
               "(brighter = cheaper), red: crash probability, blended additively (black = "
               "expensive and safe)."],
        metric=["Real panel: mean J_T over the non-crashed repeats and the crash share of the "
                "six repeats. Model panels: the predictive mean of J_T without crash (value "
                "part of the bar distribution) and the predicted crash probability for the "
                "unrun controller (`predict_unrun`)."],
        numbers=numbers, data=data + [CACHE_NOTE], provenance=prov)
    return c.Output(sc, fig=fig)


LCPFN_DECPFN_EXAMPLE = c.FigureSpec("lcpfn_decpfn_example", draw_lcpfn_decpfn_example)
FTPFN_XDECPFN_EXAMPLE = c.FigureSpec("ftpfn_xdecpfn_example", draw_ftpfn_xdecpfn_example)
LANDSCAPE_FURUTA = c.FigureSpec("landscape_furuta", draw_landscape_furuta)

__all__ = ["LCPFN_DECPFN_EXAMPLE", "FTPFN_XDECPFN_EXAMPLE", "LANDSCAPE_FURUTA", "example_trial",
           "median_task", "landscape_colors", "draw_example_panel", "six_panel_figure"]

