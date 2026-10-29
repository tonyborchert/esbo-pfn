"""Plot group ``optimization``: the closed-loop optimization figures of the thesis.

Figures (names = thesis asset basenames):

- ``optplay_cross`` (fig:sim:optplay-cross): log10 of the problem mean of the per-problem median
  scaled regret against % of the trial budget ``15 min(d, 4)`` and against % of the simulated time
  budget, nine methods.
- ``optplay_cross__sheet`` (fig:app:optplay-cross-sheet): the time curves per problem plus the
  pooled panel, shared y.
- ``optplay_violin__cross_full`` (fig:sim:optplay-violin-cross-full) and ``optplay_violin__sheet``
  (fig:app:optplay-violin-sheet): distributions of the log10 scaled regret per run at 50 % and
  100 % of the simulated time budget, display floor -4.
- ``optplay_crashes`` (fig:sim:optplay-crashes): mean number of crashed trials so far against % of
  the simulated time budget on PT2, Cartpole and Furuta.
- ``optplay_timing`` (fig:sim:optplay-timing): optimizer decision time per run split into GP
  acquisition, Transformer acquisition, Segment-GP forecast and Transformer forecast.
- ``optplay_decpfn_thresholds__violin`` and ``optplay_qx_variants__violin`` (appendix): the variant
  cells of the DecPFN-based loops and of the pure model loops.

Data: native ``optimization`` / ``optimization_timing`` units first. A problem without native runs
falls back to the thesis import (``thesis_series`` ``optplay_finals``, ``optplay_crashes``), which
holds the final incumbents and the crash curves but no incumbent traces, so the cross curves, the
50 % violins and the timing split need native units.

Scaled regret (thesis setup, optimization metrics): ``(J - J_best) / (J_ESRS - J_best)`` per
instance, ``J_best`` the best final cost of the declared reference strategies, ``J_ESRS`` the final
incumbent of ESRS on the same instance.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from esbopfn.benchmarks import metrics as M
from esbopfn.plots import PlotGroup, style
from esbopfn.plots import common as C

GROUP_NAME = "optimization"
ASSET_DIR = "simulation/optimization"

FLOOR = M.REGRET_FLOOR
FLOOR_LOG10 = math.log10(FLOOR)
FRACTIONS = (0.5, 1.0)
PAIR_ALPHA = (0.22, 0.45)
SINGLE_ALPHA = 0.35
HALF_WIDTH = 0.40
PAIR_HALF_WIDTH = 0.20
PAIR_OFFSET = 0.21
BIN_WIDTH_MIN, BIN_WIDTH_MAX = 0.15, 0.40
VIOLIN_SCALE_CAP = 0.50
JITTER_SEED = 20260920
TIMING_SEGMENTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("GP acquisition", ("gp_acq",), "#0072B2"),
    ("Transformer acquisition", ("model_acq",), "#56B4E9"),
    ("Segment-GP forecast", ("segment_gp",), "#D55E00"),
    ("Transformer forecast", ("model_forecast", "tabpfn_forecast"), "#E69F00"),
)
SHORT_PROBLEM = {"boiler": "boiler", "threetank_pi": "three-tank", "pt2_deadtime": "PT2",
                 "cartpole": "cartpole", "furuta": "furuta"}
_IGNORED_PARAMS = frozenset({"model", "checkpoint", "predictor"})


# ------------------------------------------------------------------------------- cells
@dataclass(frozen=True)
class Cell:
    """One drawn optimizer configuration: strategy id, parameter overrides, its label and the
    name of the matching thesis series."""

    label: str
    id: str
    params: tuple[tuple[str, Any], ...] = ()
    thesis: str = ""
    group: str = ""
    tick: str = ""

    @property
    def overrides(self) -> dict[str, Any]:
        return dict(self.params)


def _cell(entry: dict[str, Any] | str, **extra: Any) -> Cell:
    from esbopfn.benchmarks.kinds.optimization import strategy_label

    entry = {"id": entry} if isinstance(entry, str) else dict(entry)
    entry.update({k: v for k, v in extra.items() if v is not None})
    params = dict(entry.get("params") or {})
    label = entry.get("label")
    if not label and entry.get("group"):
        label = f"{str(entry['group']).replace('/', '-')} tau={entry.get('tick', '')}"
    label = str(label or strategy_label(entry["id"], params))
    return Cell(label=label, id=str(entry["id"]), params=tuple(sorted(params.items())),
                thesis=str(entry.get("thesis") or strategy_label(entry["id"], params)),
                group=str(entry.get("group", "")), tick=str(entry.get("tick", "")))


def _sec(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    sec = cfg.get(name)
    return dict(sec) if isinstance(sec, dict) else {}


def _figcfg(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    return _sec(_sec(cfg, "figures"), name)


def all_cells(cfg: dict[str, Any]) -> list[Cell]:
    """Every strategy entry of the ``optimization`` section."""
    return [_cell(e) for e in _sec(cfg, "optimization").get("strategies", [])]


def main_cells(cfg: dict[str, Any]) -> list[Cell]:
    """The nine methods of the cross-method figures (``main_strategies`` labels)."""
    cells = {c.label: c for c in all_cells(cfg)}
    labels = cfg.get("main_strategies") or [c.label for c in all_cells(cfg)]
    return [cells[lab] for lab in labels if lab in cells]


def variant_cells(cfg: dict[str, Any], figure: str) -> list[Cell]:
    return [_cell(e) for e in _figcfg(cfg, figure).get("cells", [])]


def n_instances(cfg: dict[str, Any]) -> int:
    spec = _sec(cfg, "optimization").get("instances", 20)
    return int(spec) if isinstance(spec, int) else len(list(spec))


def problems(cfg: dict[str, Any]) -> list[str]:
    return list(cfg.get("problems") or C.PROBLEMS)


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


@functools.lru_cache(maxsize=256)
def _defaults(strategy: str, problem: str) -> tuple[tuple[str, Any], ...]:
    from esbopfn.problems.base import cached_spec
    from esbopfn.strategies import default_params

    return tuple(sorted(default_params(strategy, cached_spec(problem)).items()))


def matches(key: dict[str, Any], cell: Cell, problem: str) -> bool:
    """True if an ``optimization`` unit key belongs to ``cell`` (native: strategy id and every
    non-model parameter equal to the registry defaults updated with the cell's overrides, thesis
    runs: their ``label``)."""
    if key.get("strategy") != cell.id:
        return False
    if key.get("provenance") == "thesis":
        return key.get("label", cell.id) == cell.thesis
    expected = dict(_defaults(cell.id, problem))
    expected.update(cell.overrides)
    params = dict(key.get("params") or {})
    for k in set(expected) | set(params):
        if k in _IGNORED_PARAMS:
            continue
        if k not in expected:
            if params[k] is not None:
                return False
            continue
        if not _eq(params.get(k), expected[k]):
            return False
    return True


# ------------------------------------------------------------------------------- data
@dataclass
class ProblemRuns:
    """What one problem contributes: per cell and instance the final incumbent, the incumbent at
    the budget fractions, and (native only) the incumbent traces and run records."""

    problem: str
    source: str                                   # "native" | "thesis"
    t_budget: float
    trials_axis: int
    finals: dict[str, dict[int, float]] = field(default_factory=dict)
    at: dict[tuple[str, float], dict[int, float]] = field(default_factory=dict)
    records: dict[str, dict[int, Any]] = field(default_factory=dict)
    reference: dict[str, dict[int, float]] = field(default_factory=dict)   # J_best set
    crash_curves: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def j_best(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for per in self.reference.values():
            for i, v in per.items():
                if np.isfinite(v):
                    out[i] = min(out.get(i, math.inf), v)
        return out

    def j_esrs(self, esrs: str) -> dict[int, float]:
        return dict(self.finals.get(esrs, {}))


def _budget(problem: str) -> tuple[int, int]:
    from esbopfn.problems.base import cached_spec

    spec = cached_spec(problem)
    return int(spec.t_budget()), 15 * int(spec.budget_d())


def _load_native(db: Any, cfg: dict[str, Any], problem: str, cells: Sequence[Cell],
                 prov: C.Provenance, kind: str = "optimization",
                 budget: str = "capped") -> ProblemRuns | None:
    from esbopfn.benchmarks.kinds.optimization import read_run

    units = [u for u in db.query(kind, problem=problem)
             if u.key.get("budget", budget) == budget]
    if not units:
        return None
    t_budget, axis = _budget(problem)
    n_inst = n_instances(cfg) if kind == "optimization" else 10 ** 9
    pr = ProblemRuns(problem, "native", t_budget, axis)
    ref_cells = all_cells(cfg)
    ref_spec = cfg.get("reference_strategies", "all")
    if isinstance(ref_spec, list):
        ref_cells = [c for c in ref_cells if c.label in ref_spec or c.id in ref_spec]
    used = []
    for cell in list(dict.fromkeys(list(cells) + ref_cells)):
        for u in units:
            if not matches(u.key, cell, problem) or int(u.key["instance"]) >= n_inst:
                continue
            rec = read_run(Path(u.blob_dir))
            inst = int(u.key["instance"])
            if rec.t_budget:
                pr.t_budget = float(rec.t_budget)
            pr.records.setdefault(cell.label, {})[inst] = rec
            final = M.final_incumbent(rec)
            pr.finals.setdefault(cell.label, {})[inst] = final
            for frac in FRACTIONS:
                pr.at.setdefault((cell.label, frac), {})[inst] = float(
                    M.incumbent_vs_time(rec, np.array([frac * float(rec.t_budget)]))[0])
            if cell in ref_cells:
                pr.reference.setdefault(cell.label, {})[inst] = final
            used.append(u)
    prov.add_units(kind, list({u.key_hash: u for u in used}.values()))
    return pr


def _load_thesis(db: Any, cfg: dict[str, Any], problem: str, cells: Sequence[Cell],
                 prov: C.Provenance) -> ProblemRuns | None:
    blobs = C.thesis_series(db, "optplay_finals", problem=problem, prov=prov)
    if not blobs:
        return None
    t_budget, axis = _budget(problem)
    pr = ProblemRuns(problem, "thesis", t_budget, axis)
    n_inst = n_instances(cfg)
    by_series = {b.series: b for b in blobs}
    for name, b in by_series.items():
        f = b.frame[b.frame.instance < n_inst]
        pr.reference[name] = {int(i): float(v) for i, v in zip(f.instance, f.incumbent_capped)}
    for cell in cells:
        b = by_series.get(cell.thesis)
        if b is None:
            continue
        f = b.frame[b.frame.instance < n_inst]
        pr.finals[cell.label] = {int(i): float(v) for i, v in zip(f.instance, f.incumbent_capped)}
        pr.at[(cell.label, 1.0)] = {int(i): float(v)
                                    for i, v in zip(f.instance, f.incumbent_time)}
    if cfg.get("thesis_reference", "optplay_reference") == "optplay_reference":
        ref = C.thesis_series(db, "optplay_reference", problem=problem, prov=prov)
        if ref:
            f = ref[0].frame[ref[0].frame.instance < n_inst]
            pr.reference = {"optplay_reference": {int(i): float(v)
                                                  for i, v in zip(f.instance, f.J_best)}}
    crashes = {b.series: b for b in C.thesis_series(db, "optplay_crashes", problem=problem,
                                                    prov=prov)}
    for cell in cells:
        b = crashes.get(cell.thesis)
        if b is None:
            continue
        tb = float(b.meta.get("t_budget", t_budget))
        y = np.array([np.nan if v is None else v for v in b.frame.mean_crash_count], float)
        pr.crash_curves[cell.label] = (100.0 * b.frame.time.to_numpy(float) / tb, y)
    return pr


def load_runs(ctx: C.Context, cells: Sequence[Cell], prov: C.Provenance
              ) -> dict[str, ProblemRuns]:
    """Per problem the native runs if any exist, else the thesis import. ESRS is always loaded
    (it defines ``J_ESRS``)."""
    cells = list(cells) + [c for c in all_cells(ctx.cfg) if c.id == "esrs" and c not in cells]
    out: dict[str, ProblemRuns] = {}
    for p in problems(ctx.cfg):
        pr = _load_native(ctx.db, ctx.cfg, p, cells, prov)
        if pr is None:
            pr = _load_thesis(ctx.db, ctx.cfg, p, cells, prov)
        if pr is not None:
            out[p] = pr
    return out


def esrs_label(cells: Sequence[Cell], cfg: dict[str, Any]) -> str:
    for c in list(cells) + all_cells(cfg):
        if c.id == "esrs":
            return c.label
    return "ESRS"


def scaled(pr: ProblemRuns, values: dict[int, float], esrs: str) -> dict[int, float]:
    """Scaled regret per instance (undefined instances left out)."""
    best, ref = pr.j_best(), pr.j_esrs(esrs)
    out = {}
    for i, v in values.items():
        if i not in best or i not in ref:
            continue
        r = float(M.scaled_regret(np.array([v]), best[i], ref[i])[0])
        if not np.isnan(r):
            out[i] = r
    return out


def log_values(vals: dict[int, float]) -> np.ndarray:
    """log10 with the display floor, unsolved runs (inf) at +inf."""
    arr = np.array([vals[i] for i in sorted(vals)], dtype=float)
    return M.log10_display(arr, FLOOR)


# ------------------------------------------------------------------------------- violins
def violin_bins(populations: Sequence[np.ndarray], ylim: tuple[float, float]) -> np.ndarray:
    """One bin grid per panel: Freedman-Diaconis at the size of one violin, clamped."""
    vals = [v[np.isfinite(v)] for v in populations if len(v)]
    pooled = np.concatenate(vals) if vals else np.zeros(1)
    q1, q3 = np.percentile(pooled, [25, 75])
    n = max(int(np.median([len(v) for v in vals])) if vals else 2, 2)
    fd = 2.0 * (q3 - q1) / n ** (1.0 / 3.0) if q3 > q1 else BIN_WIDTH_MAX
    width = float(np.clip(fd, BIN_WIDTH_MIN, BIN_WIDTH_MAX))
    lo = math.floor(ylim[0] / width) * width
    hi = math.ceil(ylim[1] / width) * width + 0.5 * width
    return np.arange(lo, hi, width)


def violin_scale(populations: Sequence[np.ndarray], edges: np.ndarray) -> float:
    best = 0.0
    for v in populations:
        v = v[np.isfinite(v)]
        if len(v) < 2 or float(np.ptp(v)) < 1e-9:
            continue
        counts, _ = np.histogram(v, bins=edges)
        best = max(best, counts.max() / len(v))
    return min(best, VIOLIN_SCALE_CAP) if best > 0 else VIOLIN_SCALE_CAP


def violin_body(ax: Any, values: np.ndarray, x: float, color: str, *, edges: np.ndarray,
                scale: float, half_width: float, alpha: float) -> None:
    """A mirrored histogram drawn smooth (PCHIP through the bin shares, zero at the support)."""
    from scipy.interpolate import PchipInterpolator

    values = values[np.isfinite(values)]
    if not len(values) or float(np.ptp(values)) < 1e-9:
        return                                  # one value (ESRS at 100 %): the median line only
    counts, _ = np.histogram(values, bins=edges)
    occupied = np.flatnonzero(counts)
    if not occupied.size:
        return
    first, last = int(occupied[0]), int(occupied[-1])
    half = np.clip(counts / len(values) / scale, 0.0, 1.0) * half_width
    centers = 0.5 * (edges[:-1] + edges[1:])
    ky = np.concatenate(([edges[first]], centers[first:last + 1], [edges[last + 1]]))
    kw = np.concatenate(([0.0], half[first:last + 1], [0.0]))
    ys = np.linspace(ky[0], ky[-1], 240)
    ws = np.clip(PchipInterpolator(ky, kw)(ys), 0.0, None)
    ax.fill_betweenx(ys, x - ws, x + ws, facecolor=color, edgecolor="none", alpha=alpha,
                     zorder=3)


def violin_points(ax: Any, values: np.ndarray, x: float, color: str, *, half_width: float,
                  rng: np.random.Generator) -> None:
    values = values[np.isfinite(values)]
    if not len(values):
        return
    offs = rng.uniform(-0.72, 0.72, len(values)) * half_width
    ax.scatter(x + offs, values, s=2.2, c=color, alpha=0.75, linewidths=0.0, zorder=5)


def violin_median(ax: Any, values: np.ndarray, x: float, color: str, *, half_width: float
                  ) -> float:
    if not len(values):
        return float("nan")
    med = float(np.median(values))
    if np.isfinite(med):
        ax.plot([x - half_width, x + half_width], [med, med], color=color, lw=1.5, zorder=6,
                solid_capstyle="butt")
    return med


def draw_violins(ax: Any, pops: dict[float, list[np.ndarray]], colors: Sequence[str], *,
                 edges: np.ndarray, scale: float, points: bool, ylim: tuple[float, float]
                 ) -> dict[float, list[float]]:
    """One slot per entry of ``colors``. ``pops[frac][slot]`` are the log10 values. With both
    fractions the slot holds the pair (50 % left, 100 % right), with one it holds one violin.
    Returns the medians per fraction."""
    fracs = [f for f in FRACTIONS if f in pops]
    paired = len(fracs) == 2
    medians: dict[float, list[float]] = {}
    for k, frac in enumerate(fracs):
        rng = np.random.default_rng(JITTER_SEED + k)
        side = (-PAIR_OFFSET, PAIR_OFFSET)[k] if paired else 0.0
        hw = PAIR_HALF_WIDTH if paired else HALF_WIDTH
        alpha = PAIR_ALPHA[k] if paired else (PAIR_ALPHA[1] if frac == 1.0 else SINGLE_ALPHA)
        medians[frac] = []
        for slot, color in enumerate(colors):
            vals = pops[frac][slot]
            shown = np.clip(vals, ylim[0], ylim[1])
            violin_body(ax, shown, slot + side, color, edges=edges, scale=scale, half_width=hw,
                        alpha=alpha)
            if points:
                violin_points(ax, shown, slot + side, color, half_width=hw, rng=rng)
            medians[frac].append(violin_median(ax, vals, slot + side, color, half_width=hw))
    ax.set_xlim(-0.6, len(colors) - 0.4)
    ax.set_ylim(*ylim)
    return medians


def budget_legend(fig: Any, fracs: Sequence[float], theme: str, y_pt: float = 4.0) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fg = C.fg_color(theme)
    alphas = PAIR_ALPHA if len(fracs) == 2 else (PAIR_ALPHA[1],)
    handles: list[Any] = [Patch(facecolor=fg, edgecolor="none", alpha=a) for a in alphas]
    labels = [f"{100 * f:.0f} {C.pct()} budget" for f in fracs]
    handles.append(Line2D([], [], color=fg, lw=1.5, solid_capstyle="butt"))
    labels.append("median")
    C.legend_above(fig, handles, labels, ncol=len(handles), y_pt=y_pt)


def violin_ylabel() -> str:
    return C.better(r"$\log_{10}$ scaled regret", higher=False)


def group_headers(ax: Any, levels: list[list[tuple[str, int, int]]], theme: str,
                  fontsize: float = 7.0) -> None:
    """Nested category titles above a categorical axes, outermost level first. Every level but
    the lowest carries a rule under its labels."""
    from matplotlib.transforms import ScaledTranslation

    fig = ax.get_figure()
    fg = C.fg_color(theme)

    def at(dy: float) -> Any:
        return ax.get_xaxis_transform() + ScaledTranslation(0.0, dy / 72.0, fig.dpi_scale_trans)

    for row, level in enumerate(reversed(levels)):
        dy = 2.5 + row * 10.5
        for label, i0, i1 in level:
            if row > 0:
                ax.plot([i0 - 0.36, i1 + 0.36], [1.0, 1.0], transform=at(dy), color=fg, lw=0.6,
                        clip_on=False, solid_capstyle="butt", zorder=10)
            ax.text(0.5 * (i0 + i1), 1.0, label, transform=at(dy + 1.5), ha="center",
                    va="bottom", fontsize=fontsize, color=fg, clip_on=False, zorder=10)


def cell_color(cell: Cell, theme: str) -> str:
    return style.method_color(style.display_name(cell.id), theme)


# ------------------------------------------------------------------------------- populations
@dataclass
class Populations:
    """log10 scaled regret per (panel, fraction, cell), panel ``pooled`` = all problems."""

    values: dict[tuple[str, float, str], np.ndarray]
    raw: dict[tuple[str, float, str], dict[int, float]]
    fracs: list[float]
    sources: dict[str, str]


def populations(runs: dict[str, ProblemRuns], cells: Sequence[Cell], esrs: str
                ) -> Populations:
    values: dict[tuple[str, float, str], np.ndarray] = {}
    raw: dict[tuple[str, float, str], dict[int, float]] = {}
    fracs = [f for f in FRACTIONS
             if any((c.label, f) in pr.at for pr in runs.values() for c in cells)]
    for frac in fracs:
        for c in cells:
            pooled = []
            for p, pr in runs.items():
                vals = scaled(pr, pr.at.get((c.label, frac), {}), esrs)
                raw[(p, frac, c.label)] = vals
                values[(p, frac, c.label)] = log_values(vals)
                pooled.append(values[(p, frac, c.label)])
            values[("pooled", frac, c.label)] = (np.concatenate(pooled) if pooled
                                                 else np.zeros(0))
    return Populations(values, raw, fracs, {p: pr.source for p, pr in runs.items()})


def _ylim(pops: Populations) -> tuple[float, float]:
    vals = [v[np.isfinite(v)] for v in pops.values.values() if len(v)]
    pooled = np.concatenate(vals) if vals else np.array([FLOOR_LOG10, 0.0])
    lo, hi = float(min(pooled.min(), FLOOR_LOG10)), float(pooled.max())
    span = max(hi - lo, 1e-6)
    return lo - 0.04 * span, hi + 0.04 * span


def _source_notes(runs: dict[str, ProblemRuns], prov: C.Provenance) -> list[str]:
    thesis = [p for p, pr in runs.items() if pr.source == "thesis"]
    native = [p for p, pr in runs.items() if pr.source == "native"]
    out = []
    if native:
        out.append("native `optimization` units on " + ", ".join(native))
    if thesis:
        jb = ("`J_best` = `optplay_reference` of the import (best successful final cost over "
              "every thesis cell on disk, the thesis definition)"
              if any("optplay_reference" in runs[p].reference for p in thesis)
              else "`J_best` recomputed over every imported cell (declared reference set)")
        out.append("thesis import (`optplay_finals`, final incumbents at the time budget) on "
                   + ", ".join(thesis) + ". " + jb)
    return out


def _require(runs: dict[str, ProblemRuns], cfg: dict[str, Any]) -> None:
    missing = [p for p in problems(cfg) if p not in runs]
    if len(missing) == len(problems(cfg)):
        raise C.MissingData(C.missing_units_message("optimization", missing, GROUP_NAME))


def _check_cells(pops: Populations, cells: Sequence[Cell], what: str) -> None:
    empty = [c.label for c in cells
             if all(not len(pops.values.get(("pooled", f, c.label), [])) for f in pops.fracs)]
    if len(empty) == len(cells):
        raise C.MissingData(C.missing_units_message("optimization", empty, GROUP_NAME),
                            f"no runs of the {what} cells")


def _violin_sidecar(asset: str, label: str, pops: Populations, cells: Sequence[Cell],
                    panels: Sequence[str], runs: dict[str, ProblemRuns],
                    prov: C.Provenance) -> C.Sidecar:
    numbers = []
    for panel in panels:
        parts = []
        for c in cells:
            seg = []
            for f in pops.fracs:
                v = pops.values.get((panel, f, c.label), np.zeros(0))
                if not len(v):
                    continue
                med = float(np.median(v))
                floor = int(np.sum(v <= FLOOR_LOG10 + 1e-12))
                seg.append(f"{100 * f:.0f}% {C.fmt_num(med)} (n={len(v)}, {floor} at floor)")
            if seg:
                parts.append(f"{c.label} " + ", ".join(seg))
        if parts:
            numbers.append(f"{C.problem_label(panel)}: median log10 scaled regret: "
                           + "; ".join(parts))
    if 0.5 not in pops.fracs:
        numbers.append("50 % budget not shown: the per-run incumbent traces (native "
                       "`optimization` units, `pfn bench run optimization`) are not in the "
                       "database, the thesis import holds final incumbents only")
    sc = C.Sidecar(
        thesis_asset=asset, label=label,
        shown=["one violin per method and budget fraction (" + ", ".join(
            f"{100 * f:.0f} %" for f in pops.fracs) + " of the simulated time budget "
            "15 T min(d, 4)), body = mirrored histogram drawn smooth, line = median",
            "methods: " + ", ".join(c.label for c in cells)],
        metric=["scaled regret (J - J_best) / (J_ESRS - J_best) per run, log10 with display "
                "floor -4 (values below are drawn at the floor), lower is better",
                "J_best: best final cost of the reference strategies on the instance, J_ESRS: "
                "the final incumbent of ESRS"],
        numbers=numbers, data=_source_notes(runs, prov), provenance=prov)
    return sc


# ------------------------------------------------------------------------------- figures
def fig_violin_cross_full(ctx: C.Context) -> C.Output:
    prov = C.Provenance()
    cells = main_cells(ctx.cfg)
    runs = load_runs(ctx, cells, prov)
    _require(runs, ctx.cfg)
    pops = populations(runs, cells, esrs_label(cells, ctx.cfg))
    _check_cells(pops, cells, "main")
    ylim = _ylim(pops)
    fig, axes = C.panel_sheet(1, 1, box_h_pt=98.3, bottom_pt=60.0, top_pt=22.0)
    ax = axes[0, 0]
    allv = [pops.values[("pooled", f, c.label)] for f in pops.fracs for c in cells]
    edges = violin_bins(allv, ylim)
    scale = violin_scale(allv, edges)
    draw_violins(ax, {f: [pops.values[("pooled", f, c.label)] for c in cells]
                      for f in pops.fracs},
                 [style.method_color(c.label, ctx.theme) for c in cells], edges=edges,
                 scale=scale, points=False, ylim=ylim)
    _method_ticks(ax, cells)
    C.finish(ax, ylabel=violin_ylabel(), pad="none", grid_axis="y")
    budget_legend(fig, pops.fracs, ctx.theme)
    sc = _violin_sidecar(f"{ASSET_DIR}/at_the_end/optplay_violin__cross_full.pdf",
                         "fig:sim:optplay-violin-cross-full", pops, cells, ["pooled"], runs,
                         prov)
    return C.Output(sc, fig=fig)


def _method_ticks(ax: Any, cells: Sequence[Cell], fontsize: float = 6.0) -> None:
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([C.tt(c.label) for c in cells], rotation=90, fontsize=fontsize)
    ax.tick_params(axis="x", length=0)


def fig_violin_sheet(ctx: C.Context) -> C.Output:
    prov = C.Provenance()
    cells = main_cells(ctx.cfg)
    runs = load_runs(ctx, cells, prov)
    _require(runs, ctx.cfg)
    pops = populations(runs, cells, esrs_label(cells, ctx.cfg))
    _check_cells(pops, cells, "main")
    panels = ["pooled"] + [p for p in problems(ctx.cfg) if p in runs]
    ylim = _ylim(pops)
    nrows = math.ceil(len(panels) / 2)
    fig, axes = C.panel_sheet(nrows, 2, n=len(panels), row_gap_pt=70.0, bottom_pt=58.0,
                              top_pt=26.0)
    colors = [style.method_color(c.label, ctx.theme) for c in cells]
    for ax, panel in zip([a for a in axes.flat if a is not None], panels):
        allv = [pops.values[(panel, f, c.label)] for f in pops.fracs for c in cells]
        edges = violin_bins(allv, ylim)
        draw_violins(ax, {f: [pops.values[(panel, f, c.label)] for c in cells]
                          for f in pops.fracs}, colors, edges=edges,
                     scale=violin_scale(allv, edges), points=panel != "pooled", ylim=ylim)
        _method_ticks(ax, cells, fontsize=5.0)
        ax.set_title(C.problem_label(panel), fontsize=7.5)
        C.finish(ax, ylabel=violin_ylabel() if ax.get_position().x0 < 0.3 else None,
                 pad="none", grid_axis="y")
    budget_legend(fig, pops.fracs, ctx.theme, y_pt=16.0)
    sc = _violin_sidecar(f"{ASSET_DIR}/at_the_end/optplay_violin__sheet.pdf",
                         "fig:app:optplay-violin-sheet", pops, cells, panels, runs, prov)
    return C.Output(sc, fig=fig)


def _variant_violin(ctx: C.Context, figure: str, asset: str, label: str, xlabel: str,
                    levels_of: Any, fracs: Sequence[float] | None) -> C.Output:
    prov = C.Provenance()
    cells = variant_cells(ctx.cfg, figure)
    if not cells:
        raise C.MissingData(f"cells of figures.{figure} in configs/benchmarks/optimization.yaml")
    runs = load_runs(ctx, cells, prov)
    _require(runs, ctx.cfg)
    pops = populations(runs, cells, esrs_label(main_cells(ctx.cfg), ctx.cfg))
    if fracs is not None:
        pops.fracs = [f for f in pops.fracs if f in fracs]
    _check_cells(pops, cells, figure)
    ylim = _ylim(pops)
    fig, axes = C.panel_sheet(1, 1, bottom_pt=34.0, top_pt=24.0 + 10.5 * len(levels_of(cells)))
    ax = axes[0, 0]
    allv = [pops.values[("pooled", f, c.label)] for f in pops.fracs for c in cells]
    edges = violin_bins(allv, ylim)
    draw_violins(ax, {f: [pops.values[("pooled", f, c.label)] for c in cells]
                      for f in pops.fracs},
                 [cell_color(c, ctx.theme) for c in cells], edges=edges,
                 scale=violin_scale(allv, edges), points=False, ylim=ylim)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c.tick or c.label for c in cells], fontsize=5.5)
    ax.tick_params(axis="x", length=0)
    C.finish(ax, xlabel=xlabel, ylabel=violin_ylabel(), pad="none", grid_axis="y")
    group_headers(ax, levels_of(cells), ctx.theme)
    budget_legend(fig, pops.fracs, ctx.theme)
    # the legend sits above the header stack
    fig.legends[-1].set_bbox_to_anchor(
        (fig.legends[-1].get_bbox_to_anchor().x0 / fig.bbox.width,
         ax.get_position().y1 + (10.5 * len(levels_of(cells)) + 8.0)
         / (fig.get_size_inches()[1] * 72.0)), transform=fig.transFigure)
    sc = _violin_sidecar(asset, label, pops, cells, ["pooled"], runs, prov)
    sc.shown.append("cells: " + "; ".join(
        f"{c.label} = `{c.id}` {c.overrides or ''} (thesis series `{c.thesis}`)" for c in cells))
    return C.Output(sc, fig=fig)


def _levels(cells: Sequence[Cell], depth: int) -> list[list[tuple[str, int, int]]]:
    """Header levels from ``group`` strings ``"A/B"`` (outermost first), contiguous spans."""
    levels: list[list[tuple[str, int, int]]] = []
    for d in range(depth):
        row: list[tuple[str, int, int]] = []
        for i, c in enumerate(cells):
            parts = c.group.split("/")
            key = "/".join(parts[:d + 1])
            name = parts[d] if d < len(parts) else ""
            if row and row[-1][0] == key:
                row[-1] = (key, row[-1][1], i)
            else:
                row.append((key, i, i))
        levels.append([(C.tt(k.split("/")[-1]), a, b) for k, a, b in row if k])
    return levels


def fig_decpfn_thresholds(ctx: C.Context) -> C.Output:
    return _variant_violin(
        ctx, "decpfn_thresholds",
        f"{ASSET_DIR}/decpfn_100_binned/optplay_decpfn_thresholds__violin.pdf",
        "fig:app:optplay-decpfn-thresholds-violin", r"$\tau$",
        lambda cells: _levels(cells, 1), None)


def fig_qx_variants(ctx: C.Context) -> C.Output:
    return _variant_violin(
        ctx, "qx_variants", f"{ASSET_DIR}/ftpfn_xdecpfn/optplay_qx_variants__violin.pdf",
        "fig:app:optplay-qx-variants-violin", r"$\tau$", lambda cells: _levels(cells, 2),
        [1.0])


# ------------------------------------------------------------------------------- curves
def regret_curves(runs: dict[str, ProblemRuns], cells: Sequence[Cell], esrs: str,
                  axis: str, n_points: int) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """``(pct grid, {cell: {problem | "pooled": median scaled regret}})`` over % of the trial
    budget (``axis="trials"``) or of the simulated time budget (``"time"``)."""
    pct = np.linspace(0.0, 100.0, n_points)
    out: dict[str, dict[str, np.ndarray]] = {}
    for c in cells:
        per: dict[str, np.ndarray] = {}
        for p, pr in runs.items():
            recs = pr.records.get(c.label, {})
            best, ref = pr.j_best(), pr.j_esrs(esrs)
            rows = []
            for i in sorted(recs):
                if i not in best or i not in ref:
                    continue
                rec = recs[i]
                if axis == "trials":
                    inc = M.incumbent_vs_trials(rec, pr.trials_axis)
                    k = np.floor(pct / 100.0 * pr.trials_axis + 1e-9).astype(int)
                    J = np.where(k >= 1, inc[np.clip(k - 1, 0, pr.trials_axis - 1)], np.inf)
                else:
                    J = M.incumbent_vs_time(rec, pct / 100.0 * float(rec.t_budget))
                rows.append(M.scaled_regret(J, best[i], ref[i]))
            if rows:
                per[p] = M.problem_median(np.stack(rows))
        if per and len(per) == len(runs):
            with np.errstate(invalid="ignore"):
                per["pooled"] = M.cross_problem_mean(list(per[p] for p in runs))
        out[c.label] = per
    return pct, out


def _native_runs(ctx: C.Context, cells: Sequence[Cell], prov: C.Provenance
                 ) -> dict[str, ProblemRuns]:
    cells = list(cells) + [c for c in all_cells(ctx.cfg) if c.id == "esrs" and c not in cells]
    runs = {}
    for p in problems(ctx.cfg):
        pr = _load_native(ctx.db, ctx.cfg, p, cells, prov)
        if pr is not None and pr.records:
            runs[p] = pr
    return runs


def _display(y: np.ndarray) -> np.ndarray:
    v = M.log10_display(y, FLOOR)
    return np.where(np.isfinite(v), v, np.nan)


def _plot_curves(ax: Any, pct: np.ndarray, curves: dict[str, dict[str, np.ndarray]],
                 cells: Sequence[Cell], panel: str, theme: str) -> dict[str, np.ndarray]:
    drawn = {}
    for c in cells:
        y = curves.get(c.label, {}).get(panel)
        if y is None:
            continue
        yy = _display(y)
        ax.plot(pct, yy, color=style.method_color(c.label, theme), lw=1.0,
                drawstyle="steps-post", label=C.tt(c.label))
        drawn[c.label] = yy
    return drawn


def _method_legend(fig: Any, cells: Sequence[Cell], theme: str, ncol: int = 5,
                   y_pt: float = 4.0) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=style.method_color(c.label, theme), lw=1.2) for c in cells]
    C.legend_above(fig, handles, [C.tt(c.label) for c in cells], ncol=ncol, fontsize=6.0,
                   y_pt=y_pt)


def _at_pct(pct: np.ndarray, y: np.ndarray, p: float) -> float:
    i = int(np.searchsorted(pct, p - 1e-9))
    return float(y[min(i, len(y) - 1)])


def _need_native_curves(runs: dict[str, ProblemRuns], cfg: dict[str, Any], what: str) -> None:
    missing = [p for p in problems(cfg) if p not in runs]
    if missing:
        raise C.MissingData(
            C.missing_units_message("optimization", missing, GROUP_NAME),
            f"{what} need the per-run incumbent traces, the thesis import holds final "
            "incumbents only")


def fig_cross(ctx: C.Context) -> C.Output:
    prov = C.Provenance()
    cells = main_cells(ctx.cfg)
    runs = _native_runs(ctx, cells, prov)
    _need_native_curves(runs, ctx.cfg, "the cross-problem curves")
    esrs = esrs_label(cells, ctx.cfg)
    n_points = int(_figcfg(ctx.cfg, "cross").get("time_points", 300))
    fig, axes = C.panel_sheet(1, 2, sharey=True, top_pt=30.0)
    numbers = []
    for ax, axis, xlabel in zip(axes[0], ("trials", "time"),
                                (f"trials / {C.pct()}", f"sim. time / {C.pct()}")):
        pct, curves = regret_curves(runs, cells, esrs, axis, n_points)
        drawn = _plot_curves(ax, pct, curves, cells, "pooled", ctx.theme)
        C.finish(ax, xlabel=xlabel, ylabel=violin_ylabel() if axis == "trials" else None)
        ax.set_xticks([0, 50, 100])
        marks = (25, 50, 75, 100) if axis == "trials" else (50, 75, 100)
        for lab, y in drawn.items():
            numbers.append(f"{axis}: {lab} " + ", ".join(
                f"{m}% {C.fmt_num(_at_pct(pct, y, m))}" for m in marks))
    _method_legend(fig, cells, ctx.theme)
    sc = C.Sidecar(
        thesis_asset=f"{ASSET_DIR}/at_the_end/optplay_cross.pdf", label="fig:sim:optplay-cross",
        shown=["(a) over % of the trial budget 15 min(d, 4), (b) over % of the simulated time "
               "budget 15 T min(d, 4)", "methods: " + ", ".join(c.label for c in cells)],
        metric=["per instance scaled regret (J - J_best) / (J_ESRS - J_best), median over the "
                "instances of a problem, mean over the problems, log10 last (floor 1e-4), lower "
                "is better", "unsolved instances count as +inf (the curve starts once the "
                "median is finite on every problem)"],
        numbers=numbers, data=_source_notes(runs, prov), provenance=prov)
    return C.Output(sc, fig=fig)


def fig_cross_sheet(ctx: C.Context) -> C.Output:
    prov = C.Provenance()
    cells = main_cells(ctx.cfg)
    runs = _native_runs(ctx, cells, prov)
    if not runs:
        _need_native_curves(runs, ctx.cfg, "the per-problem curves")
    esrs = esrs_label(cells, ctx.cfg)
    n_points = int(_figcfg(ctx.cfg, "cross").get("time_points", 300))
    pct, curves = regret_curves(runs, cells, esrs, "time", n_points)
    panels = (["pooled"] if len(runs) == len(problems(ctx.cfg)) else []) + list(runs)
    nrows = math.ceil(len(panels) / 2)
    fig, axes = C.panel_sheet(nrows, 2, n=len(panels), sharey=True, row_gap_pt=40.0,
                              top_pt=34.0)
    numbers = []
    flat = [a for a in axes.flat if a is not None]
    for k, (ax, panel) in enumerate(zip(flat, panels)):
        drawn = _plot_curves(ax, pct, curves, cells, panel, ctx.theme)
        ax.set_title(C.problem_label(panel), fontsize=7.5)
        last_row = k >= len(panels) - 2
        C.finish(ax, xlabel=f"sim. time / {C.pct()}" if last_row else None,
                 ylabel=violin_ylabel() if k % 2 == 0 else None)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(labelleft=True)
        numbers.append(f"{C.problem_label(panel)} at 100%: " + ", ".join(
            f"{lab} {C.fmt_num(_at_pct(pct, y, 100))}" for lab, y in drawn.items()))
    _method_legend(fig, cells, ctx.theme, y_pt=16.0)
    sc = C.Sidecar(
        thesis_asset=f"{ASSET_DIR}/at_the_end/optplay_cross__sheet.pdf",
        label="fig:app:optplay-cross-sheet",
        shown=["median scaled regret over % of the simulated time budget, per problem and "
               "pooled (mean of the per-problem medians), shared y"],
        metric=["log10 of the median over instances of the scaled regret (floor 1e-4)"],
        numbers=numbers, data=_source_notes(runs, prov), provenance=prov)
    return C.Output(sc, fig=fig)


# ------------------------------------------------------------------------------- crashes
def fig_crashes(ctx: C.Context) -> C.Output:
    prov = C.Provenance()
    cells = main_cells(ctx.cfg)
    fc = _figcfg(ctx.cfg, "crashes")
    plist = [p for p in fc.get("problems", C.CRASH_PROBLEMS) if p in problems(ctx.cfg)]
    n_points = int(fc.get("time_points", 300))
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    sources = {}
    for p in plist:
        pr = _load_native(ctx.db, ctx.cfg, p, cells, prov)
        if pr is not None and pr.records:
            grid = M.time_grid(pr.t_budget, n_points)
            curves[p] = {lab: (100.0 * grid / pr.t_budget,
                               M.mean_crash_count(list(recs.values()), grid))
                         for lab, recs in pr.records.items()}
            sources[p] = "native"
            continue
        pr = _load_thesis(ctx.db, ctx.cfg, p, cells, prov)
        if pr is not None and pr.crash_curves:
            curves[p] = pr.crash_curves
            sources[p] = "thesis"
    if not curves:
        raise C.MissingData(C.missing_units_message("optimization", plist, GROUP_NAME))
    fig, axes = C.panel_sheet(1, len(plist), top_pt=34.0)
    numbers = []
    for k, (ax, p) in enumerate(zip(axes[0], plist)):
        ax.set_title(C.problem_label(p), fontsize=7.5)
        parts = []
        for c in cells:
            if c.label not in curves.get(p, {}):
                continue
            x, y = curves[p][c.label]
            ax.plot(x, y, color=style.method_color(c.label, ctx.theme), lw=1.0,
                    label=C.tt(c.label))
            last = y[np.isfinite(y)]
            if len(last):
                parts.append(f"{c.label} {last[-1]:.2f}")
        numbers.append(f"{C.problem_label(p)} at 100% ({sources.get(p, 'missing')}): "
                       + ", ".join(parts))
        C.finish(ax, xlabel=f"sim. time / {C.pct()}",
                 ylabel=C.better("mean crashes", higher=False) if k == 0 else None)
        ax.set_xticks([0, 50, 100])
    _method_legend(fig, cells, ctx.theme, y_pt=16.0)
    sc = C.Sidecar(
        thesis_asset=f"{ASSET_DIR}/at_the_end/optplay_crashes.pdf",
        label="fig:sim:optplay-crashes",
        shown=["mean number of crashed trials among the trials finished so far, over % of the "
               "simulated time budget, averaged over the runs on all instances",
               "methods: " + ", ".join(c.label for c in cells)],
        metric=["crash count (lower is better)"], numbers=numbers,
        data=[f"{p}: {s}" for p, s in sources.items()]
        + ["ESBO-T-TabPFN is not in the thesis crash aggregates"
           if "thesis" in sources.values() else ""],
        provenance=prov)
    sc.data = [d for d in sc.data if d]
    return C.Output(sc, fig=fig)


# ------------------------------------------------------------------------------- timing
def timing_cells(cfg: dict[str, Any]) -> list[Cell]:
    return [_cell(e) for e in _sec(cfg, "optimization_timing").get("strategies", [])]


def fig_timing(ctx: C.Context) -> C.Output:
    prov = C.Provenance()
    cells = timing_cells(ctx.cfg)
    plist = problems(ctx.cfg)
    split: dict[tuple[str, str], dict[str, float]] = {}
    nruns: dict[tuple[str, str], int] = {}
    for p in plist:
        pr = _load_native(ctx.db, ctx.cfg, p, cells, prov, kind="optimization_timing",
                          budget="timing")
        if pr is None:
            continue
        for c in cells:
            recs = list(pr.records.get(c.label, {}).values())
            if recs:
                split[(c.label, p)] = M.timing_split(recs, {k: name for name, keys, _ in
                                                            TIMING_SEGMENTS for k in keys})
                nruns[(c.label, p)] = len(recs)
    if not split:
        raise C.MissingData(
            C.missing_units_message("optimization_timing", plist, GROUP_NAME),
            "the thesis import holds contended campaign times only")
    span = len(plist) + 1.5
    x = np.array([[mi * span + pi for pi in range(len(plist))] for mi in range(len(cells))])
    stack = np.zeros((len(TIMING_SEGMENTS), len(cells), len(plist)))
    for si, (name, _keys, _col) in enumerate(TIMING_SEGMENTS):
        for mi, c in enumerate(cells):
            for pi, p in enumerate(plist):
                stack[si, mi, pi] = split.get((c.label, p), {}).get(name, 0.0)
    totals = stack.sum(axis=0)
    flat = np.sort(totals.ravel())
    ymax = None
    if flat.size > 2 and flat[-1] > 3.0 * np.percentile(flat, 90):
        big = flat[flat > 3.0 * np.percentile(flat, 90)]
        ymax = 1.1 * float(flat[flat < big.min()].max()) if (flat < big.min()).any() else None
    fig, axes = C.panel_sheet(1, 1, bottom_pt=46.0)
    ax = axes[0, 0]
    bg = C.bg_color(ctx.theme)
    for si, (name, _keys, color) in enumerate(TIMING_SEGMENTS):
        vals = stack[si]
        bottom = stack[:si].sum(axis=0)
        if ymax is not None:
            vals = np.clip(bottom + vals, None, ymax) - np.clip(bottom, None, ymax)
            bottom = np.clip(bottom, None, ymax)
        ax.bar(x.ravel(), vals.ravel(), bottom=bottom.ravel(), width=0.8, color=color,
               edgecolor=bg, linewidth=0.3, label=name, zorder=3)
    if ymax is not None:
        for (mi, pi), tot in np.ndenumerate(totals):
            if tot > ymax:
                ax.text(x[mi, pi], 0.97 * ymax, f"{tot:.1f}", rotation=90, ha="center",
                        va="top", fontsize=4.5, color=bg, zorder=5)
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(0, 1.3 * float(totals.max()) if totals.max() > 0 else 1.0)
    ax.set_xticks(x.ravel())
    ax.set_xticklabels([SHORT_PROBLEM.get(p, p) for _ in cells for p in plist], rotation=90,
                       fontsize=5)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.set_xlim(x.min() - 0.5 - 0.8, x.max() + 0.5 + 0.8)
    C.finish(ax, ylabel="optimizer time / s", pad="none", grid_axis="y")
    C.inner_legend(ax, ncol=2, loc="upper left", fontsize=5)
    from matplotlib.transforms import blended_transform_factory

    below = blended_transform_factory(ax.transData, ax.transAxes)
    for mi, c in enumerate(cells):
        ax.annotate(C.tt(c.label), xy=(x[mi].mean(), 0.0), xycoords=below,
                    xytext=(0.0, -30.0), textcoords="offset points", ha="center", va="top",
                    fontsize=5.5)
    numbers = [f"{c.label}: " + ", ".join(
        f"{SHORT_PROBLEM.get(p, p)} {totals[mi, pi]:.1f} s" for pi, p in enumerate(plist)
        if (c.label, p) in split) for mi, c in enumerate(cells)]
    numbers.append("runs per bar: " + ", ".join(sorted({str(v) for v in nruns.values()})))
    sc = C.Sidecar(
        thesis_asset=f"{ASSET_DIR}/at_the_end/optplay_timing.pdf",
        label="fig:sim:optplay-timing",
        shown=["optimizer decision time per run (simulation excluded), stacked by "
               + ", ".join(n for n, _k, _c in TIMING_SEGMENTS), "one group per method, one bar "
               "per problem" + (", bars above the axis are clipped and labeled with their total"
                                if ymax is not None else "")],
        metric=["seconds summed per run over the strategy's timing categories, mean over the "
                "`optimization_timing` runs (one thread)"],
        numbers=numbers, data=_source_notes({}, prov), provenance=prov)
    return C.Output(sc, fig=fig)


FIGURES = [
    C.FigureSpec("optplay_cross", fig_cross),
    C.FigureSpec("optplay_cross__sheet", fig_cross_sheet),
    C.FigureSpec("optplay_violin__cross_full", fig_violin_cross_full),
    C.FigureSpec("optplay_violin__sheet", fig_violin_sheet),
    C.FigureSpec("optplay_crashes", fig_crashes),
    C.FigureSpec("optplay_timing", fig_timing),
    C.FigureSpec("optplay_decpfn_thresholds__violin", fig_decpfn_thresholds),
    C.FigureSpec("optplay_qx_variants__violin", fig_qx_variants),
]

GROUP = PlotGroup(name=GROUP_NAME, needs=C.spec_needs(GROUP_NAME),
                  render=C.make_render(GROUP_NAME, FIGURES))
