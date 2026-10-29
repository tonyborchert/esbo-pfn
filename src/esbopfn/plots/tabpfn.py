"""Plot group ``tabpfn``: the off-the-shelf TabPFN predictors (fig:sim:tabpfn-panel).

``tabpfn_panel``: median normalized log-likelihood pooled over the five problems against the
median inference time per prediction, one Pareto front over the context sizes per
Trajectory-TabPFN resolution ``N_grid`` in {25, 50, 100, 250} (``n >= 2``, darker for larger
``N_grid``) and per Segment-TabPFN mode (``rate`` dashed, ``split``, ``n >= 4``). The
normalization ``med_D`` uses the participant set of ``prediction_summary`` (the spec's
``participants``), so the panel shares its scale with fig:sim:fig3-cross. Data and timing as in
:mod:`esbopfn.plots.prediction_summary`.
"""

from __future__ import annotations

from typing import Any

from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from esbopfn.plots import PlotGroup
from esbopfn.plots import common as C
from esbopfn.plots import prediction_summary as PS

GROUP_NAME = "tabpfn"
RESOLUTIONS = (25, 50, 100, 250)
TRAJECTORY = tuple(f"tabpfn/{r}" for r in RESOLUTIONS)
SEGMENT = ("segment-tabpfn/rate", "segment-tabpfn/split")
FLOORS = {**{t: 2 for t in TRAJECTORY}, **{t: 4 for t in SEGMENT}}
ALPHA_LO, ALPHA_HI = 0.30, 0.95
YLIM = (-1.0, 2.0)
LW = 1.4


def resolution_alpha(rank: int, count: int) -> float:
    return ALPHA_LO + (ALPHA_HI - ALPHA_LO) * rank / max(count - 1, 1)


def draw_tabpfn_panel(ctx: C.Context) -> C.Output:
    templates = list(TRAJECTORY + SEGMENT)
    board = PS.fig3_board(ctx, templates)
    fr = PS.fronts(board, PS.POOLED, templates, floors=FLOORS)
    if not fr:
        raise C.MissingData("pooled TabPFN scores with inference times")
    fig, axes = C.panel_sheet(1, 1, top_pt=24.0)
    ax = axes[0, 0]
    ax.set_xscale("log")
    ax.set_xlim(*PS.x_limits(fr, YLIM))
    ax.set_ylim(*YLIM)
    PS.log_axis(ax, mult=(1, 2, 5))
    traj = [t for t in TRAJECTORY if t in fr]
    styles: dict[str, dict[str, Any]] = {}
    for i, t in enumerate(traj):
        styles[t] = dict(alpha=resolution_alpha(i, len(traj)), lw=LW, zorder=2)
    styles["segment-tabpfn/rate"] = dict(ls="--", lw=LW, zorder=3)
    styles["segment-tabpfn/split"] = dict(lw=LW, zorder=3)
    PS.front_lines(ax, fr, ctx.theme, styles=styles)
    for t, f in fr.items():           # a one-point front is a short tick at its point
        if len(f) == 1:
            ax.plot(f.seconds, f.y, "o", ms=2.5, color=C.model_color(t, ctx.theme))
    C.finish(ax, xlabel="inference time / s", ylabel=C.better(PS.YLABEL, higher=True),
             pad="none")
    ax.set_yticks([-1, 0, 1, 2])
    PS.spines_on_top(ax)
    tcol = C.model_color("tabpfn/50", ctx.theme)
    ramp = [resolution_alpha(i, 3) for i in range(3)]
    handles: list[Any] = [tuple(Line2D([], [], color=tcol, lw=LW, alpha=a) for a in ramp),
                          Line2D([], [], color=C.model_color(SEGMENT[0], ctx.theme), lw=LW,
                                 ls="--"),
                          Line2D([], [], color=C.model_color(SEGMENT[1], ctx.theme), lw=LW),
                          tuple(Line2D([], [], color="#555555", lw=LW, alpha=a) for a in ramp)]
    labels = ["Trajectory-TabPFN", "Segment-TabPFN (rate)", "Segment-TabPFN (split)",
              r"fainter = lower resolution $N_{\mathrm{grid}}$"]
    C.legend_above(fig, handles, labels, ncol=4, y_pt=6.0, fontsize=6.5,
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)}, handlelength=2.4)
    sc = C.Sidecar("simulation/prediction/tabpfn_stenger_tabpfn/tabpfn_panel.pdf",
                   label="fig:sim:tabpfn-panel")
    sc.shown = ["one Pareto front over the context sizes (staircase, steps-post) per series: "
                + ", ".join(f"`{t}`" for t in templates if t in fr),
                "Trajectory-TabPFN n >= 2 with alpha rising from "
                f"{ALPHA_LO:.2f} (N_grid 25) to {ALPHA_HI:.2f} (N_grid 250), Segment-TabPFN n >= 4, "
                "rate dashed, lines only"]
    sc.metric = ["y: median normalized conditioned LL on non-crashed queries pooled across the "
                 "five problems (higher is better), range [-1, 2]",
                 "x: median inference seconds per prediction, median across problems (log)"]
    sc.numbers = PS.summary_lines(board, PS.POOLED, fr)
    sc.data = [f"source: {board.source}"] + board.notes
    sc.provenance = board.prov
    return C.Output(sc, fig=fig)


FIGURES: list[C.FigureSpec] = [C.FigureSpec("tabpfn_panel", draw_tabpfn_panel)]

GROUP = PlotGroup(name=GROUP_NAME, needs=C.spec_needs(GROUP_NAME),
                  render=C.make_render(GROUP_NAME, FIGURES))
