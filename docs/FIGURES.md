# Figures

Every result figure is drawn by a plot group from the benchmark database
(`$PFN_HOME/benchmarks/db.sqlite`). A plot group is a module `src/esbopfn/plots/<group>.py`
exposing `GROUP = PlotGroup(name, needs, render)` with its configuration in
`src/esbopfn/configs/benchmarks/<group>.yaml`. `needs(cfg, db)` lists the benchmark units the
figures read, `render(cfg, db, out_dir, fmt, theme)` writes the figures. Units are filled by
`pfn bench run <group>` (simulation), by `pfn hw run` and `pfn hw bench-pred` (rig), or by the
thesis importers (`pfn import thesis-results`, `pfn hw import-logs`, `pfn hw import-pred`), which
store the same kinds with `provenance: thesis`.

## Usage

```bash
pfn plot --list                        # registered groups
pfn plot hardware                      # -> $PFN_HOME/figures/hardware/
pfn plot hardware --format png --theme black --out /tmp/figs   # -> /tmp/figs/hardware/
pfn bench plan prediction_summary      # the units a group needs and which are missing
pfn bench run prediction_summary --shard 0/8
```

- `--format pdf|png` (default `pdf`, png at 300 dpi), `--theme white|black` (black swaps only
  background and foreground, the method colors stay), `--out DIR` writes into `DIR/<group>/`.
- Style (`plots/style.py`): Charter through LaTeX when a TeX installation with `mathdesign` is
  found (mathtext otherwise), 9 pt, text width 6.3 in, dashed light grid, one color per method
  family (`style.METHOD_COLORS`), thesis display names (`style.DISPLAY_NAME`).
- Every figure gets a sidecar `<name>.md` beside it with four sections: what is shown, the metric,
  the data (unit kinds, key hashes, provenance, rig, methods, seeds) and the numbers the figure is
  drawn from (medians per method, per-seed values, reference values such as `J_best`).

## Catalog

"Local" means the figure renders from the thesis import on a laptop without running any
benchmark. "`--raw`" marks figures that need the cluster copies imported with
`pfn import thesis-results --raw DIR` ([BENCHMARKS.md](BENCHMARKS.md#importing-the-thesis-results)).
"Units" are the benchmark kinds the figure reads.

### Priors (`priors`, `pfn bench run priors`)

| Thesis figure | Name | Units | Local |
|---|---|---|---|
| fig:setup:prior-samples | `priors_samples` | `instances`, live rollouts | yes (cheap live rollouts) |
| fig:app:prior-landscapes | `priors_landscapes` | `instances`, `landscapes` | yes (thesis landscape cache, `--raw`) |

### Single-trajectory predictors (`single_trajectory`)

| Thesis figure | Name | Units | Local |
|---|---|---|---|
| fig:sim:lcpfn-decpfn-example | `lcpfn_decpfn_example` | live inference (checkpoints) | with the checkpoints (native or imported thesis weights) |
| fig:sim:lcpfn-decpfn-heads | `lcpfn_decpfn_heads_benchmark` | `fields`, `prediction`, `prediction_timing` | yes (thesis aggregates) |
| fig:sim:lcpfn-decpfn-stop-sheet | `lcpfn_decpfn_stop_sheet` | `fields`, `prediction` | yes (thesis aggregates and thesis predictions, `--raw`) |
| fig:app:hpo-gridsearch | `gridsearch_panels` | `$PFN_HOME/hpo` results | yes (thesis HPO tables, `--raw`) or after `pfn hpo` |

### Off-the-shelf TabPFN (`tabpfn`)

| Thesis figure | Name | Units | Local |
|---|---|---|---|
| fig:sim:tabpfn-panel | `tabpfn_panel` | `fields`, `prediction`, `prediction_timing` | yes (thesis aggregates) |

### Cross-trajectory predictors (`cross_trajectory`)

| Thesis figure | Name | Units | Local |
|---|---|---|---|
| fig:sim:context-sweep | `context_ladder` | `fields`, `prediction_ladder` | yes |
| fig:sim:ftpfn-xdecpfn-example | `ftpfn_xdecpfn_example` | live inference (checkpoints) | with the checkpoints |
| fig:sim:landscape-furuta | `landscape_furuta` | live inference, `landscapes` | with the checkpoints |
| fig:sim:ftpfn-xdecpfn-stop-sheet | `ftpfn_xdecpfn_stop_sheet` | `fields`, `prediction` | yes (thesis aggregates) |

### Prediction summary (`prediction_summary`)

| Thesis figure | Name | Units | Local |
|---|---|---|---|
| fig:sim:fig3-cross | `fig3_cross` | `prediction`, `prediction_timing`, `thesis_series` | yes |
| fig:app:fig3-sheet | `fig3_sheet` | same | yes |
| fig:sim:fig2-sheet | `fig2_sheet` | `prediction`, `thesis_series` | yes (DecPFN-no-St from the thesis predictions, `--raw`) |

### Optimization (`optimization`)

| Thesis figure | Name | Units | Local |
|---|---|---|---|
| fig:sim:optplay-cross | `optplay_cross` | `instances`, `optimization` | yes (thesis runs, `--raw`) |
| fig:app:optplay-cross-sheet | `optplay_cross__sheet` | same | yes (thesis runs, `--raw`) |
| fig:sim:optplay-violin-cross-full | `optplay_violin__cross_full` | `optimization`, `thesis_series` | yes at 100 %, at 50 % with the thesis runs (`--raw`) |
| fig:app:optplay-violin-sheet | `optplay_violin__sheet` | same | yes at 100 %, at 50 % with the thesis runs (`--raw`) |
| fig:sim:optplay-crashes | `optplay_crashes` | `optimization`, `thesis_series` | yes |
| fig:sim:optplay-timing | `optplay_timing` | `optimization_timing` | yes (one-thread thesis timing tree, `--raw`) |
| fig:app:optplay-decpfn-thresholds-violin | `optplay_decpfn_thresholds__violin` | `optimization` | yes at 100 % |
| fig:app:optplay-qx-variants-violin | `optplay_qx_variants__violin` | `optimization` | yes at 100 % |

### Generation speed (`generation_speed`)

| Thesis table | Name | Units | Local |
|---|---|---|---|
| tab:app:prior-cost | `prior_cost_table` (`.md` and `.tex`) | `generation_speed` (C++ binaries) | after `pfn data build` and `pfn bench run generation_speed` |
| tab:setup:model-cost | `model_cost_table` (`.md` and `.tex`) | the model tree (`metrics.json`, `model.pt`) | for the models present, imported or trained |

### Hardware (`hardware`, `src/esbopfn/plots/hardware.py`)

All hardware figures read the rig runs (`hardware_optimization`, one unit per rig, method and
seed, blob `run.json` + `trials.jsonl`) and the hardware prediction scores (`hardware_prediction`,
blob `scores.parquet` + `summary.json`) of the rigs in `configs/benchmarks/hardware.yaml`
(`rigs`, registry `configs/rigs.yaml`). Every number goes through `esbopfn.hardware.metrics`.
`optimization.provenance` and `prediction.provenance` select the imported thesis units (`thesis`)
or the units run with this code (`native`). All of them are local from the thesis import.

| Thesis figure | Name | Units | Sidecar numbers |
|---|---|---|---|
| fig:hw:optimization-without-weight / -with-weight | `hwplay_optimization__<rig>` | `hardware_optimization` | `J_best` with its method and seed, reset normalization, per seed final regret, crash counts, trials, run end |
| fig:hw:regret-violin | `hwplay_regret_violin` | `hardware_optimization` | per seed log10 regret at 50 % and 100 %, excluded `J_best` seed |
| fig:hw:cost-early | `hwplay_cost_1000` | `hardware_optimization` | median best cost per method at every 100 s |
| fig:hw:best-controller-without-weight / -with-weight | `hwplay_best_controller__500s__<rig>` | `hardware_optimization` | median seed, trial, cost, gains, ranked incumbent costs of all seeds |
| fig:app:hw-acq-wait | `hwplay_acq_wait` | `hardware_optimization` | mean and per-seed acquisition wait |
| fig:hw:pred-benchmark | `hwpred_benchmark` | `hardware_prediction` (ladder) | median LL and seconds per context size, drawn Pareto fronts, hidden glyphs |
| fig:hw:pred-ablation-bars | `hwpred_ablation_bars` | `hardware_prediction` (ablation) | median LL and query count per DecPFN variant |

Notes on the hardware figures:

- Wall-clock is reset-normalized: every trial's reset time is replaced by the mean reset time of
  the rig's `reset_reference_method` (DecPFN-H). `J_best` is the best successful cost recorded on
  the rig over the shown methods. Regret is unscaled `log10(J^- - J_best)`.
- `hwplay_optimization` panel (b) runs over the first `metrics.trials_axis` trials (60, as the
  caption states). Only the first rig carries the legend (`figures.legend_on_first_rig_only`).
- `hwpred_benchmark` plots `figures.ladder_metric` (`median_ll_all`, crashed queries included, as
  the thesis ladder medians) and drops medians below `figures.pred_y_cutoff` (-7). DecPFN never
  reads the context and is drawn as one pooled point. Among overlapping circles the higher
  context size stays, each series keeps its best point, the Pareto line runs through every
  surviving rung.
- `hwpred_ablation_bars` plots `figures.ablation_metric` (`median_ll`, non-crashed queries) with
  the y axis starting at the shared floor, not at 0.

## README art (`readme`, `src/esbopfn/plots/readme_art.py`)

The README images live in `assets/`, each as PNG at twice the screen resolution and as SVG, a
light (`<name>.png`) and a dark (`<name>_dark.png`) variant, and a sidecar `<name>.md`:

| Image | Content | Inputs |
|---|---|---|
| `hero_3d` | 3D view over `t`, `J_t` and one controller coordinate `theta`: three real rig trials (BO on the rig without weight, seed 4 trial 25 complete, seed 1 trial 12 and seed 0 trial 3 shown stopped early with dotted drop lines) as the only lines, the predicted total cost at `t = T` of the thesis Furuta XDecPFN along a 1-D slice (nested fills mu +- sigma and mu +- 2 sigma), a star at the next controller (argmax of the probability of improvement over the slice) and the predicted crash probability as a flat red graph above the back wall | `hardware_optimization` units, checkpoint `xdecpfn/furuta/thesis/default` (`predict_unrun`) |
| `problems_strip` | one set of cost trajectories per problem (random controllers on the nominal plant, crashes marked) | live rollouts |
| `results_cards` | headline numbers: pooled prediction score at 64 context trials, simulation regret at 100 % time, hardware regret and crash counts | `thesis_series` (variant `thesis`), `hardware_optimization` |

The slice of `hero_3d` runs along the box dimension in which the predicted mean varies most, with
the other coordinates at the incumbent of the context (the three shown trials plus six more trials
of the seed 4 run), every trial sits at its own coordinate on that dimension. The view (`HERO_VIEW`,
`HERO_BOX`) follows the wall view of the presentation slide. Every number on the cards comes from
the database, a missing unit leaves its row out and is listed in the sidecar.

Pipeline (the inputs come from the imports of [BENCHMARKS.md](BENCHMARKS.md#importing-the-thesis-results)
and [HARDWARE.md](HARDWARE.md#importing-the-thesis-logs)):

```bash
pfn hw import-logs <thesis project>/impl/prediction/hwplay/runs_holdcost5_newbox_reset_fixed
pfn import thesis-results --from <thesis project>      # the thesis series of the result cards
pfn import thesis-models --from <checkpoint dir>       # the XDecPFN checkpoint of hero_3d
python -m esbopfn.plots.readme_art assets              # writes assets/*.png|svg|md
pfn plot readme --out /tmp/art                         # the same into /tmp/art/readme/
```

## Rendered figures in `assets/figures/`

The README embeds PNG renders of the figure groups that draw from the thesis import, each with its
sidecar, and the method schematics of the thesis converted to PNG:

```bash
pfn plot prediction_summary optimization hardware cross_trajectory --format png --out assets/figures
pfn plot single_trajectory --format png --out /tmp/st   # then copy lcpfn_decpfn_example.png and .md
for s in xdecpfn_training decpfn_training lcpfn_training; do
  pdftoppm -png -r 200 -singlefile <thesis>/assets/$s.pdf assets/figures/schematics/$s
done
```

The live-inference figures cache their simulated runs and model predictions under
`$PFN_HOME/figures/_cache/<group>/`, never in the output directory, so `--out` receives only
figures, sidecars and `skipped.txt`. The live figures take the native model first and fall back
to the imported thesis weights (`pfn import thesis-models`, see
[LEGACY_MODELS.md](LEGACY_MODELS.md)). The sidecars hold `$PFN_HOME` in place of local paths. Only
figures that render from data in the database are kept there. The groups list the figures
they skip and why in `skipped.txt` next to the rendered ones (for example the optimization curves,
which need per-run records).
